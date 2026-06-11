# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

"""ParallelDims — fail-fast parallel configuration validator + mesh builder.

Centralises parallel-degree validation in a single dataclass so
misconfigurations are caught before model construction.

What this provides:

1. **Inference** — auto-fill ``dp`` (or ``dp_shard=-1``) from the product
   constraint ``dp_replicate * dp_shard * cp * tp * pp == world_size``.

2. **Validation against world_size** — raises ``ValueError`` with a clear
   message when the product mismatches.

3. **Validation against the model** — checks divisibility constraints that
   would otherwise crash deep inside ``parallelize_module``:

   - ``num_attention_heads % tp == 0`` (TP shards heads)
   - ``num_key_value_heads % tp == 0`` (GQA constraint)
   - ``num_experts % ep == 0`` (when MoE)
   - ``ulysses_degree <= cp`` and ``cp % ulysses_degree == 0``
   - ``seq_len % (cp * tp) == 0`` (sequence parallel + CP)
   - ``etp == tp or etp == 1`` (expert TP rule)

4. **Mesh building** — returns a ``DeviceMesh`` with the canonical dim order
   ``dp_replicate → dp_shard → ep → cp → tp → pp``. Backwards compatible with
   the legacy single ``dp`` field (auto-collapses to ``dp_shard``).

User experience:

- Default config (no parallel section) — works on 1 GPU, runs as DDP-1.
- Set only ``tp=4`` on world_size=8 → ``dp`` auto-inferred to 2.
- Set ``dp_shard=-1`` → fills remaining cards into FSDP shard dim.
- Misconfig (heads=12, tp=8) → fails at ``_setup`` with a single readable
  error before any model parallelization is attempted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from hyper_parallel import init_device_mesh

logger = logging.getLogger(__name__)


@dataclass
class ParallelDims:
    """Validated parallel degrees + lazy mesh builder.

    Attributes:
        dp_replicate: DDP replication degree (HSDP outer dim).
        dp_shard: FSDP shard degree. ``-1`` means "fill the rest from
            ``world_size / (dp_replicate * cp * tp * pp)``".
        cp: Context parallel degree.
        tp: Tensor parallel degree (dense path).
        pp: Pipeline parallel degree.
        ep: Expert parallel degree (MoE only).
        etp: Expert tensor parallel degree. Must equal ``tp`` or ``1``.
        ulysses_degree: Ulysses sub-degree inside ``cp``. ``None`` means
            "pure Ulysses (degree == cp)".
        world_size: Total number of ranks.
    """

    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    etp: int = 1
    ulysses_degree: Optional[int] = None
    world_size: int = 1
    # Cached after build_mesh.
    _device_mesh: object = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction & inference
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, parallel_cfg, world_size: int) -> "ParallelDims":
        """Build from a ``ParallelConfig`` (or any object with the same fields).

        Accepts the legacy single-``dp`` field. If ``dp`` is set and
        ``dp_replicate``/``dp_shard`` are at default, ``dp`` is mapped to
        ``dp_shard`` (FSDP behavior).
        """
        dp_replicate = getattr(parallel_cfg, 'dp_replicate', 1)
        dp_shard = getattr(parallel_cfg, 'dp_shard', None)
        legacy_dp = getattr(parallel_cfg, 'dp', None)

        # Backward-compat: legacy ``dp`` maps to ``dp_shard`` when both
        # dp_replicate/dp_shard fields are at defaults.
        if dp_shard is None:
            dp_shard = legacy_dp if legacy_dp is not None else 1

        return cls(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            cp=getattr(parallel_cfg, 'cp', 1),
            tp=getattr(parallel_cfg, 'tp', 1),
            pp=getattr(parallel_cfg, 'pp', 1),
            ep=getattr(parallel_cfg, 'ep', 1),
            etp=getattr(parallel_cfg, 'etp', getattr(parallel_cfg, 'tp', 1)),
            ulysses_degree=getattr(parallel_cfg, 'ulysses_degree', None),
            world_size=world_size,
        )

    def __post_init__(self) -> None:
        self._infer_and_validate()

    def _infer_and_validate(self) -> None:
        """Auto-fill ``dp_shard=-1`` then validate ``product == world_size``."""
        for name, value in (
            ("dp_replicate", self.dp_replicate),
            ("cp", self.cp),
            ("tp", self.tp),
            ("pp", self.pp),
            ("ep", self.ep),
            ("etp", self.etp),
        ):
            if value < 1:
                raise ValueError(f"Parallel degree {name}={value} must be >= 1")

        if self.dp_shard < -1 or self.dp_shard == 0:
            raise ValueError(
                f"dp_shard={self.dp_shard} must be -1 (auto) or a positive int"
            )

        # Auto-infer dp_shard when -1.  ep is an independent peer mesh dim
        # (see build_mesh) so it does NOT reduce the dp pool.
        if self.dp_shard == -1:
            non_dp = self.dp_replicate * self.cp * self.tp * self.pp * self.ep
            if self.world_size % non_dp != 0:
                raise ValueError(
                    f"Cannot auto-infer dp_shard: world_size={self.world_size} "
                    f"is not divisible by dp_replicate*cp*tp*pp*ep={non_dp}"
                )
            self.dp_shard = max(self.world_size // non_dp, 1)
            logger.info_rank0(
                "Auto-inferred dp_shard=%d (world_size=%d / dp_replicate=%d * "
                "cp=%d * tp=%d * pp=%d * ep=%d)",
                self.dp_shard, self.world_size,
                self.dp_replicate, self.cp, self.tp, self.pp, self.ep,
            )

        product = (
            self.dp_replicate * self.dp_shard
            * self.cp * self.tp * self.pp * self.ep
        )
        if product != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * "
                f"dp_shard({self.dp_shard}) * cp({self.cp}) * tp({self.tp}) * "
                f"pp({self.pp}) * ep({self.ep}) = {product} != "
                f"world_size({self.world_size}). Set dp_shard=-1 to auto-infer."
            )

        # ep is an independent peer mesh dim alongside dp/cp/tp/pp (see build_mesh).
        # It does NOT need to divide the dp_shard*cp pool; it occupies its own
        # mesh axis.  We only enforce the expert-TP compatibility rule below.
        if self.ep > 1:
            if self.etp not in (self.tp, 1):
                raise ValueError(
                    f"etp={self.etp} must equal tp={self.tp} or 1 "
                    f"(expert tensor-parallel must align with TP or be inactive)"
                )

        # No model has implemented PP wiring yet — fail fast.
        if self.pp > 1:
            raise NotImplementedError(
                f"Pipeline parallel (pp={self.pp}>1) is not yet implemented "
                "for any model. Set pp=1 or add a per-model PP path in "
                "models/<name>/parallelize.py before enabling."
            )

        # Ulysses must divide cp.
        if self.ulysses_degree is not None and self.cp > 1:
            if self.ulysses_degree > self.cp:
                raise ValueError(
                    f"ulysses_degree={self.ulysses_degree} must be <= "
                    f"cp={self.cp}"
                )
            if self.cp % self.ulysses_degree != 0:
                raise ValueError(
                    f"cp={self.cp} must be divisible by "
                    f"ulysses_degree={self.ulysses_degree}"
                )

    # ------------------------------------------------------------------
    # Validate against an actual model
    # ------------------------------------------------------------------
    def validate_against_model(
        self,
        model,
        seq_len: Optional[int] = None,
    ) -> None:
        """Cross-check parallel degrees against a built model's hyperparams.

        Reads ``model.config`` for standard transformer fields. Skips silently
        if a field is absent. Model-specific validation (e.g. "TP unsupported
        for linear-attn layers") is inlined at the top of each
        ``parallelize_<model>()`` function — convention.

        Args:
            model: The built ``nn.Module`` (must expose ``.config`` to trigger
                most checks).
            seq_len: Optional maximum sequence length used for cp/tp
                divisibility checks.

        Raises:
            ValueError: With a single readable message when a constraint is
                violated. Stops here so the user sees the real cause instead
                of a stack trace from inside ``parallelize_module``.
        """
        cfg = getattr(model, 'config', None)
        if cfg is None:
            return

        heads = getattr(cfg, 'num_attention_heads', None)
        if heads is not None and self.tp > 1 and heads % self.tp != 0:
            raise ValueError(
                f"num_attention_heads={heads} not divisible by tp={self.tp}. "
                f"Pick tp from the divisors of {heads}."
            )

        kv_heads = getattr(cfg, 'num_key_value_heads', None)
        if kv_heads is not None and self.tp > 1 and kv_heads % self.tp != 0:
            raise ValueError(
                f"num_key_value_heads={kv_heads} not divisible by tp={self.tp} "
                f"(GQA constraint). Pick tp from the divisors of {kv_heads}."
            )

        num_experts = getattr(cfg, 'num_experts', None)
        if num_experts is not None and self.ep > 1 and num_experts % self.ep != 0:
            raise ValueError(
                f"num_experts={num_experts} not divisible by ep={self.ep}. "
                f"Pick ep from the divisors of {num_experts}."
            )

        if seq_len is not None and self.cp * self.tp > 1:
            divisor = self.cp * self.tp
            if seq_len % divisor != 0:
                raise ValueError(
                    f"max_seq_len={seq_len} not divisible by cp*tp={divisor}. "
                    f"Increase seq_len to a multiple of {divisor} or reduce "
                    f"cp/tp."
                )

    # ------------------------------------------------------------------
    # Mesh building
    # ------------------------------------------------------------------
    def build_mesh(self, device_type: str):
        """Build the DeviceMesh with canonical dim order and named flatten aliases.

        Order of base dims: ``dp_replicate → dp_shard → ep → cp → tp → pp``.
        Only base dims with degree > 1 are materialized; if all are 1, a 1D
        ``dp_shard`` mesh of the world is created so the FSDP code path runs
        unchanged on single-card.

        After construction, the following flatten aliases are registered on
        the root mesh so callers can reach them with ``mesh["fsdp"]`` /
        ``mesh["dp"]`` regardless of the underlying parallel composition:

            ``"fsdp"``  – mesh used for ``fully_shard`` / reduce-scatter.
                          Always equals the ``dp_shard`` axis.
            ``"dp"``    – combined data-parallel mesh used for loss / token
                          all-reduce. ``dp_replicate × dp_shard`` when both
                          are >1 (HSDP); otherwise the single non-trivial
                          DP axis (or ``dp_shard`` for the 1-card case).

        Args:
            device_type: Backend device string (``"npu"`` / ``"cuda"``).

        Returns:
            ``DeviceMesh`` instance.
        """
        dims = []
        names = []
        for name, size in (
            ("dp_replicate", self.dp_replicate),
            ("dp_shard", self.dp_shard),
            ("ep", self.ep),
            ("cp", self.cp),
            ("tp", self.tp),
            ("pp", self.pp),
        ):
            if size > 1:
                dims.append(size)
                names.append(name)

        if not dims:
            dims = [self.world_size]
            names = ["dp_shard"]

        self._device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple(dims),
            mesh_dim_names=tuple(names),
        )
        self._register_flatten_aliases(names)
        logger.info_rank0(
            "DeviceMesh built: shape=%s, names=%s",
            tuple(dims), tuple(names),
        )
        return self._device_mesh

    def _register_flatten_aliases(self, base_names) -> None:
        """Register named flatten aliases on the root mesh.

        These aliases give the rest of the trainer a stable, intent-named
        handle on combined parallel axes so callers never need to fall back
        to the whole mesh:

            ``"fsdp"`` – the axis FSDP shards along (= ``dp_shard``).
            ``"dp"``   – combined data-parallel mesh (replicate × shard).
                         Used for grad/optimizer-state replication accounting.
            ``"loss"`` – the mesh over which loss / token counts are
                         all-reduced. Equals ``dp × cp`` when CP is enabled
                         (CP-sharded ranks see different sub-sequences and
                         must contribute their token counts to the global
                         denominator); otherwise equals ``dp``.

        Reserved names (intentionally not registered yet):
            ``"efsdp"`` – FSDP mesh for expert layers when EP > 1. Will
                          fold ``dp_shard / ep`` once real EP lands.
            ``"etp"``   – expert TP mesh (= ``ep × tp`` composition)
                          alongside dense ``tp``. Same gate.
            ``"batch"`` – per-DP batch dispatch mesh; today identical to
                          ``"dp"``, will diverge if we ever support
                          microbatch-sharded scheduling.

        Idempotent: every flatten call is gated on whether the alias is
        already on the root mesh, so repeated ``build_mesh`` calls are
        safe.

        Args:
            base_names: Sequence of base mesh-dim names that were materialized
                (degree > 1, plus the degenerate ``dp_shard`` of size 1 when
                no other dim was present).
        """
        # pylint: disable=protected-access
        mesh = self._device_mesh
        existing = set(mesh.mesh_dim_names or ())
        flatten_keys = set(mesh._get_root_mesh().get_flatten_mapping().keys())

        def _flatten_unique(source_dims, alias):
            if alias in existing or alias in flatten_keys:
                return
            mesh[source_dims].flatten(alias)
            flatten_keys.add(alias)

        has_replicate = "dp_replicate" in base_names
        has_shard = "dp_shard" in base_names
        has_cp = "cp" in base_names

        # ``fsdp`` — the axis ``fully_shard`` actually shards along.
        if has_shard:
            _flatten_unique("dp_shard", "fsdp")

        # ``dp`` — combined replicate×shard data-parallel mesh.
        if has_replicate and has_shard:
            _flatten_unique(("dp_replicate", "dp_shard"), "dp")
        elif has_replicate:
            _flatten_unique("dp_replicate", "dp")
        elif has_shard:
            _flatten_unique("dp_shard", "dp")

        # ``loss`` — dp folded with cp when context parallelism is active so
        # loss / token counts include CP-sharded contributions.
        if has_cp:
            if has_replicate and has_shard:
                _flatten_unique(("dp_replicate", "dp_shard", "cp"), "loss")
            elif has_replicate:
                _flatten_unique(("dp_replicate", "cp"), "loss")
            elif has_shard:
                _flatten_unique(("dp_shard", "cp"), "loss")
            else:
                _flatten_unique("cp", "loss")
        else:
            # No CP — ``loss`` and ``dp`` are the same group. Re-flatten
            # the existing 1D dp mesh under the ``loss`` alias so both
            # names resolve via ``__getitem__``.
            if "loss" not in flatten_keys and "dp" in flatten_keys:
                mesh["dp"].flatten("loss")
                flatten_keys.add("loss")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def dp_size(self) -> int:
        """Combined data-parallel size = dp_replicate * dp_shard."""
        return self.dp_replicate * self.dp_shard

    @property
    def non_dp_size(self) -> int:
        """Product of model-side parallel dims (tp*cp*pp*ep)."""
        return self.tp * self.cp * self.pp * self.ep

    @property
    def tp_enabled(self) -> bool:
        """Return True if tensor parallelism is enabled (tp > 1)."""
        return self.tp > 1

    @property
    def cp_enabled(self) -> bool:
        """Return True if context parallelism is enabled (cp > 1)."""
        return self.cp > 1

    @property
    def ep_enabled(self) -> bool:
        """Return True if expert parallelism is enabled (ep > 1)."""
        return self.ep > 1

    @property
    def pp_enabled(self) -> bool:
        """Return True if pipeline parallelism is enabled (pp > 1)."""
        return self.pp > 1

    @property
    def fsdp_enabled(self) -> bool:
        """FSDP is on whenever there's a shard dim or HSDP outer dim."""
        return self.dp_shard > 1 or self.dp_replicate > 1

    def summary(self) -> str:
        """Compact one-line summary for logging."""
        return (
            f"dp_replicate={self.dp_replicate} dp_shard={self.dp_shard} "
            f"cp={self.cp} tp={self.tp} pp={self.pp} ep={self.ep} "
            f"etp={self.etp} | dp={self.dp_size} world={self.world_size}"
        )

__all__ = ["ParallelDims"]
