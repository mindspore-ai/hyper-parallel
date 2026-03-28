# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Unified Context Parallel: Pure Ulysses, Pure Colossal AI, and Hybrid CP."""
from functools import partial
from typing import Optional

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.parallel_style import ParallelStyle
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module
Tensor = platform.Tensor


# ---------------------------------------------------------------------------
# Low-level communication primitives
# ---------------------------------------------------------------------------

def _ensure_1d(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return a 1-D DeviceMesh (flatten if multi-dimensional)."""
    if device_mesh.ndim == 1:
        return device_mesh
    ranks = list(device_mesh.rank_list)
    return DeviceMesh(device_mesh.device_type, ranks, mesh_dim_names=("cp",))


def _build_2d_mesh(device_mesh: DeviceMesh, ds: int, co: int) -> DeviceMesh:
    """Build or validate a 2-D ``(co × ds)`` DeviceMesh for Hybrid CP.

    If *device_mesh* is already 2-D it is returned as-is (must have
    ``mesh_dim_names`` set).  Otherwise the ranks of the 1-D mesh are tiled
    into *co* rows of *ds* adjacent ranks each.
    """
    if device_mesh.ndim == 2:
        if not device_mesh.mesh_dim_names:
            raise ValueError(
                "2-D device_mesh for Hybrid CP must have mesh_dim_names=(\"co\", \"ds\")."
            )
        return device_mesh
    ranks = list(device_mesh.rank_list)
    return DeviceMesh(
        device_mesh.device_type,
        [ranks[i * ds:(i + 1) * ds] for i in range(co)],
        mesh_dim_names=("co", "ds"),
    )


def _scatter_seq_to_head(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
    head_dim: int,
    submesh_size: int,
) -> "DTensor":
    """All-to-all: ``Shard(seq_dim)`` → ``Shard(head_dim)``. Returns DTensor."""
    if isinstance(tensor, DTensor):
        return tensor.redistribute(submesh, (Shard(head_dim),))
    if tensor.shape[head_dim] % submesh_size != 0:
        raise ValueError(
            f"num_heads ({tensor.shape[head_dim]}) must be divisible by "
            f"ulysses_degree ({submesh_size})."
        )
    return DTensor.from_local(tensor, submesh, (Shard(seq_dim),)).redistribute(
        submesh, (Shard(head_dim),)
    )


def _gather_head_to_seq(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
    head_dim: int,
) -> "DTensor":
    """Reverse all-to-all: ``Shard(head_dim)`` → ``Shard(seq_dim)``. Returns DTensor."""
    if isinstance(tensor, DTensor):
        return tensor.redistribute(submesh, (Shard(seq_dim),))
    return DTensor.from_local(tensor, submesh, (Shard(head_dim),)).redistribute(
        submesh, (Shard(seq_dim),)
    )


def _gather_seq(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
) -> "DTensor":
    """All-gather: ``Shard(seq_dim)`` → ``Replicate``. Returns DTensor."""
    if isinstance(tensor, DTensor):
        return tensor.redistribute(submesh, (Replicate(),))
    return DTensor.from_local(tensor, submesh, (Shard(seq_dim),)).redistribute(
        submesh, (Replicate(),)
    )




# ---------------------------------------------------------------------------
# Unified ContextParallel
# ---------------------------------------------------------------------------

class ContextParallel(ParallelStyle):
    """Unified Context Parallel for core-attention modules.

    Three modes controlled by ``ulysses_degree``:

    +-----------------+--------------------+------------------------------------------+
    | Mode            | ``ulysses_degree`` | Mechanism                                |
    +=================+====================+==========================================+
    | Pure Ulysses    | ``None`` (default) | seq→head A2A before attn;                |
    |                 | (= cp_size)        | head→seq A2A after.                      |
    |                 |                    | Requires ``num_heads % cp_size == 0``.   |
    +-----------------+--------------------+------------------------------------------+
    | Pure Colossal AI| ``1``              | Q stays as local Shard(seq);             |
    |                 |                    | K/V all-gathered (Replicate).            |
    |                 |                    | No head-count constraint.                |
    +-----------------+--------------------+------------------------------------------+
    | Hybrid          | ``1 < k < cp_size``| Q/K/V seq→head A2A on Ulysses sub-mesh  |
    |                 |                    | (size ``k``); K/V then all-gathered on   |
    |                 |                    | Colossal sub-mesh (size ``cp_size // k``)|
    |                 |                    | Requires ``num_heads % k == 0``.         |
    +-----------------+--------------------+------------------------------------------+

    Args:
        seq_dim:         Sequence dimension index. 1 for BSHD, 2 for BNSD.
        head_dim:        Head dimension index. 2 for BSHD, 1 for BNSD.
        ulysses_degree:  Ulysses sub-mesh size (see table above).
        qkv_indices:     Positional-argument indices for (Q, K, V).
        qkv_kwarg_names: Keyword-argument names for (Q, K, V).
        load_balance:    Enable Head-Tail Q-exchange load balancing.
                         Only valid with Pure Colossal AI (``ulysses_degree=1``).

                         **Important**: When ``load_balance=True``, ``q.shape[seq_dim]``
                         inside ``forward()`` returns ``S / 2`` (global shape / 2)
                         rather than the true global ``S``. This is because
                         ``DTensor.shape`` returns ``local_tensor_size * mesh_size``,
                         and each sub-FA call wraps a half-sized Q shard
                         (``S / (2 * cp_size)`` tokens) with a ``co_submesh`` of
                         size ``cp_size``, giving a DTensor global shape of
                         ``S / (2 * cp_size) * cp_size = S / 2``.
                         K/V are always Replicate so ``k.shape[seq_dim]`` always
                         returns the true ``S``. **When building the attention mask,
                         use ``k.shape[seq_dim]`` (not ``q.shape[seq_dim]``) to
                         obtain the correct global sequence length.**
    """

    def __init__(
        self,
        seq_dim: int = 1,
        head_dim: int = 2,
        ulysses_degree: Optional[int] = None,
        qkv_indices: tuple = (0, 1, 2),
        qkv_kwarg_names: tuple = (),
        load_balance: bool = False,
    ):
        if load_balance and ulysses_degree != 1:
            raise ValueError(
                "load_balance=True requires ulysses_degree=1 (Pure Colossal AI mode)."
            )
        self.seq_dim = seq_dim
        self.head_dim = head_dim
        self.ulysses_degree = ulysses_degree
        self.qkv_indices = qkv_indices
        self.qkv_kwarg_names = qkv_kwarg_names
        self.load_balance = load_balance

    # ------------------------------------------------------------------
    # ParallelStyle interface
    # ------------------------------------------------------------------

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register forward hooks on *module* and return it.

        Args:
            module:      attention submodule to parallelise.
            device_mesh: CP device mesh (1-D or 2-D).
        """
        cp_size = device_mesh.mesh.numel()
        ds = self.ulysses_degree if self.ulysses_degree is not None else cp_size
        if cp_size % ds != 0:
            raise ValueError(
                f"cp_size ({cp_size}) must be divisible by ulysses_degree ({ds})."
            )
        co = cp_size // ds

        if ds == 1:
            # Pure Colossal AI
            co_submesh = _ensure_1d(device_mesh)
            if self.load_balance:
                self._apply_lb_colossal(module, co_submesh)
            else:
                module.register_forward_pre_hook(
                    partial(self._pre_hook_colossal, co_submesh=co_submesh),
                    with_kwargs=True,
                )
                module.register_forward_hook(
                    partial(self._post_hook_colossal, co_submesh=co_submesh)
                )
        elif co == 1:
            # Pure Ulysses
            ds_submesh = _ensure_1d(device_mesh)
            module.register_forward_pre_hook(
                partial(self._pre_hook_ulysses, ds_submesh=ds_submesh, ds_size=ds),
                with_kwargs=True,
            )
            module.register_forward_hook(
                partial(self._post_hook_ata, ds_submesh=ds_submesh)
            )
        else:
            # Hybrid
            two_d_mesh = _build_2d_mesh(device_mesh, ds, co)
            dim_names = two_d_mesh.mesh_dim_names
            assert dim_names is not None, "2-D mesh must have mesh_dim_names (guaranteed by _build_2d_mesh)"
            ds_submesh = two_d_mesh[dim_names[1]]
            module.register_forward_pre_hook(
                partial(
                    self._pre_hook_hybrid,
                    two_d_mesh=two_d_mesh,
                    ds_submesh=ds_submesh,
                    ds_size=ds,
                ),
                with_kwargs=True,
            )
            module.register_forward_hook(
                partial(self._post_hook_ata, ds_submesh=ds_submesh)
            )

        return module

    # ------------------------------------------------------------------
    # Pre-hooks
    # ------------------------------------------------------------------

    def _pre_hook_colossal(self, module, args, kwargs, co_submesh):  # pylint: disable=unused-argument
        """Wrap Q as ``DTensor(co_submesh, Shard(seq))``; all-gather K/V."""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        q_idx = self.qkv_indices[0]
        if q_idx < len(new_args) and isinstance(new_args[q_idx], Tensor) \
                and not isinstance(new_args[q_idx], DTensor):
            new_args[q_idx] = DTensor.from_local(
                new_args[q_idx], co_submesh, (Shard(self.seq_dim),)
            )
        for idx in self.qkv_indices[1:]:
            if idx < len(new_args) and isinstance(new_args[idx], Tensor):
                new_args[idx] = _gather_seq(new_args[idx], co_submesh, self.seq_dim)

        if self.qkv_kwarg_names:
            q_name = self.qkv_kwarg_names[0]
            if q_name in new_kwargs and isinstance(new_kwargs[q_name], Tensor) \
                    and not isinstance(new_kwargs[q_name], DTensor):
                new_kwargs[q_name] = DTensor.from_local(
                    new_kwargs[q_name], co_submesh, (Shard(self.seq_dim),)
                )
            for name in self.qkv_kwarg_names[1:]:
                if name in new_kwargs and isinstance(new_kwargs[name], Tensor):
                    new_kwargs[name] = _gather_seq(new_kwargs[name], co_submesh, self.seq_dim)

        return tuple(new_args), new_kwargs

    def _pre_hook_ulysses(self, module, args, kwargs, ds_submesh, ds_size):  # pylint: disable=unused-argument
        """Seq→head all-to-all for Q, K, and V."""
        new_args = list(args)
        for idx in self.qkv_indices:
            if idx < len(new_args) and isinstance(new_args[idx], Tensor):
                new_args[idx] = _scatter_seq_to_head(
                    new_args[idx], ds_submesh, self.seq_dim, self.head_dim, ds_size
                )

        new_kwargs = dict(kwargs)
        for name in self.qkv_kwarg_names:
            if name in new_kwargs and isinstance(new_kwargs[name], Tensor):
                new_kwargs[name] = _scatter_seq_to_head(
                    new_kwargs[name], ds_submesh, self.seq_dim, self.head_dim, ds_size
                )

        return tuple(new_args), new_kwargs

    def _ata_scatter_to_2d(self, t, ds_submesh, two_d_mesh, ds_size):
        """ATA scatter: Shard(seq)→Shard(head) on ds_submesh; wrap as 2-D DTensor.

        Args:
            t: Plain local tensor to scatter.
            ds_submesh: 1-D Ulysses sub-mesh.
            two_d_mesh: 2-D mesh (co × ds).
            ds_size: Ulysses degree (world size on ds_submesh).

        Returns:
            DTensor with placements ``(Shard(seq_dim), Shard(head_dim))`` on two_d_mesh.
        """
        if t.shape[self.head_dim] % ds_size != 0:
            raise ValueError(
                f"num_heads ({t.shape[self.head_dim]}) must be divisible by "
                f"ulysses_degree ({ds_size})."
            )
        local = (
            DTensor.from_local(t, ds_submesh, (Shard(self.seq_dim),))
            .redistribute(ds_submesh, (Shard(self.head_dim),))
            .to_local()
        )
        return DTensor.from_local(local, two_d_mesh, (Shard(self.seq_dim), Shard(self.head_dim)))

    def _pre_hook_hybrid(self, module, args, kwargs, two_d_mesh, ds_submesh, ds_size):  # pylint: disable=unused-argument
        """Hybrid: seq→head ATA on ds-submesh, then all-gather K/V on co-submesh.

        After this hook, placements on ``two_d_mesh`` are:
          Q   → ``(Shard(seq_dim),  Shard(head_dim))``
          K/V → ``(Replicate(),     Shard(head_dim))``
        """
        new_args = list(args)

        # Step 1: ATA on ds_submesh for all of Q/K/V; wrap as 2-D DTensor
        for idx in self.qkv_indices:
            if idx < len(new_args) and isinstance(new_args[idx], Tensor) \
                    and not isinstance(new_args[idx], DTensor):
                new_args[idx] = self._ata_scatter_to_2d(
                    new_args[idx], ds_submesh, two_d_mesh, ds_size
                )

        # Step 2: all-gather K/V on co-dim (Shard(seq)→Replicate)
        for idx in self.qkv_indices[1:]:
            if idx < len(new_args) and isinstance(new_args[idx], DTensor):
                new_args[idx] = new_args[idx].redistribute(
                    two_d_mesh, (Replicate(), Shard(self.head_dim))
                )

        # Same for kwargs
        new_kwargs = dict(kwargs)
        for name in self.qkv_kwarg_names:
            if name in new_kwargs and isinstance(new_kwargs[name], Tensor) \
                    and not isinstance(new_kwargs[name], DTensor):
                t = new_kwargs[name]
                local = (
                    DTensor.from_local(t, ds_submesh, (Shard(self.seq_dim),))
                    .redistribute(ds_submesh, (Shard(self.head_dim),))
                    .to_local()
                )
                new_kwargs[name] = DTensor.from_local(
                    local, two_d_mesh, (Shard(self.seq_dim), Shard(self.head_dim))
                )
        for name in self.qkv_kwarg_names[1:]:
            if name in new_kwargs and isinstance(new_kwargs[name], DTensor):
                new_kwargs[name] = new_kwargs[name].redistribute(
                    two_d_mesh, (Replicate(), Shard(self.head_dim))
                )

        return tuple(new_args), new_kwargs

    # ------------------------------------------------------------------
    # Post-hooks
    # ------------------------------------------------------------------

    def _post_hook_ata(self, module, inputs, outputs, ds_submesh):  # pylint: disable=unused-argument
        """Reverse all-to-all: head→seq on ds-submesh; returns local tensor.

        Handles both Ulysses (1-D DTensor or plain tensor) and Hybrid
        (2-D DTensor — ``to_local()`` first to project onto the 1-D ds-submesh).
        """
        def _process(out):
            if isinstance(out, (Tensor, DTensor)):
                if isinstance(out, DTensor):
                    out = out.to_local()
                return _gather_head_to_seq(
                    out, ds_submesh, self.seq_dim, self.head_dim
                ).to_local()
            return out

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_process(o) for o in outputs)
        return _process(outputs)

    def _post_hook_colossal(self, module, inputs, outputs, co_submesh):  # pylint: disable=unused-argument
        """Colossal AI: convert any DTensor output to a local tensor."""
        def _process(out):
            return out.to_local() if isinstance(out, DTensor) else out

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_process(o) for o in outputs)
        return _process(outputs)

    # ------------------------------------------------------------------
    # Load-balance Colossal AI (Head-Tail Q-exchange)
    # ------------------------------------------------------------------

    def _apply_lb_colossal(self, module: Module, co_submesh: DeviceMesh) -> None:
        """Replace ``module.forward`` with the load-balanced two-sub-FA wrapper."""
        ws = co_submesh.mesh.numel()
        rank_list = list(co_submesh.rank_list)
        local_idx = rank_list.index(platform.get_rank())
        target_idx = ws - 1 - local_idx
        module.forward = partial(
            self._lb_colossal_forward,
            original_forward=module.forward,
            co_submesh=co_submesh,
            local_idx=local_idx,
            target_idx=target_idx,
            ws=ws,
            peer_rank=rank_list[target_idx],
        )

    def _lb_colossal_forward(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *args,
        original_forward,
        co_submesh: DeviceMesh,
        local_idx: int,
        target_idx: int,
        ws: int,
        peer_rank: int,
        **kwargs,
    ):
        """Head-Tail load-balanced forward for Pure Colossal AI CP.

        Splits local Q (shape ``[B, S/ws, H, D]``) into head/tail halves.
        The tail is P2P-exchanged with the paired rank ``(ws - 1 - local_idx)``.
        Two sub-FA calls are issued with adjusted causal-mask offsets:

        - FA1: ``q_keep``  at ``split_id = 2*local_idx``
        - FA2: ``q_peer``  at ``split_id = 2*target_idx + 1``

        FA2's output is exchanged back; final output = ``cat([FA1, FA2_recv])``.
        """
        from hyper_parallel.core.shard.ops.parallel_npu_flash_attention_score import (  # pylint: disable=import-outside-toplevel
            _set_lb_override, _clear_lb_override,
        )

        seq_dim = self.seq_dim
        q_idx, k_idx, v_idx = self.qkv_indices
        new_args = list(args)

        q = new_args[q_idx]
        half = q.shape[seq_dim] // 2
        q_keep = q.narrow(seq_dim, 0, half)
        q_mine = q.narrow(seq_dim, half, half)

        q_peer = platform.p2p_exchange(q_mine, peer_rank)
        k_full = _gather_seq(new_args[k_idx], co_submesh, seq_dim).to_local()
        v_full = _gather_seq(new_args[v_idx], co_submesh, seq_dim).to_local()

        # K/V are Replicate; wrap once and reuse for both FA calls
        k_full_dt = DTensor.from_local(k_full, co_submesh, (Replicate(),))
        v_full_dt = DTensor.from_local(v_full, co_submesh, (Replicate(),))

        def _fa(q_half, split_id):
            new_args[q_idx] = DTensor.from_local(q_half, co_submesh, (Shard(seq_dim),))
            new_args[k_idx] = k_full_dt
            new_args[v_idx] = v_full_dt
            _set_lb_override(split_id=split_id, split_num=2 * ws)
            out = original_forward(*new_args, **kwargs)
            _clear_lb_override()
            return out.to_local() if isinstance(out, DTensor) else out

        fa1_out = _fa(q_keep, split_id=2 * local_idx)
        fa2_out = _fa(q_peer, split_id=2 * target_idx + 1)
        fa2_our = platform.p2p_exchange(fa2_out, peer_rank)
        return platform.cat([fa1_out, fa2_our], dim=seq_dim)
