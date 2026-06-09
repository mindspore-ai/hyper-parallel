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
"""Unified Context Parallel: Pure Ulysses, Pure Colossal AI, and Hybrid CP."""
from functools import partial
from typing import Optional

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, StridedShard
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module
Tensor = platform.Tensor

_OUTPUT_LAYOUT_STACK_ATTR = "_context_parallel_output_layout_stack"
_OUTPUT_LOCAL = "local"
_OUTPUT_CP = "cp"
_OUTPUT_NON_CP = "non_cp"


# ---------------------------------------------------------------------------
# DTensor boundary helpers
# ---------------------------------------------------------------------------

def _same_mesh_rank_list(lhs: DeviceMesh, rhs: DeviceMesh) -> bool:
    """Return whether two meshes describe the same participant ranks."""
    return tuple(lhs.rank_list) == tuple(rhs.rank_list)


def _same_mesh(lhs: DeviceMesh, rhs: DeviceMesh) -> bool:
    """Return whether two meshes represent the same named layout."""
    return lhs.to_hash() == rhs.to_hash()


def _is_foreign_dtensor(tensor: Tensor, submesh: DeviceMesh) -> bool:
    """Return whether *tensor* is a DTensor whose mesh differs from *submesh*."""
    return isinstance(tensor, DTensor) and not _same_mesh(tensor.device_mesh, submesh)


def _is_cp_composed_dtensor(tensor: Tensor, cp_mesh: DeviceMesh) -> bool:
    """Return whether *tensor* already lives on a ``CP + non-CP`` mesh."""
    if not isinstance(tensor, DTensor):
        return False
    mesh = tensor.device_mesh
    if mesh.ndim <= cp_mesh.ndim or not mesh.mesh_dim_names:
        return False
    cp_names = tuple(cp_mesh.mesh_dim_names or ())
    if len(cp_names) != cp_mesh.ndim or not all(name in mesh.mesh_dim_names for name in cp_names):
        return False
    try:
        return _same_mesh_rank_list(mesh[cp_names], cp_mesh)
    except (KeyError, RuntimeError, ValueError):
        return False


def _compose_cp_non_cp_mesh(cp_mesh: DeviceMesh, non_cp_mesh: DeviceMesh) -> DeviceMesh:
    """Build a ``CP + non-CP`` mesh used while executing CP attention."""
    meshes = (cp_mesh, non_cp_mesh)
    root_mesh = cp_mesh.root_mesh or cp_mesh
    root_dim_names = tuple(root_mesh.mesh_dim_names or ())
    if root_dim_names and all(
        mesh.mesh_dim_names and all(name in root_dim_names for name in mesh.mesh_dim_names)
        for mesh in meshes
    ):
        meshes = tuple(
            sorted(meshes, key=lambda mesh: root_dim_names.index(mesh.mesh_dim_names[0]))
        )
    return DeviceMesh.concatenate(meshes)


def _compose_cp_non_cp_placements(
    composed_mesh: DeviceMesh,
    cp_mesh: DeviceMesh,
    cp_placements,
    non_cp_mesh: DeviceMesh,
    non_cp_placements,
) -> tuple:
    """Return placements aligned to the actual composed mesh dimension order."""
    placement_by_name = {}
    for name, placement in zip(cp_mesh.mesh_dim_names or (), cp_placements):
        placement_by_name[name] = placement
    for name, placement in zip(non_cp_mesh.mesh_dim_names or (), non_cp_placements):
        placement_by_name[name] = placement
    placements = [placement_by_name[name] for name in composed_mesh.mesh_dim_names]
    for mesh_idx, placement in enumerate(placements):
        if not placement.is_shard():
            continue
        split_factor = 1
        for right_mesh_idx in range(mesh_idx + 1, len(placements)):
            if placements[right_mesh_idx].is_shard(placement.dim):
                split_factor *= composed_mesh.mesh_shape[right_mesh_idx]
        if split_factor > 1:
            placements[mesh_idx] = StridedShard(placement.dim, split_factor)
    return tuple(placements)


def _compose_cp_mesh(cp_mesh: DeviceMesh, non_cp_mesh: DeviceMesh) -> DeviceMesh:
    """Compatibility wrapper for CP + non-CP mesh composition."""
    return _compose_cp_non_cp_mesh(cp_mesh, non_cp_mesh)


def _composed_placements(
    composed_mesh: DeviceMesh,
    cp_mesh: DeviceMesh,
    cp_placements,
    non_cp_mesh: DeviceMesh,
    non_cp_placements,
) -> tuple:
    """Compatibility wrapper for CP + non-CP placements composition."""
    return _compose_cp_non_cp_placements(
        composed_mesh,
        cp_mesh,
        cp_placements,
        non_cp_mesh,
        non_cp_placements,
    )


def _non_cp_placements_from_composed(
    tensor: "DTensor",
    cp_mesh: DeviceMesh,
) -> tuple:
    """Extract non-CP placements from a composed DTensor by mesh dimension name."""
    cp_names = set(cp_mesh.mesh_dim_names or ())
    return tuple(
        placement
        for name, placement in zip(tensor.device_mesh.mesh_dim_names, tensor.placements)
        if name not in cp_names
    )


def _cp_mesh_from_composed(composed_mesh: DeviceMesh, non_cp_mesh: DeviceMesh) -> DeviceMesh:
    """Return the CP portion of a composed mesh."""
    non_cp_names = set(non_cp_mesh.mesh_dim_names or ())
    cp_names = tuple(name for name in composed_mesh.mesh_dim_names if name not in non_cp_names)
    return composed_mesh[cp_names]


def _split_cp_composed_layout(tensor: "DTensor", cp_mesh: DeviceMesh):
    """Return ``(non_cp_mesh, placements, composed_mesh)`` for a composed DTensor."""
    if not _is_cp_composed_dtensor(tensor, cp_mesh):
        return None
    mesh = tensor.device_mesh
    dim_names = mesh.mesh_dim_names
    if dim_names is None:
        raise ValueError("dim_names must not be None")
    cp_names = set(cp_mesh.mesh_dim_names or ())
    non_cp_names = tuple(name for name in dim_names if name not in cp_names)
    if len(non_cp_names) == 0:
        return None
    return mesh[non_cp_names], _non_cp_placements_from_composed(tensor, cp_mesh), mesh


def _non_cp_dtensor_layout(tensor: Tensor, cp_mesh: DeviceMesh, seq_dim: int):
    """Return metadata for dropping CP and keeping the incoming non-CP layout."""
    del seq_dim
    if not isinstance(tensor, DTensor):
        return None
    composed_layout = _split_cp_composed_layout(tensor, cp_mesh)
    if composed_layout is not None:
        return composed_layout
    if not _is_foreign_dtensor(tensor, cp_mesh):
        return None
    non_cp_mesh = tensor.device_mesh
    return non_cp_mesh, tuple(tensor.placements), _compose_cp_non_cp_mesh(cp_mesh, non_cp_mesh)


def _localize_foreign_dtensor(tensor: Tensor, cp_mesh: DeviceMesh, seq_dim: int):
    """Convert a safe non-CP DTensor to its local tensor at the CP boundary."""
    del seq_dim
    if _is_cp_composed_dtensor(tensor, cp_mesh) or not _is_foreign_dtensor(tensor, cp_mesh):
        return tensor
    return tensor.to_local()


def _cp_boundary_dim_size(tensor: Tensor, submesh: DeviceMesh, dim: int) -> int:
    """Return the dimension size CP should validate at its local boundary."""
    if _is_foreign_dtensor(tensor, submesh):
        return tensor.to_local().shape[dim]
    return tensor.shape[dim]


def _to_cp_dtensor(
    tensor: Tensor,
    cp_mesh: DeviceMesh,
    src_placements,
    dst_placements,
    seq_dim: int,
) -> "DTensor":
    """Wrap a local/non-CP tensor as a CP DTensor and redistribute it."""
    if _is_cp_composed_dtensor(tensor, cp_mesh):
        non_cp_mesh, non_cp_placements, composed_mesh = _split_cp_composed_layout(tensor, cp_mesh)
        return tensor.redistribute(
            composed_mesh,
            _composed_placements(composed_mesh, cp_mesh, dst_placements, non_cp_mesh, non_cp_placements),
        )
    layout = _non_cp_dtensor_layout(tensor, cp_mesh, seq_dim)
    if layout is not None:
        non_cp_mesh, non_cp_placements, composed_mesh = layout
        src_composed_placements = _composed_placements(
            composed_mesh, cp_mesh, src_placements, non_cp_mesh, non_cp_placements
        )
        dst_composed_placements = _composed_placements(
            composed_mesh, cp_mesh, dst_placements, non_cp_mesh, non_cp_placements
        )
        return DTensor.from_local(
            tensor.to_local(),
            composed_mesh,
            src_composed_placements,
        ).redistribute(composed_mesh, dst_composed_placements)
    tensor = _localize_foreign_dtensor(tensor, cp_mesh, seq_dim)
    if isinstance(tensor, DTensor):
        return tensor.redistribute(cp_mesh, dst_placements)
    cp_tensor = DTensor.from_local(tensor, cp_mesh, src_placements)
    if not hasattr(cp_tensor, "redistribute"):
        return cp_tensor
    return cp_tensor.redistribute(cp_mesh, dst_placements)


def _output_layout_from_q(tensor: Tensor, cp_mesh: DeviceMesh, seq_dim: int):
    """Return the output-layout policy implied by the Q input."""
    if not isinstance(tensor, DTensor):
        return _OUTPUT_LOCAL, None
    non_cp_layout = _non_cp_dtensor_layout(tensor, cp_mesh, seq_dim)
    if non_cp_layout is not None:
        return _OUTPUT_NON_CP, non_cp_layout
    return _OUTPUT_CP, tensor.device_mesh


def _push_output_layout(module, layout) -> None:
    """Push the current forward's output-layout policy for the matching post-hook."""
    if module is None:
        return
    stack = getattr(module, _OUTPUT_LAYOUT_STACK_ATTR, None)
    if stack is None:
        stack = []
        setattr(module, _OUTPUT_LAYOUT_STACK_ATTR, stack)
    stack.append(layout)


def _pop_output_layout(module):
    """Pop the output-layout policy recorded by the matching pre-hook."""
    if module is None:
        return _OUTPUT_LOCAL, None
    stack = getattr(module, _OUTPUT_LAYOUT_STACK_ATTR, None)
    if not stack:
        return _OUTPUT_LOCAL, None
    layout = stack.pop()
    if not stack:
        delattr(module, _OUTPUT_LAYOUT_STACK_ATTR)
    return layout


def _drop_cp_from_output(output, layout, cp_placements):
    """Drop CP metadata from the current local shard and keep the non-CP layout."""
    if layout is None or not isinstance(output, (Tensor, DTensor)):
        return output
    non_cp_mesh, non_cp_placements, composed_mesh = layout
    cp_mesh = _cp_mesh_from_composed(composed_mesh, non_cp_mesh)
    composed_placements = _composed_placements(
        composed_mesh,
        cp_mesh,
        cp_placements,
        non_cp_mesh,
        non_cp_placements,
    )
    if isinstance(output, DTensor):
        if not _same_mesh(output.device_mesh, composed_mesh):
            output = DTensor.from_local(output.to_local(), composed_mesh, composed_placements)
    else:
        output = DTensor.from_local(output, composed_mesh, composed_placements)
    if tuple(output.placements) != composed_placements:
        output = output.redistribute(composed_mesh, composed_placements)
    return DTensor.from_local(output.to_local(), non_cp_mesh, non_cp_placements)


def _wrap_cp_output_dtensor(output, device_mesh: DeviceMesh, placements):
    """Wrap/redistribute CP output as a CP DTensor."""
    if not isinstance(output, (Tensor, DTensor)):
        return output
    if isinstance(output, DTensor):
        if _same_mesh(output.device_mesh, device_mesh):
            if tuple(output.placements) == tuple(placements):
                return output
            return output.redistribute(device_mesh, placements)
        output = output.to_local()
    return DTensor.from_local(output, device_mesh, placements)


def _finalize_output(output, output_layout, cp_mesh, cp_placements, use_local_output: bool):
    """Apply CP boundary output policy.

    With ``use_local_output=True`` the public output is always local. Otherwise
    the output mirrors the Q input boundary: local input returns local output,
    CP-DTensor input returns a CP DTensor, and non-CP DTensor input drops CP
    back to that non-CP mesh.
    """
    if use_local_output:
        return output.to_local() if isinstance(output, DTensor) else output
    output_kind, layout = output_layout
    if output_kind == _OUTPUT_NON_CP:
        return _drop_cp_from_output(output, layout, cp_placements)
    if output_kind == _OUTPUT_CP:
        return _wrap_cp_output_dtensor(output, cp_mesh, cp_placements)
    return output.to_local() if isinstance(output, DTensor) else output


def _finalize_colossal_output(output, output_layout, co_submesh, seq_dim: int, use_local_output: bool):
    """Apply Colossal output policy: local tensor or DTensor output."""
    return _finalize_output(output, output_layout, co_submesh, (Shard(seq_dim),), use_local_output)


def _finalize_ata_output(output_dtensor, output_layout, ds_submesh, seq_dim: int, use_local_output: bool):
    """Apply Ulysses/Hybrid output policy after reverse ATA."""
    return _finalize_output(output_dtensor, output_layout, ds_submesh, (Shard(seq_dim),), use_local_output)


def _is_tensor_or_dtensor(value) -> bool:
    """Return whether *value* can participate in CP tensor conversion."""
    return isinstance(value, (Tensor, DTensor))


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


def _build_hybrid_cp_mesh(cp_mesh: DeviceMesh, ds: int, co: int) -> DeviceMesh:
    """Build the Hybrid CP ``(co, ds)`` mesh, preserving root layout when possible."""
    if cp_mesh.ndim == 2:
        if not cp_mesh.mesh_dim_names:
            raise ValueError(
                "2-D device_mesh for Hybrid CP must have mesh_dim_names=(\"co\", \"ds\")."
            )
        return cp_mesh
    if cp_mesh.ndim != 1:
        raise ValueError(f"Hybrid CP expects a 1-D or 2-D CP mesh, got {cp_mesh.ndim}D.")
    if cp_mesh.root_mesh is not None and cp_mesh.mesh_dim_names == ("cp",):
        return cp_mesh._unflatten("cp", (co, ds), ("co", "ds"))  # pylint: disable=protected-access
    return _build_2d_mesh(cp_mesh, ds, co)


def _scatter_seq_to_head(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
    head_dim: int,
    submesh_size: int,
) -> "DTensor":
    """All-to-all: ``Shard(seq_dim)`` → ``Shard(head_dim)``. Returns DTensor."""
    head_size = _cp_boundary_dim_size(tensor, submesh, head_dim)
    if head_size % submesh_size != 0:
        raise ValueError(
            f"num_heads ({head_size}) must be divisible by "
            f"ulysses_degree ({submesh_size})."
        )
    return _to_cp_dtensor(tensor, submesh, (Shard(seq_dim),), (Shard(head_dim),), seq_dim)


def _gather_head_to_seq(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
    head_dim: int,
) -> "DTensor":
    """Reverse all-to-all: ``Shard(head_dim)`` → ``Shard(seq_dim)``. Returns DTensor."""
    return _to_cp_dtensor(tensor, submesh, (Shard(head_dim),), (Shard(seq_dim),), seq_dim)


def _gather_seq(
    tensor: Tensor,
    submesh: DeviceMesh,
    seq_dim: int,
) -> "DTensor":
    """All-gather: ``Shard(seq_dim)`` → ``Replicate``. Returns DTensor."""
    return _to_cp_dtensor(tensor, submesh, (Shard(seq_dim),), (Replicate(),), seq_dim)


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
        use_local_output: Return local tensors after CP when True.  When False,
                         keep CP DTensor outputs, or drop the CP axis from
                         composed CP+TP outputs and keep the non-CP layout.
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
        use_local_output: bool = False,
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
        self.use_local_output = use_local_output
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
            hybrid_cp_mesh = _build_hybrid_cp_mesh(device_mesh, ds, co)
            dim_names = hybrid_cp_mesh.mesh_dim_names
            if dim_names is None:
                raise ValueError("2-D mesh must have mesh_dim_names (guaranteed by _build_2d_mesh)")
            ds_submesh = hybrid_cp_mesh[dim_names[1]]
            module.register_forward_pre_hook(
                partial(
                    self._pre_hook_hybrid,
                    hybrid_cp_mesh=hybrid_cp_mesh,
                    ds_submesh=ds_submesh,
                    ds_size=ds,
                ),
                with_kwargs=True,
            )
            module.register_forward_hook(
                partial(
                    self._post_hook_hybrid,
                    hybrid_cp_mesh=hybrid_cp_mesh,
                    ds_submesh=ds_submesh,
                )
            )

        return module

    # ------------------------------------------------------------------
    # Pre-hooks
    # ------------------------------------------------------------------

    def _record_q_output_tp_layout(self, module, args, kwargs, submesh) -> None:
        """Remember how the CP output should cross the public boundary."""
        output_layout = (_OUTPUT_LOCAL, None)
        q_idx = self.qkv_indices[0]
        if q_idx < len(args):
            output_layout = _output_layout_from_q(args[q_idx], submesh, self.seq_dim)
        if output_layout[0] == _OUTPUT_LOCAL and self.qkv_kwarg_names:
            q_name = self.qkv_kwarg_names[0]
            if q_name in kwargs:
                output_layout = _output_layout_from_q(kwargs[q_name], submesh, self.seq_dim)
        _push_output_layout(module, output_layout)

    def _pre_hook_colossal(self, module, args, kwargs, co_submesh):  # pylint: disable=unused-argument
        """Wrap Q as ``DTensor(co_submesh, Shard(seq))``; all-gather K/V."""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        self._record_q_output_tp_layout(module, new_args, new_kwargs, co_submesh)
        q_idx = self.qkv_indices[0]
        if q_idx < len(new_args) and _is_tensor_or_dtensor(new_args[q_idx]):
            new_args[q_idx] = _to_cp_dtensor(
                new_args[q_idx],
                co_submesh,
                (Shard(self.seq_dim),),
                (Shard(self.seq_dim),),
                self.seq_dim,
            )
        for idx in self.qkv_indices[1:]:
            if idx < len(new_args) and _is_tensor_or_dtensor(new_args[idx]):
                new_args[idx] = _gather_seq(new_args[idx], co_submesh, self.seq_dim)

        if self.qkv_kwarg_names:
            q_name = self.qkv_kwarg_names[0]
            if q_name in new_kwargs and _is_tensor_or_dtensor(new_kwargs[q_name]):
                new_kwargs[q_name] = _to_cp_dtensor(
                    new_kwargs[q_name],
                    co_submesh,
                    (Shard(self.seq_dim),),
                    (Shard(self.seq_dim),),
                    self.seq_dim,
                )
            for name in self.qkv_kwarg_names[1:]:
                if name in new_kwargs and _is_tensor_or_dtensor(new_kwargs[name]):
                    new_kwargs[name] = _gather_seq(new_kwargs[name], co_submesh, self.seq_dim)

        return tuple(new_args), new_kwargs

    def _pre_hook_ulysses(self, module, args, kwargs, ds_submesh, ds_size):  # pylint: disable=unused-argument
        """Seq→head all-to-all for Q, K, and V."""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        self._record_q_output_tp_layout(module, new_args, new_kwargs, ds_submesh)
        for idx in self.qkv_indices:
            if idx < len(new_args) and _is_tensor_or_dtensor(new_args[idx]):
                new_args[idx] = _scatter_seq_to_head(
                    new_args[idx], ds_submesh, self.seq_dim, self.head_dim, ds_size
                )

        for name in self.qkv_kwarg_names:
            if name in new_kwargs and _is_tensor_or_dtensor(new_kwargs[name]):
                new_kwargs[name] = _scatter_seq_to_head(
                    new_kwargs[name], ds_submesh, self.seq_dim, self.head_dim, ds_size
                )

        return tuple(new_args), new_kwargs

    def _hybrid_cp_layout_from_input(self, t, hybrid_cp_mesh, cp_placements):
        """Return the Hybrid execution mesh and placements for one input tensor."""
        layout = _non_cp_dtensor_layout(t, hybrid_cp_mesh, self.seq_dim)
        if layout is None:
            return hybrid_cp_mesh, tuple(cp_placements)
        non_cp_mesh, non_cp_placements, composed_mesh = layout
        return composed_mesh, _compose_cp_non_cp_placements(
            composed_mesh,
            hybrid_cp_mesh,
            cp_placements,
            non_cp_mesh,
            non_cp_placements,
        )

    def _ata_scatter_to_hybrid(self, t, ds_submesh, hybrid_cp_mesh, ds_size):
        """ATA scatter on ds, then wrap as Hybrid CP or Hybrid CP + non-CP DTensor.

        Args:
            t: Plain local tensor to scatter.
            ds_submesh: 1-D Ulysses sub-mesh.
            hybrid_cp_mesh: 2-D mesh (co × ds).
            ds_size: Ulysses degree (world size on ds_submesh).

        Returns:
            DTensor with CP placements ``(Shard(seq_dim), Shard(head_dim))``,
            preserving any incoming non-CP placements.
        """
        out_mesh, out_placements = self._hybrid_cp_layout_from_input(
            t,
            hybrid_cp_mesh,
            (Shard(self.seq_dim), Shard(self.head_dim)),
        )
        t = _localize_foreign_dtensor(t, ds_submesh, self.seq_dim)
        if isinstance(t, DTensor):
            t = t.redistribute(ds_submesh, (Shard(self.seq_dim),)).to_local()
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
        return DTensor.from_local(local, out_mesh, out_placements)

    def _hybrid_kv_gather_placements(self, t, hybrid_cp_mesh):
        """Return K/V placements after co gather for a Hybrid input DTensor."""
        out_mesh, out_placements = self._hybrid_cp_layout_from_input(
            t,
            hybrid_cp_mesh,
            (Replicate(), Shard(self.head_dim)),
        )
        return out_mesh, out_placements

    def _pre_hook_hybrid(  # pylint: disable=unused-argument
        self, module, args, kwargs, hybrid_cp_mesh, ds_submesh, ds_size
    ):
        """Hybrid: seq→head ATA on ds-submesh, then all-gather K/V on co-submesh.

        After this hook, CP placements on ``hybrid_cp_mesh`` are:
          Q   → ``(Shard(seq_dim),  Shard(head_dim))``
          K/V → ``(Replicate(),     Shard(head_dim))``
        Incoming non-CP placements are preserved on a composed mesh.
        """
        new_args = list(args)
        new_kwargs = dict(kwargs)

        self._record_q_output_tp_layout(module, new_args, new_kwargs, hybrid_cp_mesh)

        # Step 1: ATA on ds_submesh for all of Q/K/V; wrap as 2-D DTensor
        for idx in self.qkv_indices:
            if idx < len(new_args) and self._needs_hybrid_ata(new_args[idx], hybrid_cp_mesh):
                new_args[idx] = self._ata_scatter_to_hybrid(
                    new_args[idx], ds_submesh, hybrid_cp_mesh, ds_size
                )

        # Step 2: all-gather K/V on co-dim (Shard(seq)→Replicate)
        for idx in self.qkv_indices[1:]:
            if idx < len(new_args) and isinstance(new_args[idx], DTensor):
                out_mesh, out_placements = self._hybrid_kv_gather_placements(
                    new_args[idx],
                    hybrid_cp_mesh,
                )
                new_args[idx] = new_args[idx].redistribute(out_mesh, out_placements)

        # Same for kwargs
        for name in self.qkv_kwarg_names:
            if name in new_kwargs and self._needs_hybrid_ata(new_kwargs[name], hybrid_cp_mesh):
                new_kwargs[name] = self._ata_scatter_to_hybrid(
                    new_kwargs[name], ds_submesh, hybrid_cp_mesh, ds_size
                )
        for name in self.qkv_kwarg_names[1:]:
            if name in new_kwargs and isinstance(new_kwargs[name], DTensor):
                out_mesh, out_placements = self._hybrid_kv_gather_placements(
                    new_kwargs[name],
                    hybrid_cp_mesh,
                )
                new_kwargs[name] = new_kwargs[name].redistribute(out_mesh, out_placements)

        return tuple(new_args), new_kwargs

    @staticmethod
    def _needs_hybrid_ata(value, hybrid_cp_mesh) -> bool:
        """Return whether Hybrid CP should run ATA before entering attention."""
        if not _is_tensor_or_dtensor(value):
            return False
        return not isinstance(value, DTensor) or _is_foreign_dtensor(value, hybrid_cp_mesh)

    # ------------------------------------------------------------------
    # Post-hooks
    # ------------------------------------------------------------------

    def _post_hook_ata(self, module, inputs, outputs, ds_submesh):  # pylint: disable=unused-argument
        """Reverse all-to-all: head→seq on ds-submesh; returns local tensor.

        Handles both Ulysses (1-D DTensor or plain tensor) and Hybrid
        (2-D DTensor — ``to_local()`` first to project onto the 1-D ds-submesh).
        """
        output_layout = _pop_output_layout(module)

        def _process(out):
            if isinstance(out, (Tensor, DTensor)):
                if isinstance(out, DTensor) and not _is_cp_composed_dtensor(out, ds_submesh):
                    out = out.to_local()
                seq_dtensor = _gather_head_to_seq(
                    out, ds_submesh, self.seq_dim, self.head_dim
                )
                return _finalize_ata_output(
                    seq_dtensor, output_layout, ds_submesh, self.seq_dim, self.use_local_output
                )
            return out

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_process(o) for o in outputs)
        return _process(outputs)

    def _post_hook_hybrid(self, module, inputs, outputs, hybrid_cp_mesh, ds_submesh):  # pylint: disable=unused-argument
        """Hybrid reverse ATA, then drop the full ``(co, ds)`` CP mesh if needed."""
        output_layout = _pop_output_layout(module)

        def _process(out):
            if not isinstance(out, (Tensor, DTensor)):
                return out
            out_local = out.to_local() if isinstance(out, DTensor) else out
            seq_dtensor = _gather_head_to_seq(
                out_local,
                ds_submesh,
                self.seq_dim,
                self.head_dim,
            )
            seq_local = seq_dtensor.to_local()
            if self.use_local_output:
                return seq_local

            output_kind, layout = output_layout
            if output_kind == _OUTPUT_NON_CP:
                non_cp_mesh, non_cp_placements, _ = layout
                return DTensor.from_local(seq_local, non_cp_mesh, non_cp_placements)
            if output_kind == _OUTPUT_CP:
                return DTensor.from_local(
                    seq_local,
                    hybrid_cp_mesh,
                    (Shard(self.seq_dim), Replicate()),
                )
            return seq_local

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_process(o) for o in outputs)
        return _process(outputs)

    def _post_hook_colossal(self, module, inputs, outputs, co_submesh):  # pylint: disable=unused-argument
        """Colossal AI: convert any DTensor output to a local tensor."""
        output_layout = _pop_output_layout(module)

        def _process(out):
            return _finalize_colossal_output(
                out, output_layout, co_submesh, self.seq_dim, self.use_local_output
            )

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
        output_layout = _output_layout_from_q(q, co_submesh, seq_dim)
        if _is_foreign_dtensor(q, co_submesh):
            q = _localize_foreign_dtensor(q, co_submesh, seq_dim)
            new_args[q_idx] = q
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
        out = platform.cat([fa1_out, fa2_our], dim=seq_dim)
        return _finalize_colossal_output(out, output_layout, co_submesh, seq_dim, self.use_local_output)
