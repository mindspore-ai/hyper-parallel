# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""AsyncContextParallel: overlap projection GEMM with CP collectives.

Supports async Pure Ulysses, async Pure Colossal AI, and Hybrid CP modes.
Hybrid keeps the original half-async path: async Ulysses A2A + sync Colossal
AllGather. Falls back to sync ContextParallel when q/k/v_proj are not provided.

Forward:  proj hooks launch async A2A → attn pre-hook waits Q/K/V → attn hook gathers output
Backward: autograd backward launches async A2A → proj pre-hooks wait before GEMMs
"""
from functools import partial
from typing import Optional

from hyper_parallel.core.context_parallel.context_parallel import (
    ContextParallel,
    _build_2d_mesh,
    _ensure_1d,
    _gather_seq,
    _gather_head_to_seq,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module
Tensor = platform.Tensor


# ---------------------------------------------------------------------------
# All-to-all helpers
# ---------------------------------------------------------------------------

def _detach_if_available(tensor: Tensor) -> Tensor:
    """Detach the communication buffer when the backend tensor exposes ``detach``."""
    detach = getattr(tensor, "detach", None)
    return detach() if detach is not None else tensor


def _launch_async_a2a_seq_to_head(
    tensor: Tensor,
    group,
    world_size: int,
    head_dim: int,
) -> tuple:
    """Launch async seq→head A2A (forward)."""
    x = tensor.contiguous()
    shape = list(x.shape)
    num_heads = shape[head_dim]
    if num_heads % world_size != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by world_size ({world_size}).")
    ndim = len(shape) + 1
    x_perm = x.reshape(
        shape[:head_dim] + [world_size, num_heads // world_size] + shape[head_dim + 1:]
    ).permute(
        [head_dim] + list(range(head_dim)) + list(range(head_dim + 1, ndim))
    ).contiguous()
    out_perm, work = platform.all_to_all_single(_detach_if_available(x_perm), list(x_perm.shape), group, async_op=True)
    return work, out_perm


def _a2a_reconstruct(out_perm: Tensor, concat_dim: int) -> Tensor:
    """Reconstruct A2A result from raw out_perm."""
    new_ndim = out_perm.dim()
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, new_ndim))
    x_recon = out_perm.permute(recon_perm).contiguous()
    shape = list(x_recon.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return x_recon.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize a possibly negative dimension index."""
    return dim + ndim if dim < 0 else dim


def _move_dim_to_front(tensor: Tensor, dim: int) -> Tensor:
    """Move ``dim`` to the leading dimension before all-gather/reduce-scatter."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [i for i in range(tensor.dim()) if i != dim]
    return tensor.permute(perm).contiguous()


def _move_dim_from_front(tensor: Tensor, dim: int) -> Tensor:
    """Inverse of :func:`_move_dim_to_front`."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [i for i in range(tensor.dim()) if i != dim]
    inverse = [0] * len(perm)
    for idx, value in enumerate(perm):
        inverse[value] = idx
    return tensor.permute(inverse).contiguous()


def _launch_async_allgather_seq(
    tensor: Tensor,
    group,
    world_size: int,
    gather_dim: int,
) -> tuple:
    """Launch async all-gather along ``gather_dim``."""
    x_perm = _move_dim_to_front(tensor.contiguous(), gather_dim)
    output_shape = list(x_perm.shape)
    output_shape[0] *= world_size
    out_perm, work = platform.all_gather_single(_detach_if_available(x_perm), output_shape, group, async_op=True)
    return work, out_perm


def _allgather_reconstruct(out_perm: Tensor, gather_dim: int) -> Tensor:
    """Move the leading communication buffer dimension back to ``gather_dim``.

    The input may be a forward all-gather output buffer or a backward
    reduce-scatter local buffer, as both are stored with ``gather_dim`` moved
    to dimension 0 before communication.
    """
    return _move_dim_from_front(out_perm, gather_dim)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _to_local(tensor: Tensor) -> Tensor:
    """Return the local Tensor from a DTensor, or the tensor itself."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


# ---------------------------------------------------------------------------
# AsyncContextParallel
# ---------------------------------------------------------------------------

class AsyncContextParallel(ContextParallel):
    """Context Parallel with projection–collective compute overlap.

    Requires ``q_proj``, ``k_proj``, ``v_proj`` in :meth:`apply`; otherwise
    falls back to synchronous :class:`ContextParallel`.

    Pure Colossal AI (``ulysses_degree=1``) overlaps K/V AllGather when the
    backend exposes async all-gather handles. Hybrid CP overlaps Ulysses A2A and
    keeps the Colossal K/V AllGather synchronous for backward-ordering safety.

    Args:
        seq_dim:         Sequence dimension (1=BSHD, 2=BNSD).
        head_dim:        Head dimension (2=BSHD, 1=BNSD).
        ulysses_degree:  Ulysses sub-mesh size (see :class:`ContextParallel`).
        qkv_indices:     Positional indices of (Q, K, V) in attention forward.
        qkv_kwarg_names: Keyword names for (Q, K, V).
        load_balance:    Load-balance flag forwarded to base class.
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
        super().__init__(
            seq_dim=seq_dim,
            head_dim=head_dim,
            ulysses_degree=ulysses_degree,
            qkv_indices=qkv_indices,
            qkv_kwarg_names=qkv_kwarg_names,
            load_balance=load_balance,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def apply(  # pylint: disable=arguments-differ
        self,
        module: Module,
        device_mesh: DeviceMesh,
        q_proj: Optional[Module] = None,
        k_proj: Optional[Module] = None,
        v_proj: Optional[Module] = None,
    ) -> Module:
        """Register async-overlap hooks and return *module*.

        Falls back to synchronous :class:`ContextParallel` if any of
        ``q/k/v_proj`` is ``None``.

        Args:
            module:      Core-attention submodule.
            device_mesh: CP device mesh (1-D or 2-D).
            q_proj:      The last module in the Q path whose output is passed
                         directly to the attention module as Q.  Its forward
                         post-hook launches the async Q all-to-all.  There
                         must be **no** intermediate ops (view, transpose, …)
                         between this module and attention; such ops would be
                         bypassed by the pre-hook substitution and could cause
                         shape mismatches.  For models with QK normalization
                         applied right before attention, pass ``qk_norm_q``
                         here instead of the raw projection.
            k_proj:      Same semantics as ``q_proj``, for the K path.  Pass
                         ``qk_norm_k`` when the model applies QK-Norm before
                         attention.
            v_proj:      Value projection module (no norm variant needed).
        """
        if q_proj is None or k_proj is None or v_proj is None:
            return super().apply(module, device_mesh)

        cp_size = device_mesh.mesh.numel()
        ds = self.ulysses_degree if self.ulysses_degree is not None else cp_size
        if cp_size % ds != 0:
            raise ValueError(
                f"cp_size ({cp_size}) must be divisible by ulysses_degree ({ds})."
            )
        co = cp_size // ds

        if ds == 1:
            if self.load_balance:
                return super().apply(module, device_mesh)
            return self._apply_colossal_async(module, device_mesh, cp_size, k_proj, v_proj)

        return self._apply_a2a_async(module, device_mesh, ds, co, q_proj, k_proj, v_proj)

    def _apply_colossal_async(
        self,
        module: Module,
        device_mesh: DeviceMesh,
        cp_size: int,
        k_proj: Module,
        v_proj: Module,
    ) -> Module:
        """Register Pure Colossal async K/V AllGather hooks."""
        co_submesh = _ensure_1d(device_mesh)
        group = co_submesh.get_group()
        fwd_ag_slots = {"k": None, "v": None}
        bwd_ag_slots = {"k": [], "v": []}
        self._register_ag_proj_hooks(
            k_proj,
            v_proj,
            group=group,
            world_size=cp_size,
            fwd_slots=fwd_ag_slots,
            bwd_slots=bwd_ag_slots,
        )
        platform.register_forward_pre_hook(
            module,
            partial(
                self._attn_pre_hook_colossal,
                co_submesh=co_submesh,
                group=group,
                world_size=cp_size,
                fwd_slots=fwd_ag_slots,
                bwd_slots=bwd_ag_slots,
            ),
            with_kwargs=True,
        )
        module.register_forward_hook(
            partial(self._post_hook_colossal, co_submesh=co_submesh)
        )
        return module

    def _apply_a2a_async(  # pylint: disable=too-many-arguments
        self,
        module: Module,
        device_mesh: DeviceMesh,
        ds: int,
        co: int,
        q_proj: Module,
        k_proj: Module,
        v_proj: Module,
    ) -> Module:
        """Register Pure Ulysses or Hybrid async A2A hooks."""
        fwd_slots = {"q": None, "k": None, "v": None}
        bwd_slots = {"q": [], "k": [], "v": []}

        if co == 1:
            ds_submesh = _ensure_1d(device_mesh)
            group = ds_submesh.get_group()
            pre_hook = partial(
                self._attn_pre_hook_ulysses,
                group=group,
                world_size=ds,
                fwd_slots=fwd_slots,
                bwd_slots=bwd_slots,
            )
        else:
            two_d_mesh = _build_2d_mesh(device_mesh, ds, co)
            dim_names = two_d_mesh.mesh_dim_names
            assert dim_names is not None, "2-D mesh must have mesh_dim_names (guaranteed by _build_2d_mesh)"
            ds_submesh = two_d_mesh[dim_names[1]]
            group = ds_submesh.get_group()
            pre_hook = partial(
                self._attn_pre_hook_hybrid,
                group=group,
                world_size=ds,
                two_d_mesh=two_d_mesh,
                fwd_slots=fwd_slots,
                bwd_slots=bwd_slots,
            )

        self._register_proj_hooks(
            q_proj,
            k_proj,
            v_proj,
            group=group,
            world_size=ds,
            fwd_slots=fwd_slots,
            bwd_slots=bwd_slots,
        )
        platform.register_forward_pre_hook(
            module,
            pre_hook,
            with_kwargs=True,
        )
        module.register_forward_hook(
            partial(self._attn_post_hook_ata, ds_submesh=ds_submesh)
        )
        return module

    # ------------------------------------------------------------------
    # Shared: projection hooks registration
    # ------------------------------------------------------------------

    def _register_proj_hooks(self, q_proj, k_proj, v_proj, group, world_size, fwd_slots, bwd_slots):
        """Register forward and backward hooks on all three projection modules."""
        for key, proj in [("q", q_proj), ("k", k_proj), ("v", v_proj)]:
            proj.register_forward_hook(
                partial(self._proj_post_hook, key=key, group=group, world_size=world_size,
                        fwd_slots=fwd_slots)
            )
            platform.register_full_backward_pre_hook(
                proj,
                partial(self._proj_bwd_pre_hook, bwd_slot=bwd_slots[key])
            )

    def _proj_post_hook(  # pylint: disable=unused-argument,too-many-arguments
        self, module, inputs, output, key, group, world_size, fwd_slots
    ):
        """Launch async seq→head A2A after projection; return original output unchanged."""
        tensor = output.to_local() if isinstance(output, DTensor) else output
        fwd_slots[key] = _launch_async_a2a_seq_to_head(
            tensor, group, world_size, self.head_dim
        )
        return output

    def _register_ag_proj_hooks(self, k_proj, v_proj, group, world_size, fwd_slots, bwd_slots):
        """Register async AllGather hooks for K/V projection modules."""
        for key, proj in [("k", k_proj), ("v", v_proj)]:
            proj.register_forward_hook(
                partial(self._proj_ag_post_hook, key=key, group=group, world_size=world_size,
                        fwd_slots=fwd_slots)
            )
            platform.register_full_backward_pre_hook(
                proj,
                partial(self._proj_ag_bwd_pre_hook, bwd_slot=bwd_slots[key])
            )

    def _proj_ag_post_hook(  # pylint: disable=unused-argument,too-many-arguments
        self, module, inputs, output, key, group, world_size, fwd_slots
    ):
        """Launch async K/V AllGather after projection; return original output."""
        tensor = output.to_local() if isinstance(output, DTensor) else output
        fwd_slots[key] = _launch_async_allgather_seq(
            tensor, group, world_size, self.seq_dim
        )
        return output

    def _get_qkv_value(self, args, kwargs, qkv_pos: int):
        """Return Q/K/V value from positional args or configured kwargs."""
        idx = self.qkv_indices[qkv_pos]
        if idx < len(args):
            return args[idx]
        if qkv_pos < len(self.qkv_kwarg_names):
            name = self.qkv_kwarg_names[qkv_pos]
            if name in kwargs:
                return kwargs[name]
        return None

    def _set_qkv_value(self, args, kwargs, qkv_pos: int, value):
        """Set Q/K/V value in positional args or configured kwargs."""
        idx = self.qkv_indices[qkv_pos]
        if idx < len(args):
            args[idx] = value
            return
        if qkv_pos < len(self.qkv_kwarg_names):
            name = self.qkv_kwarg_names[qkv_pos]
            if name in kwargs:
                kwargs[name] = value

    # ------------------------------------------------------------------
    # Internal: wait for pre-launched communication handles
    # ------------------------------------------------------------------

    def _wait_a2a(self, tensor, group, world_size, fwd_slots, key, bwd_slot):
        """Wait for pre-launched A2A; returns head-scattered tensor (differentiable)."""
        work, out_perm = fwd_slots[key]
        fwd_slots[key] = None
        return platform.differentiable_async_a2a_wait(
            tensor, work, out_perm, group, world_size,
            self.seq_dim, self.head_dim,  # concat_dim=seq_dim, split_dim=head_dim
            bwd_slot,
        )

    def _wait_allgather(self, tensor, group, world_size, work, out_perm, bwd_slot=None):
        """Wait for pre-launched AllGather and return gathered tensor."""
        return platform.differentiable_async_allgather_wait(
            tensor,
            work,
            out_perm,
            group,
            world_size,
            self.seq_dim,
            bwd_slot,
        )

    # ------------------------------------------------------------------
    # QKV transform helpers
    # ------------------------------------------------------------------

    def _apply_qkv_transforms(
        self,
        args,
        kwargs,
        transform_q,
        transform_k,
        transform_v,
    ):
        """Apply transforms to Q/K/V and return modified ``(args, kwargs)``.

        Each transform receives the local tensor (DTensor unwrapped if needed)
        and returns the value to place back into the attention call.
        """
        new_args = list(args)
        new_kwargs = dict(kwargs)

        transforms = (transform_q, transform_k, transform_v)
        for pos, transform in enumerate(transforms):
            value = self._get_qkv_value(new_args, new_kwargs, pos)
            if value is None:
                continue
            self._set_qkv_value(
                new_args,
                new_kwargs,
                pos,
                transform(_to_local(value)),
            )
        return tuple(new_args), new_kwargs

    # ------------------------------------------------------------------
    # Attention pre-hooks
    # ------------------------------------------------------------------

    def _attn_pre_hook_ulysses(  # pylint: disable=unused-argument,too-many-arguments
        self, module, args, kwargs, group, world_size, fwd_slots, bwd_slots
    ):
        """Wait Q/K/V A2A and return head-scattered args/kwargs."""
        def make_a2a(key):
            return lambda value: self._wait_a2a(
                value,
                group,
                world_size,
                fwd_slots,
                key,
                bwd_slots[key],
            )

        return self._apply_qkv_transforms(
            args,
            kwargs,
            transform_q=make_a2a("q"),
            transform_k=make_a2a("k"),
            transform_v=make_a2a("v"),
        )

    def _attn_pre_hook_colossal(  # pylint: disable=unused-argument,too-many-arguments
        self, module, args, kwargs, co_submesh, group, world_size, fwd_slots, bwd_slots
    ):
        """Wait K/V AllGather, wrap Q as local shard and return CP-ready args/kwargs."""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        q_value = self._get_qkv_value(new_args, new_kwargs, 0)
        if q_value is not None and not isinstance(q_value, DTensor):
            self._set_qkv_value(
                new_args,
                new_kwargs,
                0,
                DTensor.from_local(
                    _to_local(q_value),
                    co_submesh,
                    (Shard(self.seq_dim),),
                ),
            )

        for pos, key in enumerate(("k", "v"), start=1):
            value = self._get_qkv_value(new_args, new_kwargs, pos)
            if value is None:
                continue
            local = _to_local(value)
            work, out_perm = fwd_slots[key]
            fwd_slots[key] = None
            gathered = self._wait_allgather(
                local,
                group,
                world_size,
                work,
                out_perm,
                bwd_slots[key],
            )
            self._set_qkv_value(
                new_args,
                new_kwargs,
                pos,
                DTensor.from_local(
                    _to_local(gathered),
                    co_submesh,
                    (Replicate(),),
                ),
            )

        return tuple(new_args), new_kwargs

    def _attn_pre_hook_hybrid(  # pylint: disable=too-many-locals,too-many-arguments
        self, module, args, kwargs, group, world_size, two_d_mesh,  # pylint: disable=unused-argument
        fwd_slots, bwd_slots
    ):
        """Wait Ulysses A2A for Q/K/V, then synchronously gather K/V on the co-submesh."""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        co_submesh = two_d_mesh[two_d_mesh.mesh_dim_names[0]]

        q_value = self._get_qkv_value(new_args, new_kwargs, 0)
        if q_value is not None:
            self._set_qkv_value(
                new_args,
                new_kwargs,
                0,
                DTensor.from_local(
                    self._wait_a2a(
                        _to_local(q_value),
                        group,
                        world_size,
                        fwd_slots,
                        "q",
                        bwd_slots["q"],
                    ),
                    two_d_mesh,
                    (Shard(self.seq_dim), Shard(self.head_dim)),
                ),
            )

        k_value = self._get_qkv_value(new_args, new_kwargs, 1)
        if k_value is not None:
            k_ul = self._wait_a2a(
                _to_local(k_value),
                group,
                world_size,
                fwd_slots,
                "k",
                bwd_slots["k"],
            )
            k_full = _gather_seq(k_ul, co_submesh, self.seq_dim)
            self._set_qkv_value(
                new_args,
                new_kwargs,
                1,
                DTensor.from_local(
                    _to_local(k_full),
                    two_d_mesh,
                    (Replicate(), Shard(self.head_dim)),
                ),
            )

        v_value = self._get_qkv_value(new_args, new_kwargs, 2)
        if v_value is not None:
            v_ul = self._wait_a2a(
                _to_local(v_value),
                group,
                world_size,
                fwd_slots,
                "v",
                bwd_slots["v"],
            )
            v_full = _gather_seq(v_ul, co_submesh, self.seq_dim)
            self._set_qkv_value(
                new_args,
                new_kwargs,
                2,
                DTensor.from_local(
                    _to_local(v_full),
                    two_d_mesh,
                    (Replicate(), Shard(self.head_dim)),
                ),
            )

        return tuple(new_args), new_kwargs

    # ------------------------------------------------------------------
    # Attention post-hook (Ulysses and Hybrid share the same reverse ATA)
    # ------------------------------------------------------------------

    def _attn_post_hook_ata(self, module, args, output, ds_submesh):  # pylint: disable=unused-argument
        """Reverse head→seq gather on ds_submesh; returns local tensor."""
        def _process(o):
            if isinstance(o, (Tensor, DTensor)):
                if isinstance(o, DTensor):
                    o = o.to_local()
                return _gather_head_to_seq(
                    o, ds_submesh, self.seq_dim, self.head_dim
                ).to_local()
            return o

        if isinstance(output, (tuple, list)):
            return type(output)(_process(o) for o in output)
        return _process(output)

    # ------------------------------------------------------------------
    # Backward: wait A2A handle (launched by autograd) before proj GEMM
    # ------------------------------------------------------------------

    def _proj_bwd_pre_hook(self, module, grad_output, bwd_slot):  # pylint: disable=unused-argument
        """Wait backward A2A just before proj GEMM; replace grad with seq-form.

        The async head→seq A2A is launched inside _TorchAsyncA2AFunction.backward
        and appended to ``bwd_slot``. Waiting here lets the A2A overlap with the
        preceding proj GEMM.
        """
        work, out_perm = bwd_slot.pop()
        work.wait()
        d_seq = _a2a_reconstruct(out_perm, self.head_dim)
        return (d_seq,) + grad_output[1:] if isinstance(grad_output, tuple) else (d_seq,)

    def _proj_ag_bwd_pre_hook(self, module, grad_output, bwd_slot):  # pylint: disable=unused-argument
        """Wait backward reduce-scatter just before K/V projection GEMM."""
        work, out_perm, gather_dim = bwd_slot.pop()
        work.wait()
        d_local = _allgather_reconstruct(out_perm, gather_dim)
        return (d_local,) + grad_output[1:] if isinstance(grad_output, tuple) else (d_local,)
