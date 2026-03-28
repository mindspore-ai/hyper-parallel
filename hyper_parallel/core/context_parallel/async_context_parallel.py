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
"""AsyncContextParallel: overlap projection GEMM with all-to-all communication.

Supports Pure Ulysses, Hybrid CP modes. Falls back to sync ContextParallel
when q/k/v_proj not provided or in Pure Colossal AI mode.

Forward:  proj hooks launch async A2A → attn pre-hook waits Q/K/V → attn hook gathers output
Backward: autograd backward launches async A2A → proj pre-hooks wait before GEMMs
"""
from functools import partial
from typing import Optional, cast

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
    out_perm, work = platform.all_to_all_single(x_perm, list(x_perm.shape), group, async_op=True)
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


# ---------------------------------------------------------------------------
# AsyncContextParallel
# ---------------------------------------------------------------------------

class AsyncContextParallel(ContextParallel):
    """Context Parallel with projection–A2A compute overlap.

    Requires ``q_proj``, ``k_proj``, ``v_proj`` in :meth:`apply`; otherwise
    falls back to synchronous :class:`ContextParallel`.

    Pure Colossal AI (``ulysses_degree=1``) automatically falls back to sync
    because K/V AllGather is a barrier collective.

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
        ``q/k/v_proj`` is ``None`` or in Pure Colossal AI mode.

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
            # Pure Colossal AI: K/V AllGather cannot be made async. Fall back.
            return super().apply(module, device_mesh)

        # Per-layer handle slots — local to this apply() call, bound to hooks via partial.
        #
        # fwd_slots is a plain dict.  _proj_post_hook and _wait_a2a both receive the
        # same dict reference via partial, so a simple assignment fwd_slots[key] = ...
        # in _proj_post_hook is immediately visible to _wait_a2a — no list wrapper needed.
        #
        # bwd_slots[key] is a list held by both _wait_a2a and the autograd wait function
        # The autograd function receives the list object itself (as handle_box) and appends
        # to it; _proj_bwd_pre_hook pops from the same list.  We cannot use a plain dict
        # value here because the autograd function would hold a stale reference if we later
        # reassigned bwd_slots[key].
        fwd_slots = {"q": None, "k": None, "v": None}
        bwd_slots = {"q": [], "k": [], "v": []}

        if co == 1:
            # Pure Ulysses
            ds_submesh = _ensure_1d(device_mesh)
            group = ds_submesh.get_group()
            self._register_proj_hooks(q_proj, k_proj, v_proj, group=group, world_size=ds,
                                      fwd_slots=fwd_slots, bwd_slots=bwd_slots)
            module.register_forward_pre_hook(
                partial(self._attn_pre_hook_ulysses, group=group, world_size=ds,
                        fwd_slots=fwd_slots, bwd_slots=bwd_slots)
            )
        else:
            # Hybrid: async Ulysses A2A + sync Colossal AllGather
            two_d_mesh = _build_2d_mesh(device_mesh, ds, co)
            dim_names = two_d_mesh.mesh_dim_names
            assert dim_names is not None, "2-D mesh must have mesh_dim_names (guaranteed by _build_2d_mesh)"
            ds_submesh = two_d_mesh[dim_names[1]]
            group = ds_submesh.get_group()
            self._register_proj_hooks(q_proj, k_proj, v_proj, group=group, world_size=ds,
                                      fwd_slots=fwd_slots, bwd_slots=bwd_slots)
            module.register_forward_pre_hook(
                partial(self._attn_pre_hook_hybrid, group=group, world_size=ds,
                        two_d_mesh=two_d_mesh, fwd_slots=fwd_slots, bwd_slots=bwd_slots)
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
            proj.register_full_backward_pre_hook(
                partial(self._proj_bwd_pre_hook, bwd_slot=bwd_slots[key])
            )

    def _proj_post_hook(self, module, inputs, output, key, group, world_size, fwd_slots):  # pylint: disable=unused-argument,too-many-arguments
        """Launch async seq→head A2A after projection; return original output unchanged."""
        tensor = output.to_local() if isinstance(output, DTensor) else output
        fwd_slots[key] = _launch_async_a2a_seq_to_head(
            tensor, group, world_size, self.head_dim
        )
        return output

    # ------------------------------------------------------------------
    # Internal: wait for a single pre-launched A2A handle
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

    # ------------------------------------------------------------------
    # Attention pre-hooks
    # ------------------------------------------------------------------

    def _attn_pre_hook_ulysses(self, module, args, group, world_size,  # pylint: disable=unused-argument,too-many-arguments
                               fwd_slots, bwd_slots):
        """Wait Q/K/V A2A; return head-scattered args."""
        q_idx, k_idx, v_idx = self.qkv_indices
        new_args = list(args)

        def _local(t):
            return t.to_local() if isinstance(t, DTensor) else t

        new_args[q_idx] = self._wait_a2a(_local(new_args[q_idx]), group, world_size,
                                          fwd_slots, "q", bwd_slots["q"])
        new_args[k_idx] = self._wait_a2a(_local(new_args[k_idx]), group, world_size,
                                          fwd_slots, "k", bwd_slots["k"])
        new_args[v_idx] = self._wait_a2a(_local(new_args[v_idx]), group, world_size,
                                          fwd_slots, "v", bwd_slots["v"])
        return tuple(new_args)

    def _attn_pre_hook_hybrid(  # pylint: disable=too-many-locals,too-many-arguments
        self, module, args, group, world_size, two_d_mesh,  # pylint: disable=unused-argument
        fwd_slots, bwd_slots
    ):
        """Wait Ulysses A2A for Q/K/V, AllGather K/V on co-submesh, wrap as 2-D DTensors."""
        q_idx, k_idx, v_idx = self.qkv_indices
        new_args = list(args)

        def _local(t):
            return t.to_local() if isinstance(t, DTensor) else t

        # Wait Ulysses A2A for Q and K
        q_ul = cast(Tensor, self._wait_a2a(_local(new_args[q_idx]), group, world_size,
                                            fwd_slots, "q", bwd_slots["q"]))
        k_ul = cast(Tensor, self._wait_a2a(_local(new_args[k_idx]), group, world_size,
                                            fwd_slots, "k", bwd_slots["k"]))

        # AllGather K on co-submesh (while V A2A is still in flight)
        co_submesh = two_d_mesh[two_d_mesh.mesh_dim_names[0]]
        k_full = _gather_seq(k_ul, co_submesh, self.seq_dim)

        # Wait V A2A, then AllGather V
        v_ul = cast(Tensor, self._wait_a2a(_local(new_args[v_idx]), group, world_size,
                                            fwd_slots, "v", bwd_slots["v"]))
        v_full = _gather_seq(v_ul, co_submesh, self.seq_dim)

        def _local_dt(dt):
            return dt.to_local() if isinstance(dt, DTensor) else dt

        new_args[q_idx] = DTensor.from_local(
            q_ul, two_d_mesh, (Shard(self.seq_dim), Shard(self.head_dim))
        )
        new_args[k_idx] = DTensor.from_local(
            _local_dt(k_full), two_d_mesh, (Replicate(), Shard(self.head_dim))
        )
        new_args[v_idx] = DTensor.from_local(
            _local_dt(v_full), two_d_mesh, (Replicate(), Shard(self.head_dim))
        )
        return tuple(new_args)

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
