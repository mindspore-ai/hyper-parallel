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
"""cp_utils: Context Parallel utilities (05 §4.4.2 / §6.3.4 canonical).

- ``flex_cp_allgather``: K/V all-gather along the CP dim (reuses
  cp_mesh.get_group(); new_group is forbidden);
- ``ulysses_seq_to_head`` / ``ulysses_head_to_seq``: differentiable Pure
  Ulysses all-to-all layout transforms;
- ``shard_batch_for_cp``: data-pipeline CP sharding (aligned with the THD
  contract of the 02 collater);
- ``_shard_seq_lens_for_cp``: recompute seq_lens/seq_lens_padded per CP rank.
- Ulysses collectives and MoME/MLA/DSA model-side adaptations.

Note (G5): the seq_len % (2*cp) padding constraint originates from the
zigzag/ring load-balancing scheme; this design uses all-gather K/V +
contiguous chunk (D-01'' rejected ring), so each rank's Q chunk is
equal-length and FLOPs are naturally balanced -- the constraint is redundant
but harmless: the implementation is kept and documented here.
"""

import contextvars
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform


_ULYSSES_WRAPPED_FLAG = "_hyper_ulysses_wrapped"
platform = get_platform()

_HYBRID_MESH_CACHE = {}


class _SeqAllToAll(torch.autograd.Function):
    """Autograd-aware exchange from one tensor dimension to another."""

    @staticmethod
    def forward(ctx, group, tensor, scatter_dim, gather_dim):
        """Exchange tensor shards between the sequence and head dimensions."""
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group)
        if tensor.size(scatter_dim) % world_size:
            raise ValueError(
                f"dimension {scatter_dim} size {tensor.size(scatter_dim)} "
                f"is not divisible by CP size {world_size}")
        inputs = [
            part.contiguous()
            for part in tensor.chunk(world_size, dim=scatter_dim)
        ]
        outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
        dist.all_to_all(outputs, inputs, group=group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _SeqAllToAll.apply(
            ctx.group, grad_output, ctx.gather_dim, ctx.scatter_dim)
        return None, grad_input, None, None


@dataclass(frozen=True)
class _UlyssesContext:
    """Communication context derived from the framework-owned CP mesh."""

    cp_mesh: Any

    @property
    def group(self):
        return self.cp_mesh.get_group()

    @property
    def size(self):
        return self.cp_mesh.size()

    @property
    def rank(self):
        return self.cp_mesh.get_local_rank()


def _sequence_to_head(tensor, context):
    """[B, S/CP, H, D] -> [B, S, H/CP, D]."""
    return _SeqAllToAll.apply(context.group, tensor, 2, 1)


def _head_to_sequence(tensor, context):
    """[B, S, H/CP, D] -> [B, S/CP, H, D]."""
    return _SeqAllToAll.apply(context.group, tensor, 1, 2)


def _gather_sequence(tensor, context):
    """Gather sequence shards and sum their gradients during backward."""
    return torch.cat(
        differentiable_all_gather(
            tensor.contiguous(), group=context.group), dim=1)


def _slice_sequence(tensor, dim, context):
    if tensor.size(dim) % context.size:
        raise ValueError(
            f"sequence length {tensor.size(dim)} is not divisible by "
            f"CP size {context.size}")
    length = tensor.size(dim) // context.size
    return tensor.narrow(
        dim, context.rank * length, length).contiguous()


def _global_seq_len(actual, length, device):
    if actual is None:
        return [length]
    if isinstance(actual, Tensor):
        return actual.to(device=device, dtype=torch.int32)
    return actual


def _slice_sink(tensor, context):
    if tensor is None or tensor.size(2) == 1:
        return tensor
    if tensor.size(2) % context.size:
        raise ValueError(
            f"parameter-sink heads {tensor.size(2)} are not divisible by "
            f"CP size {context.size}")
    count = tensor.size(2) // context.size
    start = context.rank * count
    return tensor[:, :, start:start + count].contiguous()


@dataclass
class _DSATensorContext:
    query: Tensor
    key: Tensor
    q_pe: Tensor
    k_pe: Tensor


_dsa_tensor_context = contextvars.ContextVar(
    "hyper_dsa_tensor_context", default=None)


def _mome_cp_halo_exchange(attention_module, context):
    """Configure cross-rank halo exchange for MoME convolution."""
    original = getattr(attention_module, "_apply_mome")
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def apply_mome_with_halo(hidden_states, mome_mask, conv, use_fused):
        halo = conv.kernel_size[0] - 1
        if halo == 0:
            return original(hidden_states, mome_mask, conv, use_fused)
        if hidden_states.size(1) < halo:
            raise ValueError(
                f"local sequence {hidden_states.size(1)} is shorter than "
                f"MOME halo {halo}")
        tails = differentiable_all_gather(
            hidden_states[:, -halo:].contiguous(), group=context.group)
        mask_tail = mome_mask[:, -halo:].to(
            hidden_states.dtype).contiguous()
        masks = differentiable_all_gather(mask_tail, group=context.group)
        if context.rank == 0:
            left_states = tails[-1] * 0
            left_mask = torch.zeros_like(mask_tail, dtype=torch.bool)
        else:
            left_states = tails[context.rank - 1]
            left_mask = masks[context.rank - 1].bool()
        output = original(
            torch.cat((left_states, hidden_states), dim=1),
            torch.cat((left_mask, mome_mask.bool()), dim=1), conv, use_fused)
        return output[:, halo:].contiguous()

    setattr(apply_mome_with_halo, _ULYSSES_WRAPPED_FLAG, True)
    setattr(attention_module, "_apply_mome", apply_mome_with_halo)


def _mla_cp_alltoall(attention_functions, context):
    """Configure CP all-to-all around the MLA backend."""
    original = attention_functions["npu_fa_rescale"]
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def mla_with_sequence_head_exchange(
            module, query, key, value, attention_mask, **kwargs):
        if module.attention_type != "mla":
            return original(
                module, query, key, value, attention_mask, **kwargs)
        if not module.apply_FA_rescale or module.use_fused_sink_fa:
            raise ValueError(
                "MLA CP supports only non-fused npu_fa_rescale")
        local_shape = tuple(query.shape)
        query = _sequence_to_head(query, context)
        key = _sequence_to_head(key, context)
        value = _sequence_to_head(value, context)
        length = query.size(1)
        call_kwargs = kwargs.copy()
        call_kwargs.update(
            seq_length=length,
            n_head=query.size(2),
            actual_q_len=_global_seq_len(
                kwargs.get("actual_q_len"), length, query.device),
            actual_kv_len=_global_seq_len(
                kwargs.get("actual_kv_len"), length, query.device),
            param_sink_key=_slice_sink(
                kwargs.get("param_sink_key"), context),
            param_sink_value=_slice_sink(
                kwargs.get("param_sink_value"), context),
        )
        output = _head_to_sequence(
            original(
                module, query, key, value, attention_mask, **call_kwargs),
            context)
        if output.shape[:3] != torch.Size(local_shape[:3]):
            raise RuntimeError(
                f"MLA CP output {tuple(output.shape)} does not restore "
                f"{local_shape}")
        return output

    setattr(mla_with_sequence_head_exchange, _ULYSSES_WRAPPED_FLAG, True)
    attention_functions["npu_fa_rescale"] = mla_with_sequence_head_exchange


def _dsa_cp_alltoall(attention_module, attention_functions, context):
    """Configure CP all-to-all for the DSA indexer, attention, and KL loss."""
    original_indexer = attention_module.dsa_lightning_indexer_forward
    original_sparse = attention_functions["dsa_sparse_attention"]
    original_kl = attention_module.SparseLightningIndexerKLLossTrainFunction

    if not getattr(original_indexer, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_indexer)
        def index_with_gathered_sequence(
                module, index_query, index_key, merge_weight,
                actual_q_len, actual_kv_len):
            index_query = _sequence_to_head(index_query, context)
            merge_weight = _sequence_to_head(
                merge_weight.unsqueeze(-1), context).squeeze(-1)
            index_key = _gather_sequence(index_key, context)
            length = index_query.size(1)
            return original_indexer(
                module, index_query, index_key, merge_weight,
                _global_seq_len(
                    actual_q_len, length, index_query.device),
                _global_seq_len(
                    actual_kv_len, length, index_query.device))

        setattr(index_with_gathered_sequence, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.dsa_lightning_indexer_forward = (
            index_with_gathered_sequence)

    if not getattr(original_sparse, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_sparse)
        def sparse_attention_with_gathered_kv(
                module, query, key, value, attention_mask, **kwargs):
            del attention_mask
            local_shape = tuple(query.shape)
            query = _sequence_to_head(query, context)
            q_pe = _sequence_to_head(kwargs["q_pe"], context)
            key = _gather_sequence(key, context)
            value = _gather_sequence(value, context)
            k_pe = _gather_sequence(kwargs["k_pe"], context)
            length = query.size(1)
            call_kwargs = kwargs.copy()
            call_kwargs.update(
                q_pe=q_pe, k_pe=k_pe, seq_length=length,
                n_head=query.size(2),
                actual_q_len=_global_seq_len(
                    kwargs.get("actual_q_len"), length, query.device),
                actual_kv_len=_global_seq_len(
                    kwargs.get("actual_kv_len"), length, query.device),
            )
            old_heads = module.num_heads
            module.num_heads = query.size(2)
            try:
                output, softmax_max, softmax_sum = original_sparse(
                    module, query, key, value, None, **call_kwargs)
            finally:
                module.num_heads = old_heads
            if module.training and not module.freeze_dsa:
                _dsa_tensor_context.set(_DSATensorContext(
                    query=query, key=key, q_pe=q_pe, k_pe=k_pe))
            output = _head_to_sequence(output, context)
            if output.shape[:3] != torch.Size(local_shape[:3]):
                raise RuntimeError(
                    f"DSA CP output {tuple(output.shape)} does not restore "
                    f"{local_shape}")
            return output, softmax_max, softmax_sum

        setattr(sparse_attention_with_gathered_kv, _ULYSSES_WRAPPED_FLAG, True)
        attention_functions["dsa_sparse_attention"] = (
            sparse_attention_with_gathered_kv)

    if not getattr(original_kl, _ULYSSES_WRAPPED_FLAG, False):
        class CPDSAKLLoss:
            """Proxy the DSA KL loss with CP-transformed attention inputs."""

            @staticmethod
            def apply(index_query, index_key, merge_weight, query, key,
                      topk_indices, softmax_max, softmax_sum, query_rope,
                      key_rope, actual_seq_qlen, actual_seq_klen, scale,
                      loss_coeff):
                """Apply the original KL loss with saved global sequence tensors."""
                saved = _dsa_tensor_context.get()
                if saved is None:
                    return original_kl.apply(
                        index_query, index_key, merge_weight, query, key,
                        topk_indices, softmax_max, softmax_sum, query_rope,
                        key_rope, actual_seq_qlen, actual_seq_klen, scale,
                        loss_coeff)
                _dsa_tensor_context.set(None)
                query_tnd, key_tnd, q_pe_tnd, k_pe_tnd = [
                    tensor.flatten(0, 1) for tensor in
                    (saved.query, saved.key, saved.q_pe, saved.k_pe)]
                length = saved.query.size(1)
                return original_kl.apply(
                    index_query, index_key, merge_weight, query_tnd, key_tnd,
                    topk_indices, softmax_max, softmax_sum, q_pe_tnd,
                    k_pe_tnd,
                    _global_seq_len(
                        actual_seq_qlen, length, saved.query.device),
                    _global_seq_len(
                        actual_seq_klen, length, saved.query.device),
                    scale, loss_coeff)

        setattr(CPDSAKLLoss, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.SparseLightningIndexerKLLossTrainFunction = (
            CPDSAKLLoss)


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize and validate a tensor dimension."""
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is out of range for a {ndim}D tensor")
    return normalized


def _reconstruct_all_to_all(output: Tensor, concat_dim: int) -> Tensor:
    """Move the source-rank axis next to ``concat_dim`` and merge it."""
    rank_axis_position = concat_dim + 1
    permutation = (
        list(range(1, rank_axis_position))
        + [0]
        + list(range(rank_axis_position, output.dim()))
    )
    reconstructed = output.permute(permutation).contiguous()
    shape = list(reconstructed.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return reconstructed.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _move_dim_to_front(tensor: Tensor, dim: int) -> Tensor:
    """Move one dimension to the leading communication dimension."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    return tensor.movedim(dim, 0).contiguous()


def _move_dim_from_front(tensor: Tensor, dim: int) -> Tensor:
    """Restore a leading communication dimension to its original position."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    return tensor.movedim(0, dim).contiguous()


def _prepare_ulysses_send(
    tensor: Tensor,
    scatter_dim: int,
    world_size: int,
) -> Tensor:
    """Split one tensor dimension and move the destination-rank axis first."""
    ndim = tensor.dim()
    scatter_dim = _normalize_dim(scatter_dim, ndim)
    shape = list(tensor.shape)
    scatter_size = shape[scatter_dim]
    if scatter_size % world_size:
        raise ValueError(
            f"tensor dimension {scatter_dim} size ({scatter_size}) must be "
            f"divisible by Ulysses degree ({world_size})"
        )
    split_shape = (
        shape[:scatter_dim]
        + [world_size, scatter_size // world_size]
        + shape[scatter_dim + 1:]
    )
    permutation = (
        [scatter_dim]
        + list(range(scatter_dim))
        + list(range(scatter_dim + 1, ndim + 1))
    )
    return tensor.contiguous().reshape(split_shape).permute(permutation).contiguous()


class _AsyncAllGatherWait(torch.autograd.Function):
    """Attach a pre-launched all-gather to autograd at its wait point."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        work: Any,
        output: Tensor,
        group: Any,
        world_size: int,
        gather_dim: int,
    ) -> Tensor:
        """Wait for a launched all-gather and expose its autograd edge."""
        del tensor
        ctx.group = group
        ctx.world_size = world_size
        ctx.gather_dim = gather_dim
        work.wait()
        return _move_dim_from_front(output, gather_dim)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None, None]:
        """Reduce and select this rank's input-gradient sequence shard."""
        grad = grad_output.contiguous().clone()
        work = dist.all_reduce(grad, group=ctx.group, async_op=True)
        work.wait()
        rank = dist.get_rank(ctx.group)
        local = torch.chunk(
            grad, ctx.world_size, dim=ctx.gather_dim
        )[rank].contiguous()
        return (
            local,
            None,
            None,
            None,
            None,
            None,
        )


class _AsyncUlyssesWait(torch.autograd.Function):
    """Attach a pre-launched sequence-to-head A2A to autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        work: Any,
        output: Tensor,
        group: Any,
        world_size: int,
        seq_dim: int,
        head_dim: int,
    ) -> Tensor:
        """Wait for a launched A2A and expose its autograd edge."""
        del tensor
        ctx.group = group
        ctx.world_size = world_size
        ctx.seq_dim = seq_dim
        ctx.head_dim = head_dim
        work.wait()
        return _reconstruct_all_to_all(output, seq_dim)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        """Apply the inverse A2A to restore the input-gradient layout."""
        send = _prepare_ulysses_send(
            grad_output,
            scatter_dim=ctx.seq_dim,
            world_size=ctx.world_size,
        )
        output, work = platform.all_to_all_single(
            send, list(send.shape), ctx.group, async_op=True
        )
        work.wait()
        grad_input = _reconstruct_all_to_all(output, ctx.head_dim)
        return grad_input, None, None, None, None, None, None


@dataclass
class AsyncCPCollective:
    """A local-tensor CP collective launched before its consumer."""

    tensor: Tensor
    work: Any
    output: Tensor
    group: Any
    world_size: int
    kind: str
    seq_dim: int
    head_dim: Optional[int] = None
    waited: bool = False

    def wait(self) -> Tensor:
        """Materialize the collective result exactly once."""
        if self.waited:
            raise RuntimeError("an async CP collective handle cannot be waited twice")
        self.waited = True
        if self.work is None:
            return self.tensor
        if self.kind == "allgather":
            return _AsyncAllGatherWait.apply(
                self.tensor,
                self.work,
                self.output,
                self.group,
                self.world_size,
                self.seq_dim,
            )
        if self.kind == "ulysses":
            return _AsyncUlyssesWait.apply(
                self.tensor,
                self.work,
                self.output,
                self.group,
                self.world_size,
                self.seq_dim,
                self.head_dim,
            )
        raise RuntimeError(f"unknown async CP collective kind {self.kind!r}")


def async_cp_allgather_launch(
    tensor: Tensor,
    gather_dim: int,
    cp_mesh: DeviceMesh,
) -> AsyncCPCollective:
    """Launch an asynchronous all-gather along gather_dim."""
    world_size = cp_mesh.size()
    if world_size <= 1:
        return AsyncCPCollective(
            tensor, None, tensor, None, world_size, "allgather", gather_dim
        )
    gather_dim = _normalize_dim(gather_dim, tensor.dim())
    send = _move_dim_to_front(tensor.detach(), gather_dim)
    output_shape = list(send.shape)
    output_shape[0] *= world_size
    group = cp_mesh.get_group()
    output, work = platform.all_gather_single(
        send, output_shape, group, async_op=True
    )
    return AsyncCPCollective(
        tensor, work, output, group, world_size, "allgather", gather_dim
    )


def async_ulysses_seq_to_head_launch(
    tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> AsyncCPCollective:
    """Launch an asynchronous Ulysses sequence-to-head A2A."""
    world_size = cp_mesh.size()
    if world_size <= 1:
        return AsyncCPCollective(
            tensor,
            None,
            tensor,
            None,
            world_size,
            "ulysses",
            seq_dim,
            head_dim,
        )
    seq_dim = _normalize_dim(seq_dim, tensor.dim())
    head_dim = _normalize_dim(head_dim, tensor.dim())
    if seq_dim == head_dim:
        raise ValueError("Ulysses seq_dim and head_dim must be different")
    send = _prepare_ulysses_send(
        tensor.detach(), scatter_dim=head_dim, world_size=world_size
    )
    group = cp_mesh.get_group()
    output, work = platform.all_to_all_single(
        send, list(send.shape), group, async_op=True
    )
    return AsyncCPCollective(
        tensor,
        work,
        output,
        group,
        world_size,
        "ulysses",
        seq_dim,
        head_dim,
    )


def _ulysses_all_to_all(
    tensor: Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Split one tensor dimension and gather another through all-to-all."""
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return tensor

    ndim = tensor.dim()
    scatter_dim = _normalize_dim(scatter_dim, ndim)
    gather_dim = _normalize_dim(gather_dim, ndim)
    if scatter_dim == gather_dim:
        raise ValueError("Ulysses scatter_dim and gather_dim must be different")

    shape = list(tensor.shape)
    scatter_size = shape[scatter_dim]
    if scatter_size % cp_size != 0:
        raise ValueError(
            f"tensor dimension {scatter_dim} size ({scatter_size}) must be "
            f"divisible by Ulysses degree ({cp_size})"
        )

    split_shape = (
        shape[:scatter_dim]
        + [cp_size, scatter_size // cp_size]
        + shape[scatter_dim + 1:]
    )
    split_ndim = ndim + 1
    permutation = (
        [scatter_dim]
        + list(range(scatter_dim))
        + list(range(scatter_dim + 1, split_ndim))
    )
    send = tensor.contiguous().reshape(split_shape).permute(permutation).contiguous()
    received = platform.differentiable_all_to_all(
        send, list(send.shape), cp_mesh.get_group())
    return _reconstruct_all_to_all(received, gather_dim)


def ulysses_seq_to_head(
    tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Convert a sequence shard into a full-sequence head shard.

    Args:
        tensor: Local Q, K, or V tensor.
        seq_dim: Sequence dimension to gather across the CP group.
        head_dim: Head dimension to split across the CP group.
        cp_mesh: One-dimensional CP DeviceMesh.

    Returns:
        Tensor with the sequence dimension gathered and head dimension sharded.
    """
    return _ulysses_all_to_all(
        tensor,
        scatter_dim=head_dim,
        gather_dim=seq_dim,
        cp_mesh=cp_mesh,
    )


def ulysses_head_to_seq(
    tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Convert a full-sequence head shard back into a sequence shard.

    Args:
        tensor: Local attention output in Ulysses head-sharded layout.
        seq_dim: Full sequence dimension to split across the CP group.
        head_dim: Head dimension to gather across the CP group.
        cp_mesh: One-dimensional CP DeviceMesh.

    Returns:
        Tensor restored to the original local sequence-sharded layout.
    """
    return _ulysses_all_to_all(
        tensor,
        scatter_dim=seq_dim,
        gather_dim=head_dim,
        cp_mesh=cp_mesh,
    )


class _AllGatherAlongDim(torch.autograd.Function):
    """all-gather along cp_dim + backward reduce-scatter semantics (sum across
    ranks, then take this rank's chunk)."""

    @staticmethod
    def forward(ctx, t, cp_dim, group, cp_size):
        ctx.cp_dim = cp_dim
        ctx.group = group
        ctx.cp_size = cp_size
        world_t = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(world_t, t.contiguous(), group=group)
        # cat in cp_rank order: [chunk_rank0, chunk_rank1, ...]
        return torch.cat(world_t, dim=cp_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # reduce-scatter: sum the gradient across ranks, take this rank's chunk
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        rank = dist.get_rank(ctx.group)
        local = torch.chunk(grad, ctx.cp_size, dim=ctx.cp_dim)[rank]
        return local.contiguous(), None, None, None


def flex_cp_allgather(k, v, cp_dim: int, cp_mesh):
    """All-gather K/V along CP dimension for context parallel attention.

    Forward: all-gather K/V along cp_dim (each rank ends up holding the full K/V).
    Backward: reduce-scatter semantics (gradients summed across ranks, then this
      rank's chunk is taken -- implemented explicitly by the _AllGatherAlongDim
      autograd.Function, since plain ``dist.all_gather`` has no autograd kernel).

    Args:
        k, v: [B, N, S_local, H] (cp_dim=2 is the sequence dim).
        cp_dim: the gather dimension.
        cp_mesh: DeviceMesh of the CP dim. The communication group is taken from
            ``cp_mesh.get_group()`` -- already created and cached at DeviceMesh
            construction; **dist.new_group must NOT be called here** (otherwise
            every forward leaks one process group, and the semantics would be
            misaligned).
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return k, v
    group = cp_mesh.get_group()
    return (_AllGatherAlongDim.apply(k, cp_dim, group, cp_size),
            _AllGatherAlongDim.apply(v, cp_dim, group, cp_size))


def _build_hybrid_cp_submeshes(cp_mesh, ulysses_degree: int):
    """Build cached communication submeshes for local-tensor Hybrid CP."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Hybrid CP requires an active CP mesh")
    if isinstance(ulysses_degree, bool) or not isinstance(ulysses_degree, int):
        raise TypeError(
            "Hybrid CP requires integer ulysses_degree, got "
            f"{type(ulysses_degree).__name__}"
        )
    cp_size = cp_mesh.size()
    if not 1 < ulysses_degree < cp_size:
        raise ValueError(
            "Hybrid CP requires 1 < ulysses_degree < cp_size, got "
            f"ulysses_degree={ulysses_degree}, cp_size={cp_size}"
        )
    if cp_size % ulysses_degree:
        raise ValueError(
            f"cp_size ({cp_size}) must be divisible by ulysses_degree "
            f"({ulysses_degree})"
        )

    cache_key = (cp_mesh.to_hash(), ulysses_degree)
    cached = _HYBRID_MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    colossal_degree = cp_size // ulysses_degree
    if cp_mesh.ndim == 2:
        hybrid_mesh = cp_mesh
    elif (cp_mesh.root_mesh is not None
          and cp_mesh.mesh_dim_names == ("cp",)):
        hybrid_mesh = cp_mesh._unflatten(  # pylint: disable=protected-access
            "cp",
            (colossal_degree, ulysses_degree),
            ("colossal", "ulysses"),
        )
    else:
        ranks = list(cp_mesh.rank_list)
        hybrid_mesh = DeviceMesh(
            cp_mesh.device_type,
            [
                ranks[index * ulysses_degree:(index + 1) * ulysses_degree]
                for index in range(colossal_degree)
            ],
            mesh_dim_names=("colossal", "ulysses"),
        )
    dim_names = hybrid_mesh.mesh_dim_names
    if not dim_names or len(dim_names) != 2:
        raise ValueError("Hybrid CP requires a named two-dimensional submesh")
    colossal_mesh = hybrid_mesh[dim_names[0]]
    ulysses_mesh = hybrid_mesh[dim_names[1]]
    result = (ulysses_mesh, colossal_mesh)
    _HYBRID_MESH_CACHE[cache_key] = result
    return result


def hybrid_cp_attention(
        attention_fn: Callable[[Tensor, Tensor, Tensor, dict[str, Any]], Any],
        query: Tensor, key: Tensor, value: Tensor,
        attention_kwargs: dict[str, Any], cp_mesh: Any,
        ulysses_degree: int) -> Any:
    """Run local-tensor Hybrid communication around an attention callable.

    ``attention_kwargs`` are forwarded unchanged. Mask construction and other
    model-input semantics belong to the caller's input-preparation contract.
    """
    ulysses_mesh, colossal_mesh = (
        _build_hybrid_cp_submeshes(cp_mesh, ulysses_degree)
    )
    local_shape = tuple(query.shape)
    query, key, value = (
        ulysses_seq_to_head(tensor, 2, 1, ulysses_mesh)
        for tensor in (query, key, value)
    )
    key, value = flex_cp_allgather(key, value, 2, colossal_mesh)
    output = attention_fn(query, key, value, attention_kwargs)
    output = ulysses_head_to_seq(output, 2, 1, ulysses_mesh)
    if tuple(output.shape) != local_shape:
        raise RuntimeError(
            f"Hybrid CP output shape {tuple(output.shape)} does not restore "
            f"the local query shape {local_shape}"
        )
    return output


def head_tail_load_balance_attention(
        attention_fn: Callable[[Tensor, Tensor, Tensor, dict[str, Any]], Any],
        query: Tensor, key: Tensor, value: Tensor,
        attention_kwargs: dict[str, Any], cp_mesh: Any) -> Any:
    """Run local-tensor Colossal Head-Tail communication.

    ``attention_kwargs`` are forwarded unchanged to both attention calls.
    Any split-specific mask or position metadata must be prepared by the
    caller before entering this communication helper.
    """
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Head-Tail load balance requires an active CP mesh")
    local_q_len = query.shape[2]
    if local_q_len % 2:
        raise ValueError(
            "Head-Tail load balance requires an even local Q sequence "
            f"length, got {local_q_len}; pad the global sequence to a "
            f"multiple of 2 * cp_size ({2 * cp_mesh.size()})"
        )

    rank_list = list(cp_mesh.rank_list)
    local_rank = rank_list.index(platform.get_rank())
    peer_index = cp_mesh.size() - 1 - local_rank
    peer_rank = rank_list[peer_index]
    half = local_q_len // 2
    query_keep = query.narrow(2, 0, half)
    query_tail = query.narrow(2, half, half)
    query_peer = platform.p2p_exchange(query_tail, peer_rank)
    global_key, global_value = flex_cp_allgather(key, value, 2, cp_mesh)

    def run_half(query_half: Tensor) -> Tensor:
        """Run attention for one Head-Tail query half."""
        output = attention_fn(
            query_half, global_key, global_value, attention_kwargs
        )
        if not isinstance(output, Tensor):
            raise TypeError(
                "Head-Tail load balance requires the attention callable to "
                f"return a Tensor, got {type(output).__name__}"
            )
        return output

    keep_output = run_half(query_keep)
    peer_output = run_half(query_peer)
    tail_output = platform.p2p_exchange(peer_output, peer_rank)
    return platform.cat([keep_output, tail_output], dim=2)


def shard_batch_for_cp(batch: dict, cp_mesh) -> dict:
    """Shard the sequence-dim tensors of a batch along the CP mesh
    (05 §6.3.4 canonical).

    Contract (aligned with the 02 collater output):
      - input_ids/labels/position_ids: [B, S] int64
      - seq_lens / seq_lens_padded: [B, max_num_packs] int64, padded with the
        -1000 sentinel
      - qkv_format: "thd" (passthrough)

    Sharding strategy: pad to a multiple of 2*cp, then slice the token
    interval [cp_rank*chunk, (cp_rank+1)*chunk); the seq_lens family is
    recomputed separately (_shard_seq_lens_for_cp).
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return batch

    cp_rank = cp_mesh.get_local_rank()
    seq_len = batch["input_ids"].shape[1]
    pad_len = (-seq_len) % (cp_size * 2)
    chunk = (seq_len + pad_len) // cp_size
    lo = cp_rank * chunk
    hi = lo + chunk
    slc = slice(lo, hi)

    pad_values = {"labels": -100, "input_ids": 0, "attention_mask": 0}
    padded = dict(batch)
    if pad_len > 0:
        for k, v in batch.items():
            if k == "qkv_format" or not isinstance(v, torch.Tensor) or v.ndim < 1:
                continue
            if k in ("seq_lens", "seq_lens_padded"):
                continue  # recomputed separately, not padded
            if k == "position_ids":
                # position_ids increment-pad: continue incrementing from the last value
                last = v[..., -1:].to(torch.long)
                inc = torch.arange(1, pad_len + 1, device=v.device,
                                   dtype=v.dtype)
                inc = inc.reshape(*([1] * (v.ndim - 1)), pad_len)
                pad_block = inc.expand(*v.shape[:-1], pad_len) + last
            else:
                shape = list(v.shape)
                shape[-1] = pad_len
                pad_block = torch.full(shape, pad_values.get(k, 0),
                                       dtype=v.dtype, device=v.device)
            padded[k] = torch.cat([v, pad_block], dim=-1)

    out = {}
    for k, v in padded.items():
        if k in ("seq_lens", "seq_lens_padded"):
            continue
        if k == "qkv_format":
            out[k] = v
        elif isinstance(v, torch.Tensor) and v.ndim >= 1:
            out[k] = v[..., slc]
        else:
            out[k] = v

    if "seq_lens" in batch and "seq_lens_padded" in batch:
        out["seq_lens"], out["seq_lens_padded"] = _shard_seq_lens_for_cp(
            batch["seq_lens"], batch["seq_lens_padded"],
            cp_rank=cp_rank, chunk=chunk,
        )
    return out


def _shard_seq_lens_for_cp(seq_lens, seq_lens_padded, *, cp_rank: int, chunk: int):
    """Recompute seq_lens/seq_lens_padded per CP shard (preserving the -1000
    sentinel semantics).

    Walk the cumulative pack offsets of each sample (accumulated by
    seq_lens_padded); for each pack:
    - fully inside [lo, hi): kept as-is;
    - crossing the boundary: truncated to [lo, hi), with the actual /
      padding-inclusive lengths recomputed after truncation;
    - fully outside: skipped.
    The output is shifted to the local coordinate system; when
    max_local_packs=0 it is set to 1 to avoid an empty tensor.
    """
    batch_size = seq_lens.shape[0]
    lo = cp_rank * chunk
    hi = lo + chunk
    device = seq_lens.device
    sentinel = -1000

    local_lens_b, local_lens_padded_b = [], []
    max_local_packs = 0
    for b in range(batch_size):
        row_lens = seq_lens[b].tolist()
        row_padded = seq_lens_padded[b].tolist()
        local_lens, local_padded = [], []
        offset = 0
        for raw_len, raw_pad in zip(row_lens, row_padded):
            if raw_len == sentinel:
                break
            pack_start = offset
            pack_end = offset + raw_pad
            offset = pack_end
            inter_start = max(pack_start, lo)
            inter_end = min(pack_end, hi)
            if inter_start >= inter_end:
                continue
            actual_start = max(pack_start, lo)
            actual_end = min(pack_start + raw_len, hi)
            local_actual = max(actual_end - actual_start, 0)
            local_pad = inter_end - inter_start
            if local_actual > 0 or local_pad > 0:
                local_lens.append(local_actual)
                local_padded.append(local_pad)
        local_lens_b.append(local_lens)
        local_lens_padded_b.append(local_padded)
        max_local_packs = max(max_local_packs, len(local_lens))

    if max_local_packs == 0:
        max_local_packs = 1

    out_lens = torch.full((batch_size, max_local_packs), sentinel,
                          dtype=seq_lens.dtype, device=device)
    out_padded = torch.full((batch_size, max_local_packs), sentinel,
                            dtype=seq_lens_padded.dtype, device=device)
    for b in range(batch_size):
        n = len(local_lens_b[b])
        if n > 0:
            out_lens[b, :n] = torch.tensor(
                local_lens_b[b], dtype=seq_lens.dtype, device=device)
            out_padded[b, :n] = torch.tensor(
                local_lens_padded_b[b], dtype=seq_lens_padded.dtype, device=device)
    return out_lens, out_padded


def _cp_offset_causal_mask(q_len: int, kv_len: int, lo: int,
                           device, dtype=torch.bool):
    """D-04: offset-aware causal mask (this rank's Q chunk has global offset lo).

    Attendable positions: j <= lo + i (i is the local Q row index).
    Replaces is_causal=True -- torch SDPA's is_causal is top-left aligned when
    q_len != kv_len (equivalent to assuming Q starts at global position 0), so
    under CP the chunks of rank>0 would be incorrectly masked (G4).
    """
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(kv_len, device=device).view(1, -1)
    return (j <= (lo + i)).to(dtype)
