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

"""context_parallel.collectives: CP collective primitives (05 §4.4.2 / §6.3.4).

- ``flex_cp_allgather``: K/V all-gather along the CP dim (reuses
  cp_mesh.get_group(); new_group is forbidden);
- ``ulysses_seq_to_head`` / ``ulysses_head_to_seq``: differentiable Pure
  Ulysses all-to-all layout transforms;
- async collective launches (``AsyncCPCollective`` /
  ``async_cp_allgather_launch`` / ``async_ulysses_seq_to_head_launch``);
- hybrid (Ulysses + all-gather) CP sub-mesh construction and attention
  orchestration (``_build_hybrid_cp_submeshes`` / ``hybrid_cp_attention``).

Note (G5): the seq_len % (2*cp) padding constraint originates from the
zigzag/ring load-balancing scheme; this design uses all-gather K/V +
contiguous chunk (D-01'' rejected ring), so each rank's Q chunk is
equal-length and FLOPs are naturally balanced -- the constraint is redundant
but harmless: the implementation is kept and documented here.

Split out of components/distributed/cp_utils.py in stage 4e; the attention
adaptations live in context_parallel/attention.py and the batch sharding in
hyper_parallel/data/parallel/batch_parallel.py.
"""

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
    def forward(
        ctx: Any,
        group: Any,
        tensor: Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
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
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[None, Tensor, None, None]:
        """Apply the inverse all-to-all to the output gradient."""
        grad_input = _SeqAllToAll.apply(
            ctx.group, grad_output, ctx.gather_dim, ctx.scatter_dim)
        return None, grad_input, None, None


@dataclass(frozen=True)
class _UlyssesContext:
    """Communication context derived from the framework-owned CP mesh."""

    cp_mesh: Any

    @property
    def group(self) -> Any:
        """Process group of the CP mesh."""
        return self.cp_mesh.get_group()

    @property
    def size(self) -> int:
        """Number of ranks in the CP mesh."""
        return self.cp_mesh.size()

    @property
    def rank(self) -> int:
        """Local rank of this process within the CP mesh."""
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
    def forward(
        ctx: Any,
        t: Tensor,
        cp_dim: int,
        group: Any,
        cp_size: int,
    ) -> Tensor:
        """All-gather the tensor along cp_dim in CP-rank order."""
        ctx.cp_dim = cp_dim
        ctx.group = group
        ctx.cp_size = cp_size
        world_t = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(world_t, t.contiguous(), group=group)
        # cat in cp_rank order: [chunk_rank0, chunk_rank1, ...]
        return torch.cat(world_t, dim=cp_dim)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None]:
        """Sum the gradient across ranks and take this rank's chunk."""
        # reduce-scatter: sum the gradient across ranks, take this rank's chunk
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        rank = dist.get_rank(ctx.group)
        local = torch.chunk(grad, ctx.cp_size, dim=ctx.cp_dim)[rank]
        return local.contiguous(), None, None, None


def flex_cp_allgather(
    k: Tensor,
    v: Tensor,
    cp_dim: int,
    cp_mesh: DeviceMesh,
) -> tuple[Tensor, Tensor]:
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
