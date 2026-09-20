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
"""Loads of checkpoints whose tensors are named and laid out differently from the state dict they fill.

A Hugging Face checkpoint stores the query, key and value projections of a layer as three tensors
where a model may hold one fused, interleaved tensor, and the experts of a block one tensor apiece
where a model stacks them. :class:`RemapLoadPlanner` loads such a checkpoint without ever putting a
checkpoint tensor together in memory: it is told, per destination tensor, which box of it comes from
which region of which checkpoint tensor, as :class:`RemapBlock` entries, and plans each rank's reads
from the shards that rank holds, exactly as :class:`StandardLoadPlanner` plans them for a checkpoint
laid out like its state dict. Replicated shards are still read once and broadcast.

What a block cannot describe is handed to :class:`DeferredRead`: its checkpoint tensors are read
whole and given to a function that puts the result in place itself.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from hyper_parallel.core.distributed_checkpoint.metadata import (
    MetadataIndex,
    TensorStorageMetadata,
    dtype_element_size,
)
from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType, LoadPlan, ReadItem
from hyper_parallel.core.distributed_checkpoint.ragged import get_ragged_box_tensor
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner, _is_on_host
from hyper_parallel.core.distributed_checkpoint.utils import (
    chunk_to_area,
    create_chunk_list_for_tensor,
    dcp_timer_decorator,
    infer_intersection,
    narrow_tensor_by_index,
)
from hyper_parallel.core.dtensor.dtensor import DTensor

# Deferred reads land in no state dict entry, so their items name a shard of their own. The prefix
# cannot start a state dict key, which is a dotted module path.
_DEFERRED_FQN_PREFIX = "<deferred>/"


@dataclass(frozen=True)
class RemapBlock:
    """
    A box of a destination tensor, and where in one checkpoint tensor each element of it is read from.

    The element at index ``x`` of the destination tensor, for ``offsets <= x < offsets + lengths``, is
    read from index ``base + coeff @ (x - offsets)`` of the checkpoint tensor ``source``: ``coeff[s][a]``
    is how far along checkpoint dimension ``s`` one step along destination dimension ``a`` moves.

    A block whose every varying destination dimension walks a checkpoint dimension of its own one
    element at a time is a plain region of the checkpoint tensor in another order, and the read is
    copied straight into the destination. Anything else, and any block with ``post`` operations, is
    read into a buffer of its own and put in place from there.

    Attributes:
        offsets (tuple[int, ...]): Where the box starts in the whole destination tensor.
        lengths (tuple[int, ...]): Size of the box, positive in every dimension.
        source (str): Name of the checkpoint tensor the box is read from.
        base (tuple[int, ...]): Index of the checkpoint tensor the first element of the box comes from.
        coeff (tuple[tuple[int, ...], ...]): Non-negative steps, one row per checkpoint dimension and one
            column per destination dimension.
        post (tuple[Callable[[Any], Any], ...]): Elementwise operations applied in order to the values
            read, in the dtype the checkpoint stores them in, before they are copied into place.
            Default ().
    """

    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    source: str
    base: tuple[int, ...]
    coeff: tuple[tuple[int, ...], ...]
    post: tuple[Callable[[Any], Any], ...] = ()

    def __post_init__(self) -> None:
        """Reject a block whose parts disagree on the number of dimensions or step backwards."""
        ndim = len(self.offsets)
        if len(self.lengths) != ndim or any(length <= 0 for length in self.lengths):
            raise ValueError(
                f"RemapBlock of {self.source!r} needs one positive length per offset, "
                f"got offsets={self.offsets} lengths={self.lengths}"
            )
        if len(self.coeff) != len(self.base) or any(
                len(row) != ndim or any(step < 0 for step in row) for row in self.coeff
        ):
            raise ValueError(
                f"RemapBlock of {self.source!r} needs one row of {ndim} non-negative steps per checkpoint "
                f"dimension, got base={self.base} coeff={self.coeff}"
            )


@dataclass(frozen=True)
class DeferredRead:
    """
    Checkpoint tensors read whole, and what to do with them once all of them are.

    For destination tensors no block can describe. Every rank that plans the read reads the tensors
    into host memory itself, never through a broadcast, and ``complete`` is called once with all of
    them to put whatever it computes out of them in place.

    Attributes:
        sources (tuple[str, ...]): Names of the checkpoint tensors to read, each once.
        complete (Callable[[dict[str, Any]], None]): Called with ``{name: tensor}`` once every source
            has been read.
    """

    sources: tuple[str, ...]
    complete: Callable[[dict[str, Any]], None]

    def __post_init__(self) -> None:
        """Reject an empty read and a source named twice, which would never be counted complete."""
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError(f"DeferredRead needs distinct sources, got {self.sources}")


@dataclass(frozen=True)
class _Recipe:
    """
    How the region one read item stands for is put in place.

    Attributes:
        block (RemapBlock): The block the item was cut from.
        sub_lengths (tuple[int, ...]): Size of the destination box the item fills.
        axes (Optional[tuple[tuple[int, int], ...]]): ``(checkpoint dimension, destination dimension)``
            pairs of a read copied straight into place, or None for one gathered from a buffer.
    """

    block: RemapBlock
    sub_lengths: tuple[int, ...]
    axes: Optional[tuple[tuple[int, int], ...]]


class _Capture:
    """
    Stands where a read would be copied to, and keeps what was read for the planner to place itself.

    The storage reader checks the size of what it acquires against the size of what it read, then
    copies with ``copy_``. A capture answers both, holding on to the tensor instead of copying it.
    """

    __slots__ = ("_size", "value")

    def __init__(self, size: Sequence[int]) -> None:
        """
        Args:
            size (Sequence[int]): Size of the region the read returns.
        """
        self._size = tuple(size)
        self.value: Any = None

    def size(self) -> tuple[int, ...]:
        """Size of the region the read returns."""
        return self._size

    def copy_(self, tensor: Any) -> "_Capture":
        """Keep ``tensor``, the region as read."""
        self.value = tensor
        return self


def _recipe_key(item: ReadItem) -> tuple:
    """What tells the reads of one plan apart, before and after a global plan has marked them."""
    return item.dest_index, item.dest_offsets, item.storage_index, item.storage_offsets


def _source_start(block: RemapBlock, sub_offsets: Sequence[int]) -> tuple[int, ...]:
    """Checkpoint index the first element of the destination box starting at ``sub_offsets`` comes from."""
    steps = [offset - origin for offset, origin in zip(sub_offsets, block.offsets)]
    return tuple(start + sum(c * step for c, step in zip(row, steps)) for start, row in zip(block.base, block.coeff))


def _source_lengths(block: RemapBlock, sub_lengths: Sequence[int]) -> tuple[int, ...]:
    """Size of the checkpoint region a destination box of ``sub_lengths`` reads from, gaps included."""
    return tuple(1 + sum(c * (length - 1) for c, length in zip(row, sub_lengths)) for row in block.coeff)


def _direct_axes(block: RemapBlock, sub_lengths: Sequence[int]) -> Optional[tuple[tuple[int, int], ...]]:
    """
    Pair the dimensions of a read that can be copied straight into its destination box.

    Args:
        block (RemapBlock): The block the box belongs to.
        sub_lengths (Sequence[int]): Size of the box.

    Returns:
        Optional[tuple[tuple[int, int], ...]]: ``(checkpoint dimension, destination dimension)`` for
        every destination dimension the box is longer than one along, when each of them walks a
        checkpoint dimension of its own one element at a time and nothing is left to apply. None
        otherwise.
    """
    if block.post:
        return None
    pairs: dict[int, int] = {}
    for axis, length in enumerate(sub_lengths):
        if length == 1:
            continue
        walked = [dim for dim, row in enumerate(block.coeff) if row[axis]]
        if len(walked) != 1 or block.coeff[walked[0]][axis] != 1 or walked[0] in pairs:
            return None
        pairs[walked[0]] = axis
    return tuple(sorted(pairs.items()))


def _direct_view(region: Any, axes: tuple[tuple[int, int], ...], source_lengths: tuple[int, ...]) -> Any:
    """
    View a destination box in the order and shape of the checkpoint region it is copied from.

    Args:
        region (Any): The destination box.
        axes (tuple[tuple[int, int], ...]): Pairs from :func:`_direct_axes`.
        source_lengths (tuple[int, ...]): Size of the checkpoint region.

    Returns:
        Any: A view of ``region``. Only dimensions of size one are added or dropped, so it still
        writes through to the destination.
    """
    walked = [axis for _, axis in axes]
    order = walked + [axis for axis in range(len(region.shape)) if axis not in walked]
    if order != sorted(order):
        region = region.permute(order)
    return region.reshape(source_lengths)


def _gather(block: RemapBlock, sub_lengths: tuple[int, ...], read: Any) -> Any:
    """
    Pick the elements of a destination box out of the checkpoint region read for it.

    Args:
        block (RemapBlock): The block the box belongs to.
        sub_lengths (tuple[int, ...]): Size of the box.
        read (Any): The checkpoint region, starting at the element the box starts with.

    Returns:
        Any: The values of the box in destination order, with the block's post operations applied.
    """
    strides = read.stride()
    stride = tuple(
        sum(row[axis] * step for row, step in zip(block.coeff, strides)) for axis in range(len(sub_lengths))
    )
    values = read.as_strided(sub_lengths, stride, read.storage_offset())
    for operation in block.post:
        values = operation(values)
    return values


class RemapLoadPlanner(StandardLoadPlanner):
    """
    Load planner for a checkpoint whose tensors are named and laid out differently from the state dict.

    Every state dict entry to fill is described by ``table``: blocks that tile it, each saying which
    region of which checkpoint tensor it is read from. Entries without blocks are left alone, and
    ``deferred`` reads whole checkpoint tensors for whatever the blocks leave out.

    Each rank intersects the shards it holds with the blocks and reads only those regions. A read that
    is a plain region of a checkpoint tensor is copied straight into place; one that has to be gathered
    or changed is read into a buffer first. Only whole checkpoint tensors are read, one chunk each, as
    :class:`HuggingFaceStorageReader` describes them.

    Example::

        planner = RemapLoadPlanner({"linear.weight": [block_from_q, block_from_k]})
        load(state_dict, storage_reader=HuggingFaceStorageReader(path), planner=planner)
    """

    def __init__(
            self,
            table: Optional[Mapping[str, Sequence[RemapBlock]]] = None,
            deferred: Sequence[DeferredRead] = (),
            broadcast_replicated_tensors: bool = False,
    ) -> None:
        """
        Args:
            table (Optional[Mapping[str, Sequence[RemapBlock]]]): Blocks of every state dict entry to fill,
                by its flattened name. Default None, for a subclass that fills ``table`` while configuring.
            deferred (Sequence[DeferredRead]): Whole-tensor reads for what no block describes. Default ().
            broadcast_replicated_tensors (bool): See :class:`StandardLoadPlanner`; ``load`` overrides it.
                Default False.
        """
        super().__init__(broadcast_replicated_tensors=broadcast_replicated_tensors)
        self.table: dict[str, tuple[RemapBlock, ...]] = {
            name: tuple(blocks) for name, blocks in (table or {}).items()
        }
        self.deferred: tuple[DeferredRead, ...] = tuple(deferred)
        self._recipes: dict[tuple, _Recipe] = {}
        self._deferred_slots: dict[tuple, tuple[int, str]] = {}
        self._captured: dict[int, dict[str, Any]] = {}

    @dcp_timer_decorator
    def build_local_plan(self) -> LoadPlan:
        """
        Plan this rank's reads: the part of every block inside a shard it holds, and the deferred reads.

        Returns:
            LoadPlan: One read per (shard, block) overlap, then one per deferred checkpoint tensor.

        Raises:
            RuntimeError: If the planner is not configured.
            ValueError: If the table names an entry the state dict does not have, or a checkpoint
                tensor that is missing, stored in shards, or smaller than a block reads.
        """
        if self.state_dict is None or self.metadata is None:
            raise RuntimeError("Planner not configured")
        self._recipes.clear()
        self._deferred_slots.clear()
        self._captured.clear()

        items: list[ReadItem] = []
        for fqn, blocks in self.table.items():
            if fqn not in self.state_dict:
                raise ValueError(f"Remap table names {fqn!r}, which the state dict does not have")
            items.extend(self._plan_entry(fqn, self.state_dict[fqn], blocks))
        for group_index, group in enumerate(self.deferred):
            items.extend(self._plan_deferred(group_index, source) for source in group.sources)
        return LoadPlan(items=items)

    def acquire_tensor(self, read_item: ReadItem) -> Any:
        """
        Where one read is copied to.

        Args:
            read_item (ReadItem): A read of this planner's plan.

        Returns:
            Any: A view of the destination for a read copied straight into place, otherwise a
            capture that keeps the read for :meth:`apply_tensor`.
        """
        key = _recipe_key(read_item)
        if key in self._deferred_slots:
            return _Capture(read_item.lengths)
        recipe = self._recipe(key)
        if recipe.axes is None:
            return _Capture(read_item.lengths)
        return _direct_view(self._destination_box(read_item, recipe), recipe.axes, read_item.lengths)

    def apply_tensor(self, read_item: ReadItem, tensor: Any) -> None:
        """
        Put a captured read in place: gather a box out of it, or keep it for its deferred read.

        Args:
            read_item (ReadItem): A read of this planner's plan.
            tensor (Any): What :meth:`acquire_tensor` returned, the read already copied into it.
        """
        key = _recipe_key(read_item)
        slot = self._deferred_slots.get(key)
        if slot is not None:
            self._keep_deferred(slot, tensor.value)
            return
        recipe = self._recipe(key)
        if recipe.axes is not None:
            return
        self._destination_box(read_item, recipe).copy_(_gather(recipe.block, recipe.sub_lengths, tensor.value))

    def _shard_element_size(self, index: MetadataIndex) -> int:
        """
        Bytes one element of a destination shard takes, which is what a broadcast of it moves.

        Args:
            index (MetadataIndex): Names the shard.

        Returns:
            int: Element size of the state dict entry, or the default size for a deferred read.
        """
        dtype = getattr(self.state_dict.get(index.fqn), "dtype", None) if self.state_dict else None
        return dtype_element_size(None if dtype is None else str(dtype))

    def _plan_entry(self, fqn: str, obj: Any, blocks: Sequence[RemapBlock]) -> list[ReadItem]:
        """Plan the reads of one state dict entry, for the shards of it this rank holds."""
        if not self._rank_owns_dtensor_shard(obj):
            return []
        broadcastable = not _is_on_host(obj)
        items = []
        for local_index, chunk in enumerate(create_chunk_list_for_tensor(obj)):
            dest_index = MetadataIndex(fqn=fqn, offset=chunk.offsets, index=local_index)
            chunk_area = chunk_to_area(chunk)
            for block in blocks:
                block_area = tuple((start, start + length) for start, length in zip(block.offsets, block.lengths))
                overlap = infer_intersection(chunk_area, block_area)
                if overlap is not None:
                    items.append(self._plan_block_read(dest_index, chunk.offsets, block, overlap, broadcastable))
        return items

    def _plan_block_read(
            self,
            dest_index: MetadataIndex,
            chunk_offsets: tuple[int, ...],
            block: RemapBlock,
            overlap: tuple[tuple[int, int], ...],
            broadcastable: bool,
    ) -> ReadItem:
        """Plan the read of the part of ``block`` that lands in one shard, and remember how to place it."""
        sub_offsets = tuple(start for start, _ in overlap)
        sub_lengths = tuple(stop - start for start, stop in overlap)
        metadata = self._whole_tensor(block.source)
        storage_offsets = _source_start(block, sub_offsets)
        lengths = _source_lengths(block, sub_lengths)
        if len(lengths) != len(metadata.size) or any(
                start + length > size for start, length, size in zip(storage_offsets, lengths, metadata.size)
        ):
            raise ValueError(
                f"A block reads {lengths} at {storage_offsets} of checkpoint tensor {block.source!r}, "
                f"which has size {tuple(metadata.size)}"
            )
        item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index,
            dest_offsets=tuple(offset - origin for offset, origin in zip(sub_offsets, chunk_offsets)),
            storage_index=MetadataIndex(fqn=block.source, offset=metadata.chunks[0].offsets),
            storage_offsets=storage_offsets,
            lengths=lengths,
            broadcastable=broadcastable,
        )
        self._recipes[_recipe_key(item)] = _Recipe(block, sub_lengths, _direct_axes(block, sub_lengths))
        return item

    def _plan_deferred(self, group_index: int, source: str) -> ReadItem:
        """Plan the whole-tensor read of one source of a deferred read."""
        metadata = self._whole_tensor(source)
        size = tuple(metadata.size)
        item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=f"{_DEFERRED_FQN_PREFIX}{group_index}/{source}"),
            dest_offsets=(),
            storage_index=MetadataIndex(fqn=source, offset=metadata.chunks[0].offsets),
            storage_offsets=(0,) * len(size),
            lengths=size,
            # Kept in host memory for the conversion that follows, which no collective can write into.
            broadcastable=False,
        )
        self._deferred_slots[_recipe_key(item)] = (group_index, source)
        return item

    def _whole_tensor(self, source: str) -> TensorStorageMetadata:
        """
        Metadata of a checkpoint tensor stored whole, in one chunk.

        Raises:
            ValueError: If the checkpoint has no such tensor, or stores it in shards.
        """
        metadata = self.metadata.state_dict_metadata.get(source)
        if not isinstance(metadata, TensorStorageMetadata):
            raise ValueError(f"Checkpoint has no tensor {source!r}")
        chunks = metadata.chunks
        if len(chunks) != 1 or any(chunks[0].offsets) or tuple(chunks[0].sizes) != tuple(metadata.size):
            raise ValueError(
                f"Checkpoint tensor {source!r} is stored in {len(chunks)} shards; "
                "RemapLoadPlanner reads whole tensors only"
            )
        return metadata

    def _recipe(self, key: tuple) -> _Recipe:
        """The recipe of a read of the current plan."""
        recipe = self._recipes.get(key)
        if recipe is None:
            raise RuntimeError(f"Read of {key[2].fqn!r} into {key[0].fqn!r} is not part of the current plan")
        return recipe

    def _destination_box(self, read_item: ReadItem, recipe: _Recipe) -> Any:
        """The box of the destination shard one read fills, detached so it is written in place."""
        target = self.state_dict[read_item.dest_index.fqn]
        if isinstance(target, DTensor) and target.layout is not None and target.layout.ragged_shard is not None:
            local = get_ragged_box_tensor(target, read_item.dest_index)
        elif isinstance(target, DTensor):
            local = target.to_local()
        else:
            local = target
        return narrow_tensor_by_index(local.detach(), read_item.dest_offsets, recipe.sub_lengths)

    def _keep_deferred(self, slot: tuple[int, str], value: Any) -> None:
        """Keep one source of a deferred read, and complete the read once it has all of them."""
        group_index, source = slot
        captured = self._captured.setdefault(group_index, {})
        captured[source] = value
        group = self.deferred[group_index]
        if len(captured) == len(group.sources):
            del self._captured[group_index]
            group.complete(captured)
