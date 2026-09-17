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
"""Symbolic checkpoint tensors, which track where every element of a converted tensor is read from.

A weight conversion cuts, stacks, reorders and reshapes the tensors of a checkpoint into the tensors of
a model. Given :class:`RegionTensor` inputs instead of real tensors, the same conversion code computes
no values. For every element of each result it tracks which checkpoint tensor and index the element is
copied from, and which elementwise scalar operations are applied to it on the way.
:meth:`RegionTensor.remap_blocks` hands that over as :class:`RemapBlock` entries, which
:class:`RemapLoadPlanner` loads without ever assembling a whole checkpoint tensor or a whole result.

What cannot be tracked that way raises :class:`UnsupportedRegionOp`: an operation that combines two
tensors elementwise, reads values, writes in place, or indexes with tensors.
"""
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.distributed_checkpoint.remap_planner import RemapBlock

# A region cut into more blocks than this is not worth describing block by block. The conversion is
# run on the real tensors instead.
MAX_BLOCKS = 1 << 16


class UnsupportedRegionOp(TypeError):
    """Raised by an operation whose result cannot be described as regions of checkpoint tensors."""


@dataclass(frozen=True)
class RegionBlock:
    """
    A box of a symbolic tensor, and where in one checkpoint tensor each element of it comes from.

    Attributes:
        offsets (tuple[int, ...]): Where the box starts in the symbolic tensor.
        lengths (tuple[int, ...]): Size of the box, positive in every dimension.
        source (str): Name of the checkpoint tensor.
        base (tuple[int, ...]): Checkpoint index of the first element of the box.
        steps (tuple[tuple[int, ...], ...]): One column per dimension of the symbolic tensor, giving how
            far along each checkpoint dimension one step along that dimension moves.
        dtype (torch.dtype): Dtype of the values once ``post`` has been applied.
        post (tuple[Callable[[Any], Any], ...]): Elementwise operations applied in order. Default ().
    """

    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    source: str
    base: tuple[int, ...]
    steps: tuple[tuple[int, ...], ...]
    dtype: torch.dtype
    post: tuple[Callable[[Any], Any], ...] = ()


def _replaced(values: tuple, index: int, value: Any) -> tuple:
    """``values`` with the entry at ``index`` replaced by ``value``."""
    return values[:index] + (value,) + values[index + 1:]


def _strides(sizes: Sequence[int]) -> tuple[int, ...]:
    """How many elements one step along each of ``sizes`` skips, in row-major order."""
    strides, stride = [], 1
    for size in reversed(sizes):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def _check_count(count: int) -> None:
    """Refuse to go on describing a region cut into more than :data:`MAX_BLOCKS` blocks."""
    if count > MAX_BLOCKS:
        raise UnsupportedRegionOp(f"the result is cut into more than {MAX_BLOCKS} blocks")


def _restrict(block: RegionBlock, axis: int, begin: int, length: int) -> RegionBlock:
    """The part of ``block`` from ``begin`` to ``begin + length`` along ``axis``, in the same coordinates."""
    delta = begin - block.offsets[axis]
    return replace(
        block,
        offsets=_replaced(block.offsets, axis, begin),
        lengths=_replaced(block.lengths, axis, length),
        base=tuple(origin + step * delta for origin, step in zip(block.base, block.steps[axis])),
    )


def _slice_axis(blocks: Sequence[RegionBlock], axis: int, start: int, stop: int, step: int) -> list[RegionBlock]:
    """
    The blocks of ``tensor[..., start:stop:step, ...]`` along ``axis``, in the coordinates of the slice.

    Args:
        blocks (Sequence[RegionBlock]): Blocks of the tensor being sliced.
        axis (int): Dimension to slice.
        start (int): First index kept, already clamped into the dimension.
        stop (int): Index the slice stops before, already clamped into the dimension.
        step (int): Positive distance between the indices kept.

    Returns:
        list[RegionBlock]: The part of every block the slice keeps.
    """
    sliced = []
    for block in blocks:
        low = max(block.offsets[axis], start)
        high = min(block.offsets[axis] + block.lengths[axis], stop)
        if low >= high:
            continue
        first = -(-(low - start) // step)
        last = (high - 1 - start) // step
        if first > last:
            continue
        delta = start + step * first - block.offsets[axis]
        column = block.steps[axis]
        sliced.append(replace(
            block,
            offsets=_replaced(block.offsets, axis, first),
            lengths=_replaced(block.lengths, axis, last - first + 1),
            base=tuple(origin + c * delta for origin, c in zip(block.base, column)),
            steps=_replaced(block.steps, axis, tuple(c * step for c in column)),
        ))
    return sliced


def _drop_axis(block: RegionBlock, axis: int) -> RegionBlock:
    """``block`` without the dimension ``axis``, along which it is one element long."""
    return replace(
        block,
        offsets=block.offsets[:axis] + block.offsets[axis + 1:],
        lengths=block.lengths[:axis] + block.lengths[axis + 1:],
        steps=block.steps[:axis] + block.steps[axis + 1:],
    )


def _insert_axis(block: RegionBlock, axis: int) -> RegionBlock:
    """``block`` with a new dimension of size one at ``axis``."""
    return replace(
        block,
        offsets=block.offsets[:axis] + (0,) + block.offsets[axis:],
        lengths=block.lengths[:axis] + (1,) + block.lengths[axis:],
        steps=block.steps[:axis] + ((0,) * len(block.base),) + block.steps[axis:],
    )


def _shift(block: RegionBlock, axis: int, distance: int) -> RegionBlock:
    """``block`` moved ``distance`` elements along ``axis``."""
    return replace(block, offsets=_replaced(block.offsets, axis, block.offsets[axis] + distance))


def _permute_block(block: RegionBlock, order: Sequence[int]) -> RegionBlock:
    """``block`` with its dimensions in ``order``."""
    return replace(
        block,
        offsets=tuple(block.offsets[axis] for axis in order),
        lengths=tuple(block.lengths[axis] for axis in order),
        steps=tuple(block.steps[axis] for axis in order),
    )


def _merge_axes(blocks: Sequence[RegionBlock], start: int, sizes: Sequence[int]) -> list[RegionBlock]:
    """
    The blocks of a tensor whose dimensions ``start`` to ``start + len(sizes)`` are flattened into one.

    A block stays whole when it covers consecutive flat indices and walks the checkpoint the same way
    along them. That holds when every dimension after its first varying one is covered whole and moves
    through the checkpoint as a multiple of the last. Any other block is cut into pieces one element
    long along its first varying dimension, and those pieces are merged in turn.

    Args:
        blocks (Sequence[RegionBlock]): Blocks of the tensor.
        start (int): First dimension to flatten.
        sizes (Sequence[int]): Sizes of the dimensions to flatten, all larger than one.

    Returns:
        list[RegionBlock]: Blocks of the flattened tensor.
    """
    stop = start + len(sizes)
    strides = _strides(sizes)
    merged, pending = [], list(blocks)
    while pending:
        block = pending.pop()
        varying = [axis for axis in range(len(sizes)) if block.lengths[start + axis] > 1]
        column = block.steps[stop - 1] if varying else (0,) * len(block.base)
        length = 1
        if varying:
            first = varying[0]
            whole = all(
                block.offsets[start + axis] == 0 and block.lengths[start + axis] == sizes[axis]
                for axis in range(first + 1, len(sizes))
            )
            in_step = all(
                block.steps[start + axis] == tuple(c * strides[axis] for c in column)
                for axis in range(first, len(sizes))
            )
            if not (whole and in_step):
                origin = block.offsets[start + first]
                pending.extend(
                    _restrict(block, start + first, origin + index, 1)
                    for index in range(block.lengths[start + first])
                )
                _check_count(len(merged) + len(pending))
                continue
            length = block.lengths[start + first] * strides[first]
        offset = sum(index * stride for index, stride in zip(block.offsets[start:stop], strides))
        merged.append(replace(
            block,
            offsets=block.offsets[:start] + (offset,) + block.offsets[stop:],
            lengths=block.lengths[:start] + (length,) + block.lengths[stop:],
            steps=block.steps[:start] + (column,) + block.steps[stop:],
        ))
    return merged


def _decompose(start: int, length: int, sizes: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Cut a run of flat indices over dimensions of ``sizes`` into boxes.

    Args:
        start (int): First flat index of the run.
        length (int): Number of flat indices in the run.
        sizes (Sequence[int]): Sizes of the dimensions the flat index is spread over.

    Returns:
        list[tuple[tuple[int, ...], tuple[int, ...]]]: ``(offsets, lengths)`` of every box: a partial
        first row, the whole rows, then a partial last row.
    """
    if len(sizes) == 1:
        return [((start,), (length,))]
    inner = tuple(sizes[1:])
    stride = math.prod(inner)
    boxes = []
    row, column = divmod(start, stride)
    if column:
        head = min(stride - column, length)
        boxes.extend(((row,) + offsets, (1,) + lengths) for offsets, lengths in _decompose(column, head, inner))
        row, length = row + 1, length - head
    rows, tail = divmod(length, stride)
    if rows:
        boxes.append(((row,) + (0,) * len(inner), (rows,) + inner))
    if tail:
        boxes.extend(((row + rows,) + offsets, (1,) + lengths) for offsets, lengths in _decompose(0, tail, inner))
    return boxes


def _split_axis(blocks: Sequence[RegionBlock], axis: int, sizes: Sequence[int]) -> list[RegionBlock]:
    """
    The blocks of a tensor whose dimension ``axis`` is unflattened into dimensions of ``sizes``.

    Args:
        blocks (Sequence[RegionBlock]): Blocks of the tensor.
        axis (int): Dimension to unflatten.
        sizes (Sequence[int]): Sizes it is unflattened into, all larger than one.

    Returns:
        list[RegionBlock]: Blocks of the unflattened tensor.
    """
    strides = _strides(sizes)
    pieces = []
    for block in blocks:
        column = block.steps[axis]
        scaled = tuple(tuple(c * stride for c in column) for stride in strides)
        for offsets, lengths in _decompose(block.offsets[axis], block.lengths[axis], sizes):
            delta = sum(index * stride for index, stride in zip(offsets, strides)) - block.offsets[axis]
            pieces.append(replace(
                block,
                offsets=block.offsets[:axis] + offsets + block.offsets[axis + 1:],
                lengths=block.lengths[:axis] + lengths + block.lengths[axis + 1:],
                base=tuple(origin + c * delta for origin, c in zip(block.base, column)),
                steps=block.steps[:axis] + scaled + block.steps[axis + 1:],
            ))
    return pieces


def _runs(old: Sequence[int], new: Sequence[int]) -> list[tuple[int, int, int, int]]:
    """
    Match the dimensions of two shapes up into runs holding as many elements on either side.

    Args:
        old (Sequence[int]): Sizes of one shape, none of them one.
        new (Sequence[int]): Sizes of the other, none of them one, holding as many elements in all.

    Returns:
        list[tuple[int, int, int, int]]: ``(old start, old stop, new start, new stop)`` of every run.
    """
    runs, i, j = [], 0, 0
    while i < len(old):
        i0, j0 = i, j
        old_size, new_size = old[i], new[j]
        i, j = i + 1, j + 1
        while old_size != new_size:
            if old_size < new_size:
                old_size, i = old_size * old[i], i + 1
            else:
                new_size, j = new_size * new[j], j + 1
        runs.append((i0, i, j0, j))
    return runs


def _reshape(
        blocks: Sequence[RegionBlock], old_shape: tuple[int, ...], new_shape: tuple[int, ...]
) -> list[RegionBlock]:
    """
    The blocks of a tensor of ``old_shape`` reshaped to ``new_shape``, which holds as many elements.

    Dimensions of size one are dropped first and put back last. The rest are matched up into runs
    holding as many elements on either side, and each run is flattened and then unflattened.

    Args:
        blocks (Sequence[RegionBlock]): Blocks of the tensor.
        old_shape (tuple[int, ...]): Its shape.
        new_shape (tuple[int, ...]): The shape it is reshaped to.

    Returns:
        list[RegionBlock]: Blocks of the reshaped tensor.
    """
    if math.prod(old_shape) == 0:
        return []
    blocks = list(blocks)
    for axis in reversed([axis for axis, size in enumerate(old_shape) if size == 1]):
        blocks = [_drop_axis(block, axis) for block in blocks]
    old = [size for size in old_shape if size != 1]
    new = [size for size in new_shape if size != 1]
    for i0, i1, j0, j1 in reversed(_runs(old, new)):
        if i1 - i0 > 1:
            blocks = _merge_axes(blocks, i0, old[i0:i1])
        if j1 - j0 > 1:
            blocks = _split_axis(blocks, i0, new[j0:j1])
        _check_count(len(blocks))
    for axis, size in enumerate(new_shape):
        if size == 1:
            blocks = [_insert_axis(block, axis) for block in blocks]
    return blocks


def _shape_argument(shape: tuple) -> tuple:
    """A shape passed either as one sequence or as separate sizes."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        return tuple(shape[0])
    return tuple(shape)


def _expand_index(index: Any, ndim: int) -> list:
    """
    The items of an index into a region, with Ellipsis spelled out as whole slices.

    Args:
        index (Any): What ``__getitem__`` was given.
        ndim (int): Number of dimensions of the region.

    Returns:
        list: Integers, slices and None, one integer or slice per dimension indexed.

    Raises:
        UnsupportedRegionOp: If the index holds anything but integers, slices, None and Ellipsis.
    """
    items = index if isinstance(index, tuple) else (index,)
    supported = (int, slice, type(None), type(Ellipsis))
    if any(isinstance(item, bool) or not isinstance(item, supported) for item in items):
        raise UnsupportedRegionOp(f"indexing a region with {index!r} is not supported")
    consumed = sum(1 for item in items if isinstance(item, (int, slice)))
    expanded = []
    for item in items:
        expanded.extend([slice(None)] * (ndim - consumed) if item is Ellipsis else [item])
    return expanded


def _check_scalar(other: Any) -> None:
    """Refuse an operand that is not a Python number."""
    if not isinstance(other, (int, float)):
        raise UnsupportedRegionOp(f"only Python numbers can be combined with a region, got {type(other).__name__}")


def _to_dtype(tensor: Any, dtype: torch.dtype) -> Any:
    """``tensor`` converted to ``dtype``."""
    return tensor.to(dtype)


def _forward(function: Callable[[Any, Any], Any], other: Any, tensor: Any) -> Any:
    """``function(tensor, other)``."""
    return function(tensor, other)


def _reflected(function: Callable[[Any, Any], Any], other: Any, tensor: Any) -> Any:
    """``function(other, tensor)``."""
    return function(other, tensor)


def _divide(other: Any, rounding_mode: str, tensor: Any) -> Any:
    """``tensor / other``, rounded as ``rounding_mode`` says."""
    return torch.div(tensor, other, rounding_mode=rounding_mode)


def _unsupported(name: str) -> Callable[..., Any]:
    """A method that refuses the operation ``name``."""

    def refuse(self: "RegionTensor", *args: Any, **kwargs: Any) -> Any:
        """Refuse an operation a region cannot describe."""
        del args, kwargs
        raise UnsupportedRegionOp(f"{name} is not supported on {self!r}")

    refuse.__name__ = name
    return refuse


class RegionTensor:
    """
    A tensor that knows, for each of its elements, which checkpoint tensor and index it is read from.

    Supports what weight conversions do to lay tensors out differently: concatenating, stacking,
    splitting, slicing, reordering and reshaping, plus elementwise arithmetic with Python scalars and
    dtype conversions, as methods and through the matching ``torch`` functions. Shapes and dtypes follow
    torch exactly, since every operation is first run on a meta tensor of the same shape and dtype.

    Example::

        q = RegionTensor.leaf("q_proj.weight", (8, 16), torch.bfloat16)
        k = RegionTensor.leaf("k_proj.weight", (4, 16), torch.bfloat16)
        fused = torch.cat([q, k])
        fused.remap_blocks()  # two blocks: rows 0-7 read from q_proj, rows 8-11 from k_proj
    """

    __slots__ = ("_shape", "_dtype", "_blocks")

    def __init__(self, shape: Sequence[int], dtype: torch.dtype, blocks: Sequence[RegionBlock] = ()) -> None:
        """
        Args:
            shape (Sequence[int]): Shape of the tensor.
            dtype (torch.dtype): Dtype of the tensor.
            blocks (Sequence[RegionBlock]): Blocks tiling the tensor. Default (), for an empty tensor.
        """
        self._shape = tuple(int(size) for size in shape)
        self._dtype = dtype
        self._blocks = tuple(blocks)
        _check_count(len(self._blocks))

    @classmethod
    def leaf(cls, source: str, shape: Sequence[int], dtype: torch.dtype) -> "RegionTensor":
        """
        The whole of one checkpoint tensor.

        Args:
            source (str): Name of the checkpoint tensor.
            shape (Sequence[int]): Its shape.
            dtype (torch.dtype): Its dtype.

        Returns:
            RegionTensor: A tensor of one block reading the checkpoint tensor in place.
        """
        shape = tuple(int(size) for size in shape)
        ndim = len(shape)
        blocks = ()
        if math.prod(shape):
            identity = tuple(tuple(int(row == column) for row in range(ndim)) for column in range(ndim))
            blocks = (RegionBlock((0,) * ndim, shape, source, (0,) * ndim, identity, dtype),)
        return cls(shape, dtype, blocks)

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: tuple = (), kwargs: Any = None) -> Any:
        """Route the ``torch`` functions a region supports to its methods, and refuse the rest."""
        del types
        kwargs = dict(kwargs or {})
        name = _FUNCTION_NAMES.get(func)
        if name is None or kwargs.get("out") is not None:
            raise UnsupportedRegionOp(f"torch.{getattr(func, '__name__', func)} is not supported on regions")
        kwargs.pop("out", None)
        if name in _SEQUENCE_FUNCTIONS:
            return _SEQUENCE_FUNCTIONS[name](*args, **kwargs)
        args = list(args)
        tensor = kwargs.pop("input") if "input" in kwargs else args.pop(0)
        if isinstance(tensor, RegionTensor):
            return getattr(tensor, name)(*args, **kwargs)
        other = kwargs.pop("other") if "other" in kwargs else (args.pop(0) if args else None)
        if name in _REFLECTED and isinstance(other, RegionTensor) and not args and not kwargs:
            return getattr(other, _REFLECTED[name])(tensor)
        raise UnsupportedRegionOp(f"torch.{name} is not supported with {type(tensor).__name__} as its tensor")

    @property
    def shape(self) -> torch.Size:
        """Shape of the tensor."""
        return torch.Size(self._shape)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the tensor."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def device(self) -> torch.device:
        """The meta device: a region holds no values anywhere."""
        return torch.device("meta")

    @property
    def requires_grad(self) -> bool:
        """Always False."""
        return False

    @property
    def blocks(self) -> tuple[RegionBlock, ...]:
        """Blocks tiling the tensor."""
        return self._blocks

    def sources(self) -> frozenset[str]:
        """Names of the checkpoint tensors the elements are read from."""
        return frozenset(block.source for block in self._blocks)

    def remap_blocks(self) -> list[RemapBlock]:
        """
        The blocks as :class:`RemapLoadPlanner` loads them.

        A block kept in another dtype than the tensor, which a concatenation of tensors of different
        dtypes leaves behind, is converted to the dtype of the tensor last, as the conversion would.

        Returns:
            list[RemapBlock]: One entry per block.
        """
        remapped = []
        for block in self._blocks:
            post = block.post
            if block.dtype != self._dtype:
                post += (partial(_to_dtype, dtype=self._dtype),)
            coeff = tuple(tuple(column[dim] for column in block.steps) for dim in range(len(block.base)))
            remapped.append(RemapBlock(block.offsets, block.lengths, block.source, block.base, coeff, post))
        return remapped

    def size(self, dim: Any = None) -> Any:
        """The shape, or the size of dimension ``dim``."""
        return self.shape if dim is None else self._meta().size(dim)

    def dim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    def numel(self) -> int:
        """Number of elements."""
        return math.prod(self._shape)

    def nelement(self) -> int:
        """Number of elements."""
        return self.numel()

    def element_size(self) -> int:
        """Bytes one element takes."""
        return self._meta().element_size()

    def is_floating_point(self) -> bool:
        """Whether the dtype is a floating point one."""
        return self._dtype.is_floating_point

    def is_complex(self) -> bool:
        """Whether the dtype is a complex one."""
        return self._dtype.is_complex

    def __len__(self) -> int:
        """Size of the first dimension."""
        if not self._shape:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __iter__(self) -> Iterator["RegionTensor"]:
        """The tensor one index of its first dimension at a time."""
        if not self._shape:
            raise TypeError("iteration over a 0-d tensor")
        return iter(self.unbind(0))

    def __repr__(self) -> str:
        """Shape, dtype and blocks."""
        return f"RegionTensor(shape={self._shape}, dtype={self._dtype}, blocks={len(self._blocks)})"

    def reshape(self, *shape: Any) -> "RegionTensor":
        """The tensor reshaped, as ``torch.reshape`` does it."""
        return self._reshaped(tuple(self._meta().reshape(_shape_argument(shape)).shape))

    def view(self, *shape: Any) -> "RegionTensor":
        """The tensor reshaped. Viewing it as another dtype is not supported."""
        if len(shape) == 1 and isinstance(shape[0], torch.dtype):
            raise UnsupportedRegionOp("viewing a region as another dtype reinterprets its bytes")
        return self.reshape(*shape)

    def reshape_as(self, other: Any) -> "RegionTensor":
        """The tensor reshaped to the shape of ``other``."""
        return self.reshape(tuple(other.shape))

    def view_as(self, other: Any) -> "RegionTensor":
        """The tensor reshaped to the shape of ``other``."""
        return self.reshape(tuple(other.shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "RegionTensor":
        """The tensor with dimensions ``start_dim`` to ``end_dim`` flattened into one."""
        return self._reshaped(tuple(self._meta().flatten(start_dim, end_dim).shape))

    def unflatten(self, dim: int, sizes: Sequence[int]) -> "RegionTensor":
        """The tensor with dimension ``dim`` unflattened into ``sizes``."""
        return self._reshaped(tuple(self._meta().unflatten(dim, sizes).shape))

    def transpose(self, dim0: int, dim1: int) -> "RegionTensor":
        """The tensor with dimensions ``dim0`` and ``dim1`` swapped."""
        self._meta().transpose(dim0, dim1)
        if not self._shape:
            return self
        order = list(range(len(self._shape)))
        first, second = dim0 % len(order), dim1 % len(order)
        order[first], order[second] = order[second], order[first]
        return self._permuted(order)

    def swapaxes(self, axis0: int, axis1: int) -> "RegionTensor":
        """The tensor with dimensions ``axis0`` and ``axis1`` swapped."""
        return self.transpose(axis0, axis1)

    def swapdims(self, dim0: int, dim1: int) -> "RegionTensor":
        """The tensor with dimensions ``dim0`` and ``dim1`` swapped."""
        return self.transpose(dim0, dim1)

    def t(self) -> "RegionTensor":
        """The tensor with its two dimensions swapped, or itself when it has fewer."""
        self._meta().t()
        return self.transpose(0, 1) if len(self._shape) == 2 else self

    def permute(self, *dims: Any) -> "RegionTensor":
        """The tensor with its dimensions reordered."""
        dims = _shape_argument(dims)
        self._meta().permute(dims)
        return self._permuted([dim % len(self._shape) for dim in dims]) if self._shape else self

    def squeeze(self, dim: Any = None) -> "RegionTensor":
        """The tensor without the dimensions of size one ``dim`` names, or without all of them."""
        expected = tuple((self._meta().squeeze() if dim is None else self._meta().squeeze(dim)).shape)
        if not self._shape:
            return self
        if dim is None:
            axes = {axis for axis, size in enumerate(self._shape) if size == 1}
        else:
            dims = dim if isinstance(dim, (tuple, list)) else (dim,)
            axes = {d % len(self._shape) for d in dims if self._shape[d % len(self._shape)] == 1}
        blocks = list(self._blocks)
        for axis in sorted(axes, reverse=True):
            blocks = [_drop_axis(block, axis) for block in blocks]
        return self._checked(expected, blocks)

    def unsqueeze(self, dim: int) -> "RegionTensor":
        """The tensor with a new dimension of size one at ``dim``."""
        expected = tuple(self._meta().unsqueeze(dim).shape)
        axis = dim % (len(self._shape) + 1)
        return self._checked(expected, [_insert_axis(block, axis) for block in self._blocks])

    def split(self, split_size_or_sections: Any, dim: int = 0) -> tuple["RegionTensor", ...]:
        """The tensor cut along ``dim``, as ``torch.split`` cuts it."""
        return self._pieces(self._meta().split(split_size_or_sections, dim), dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["RegionTensor", ...]:
        """The tensor cut along ``dim``, as ``torch.chunk`` cuts it."""
        return self._pieces(self._meta().chunk(chunks, dim), dim)

    def unbind(self, dim: int = 0) -> tuple["RegionTensor", ...]:
        """Every index of dimension ``dim``, without that dimension."""
        size = self._meta().size(dim)
        return tuple(self.select(dim, index) for index in range(size))

    def narrow(self, dim: int, start: int, length: int) -> "RegionTensor":
        """``length`` indices of dimension ``dim`` from ``start``."""
        expected = tuple(self._meta().narrow(dim, start, length).shape)
        axis = dim % len(self._shape)
        start = start + self._shape[axis] if start < 0 else start
        return self._checked(expected, _slice_axis(self._blocks, axis, start, start + length, 1))

    def select(self, dim: int, index: int) -> "RegionTensor":
        """Index ``index`` of dimension ``dim``, without that dimension."""
        expected = tuple(self._meta().select(dim, index).shape)
        axis = dim % len(self._shape)
        index = index + self._shape[axis] if index < 0 else index
        blocks = [_drop_axis(block, axis) for block in _slice_axis(self._blocks, axis, index, index + 1, 1)]
        return self._checked(expected, blocks)

    def __getitem__(self, index: Any) -> "RegionTensor":
        """The part of the tensor integers, slices with a positive step, None and Ellipsis pick out."""
        expanded = _expand_index(index, len(self._shape))
        expected = tuple(self._meta()[index].shape)
        shape, blocks, axis = list(self._shape), list(self._blocks), 0
        for item in expanded:
            if item is None:
                blocks = [_insert_axis(block, axis) for block in blocks]
                shape.insert(axis, 1)
                axis += 1
            elif isinstance(item, slice):
                start, stop, step = item.indices(shape[axis])
                if step <= 0:
                    raise UnsupportedRegionOp("slicing a region with a negative step is not supported")
                blocks = _slice_axis(blocks, axis, start, stop, step)
                shape[axis] = len(range(start, stop, step))
                axis += 1
            else:
                position = item + shape[axis] if item < 0 else item
                blocks = [_drop_axis(block, axis) for block in _slice_axis(blocks, axis, position, position + 1, 1)]
                del shape[axis]
        return self._checked(expected, blocks)

    def contiguous(self, *args: Any, **kwargs: Any) -> "RegionTensor":
        """The tensor itself: a region has no memory layout."""
        del args, kwargs
        return self

    def clone(self, *args: Any, **kwargs: Any) -> "RegionTensor":
        """The tensor itself: a region is never written to."""
        del args, kwargs
        return self

    def detach(self) -> "RegionTensor":
        """The tensor itself."""
        return self

    def cpu(self) -> "RegionTensor":
        """The tensor itself: moving it does not change where its values come from."""
        return self

    def to(self, *args: Any, **kwargs: Any) -> "RegionTensor":
        """The tensor converted to the dtype among the arguments, if any. Devices are ignored."""
        dtype = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, (RegionTensor, torch.Tensor)):
                raise UnsupportedRegionOp("converting a region to the dtype of another tensor is not supported")
            if isinstance(arg, torch.dtype):
                dtype = arg
        return self if dtype is None else self._cast(dtype)

    def add(self, other: Any) -> "RegionTensor":
        """``self + other``."""
        return self._arithmetic(operator.add, other, reflected=False)

    def sub(self, other: Any) -> "RegionTensor":
        """``self - other``."""
        return self._arithmetic(operator.sub, other, reflected=False)

    def mul(self, other: Any) -> "RegionTensor":
        """``self * other``."""
        return self._arithmetic(operator.mul, other, reflected=False)

    def div(self, other: Any, *, rounding_mode: Any = None) -> "RegionTensor":
        """``self / other``, rounded as ``rounding_mode`` says."""
        if rounding_mode is None:
            return self._arithmetic(operator.truediv, other, reflected=False)
        _check_scalar(other)
        scalar = torch.empty((), dtype=self._dtype, device="meta")
        dtype = torch.div(scalar, other, rounding_mode=rounding_mode).dtype
        return self._elementwise(partial(_divide, other, rounding_mode), dtype)

    def neg(self) -> "RegionTensor":
        """``-self``."""
        return self._elementwise(operator.neg, (-torch.empty((), dtype=self._dtype, device="meta")).dtype)

    subtract = sub
    multiply = mul
    divide = div
    true_divide = div
    negative = neg
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = div
    __neg__ = neg

    def __radd__(self, other: Any) -> "RegionTensor":
        """``other + self``."""
        return self._arithmetic(operator.add, other, reflected=True)

    def __rsub__(self, other: Any) -> "RegionTensor":
        """``other - self``."""
        return self._arithmetic(operator.sub, other, reflected=True)

    def __rmul__(self, other: Any) -> "RegionTensor":
        """``other * self``."""
        return self._arithmetic(operator.mul, other, reflected=True)

    def __rtruediv__(self, other: Any) -> "RegionTensor":
        """``other / self``."""
        return self._arithmetic(operator.truediv, other, reflected=True)

    def __pos__(self) -> "RegionTensor":
        """The tensor itself."""
        return self

    __hash__ = object.__hash__
    __eq__ = _unsupported("__eq__")
    __ne__ = _unsupported("__ne__")
    __lt__ = _unsupported("__lt__")
    __le__ = _unsupported("__le__")
    __gt__ = _unsupported("__gt__")
    __ge__ = _unsupported("__ge__")
    __bool__ = _unsupported("__bool__")
    __index__ = _unsupported("__index__")
    __setitem__ = _unsupported("__setitem__")
    __iadd__ = _unsupported("__iadd__")
    __isub__ = _unsupported("__isub__")
    __imul__ = _unsupported("__imul__")
    __itruediv__ = _unsupported("__itruediv__")
    __pow__ = _unsupported("__pow__")
    item = _unsupported("item")
    tolist = _unsupported("tolist")
    numpy = _unsupported("numpy")
    copy_ = _unsupported("copy_")
    add_ = _unsupported("add_")
    mul_ = _unsupported("mul_")

    def _meta(self) -> torch.Tensor:
        """A meta tensor of the same shape and dtype, which torch validates arguments against."""
        return torch.empty(self._shape, dtype=self._dtype, device="meta")

    def _checked(self, expected: tuple[int, ...], blocks: Sequence[RegionBlock]) -> "RegionTensor":
        """A tensor of these blocks, whose shape the blocks' dimensions have to agree with."""
        ndim = len(expected)
        if any(len(block.offsets) != ndim for block in blocks):
            raise UnsupportedRegionOp(f"blocks disagree with the shape {expected} torch computes")
        return RegionTensor(expected, self._dtype, blocks)

    def _reshaped(self, shape: tuple[int, ...]) -> "RegionTensor":
        """The tensor reshaped to ``shape``, already validated by torch."""
        if shape == self._shape:
            return self
        return self._checked(shape, _reshape(self._blocks, self._shape, shape))

    def _permuted(self, order: Sequence[int]) -> "RegionTensor":
        """The tensor with its dimensions in ``order``."""
        shape = tuple(self._shape[axis] for axis in order)
        return RegionTensor(shape, self._dtype, [_permute_block(block, order) for block in self._blocks])

    def _pieces(self, metas: Sequence[torch.Tensor], dim: int) -> tuple["RegionTensor", ...]:
        """The tensor cut along ``dim`` into pieces as long as ``metas``, torch's own cut of it."""
        axis = dim % len(self._shape)
        pieces, start = [], 0
        for meta in metas:
            size = meta.shape[axis]
            pieces.append(self._checked(tuple(meta.shape), _slice_axis(self._blocks, axis, start, start + size, 1)))
            start += size
        return tuple(pieces)

    def _arithmetic(self, function: Callable[[Any, Any], Any], other: Any, reflected: bool) -> "RegionTensor":
        """The tensor combined elementwise with a Python number."""
        _check_scalar(other)
        scalar = torch.empty((), dtype=self._dtype, device="meta")
        dtype = (function(other, scalar) if reflected else function(scalar, other)).dtype
        operation = partial(_reflected if reflected else _forward, function, other)
        return self._elementwise(operation, dtype)

    def _elementwise(self, operation: Callable[[Any], Any], dtype: torch.dtype) -> "RegionTensor":
        """
        The tensor with ``operation`` applied to every element.

        A block still in another dtype than the tensor is converted to it first, so the operation sees
        the values in the dtype it would see them in.
        """
        blocks = []
        for block in self._blocks:
            post = block.post
            if block.dtype != self._dtype:
                post += (partial(_to_dtype, dtype=self._dtype),)
            blocks.append(replace(block, post=post + (operation,), dtype=dtype))
        return RegionTensor(self._shape, dtype, blocks)

    def _cast(self, dtype: torch.dtype) -> "RegionTensor":
        """The tensor converted to ``dtype``."""
        if dtype == self._dtype:
            return self
        return self._elementwise(partial(_to_dtype, dtype=dtype), dtype)

    def float(self) -> "RegionTensor":
        """The tensor converted to float32."""
        return self._cast(torch.float32)

    def double(self) -> "RegionTensor":
        """The tensor converted to float64."""
        return self._cast(torch.float64)

    def half(self) -> "RegionTensor":
        """The tensor converted to float16."""
        return self._cast(torch.float16)

    def bfloat16(self) -> "RegionTensor":
        """The tensor converted to bfloat16."""
        return self._cast(torch.bfloat16)

    def int(self) -> "RegionTensor":
        """The tensor converted to int32."""
        return self._cast(torch.int32)

    def long(self) -> "RegionTensor":
        """The tensor converted to int64."""
        return self._cast(torch.int64)


def _regions(tensors: Any, function: str) -> list[RegionTensor]:
    """The tensors passed to ``function``, which all have to be regions."""
    tensors = list(tensors)
    if not tensors or not all(isinstance(tensor, RegionTensor) for tensor in tensors):
        raise UnsupportedRegionOp(f"torch.{function} of regions mixed with other tensors is not supported")
    return tensors


def _cat(tensors: Sequence[Any], dim: int = 0) -> RegionTensor:
    """Regions concatenated along ``dim``, as ``torch.cat`` concatenates tensors."""
    tensors = _regions(tensors, "cat")
    meta = torch.cat([tensor._meta() for tensor in tensors], dim)  # pylint: disable=protected-access
    axis = dim % meta.dim()
    blocks, offset = [], 0
    for tensor in tensors:
        if tensor.ndim != meta.dim():
            # A one-dimensional empty tensor, which torch.cat leaves out.
            continue
        blocks.extend(_shift(block, axis, offset) for block in tensor.blocks)
        offset += tensor.shape[axis]
    return RegionTensor(meta.shape, meta.dtype, blocks)


def _stack(tensors: Sequence[Any], dim: int = 0) -> RegionTensor:
    """Regions stacked along a new dimension ``dim``, as ``torch.stack`` stacks tensors."""
    tensors = _regions(tensors, "stack")
    meta = torch.stack([tensor._meta() for tensor in tensors], dim)  # pylint: disable=protected-access
    axis = dim % meta.dim()
    blocks = [
        _shift(_insert_axis(block, axis), axis, position)
        for position, tensor in enumerate(tensors)
        for block in tensor.blocks
    ]
    return RegionTensor(meta.shape, meta.dtype, blocks)


_SEQUENCE_FUNCTIONS: dict[str, Callable[..., RegionTensor]] = {
    "cat": _cat,
    "concat": _cat,
    "concatenate": _cat,
    "stack": _stack,
}
_TENSOR_FUNCTIONS = (
    "reshape", "flatten", "unflatten", "transpose", "swapaxes", "swapdims", "t", "permute", "squeeze",
    "unsqueeze", "split", "chunk", "unbind", "narrow", "select", "clone", "detach", "add", "sub", "subtract",
    "mul", "multiply", "div", "divide", "true_divide", "neg", "negative",
)
_FUNCTION_NAMES = {
    getattr(torch, name): name for name in (*_SEQUENCE_FUNCTIONS, *_TENSOR_FUNCTIONS) if hasattr(torch, name)
}
_REFLECTED = {
    "add": "__radd__",
    "sub": "__rsub__",
    "subtract": "__rsub__",
    "mul": "__rmul__",
    "multiply": "__rmul__",
    "div": "__rtruediv__",
    "divide": "__rtruediv__",
    "true_divide": "__rtruediv__",
}


def materialize(region: RegionTensor, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute the values of a region out of the whole checkpoint tensors it reads.

    Args:
        region (RegionTensor): The region.
        sources (Mapping[str, torch.Tensor]): Every checkpoint tensor the region reads, by name.

    Returns:
        torch.Tensor: A new tensor of the region's shape and dtype, on the device of the sources.
    """
    device = next((sources[name].device for name in region.sources()), None)
    result = torch.empty(tuple(region.shape), dtype=region.dtype, device=device)
    for block in region.remap_blocks():
        source = sources[block.source]
        strides = source.stride()
        stride = tuple(
            sum(row[axis] * step for row, step in zip(block.coeff, strides)) for axis in range(len(block.lengths))
        )
        offset = source.storage_offset() + sum(start * step for start, step in zip(block.base, strides))
        values = source.as_strided(block.lengths, stride, offset)
        for operation in block.post:
            values = operation(values)
        box = tuple(slice(start, start + length) for start, length in zip(block.offsets, block.lengths))
        result[box].copy_(values)
    return result
