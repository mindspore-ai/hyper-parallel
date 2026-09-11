"""safe_open-based tensor load behavior for DCP torch reader"""
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.torch.utils import init_dist


class _FakeTensor:
    """Minimal tensor-like object for reader-path tests."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.copied_from = None

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        out_shape = []
        for item in slices:
            out_shape.append(int(item.stop) - int(item.start))
        if len(self.shape) > len(slices):
            out_shape.extend(self.shape[len(slices):])
        return _FakeTensor(tuple(out_shape))

    def copy_(self, other):
        self.copied_from = other
        self.shape = getattr(other, "shape", self.shape)
        return self


class _DummyPlanner:
    """Minimal planner stub used by the tensor-file reader tests."""

    def __init__(self, target_shape):
        self.target = _FakeTensor(target_shape)

    def acquire_tensor(self, read_item):
        _ = read_item
        return self.target

    def apply_tensor(self, read_item, tensor):
        _ = read_item, tensor


class _FakeSliceFile:
    """Fake safetensors handle recording which read API the reader picks."""

    def __init__(self, shape):
        self.shape = shape
        self.slice_calls = []
        self.tensor_calls = []

    def keys(self):
        return ["layer.weight"]

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_slice(self, key):
        self.slice_calls.append(key)
        return _FakeTensor(self.shape)

    def get_tensor(self, key):
        self.tensor_calls.append(key)
        return _FakeTensor(self.shape)


def _runtime_imports():
    """Import the reader internals under test, and wrap them the way execute_read calls them.

    The imports sit inside the function because every case calls this only after
    ``init_dist()``, so the DCP modules are first imported with the platform already up.
    The wrapper saves each case from repeating how a read of one file is put together.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.distributed_checkpoint.filesystem_storage import (
        _apply_fetched,
        _fetch_tensor_file,
        _open_checkpoint_files,
    )
    from hyper_parallel.core.distributed_checkpoint.metadata import MetadataIndex
    from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType, ReadItem
    from hyper_parallel.core.distributed_checkpoint.storage import StorageInfo

    def load_tensor_file(path: str, reqs: list, planner: Any, storage_data: dict) -> None:
        """Both halves of a read of one file, as execute_read puts them together."""
        open_files = _open_checkpoint_files()
        try:
            _apply_fetched(_fetch_tensor_file(open_files.reader(path), reqs, storage_data), planner)
        finally:
            open_files.close()

    return load_tensor_file, MetadataIndex, LoadItemType, ReadItem, StorageInfo


def _build_read_item(metadata_index_cls, load_item_type_cls, read_item_cls, storage_offsets=(), lengths=()):
    return read_item_cls(
        type=load_item_type_cls.TENSOR,
        dest_index=metadata_index_cls(fqn="layer.weight", offset=(0, 0), index=0),
        dest_offsets=(0, 0),
        storage_index=metadata_index_cls(fqn="layer.weight", offset=(0, 0), index=0),
        storage_offsets=storage_offsets,
        lengths=lengths,
    )


def _build_storage_data(storage_info_cls, read_item):
    return {
        read_item.storage_index: storage_info_cls(
            relative_path="dummy.safetensors",
            offset=0,
            length=-1,
            tensor_key="layer.weight",
        )
    }


def test_dcp_safe_open_lazy_tensor_lookup():
    """
    Feature: DCP tensor reader uses safe_open for torch safetensors.
    Description: A read item with no offsets and no lengths, which is what the planner emits
        for a zero-dimensional tensor (an optimizer ``step``, say): the slice tuple comes out
        empty and the whole tensor is wanted. Every other tensor is read through a full range
        slice per dimension, so this is the one shape that exercises the empty slice.
    Expectation: The reader still resolves the fqn through the ``safe_open`` handle and never
        falls back to loading the entire file with ``load_checkpoint``. It asks for the whole
        tensor with ``get_tensor()`` instead of indexing a slice with the empty tuple, which
        safetensors below 0.4.3 rejects outright for a scalar and answers with a wrongly shaped
        tensor for anything else.
    """
    init_dist()
    load_tensor_file, metadata_index_cls, load_item_type_cls, read_item_cls, storage_info_cls = _runtime_imports()
    req = _build_read_item(metadata_index_cls, load_item_type_cls, read_item_cls)
    storage_data = _build_storage_data(storage_info_cls, req)
    planner = _DummyPlanner(target_shape=(2, 2))
    tensor_file = _FakeSliceFile(shape=(2, 2))

    with patch(
        "hyper_parallel.core.distributed_checkpoint.filesystem_storage.safe_open",
        side_effect=lambda *args, **kwargs: tensor_file,
    ), patch(
        "hyper_parallel.platform.torch.platform.TorchPlatform.load_checkpoint",
        side_effect=AssertionError("safe_open path should not call load_checkpoint"),
    ):
        load_tensor_file(str(Path("./dummy.safetensors")), [req], planner, storage_data)

    assert tensor_file.tensor_calls == ["layer.weight"]
    assert not tensor_file.slice_calls
    assert planner.target.copied_from is not None
    assert planner.target.shape == (2, 2)


def test_dcp_safe_open_slice_lookup():
    """
    Feature: DCP tensor reader prefers safetensors slice API.
    Description: When a tensor region is requested, use get_slice() instead of materializing the full tensor.
    Expectation: Tensor reader serves the request through safe_open.get_slice().
    """
    init_dist()
    load_tensor_file, metadata_index_cls, load_item_type_cls, read_item_cls, storage_info_cls = _runtime_imports()
    req = _build_read_item(
        metadata_index_cls,
        load_item_type_cls,
        read_item_cls,
        storage_offsets=(1, 2),
        lengths=(2, 3),
    )
    storage_data = _build_storage_data(storage_info_cls, req)
    planner = _DummyPlanner(target_shape=(2, 3))
    tensor_file = _FakeSliceFile(shape=(8, 8))

    with patch(
        "hyper_parallel.core.distributed_checkpoint.filesystem_storage.safe_open",
        side_effect=lambda *args, **kwargs: tensor_file,
    ), patch(
        "hyper_parallel.platform.torch.platform.TorchPlatform.load_checkpoint",
        side_effect=AssertionError("safe_open path should not call load_checkpoint"),
    ):
        load_tensor_file(str(Path("./dummy.safetensors")), [req], planner, storage_data)

    assert tensor_file.slice_calls == ["layer.weight"]
    assert not tensor_file.tensor_calls
    assert planner.target.shape == (2, 3)
