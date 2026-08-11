"""safe_open-based tensor load behavior for DCP torch reader"""
from pathlib import Path
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
    """Fake safetensors handle that only supports slice reads."""

    def __init__(self, shape):
        self.shape = shape
        self.slice_calls = []
        self.tensor_calls = []

    def keys(self):
        return ["layer.weight"]

    def get_slice(self, key):
        self.slice_calls.append(key)
        return _FakeTensor(self.shape)

    def get_tensor(self, key):
        self.tensor_calls.append(key)
        raise AssertionError("slice path should not call get_tensor")


class _FakeTensorFile:
    """Fake safetensors handle that only supports full-tensor reads."""

    def __init__(self, shape):
        self.shape = shape
        self.tensor_calls = []

    def keys(self):
        return ["layer.weight"]

    def get_tensor(self, key):
        self.tensor_calls.append(key)
        return _FakeTensor(self.shape)


class _SafeOpenContext:
    def __init__(self, tensor_file):
        self.tensor_file = tensor_file

    def __enter__(self):
        return self.tensor_file

    def __exit__(self, exc_type, exc, tb):
        return False


def _runtime_imports():
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.distributed_checkpoint.filesystem_storage import _load_tensor_file
    from hyper_parallel.core.distributed_checkpoint.metadata import MetadataIndex
    from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType, ReadItem
    from hyper_parallel.core.distributed_checkpoint.storage import StorageInfo

    return _load_tensor_file, MetadataIndex, LoadItemType, ReadItem, StorageInfo


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
    Description: Replace whole-file load with lazy key lookup on the safetensors handle.
    Expectation: Tensor reader resolves the requested fqn through safe_open.get_tensor().
    """
    init_dist()
    load_tensor_file, metadata_index_cls, load_item_type_cls, read_item_cls, storage_info_cls = _runtime_imports()
    req = _build_read_item(metadata_index_cls, load_item_type_cls, read_item_cls)
    storage_data = _build_storage_data(storage_info_cls, req)
    planner = _DummyPlanner(target_shape=(2, 2))
    tensor_file = _FakeTensorFile(shape=(2, 2))

    with patch(
        "hyper_parallel.core.distributed_checkpoint.filesystem_storage.safe_open",
        side_effect=lambda *args, **kwargs: _SafeOpenContext(tensor_file),
    ), patch(
        "hyper_parallel.platform.torch.platform.TorchPlatform.load_checkpoint",
        side_effect=AssertionError("safe_open path should not call load_checkpoint"),
    ):
        load_tensor_file(str(Path("./dummy.safetensors")), [req], planner, storage_data)

    assert tensor_file.tensor_calls == ["layer.weight"]
    assert planner.target.copied_from is not None


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
        side_effect=lambda *args, **kwargs: _SafeOpenContext(tensor_file),
    ), patch(
        "hyper_parallel.platform.torch.platform.TorchPlatform.load_checkpoint",
        side_effect=AssertionError("safe_open path should not call load_checkpoint"),
    ):
        load_tensor_file(str(Path("./dummy.safetensors")), [req], planner, storage_data)

    assert tensor_file.slice_calls == ["layer.weight"]
    assert not tensor_file.tensor_calls
    assert planner.target.shape == (2, 3)
