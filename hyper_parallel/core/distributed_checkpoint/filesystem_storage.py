# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""File system storage implementations for checkpoint save and load."""
import pickle
from collections import Counter, deque
from pathlib import Path
from typing import Any, Optional, Union

from safetensors import safe_open

from hyper_parallel.core.distributed_checkpoint.metadata import (
    Metadata,
    MetadataIndex,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from hyper_parallel.core.distributed_checkpoint.storage import (
    StorageInfo,
    StorageReader,
    StorageWriter,
    WriteResult,
    METADATA_FILE_NAME,
)
from hyper_parallel.core.distributed_checkpoint.util import (
    narrow_tensor_by_index,
    BroadcastBatcher,
    dcp_timer_decorator,
    logger,
    platform,
    wait_broadcasts,
)

from hyper_parallel.platform.platform import (
    PlatformType,
)


class _MetadataUnpickler(pickle.Unpickler):
    """Unpickler for the metadata file, remapping module names while loading.

    ``_FOREIGN_MODULE_MAP`` maps a module name recorded in the pickle to the module to import
    in its place, so that metadata written by a tree where these classes live under other
    module names still loads. Add an entry per renamed module to support such a tree. It
    is empty here: metadata written by this tree names bare ``hyper_parallel.*`` modules,
    which are imported as they are.
    """

    _FOREIGN_MODULE_MAP = {}

    def find_class(self, module: str, name: str) -> Any:
        """
        Resolve a pickled reference, sending the modules that moved to where they now live.

        Args:
            module (str): Module the pickle names.
            name (str): Name to resolve within it.

        Returns:
            Any: The class or function the reference stands for.
        """
        if module in _MetadataUnpickler._FOREIGN_MODULE_MAP:
            module = _MetadataUnpickler._FOREIGN_MODULE_MAP[module]
        return super().find_class(module, name)


class FileSystemWriter(StorageWriter):
    """
    File system storage writer implementation.

    Saves checkpoint data to the local file system, organizing tensors
    into safetensors files and bytes into separate files.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        """
        Args:
            checkpoint_dir (Union[Path, str]): Directory the checkpoint is written into,
                made if it is not there yet.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank: int = 0
        self.is_coordinator: bool = False
        self.use_collectives: bool = True

    def initialize_writer(self, checkpoint_id: Optional[Union[Path, str]] = None) -> None:
        """
        Initialize storage writer with new checkpoint directory.

        Args:
            checkpoint_id (Optional[Union[Path, str]]): New checkpoint directory path. Default None.
        """
        if checkpoint_id:
            self.checkpoint_dir = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def configure_writer(self, is_coordinator: bool, **kwargs: Any) -> None:
        """
        Configure storage writer.

        Args:
            is_coordinator (bool): Whether this rank is the coordinator.
            **kwargs: Additional keyword arguments (e.g., rank, use_collectives).
        """
        self.is_coordinator = is_coordinator
        self.rank = kwargs.get("rank") if "rank" in kwargs else platform.get_rank()
        self.use_collectives = kwargs.get("use_collectives", True)

    def optimize_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Optimize local plan.

        Args:
            plan (SavePlan): Local save plan.

        Returns:
            SavePlan: Optimized local plan.
        """
        return plan

    def optimize_global_plan(self, plan: SavePlan) -> SavePlan:
        """
        Optimize this rank's save plan.

        Args:
            plan (SavePlan): This rank's save plan, with storage indices assigned.

        Returns:
            SavePlan: Optimized save plan.
        """
        return plan


    def _serialize_bytes_item(self, item: WriteItem, planner: SavePlanner) -> bytes:
        """Serialize a BYTE_IO item payload while preserving current behavior."""
        data = planner.get_data(item)
        if isinstance(data, bytes):
            return data
        return pickle.dumps(data)


    @dcp_timer_decorator
    def _write_bytes_items(self, plan: SavePlan, planner: SavePlanner) -> list[WriteResult]:
        """
        Write all BYTE_IO items into one per-rank bytes file.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            planner (SavePlanner): Save planner used to resolve runtime data.

        Returns:
            list[WriteResult]: Write results for BYTE_IO items.
        """
        byte_items = [item for item in plan.items if item.type.value == "byte_io"]
        if not byte_items:
            return []

        file_name = f"_rank{self.rank}_.bytes"
        file_path = self.checkpoint_dir / file_name

        results: list[WriteResult] = []

        with open(file_path, "wb") as f:
            for item in byte_items:
                payload = self._serialize_bytes_item(item, planner)
                offset = f.tell()
                f.write(payload)
                length = len(payload)
                storage_info = StorageInfo(
                    relative_path=file_name,
                    offset=offset,
                    length=length,
                )
                results.append(
                    WriteResult(
                        index=item.index,
                        storage_data=storage_info,
                    )
                )

        return results

    def _collect_tensors(
            self, plan: SavePlan, planner: SavePlanner
    ) -> tuple[dict[str, Any], dict[MetadataIndex, str]]:
        """
        Collect tensor data from planner runtime lookup.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            planner (SavePlanner): Save planner.

        Returns:
            tuple[dict[str, Any], dict[MetadataIndex, str]]: Tensor data keyed by
                physical safetensors key, plus logical index-to-key mapping.

        Raises:
            RuntimeError: If tensor data cannot be resolved for an item.
        """
        tensor_items = [
            item for item in plan.items
            if item.type.value == "tensor" and item.tensor_data
        ]
        fqn_counts = Counter(item.index.fqn for item in tensor_items)
        reserved_keys = set(fqn_counts)
        used_keys: set[str] = set()
        next_chunk_index: dict[str, int] = {}
        tensor_dict: dict[str, Any] = {}
        tensor_keys: dict[MetadataIndex, str] = {}

        for item in tensor_items:
            tensor = planner.get_data(item)
            if tensor is None:
                raise RuntimeError(
                    f"Tensor data could not be resolved for index {item.index}. "
                    f"FQN: {item.index.fqn}"
                )

            fqn = item.index.fqn
            tensor_key = fqn
            if fqn_counts[fqn] > 1:
                chunk_index = next_chunk_index.get(fqn, 0)
                next_chunk_index[fqn] = chunk_index + 1
                tensor_key = f"{fqn}.__dcp_chunk_{chunk_index}"
                while tensor_key in reserved_keys or tensor_key in used_keys:
                    tensor_key += "_"

            used_keys.add(tensor_key)
            tensor_dict[tensor_key] = tensor
            tensor_keys[item.index] = tensor_key
        return tensor_dict, tensor_keys

    @dcp_timer_decorator
    def _write_tensors(
            self,
            plan: SavePlan,
            tensor_dict: dict[str, Any],
            tensor_keys: dict[MetadataIndex, str],
    ) -> list[WriteResult]:
        """
        Write all tensors to safetensors file and create WriteResults.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            tensor_dict (dict[str, Any]): Dictionary mapping physical keys to tensor data.
            tensor_keys (dict[MetadataIndex, str]): Logical index-to-key mapping.

        Returns:
            list[WriteResult]: List of write results for tensor items.
        """
        if not tensor_dict:
            return []

        file_name = f"_rank{self.rank}_.safetensors"
        file_path = self.checkpoint_dir / file_name
        platform.save_checkpoint(tensor_dict, str(file_path))

        # Record StorageInfo for each tensor
        # Note: we don't know per-tensor byte offsets, so offset=0, length=-1
        results: list[WriteResult] = []
        for item in plan.items:
            if item.type.value == "tensor" and item.tensor_data:
                storage_info = StorageInfo(
                    relative_path=file_name,
                    offset=0,
                    length=-1,
                    tensor_key=tensor_keys[item.index],
                )
                results.append(
                    WriteResult(
                        index=item.index,
                        storage_data=storage_info,
                    )
                )
        return results

    @dcp_timer_decorator
    def execute_write(self, plan: SavePlan, planner: SavePlanner) -> list[WriteResult]:
        """
        Write data to storage and return per-item storage metadata.

        Group tensors into safetensors files and bytes into separate files, recording StorageInfo for each item.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            planner (SavePlanner): Save planner.

        Returns:
            list[WriteResult]: List of write results with storage metadata.
        """
        results: list[WriteResult] = []

        # Write all BYTE_IO items into one file per rank
        results.extend(self._write_bytes_items(plan, planner))

        # Collect and write tensors
        tensor_dict, tensor_keys = self._collect_tensors(plan, planner)
        results.extend(self._write_tensors(plan, tensor_dict, tensor_keys))

        return results

    def finalize_checkpoint(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """
        Finish writing checkpoint and populate metadata.storage_data.

        When use_collectives=True: only coordinator saves global metadata to .metadata.
        When use_collectives=False: each rank saves its own metadata to .rank{rank}_metadata,
        no cross-rank interaction.

        Args:
            metadata (Metadata): Checkpoint metadata to update.
            results (list[list[WriteResult]]): Write results from all ranks (or single rank when use_collectives=False).
        """
        should_save = not self.use_collectives or (self.use_collectives and self.is_coordinator)
        if not should_save:
            return

        # Build storage_data: map MetadataIndex -> StorageInfo
        storage_md: dict[MetadataIndex, StorageInfo] = {}
        for wr_list in results:
            for wr in wr_list:
                storage_md[wr.index] = wr.storage_data
        metadata.storage_data = storage_md

        # Save metadata file
        if self.use_collectives:
            metadata_file = self.checkpoint_dir / METADATA_FILE_NAME
        else:
            metadata_file = self.checkpoint_dir / f"{self.rank}{METADATA_FILE_NAME}"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)


def _copy_tensor_to_target(
        req: ReadItem, tensor: Any, target_tensor: Any, planner: LoadPlanner
) -> None:
    """
    Copy tensor data to target tensor and commit.

    Args:
        req (ReadItem): ReadItem request.
        tensor (Any): Source tensor (tensor-like object).
        target_tensor (Any): Target tensor (tensor-like object).
        planner (LoadPlanner): Load planner for committing.
    """
    if hasattr(target_tensor, "copy_"):
        # for torch and ms, call 'copy_' to copy data to target tensor
        target_tensor.copy_(tensor)
    else:
        target_tensor[...] = tensor
    planner.apply_tensor(req, target_tensor)


def _fetch_bytes_file(
        path: str,
        reqs: list[ReadItem],
        storage_data: dict[MetadataIndex, StorageInfo],
) -> list[tuple[ReadItem, Any]]:
    """
    Read the payload of each item out of a bytes file, leaving it packed.

    Args:
        path (str): Path to the bytes file.
        reqs (list[ReadItem]): List of ReadItems for this file.
        storage_data (dict[MetadataIndex, StorageInfo]): Physical storage mapping.

    Returns:
        list[tuple[ReadItem, Any]]: Each item with the bytes it asked for. Unpacking them
        into objects needs the planner, so it is left to :func:`_apply_fetched`.
    """
    fetched: list[tuple[ReadItem, Any]] = []
    with open(path, "rb") as f:
        for req in reqs:
            storage_info = _get_storage_info(req, storage_data)
            f.seek(storage_info.offset)
            fetched.append((req, f.read(storage_info.length)))
    return fetched


def _get_tensor_size(tensor: Any) -> Optional[tuple]:
    """
    Get size/shape of a tensor.

    Args:
        tensor (Any): Tensor object (tensor-like with shape/size attribute).

    Returns:
        Optional[tuple]: Tuple of tensor size or None if not available.
    """
    if hasattr(tensor, "size") and callable(tensor.size):
        return tuple(tensor.size())
    return getattr(tensor, "shape", None)


def _get_storage_info(
        req: ReadItem,
        storage_data: dict[MetadataIndex, StorageInfo],
) -> StorageInfo:
    """Return physical storage metadata for one read request."""
    storage_info = storage_data.get(req.storage_index)
    if storage_info is None:
        raise KeyError(f"StorageInfo not found for index {req.storage_index}")
    return storage_info


def _validate_and_copy_tensor(
        req: ReadItem,
        tensor: Any,
        planner: LoadPlanner,
) -> None:
    """Validate a loaded tensor slice and copy it to its planner destination."""
    target_tensor = planner.acquire_tensor(req)
    if platform.is_tensor(target_tensor):
        target_tensor = platform.detach(target_tensor)

    target_size = _get_tensor_size(target_tensor)
    tensor_size = _get_tensor_size(tensor)
    if target_size is not None and tensor_size is not None and target_size != tensor_size:
        raise AssertionError(
            f"req {req.storage_index} mismatch sizes "
            f"{target_size} vs {tensor_size}"
        )
    _copy_tensor_to_target(req, tensor, target_tensor, planner)


def _fetch_torch_tensor_file(
        tensor_file: Any,
        reqs: list[ReadItem],
        storage_data: dict[MetadataIndex, StorageInfo],
) -> list[tuple[ReadItem, Any]]:
    """Slice the region each item asked for off an open Torch safetensors file.

    A name the file does not hold is left to safetensors to report - "File does not contain
    tensor <name>", which says as much as a check here could. Checking first meant listing
    the file, and listing brings every name in it across from the reader whether one shard is
    being read or all of them: 6.5 ms on a file of nine thousand tensors, paid again every
    time the file is opened, to say ahead of time what the next line says anyway.
    """
    fetched: list[tuple[ReadItem, Any]] = []
    for req in reqs:
        storage_info = _get_storage_info(req, storage_data)
        tensor_key = storage_info.tensor_key or req.storage_index.fqn
        tensor_slices = tuple(
            slice(int(off), int(off) + int(length))
            for off, length in zip(req.storage_offsets, req.lengths)
        )
        if tensor_slices:
            tensor = tensor_file.get_slice(tensor_key)[tensor_slices]
        else:
            # Scalar entries (rank-0 tensors such as the AdamW ``step``) have no slices to
            # narrow by, and safetensors before 0.4.3 rejects the empty index with
            # "too many indices for tensor of dimension 0" - read the whole tensor instead.
            tensor = tensor_file.get_tensor(tensor_key)
        fetched.append((req, tensor))
    return fetched


def _fetch_ms_tensor_file(
        param_dict: Any,
        reqs: list[ReadItem],
        storage_data: dict[MetadataIndex, StorageInfo],
) -> list[tuple[ReadItem, Any]]:
    """Narrow the region each item asked for out of a file the ms adapter has read in."""
    fetched: list[tuple[ReadItem, Any]] = []
    for req in reqs:
        storage_info = _get_storage_info(req, storage_data)
        tensor_key = storage_info.tensor_key or req.storage_index.fqn
        if tensor_key not in param_dict:
            raise KeyError(f"Key {tensor_key} not found in checkpoint file")
        fetched.append((req, narrow_tensor_by_index(
            param_dict[tensor_key],
            req.storage_offsets,
            req.lengths,
        )))
    return fetched


def _fetch_tensor_file(
        tensor_file: Any,
        reqs: list[ReadItem],
        storage_data: dict[MetadataIndex, StorageInfo],
) -> list[tuple[ReadItem, Any]]:
    """
    Take what a set of items wants out of an already open checkpoint file.

    Nothing but the file is touched: no planner, no device, which is what leaves the caller
    free to decide when what was read is put in place.

    Args:
        tensor_file (Any): The open file, as :class:`_OpenFiles` hands it over.
        reqs (list[ReadItem]): List of ReadItems for this file.
        storage_data (dict[MetadataIndex, StorageInfo]): Physical storage mapping.

    Returns:
        list[tuple[ReadItem, Any]]: Each item with the region it asked for.
    """
    if platform.platform_type == PlatformType.PYTORCH:
        return _fetch_torch_tensor_file(tensor_file, reqs, storage_data)
    return _fetch_ms_tensor_file(tensor_file, reqs, storage_data)


def _apply_fetched(
        fetched: list[tuple[ReadItem, Any]],
        planner: LoadPlanner,
) -> None:
    """
    Put what was read where the load wants it.

    The other half of a read, and the half that has to stay with whoever owns the device and
    the planner: it copies into the state dict and hands each item back to the planner.

    Args:
        fetched (list[tuple[ReadItem, Any]]): Items with what was read for them, as
            :func:`_fetch_tensor_file` and :func:`_fetch_bytes_file` return them.
        planner (LoadPlanner): Load planner for resolving and committing.
    """
    for req, payload in fetched:
        if req.type is LoadItemType.BYTE_IO:
            planner.apply_bytes(req, payload)
        else:
            _validate_and_copy_tensor(req, payload, planner)


# How many checkpoint files a read keeps open. Going through the shards in the order every
# rank agrees on means walking the files over and over, where the old file-at-a-time read
# went through each one once, and opening one costs far more than the slice read it wraps --
# 0.70 ms against 0.045 ms, measured on a file holding four hundred tensors. So the files
# stay open. How many depends on what open means: a torch reader is a memory map over the
# file and several cost next to nothing to hold, while the ms adapter reads every tensor of
# the file into memory, so only the one in use is kept, as before.
_TORCH_FILES_KEPT = 8
_MS_FILES_KEPT = 1


class _OpenFiles:
    """The checkpoint files a read is holding open, least recently reached for first."""

    def __init__(self, capacity: int, open_one: Any, close_one: Any) -> None:
        """
        Args:
            capacity (int): How many files to hold before letting the oldest go.
            open_one (Any): Called with a path, returns a reader for it.
            close_one (Any): Called with a reader that is being let go.
        """
        self._capacity = capacity
        self._open_one = open_one
        self._close_one = close_one
        self._readers: dict[str, Any] = {}

    def reader(self, path: str) -> Any:
        """
        The reader for this file, opening it when it is not already held.

        Args:
            path (str): Path of the checkpoint file.

        Returns:
            Any: The open reader, now the most recently reached for.
        """
        reader = self._readers.pop(path, None)
        if reader is None:
            while len(self._readers) >= self._capacity:
                self._close_one(self._readers.pop(next(iter(self._readers))))
            reader = self._open_one(path)
        self._readers[path] = reader
        return reader

    def close(self) -> None:
        """Let go of every file still held."""
        for reader in self._readers.values():
            self._close_one(reader)
        self._readers.clear()


def _open_checkpoint_files() -> _OpenFiles:
    """
    A place to keep the checkpoint files of one read open, for the platform in use.

    Returns:
        _OpenFiles: Opens through safetensors on torch and through the platform adapter
        otherwise, holding as many as that kind of reader is cheap to hold.
    """
    if platform.platform_type == PlatformType.PYTORCH:
        return _OpenFiles(
            _TORCH_FILES_KEPT,
            lambda path: safe_open(path, framework="pt", device="cpu"),
            lambda reader: reader.__exit__(None, None, None),
        )
    return _OpenFiles(_MS_FILES_KEPT, platform.load_checkpoint, lambda reader: None)


def _broadcast_batch_bytes(requested: int) -> int:
    """
    How small a shard has to be to travel with others, for the platform in use.

    Returns:
        int: What the caller asked for on torch. Zero on the ms platform, where gathering
        shards into one buffer would go through reshaped views whose writes are not known to
        reach the buffer behind them; there every shard is sent on its own, as before.
    """
    if platform.platform_type == PlatformType.PYTORCH:
        return requested
    return 0


class FileSystemReader(StorageReader):
    """
    File system storage reader implementation.

    Reads checkpoint data from the local file system, loading tensors
    from safetensors files and bytes from separate files.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        """
        Args:
            checkpoint_dir (Union[Path, str]): Directory the checkpoint is read out of.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        # Cached storage layout: MetadataIndex -> StorageInfo (torch-aligned)
        self.storage_data: dict[MetadataIndex, StorageInfo] = {}
        self.rank: int = 0
        self.is_coordinator: bool = False

    def initialize_reader(self, checkpoint_id: Optional[Union[Path, str]] = None) -> None:
        """
        Initialize storage reader with new checkpoint directory.

        Args:
            checkpoint_id (Optional[Union[Path, str]]): New checkpoint directory path. Default None.
        """
        if checkpoint_id:
            self.checkpoint_dir = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    @dcp_timer_decorator
    def load_metadata(self, **kwargs: Any) -> Metadata:
        """
        Load checkpoint metadata from file.

        When rank is provided in kwargs: load rank-local metadata from .rank{rank}_metadata
        (for checkpoints saved with use_collectives=False).
        Otherwise: load global metadata from .metadata.

        Args:
            **kwargs: Optional arguments (e.g., rank for rank-local metadata).

        Returns:
            Metadata: Metadata object loaded from file.
        """
        rank = kwargs.get("rank")
        if rank is not None:
            metadata_file = self.checkpoint_dir / f"{rank}{METADATA_FILE_NAME}"
        else:
            metadata_file = self.checkpoint_dir / METADATA_FILE_NAME

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        with open(metadata_file, "rb") as f:
            metadata = _MetadataUnpickler(f).load()
        return metadata

    def configure_reader(self, metadata: Metadata, is_coordinator: bool,
                         **kwargs: Any) -> None:
        """Configure storage reader."""
        # Cache storage_data separately for quick lookup in execute_read.
        # This mirrors torch.filesystem, where reader keeps a storage_data dict.
        self.storage_data = getattr(metadata, "storage_data", None)
        self.is_coordinator = is_coordinator
        # Do not evaluate get_rank() when an offline caller supplies rank. The
        # default process group is intentionally not initialized by converters.
        self.rank = kwargs["rank"] if "rank" in kwargs else platform.get_rank()

    def optimize_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Optimize local plan.

        Args:
            plan (LoadPlan): Local load plan.

        Returns:
            LoadPlan: Optimized local plan.
        """
        return plan

    def optimize_global_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Optimize this rank's load plan.

        Args:
            plan (LoadPlan): This rank's load plan.

        Returns:
            LoadPlan: Optimized load plan.
        """
        return plan

    def _get_storage_path(self, read_item: ReadItem) -> str:
        """
        Get storage file path for a read item.

        Args:
            read_item (ReadItem): ReadItem to get path for.

        Returns:
            str: Absolute path to the storage file.
        """
        if self.storage_data is None:
            raise KeyError("Checkpoint metadata.storage_data is required for filesystem read")
        storage_info = self.storage_data.get(read_item.storage_index)
        if storage_info is None:
            raise KeyError(f"StorageInfo not found for index {read_item.storage_index}")
        return str(self.checkpoint_dir / storage_info.relative_path)

    def _group_items_by_file(self, items: list) -> dict[str, list]:
        """
        Group ReadItems by the checkpoint file they read from.

        Which items a rank goes to storage for at all is settled by :meth:`execute_read`,
        which only passes on the ones this rank reads itself.

        Args:
            items (list[ReadItem]): ReadItems to load from storage.

        Returns:
            dict[str, list[ReadItem]]: Dictionary mapping file paths to lists of ReadItems.
        """
        per_file: dict[str, list] = {}
        for read_item in items:
            path = self._get_storage_path(read_item)
            per_file.setdefault(path, []).append(read_item)
        return per_file

    def _private_batches(self, by_shard: dict) -> dict[str, list]:
        """
        Gather the shards no other rank shares into one batch per checkpoint file.

        Nothing waits on these - they are read after the last broadcast has been handed
        over - so unlike the shared shards they are under no obligation to be read in the
        order every rank agreed on, and are free to be read in whatever order costs least.
        Asking for a file's worth at once is that order: the file is opened, read for every
        shard that wants it, and left, however far apart in the plan those shards sit.

        What this is for above all is the pickled entries. Those go through the save side's
        dedup like everything else, so the one that writes an entry is whichever rank was
        carrying the least at the time, but the load side never broadcasts them - each rank
        rebuilds its own object - so every rank reads every entry it wants, out of whatever
        file the dedup put it in. Nothing holds a bytes file open between reads either, so
        one batch per shard opened a file for each: eighty entries spread over three files
        cost eighty opens where three would do.

        Args:
            by_shard (dict[MetadataIndex, list[ReadItem]]): The plan's items, gathered by
                the shard they land in.

        Returns:
            dict[str, list[ReadItem]]: The items of every file, keyed by its path, in the
            order the files first come up in the plan.
        """
        private: dict[str, list] = {}
        for items in by_shard.values():
            if items[0].source is not None:
                continue
            for item in items:
                private.setdefault(self._get_storage_path(item), []).append(item)
        return private

    def _fetch_from_storage(self, items: list, open_files: _OpenFiles) -> list[tuple[ReadItem, Any]]:
        """
        Take a set of items off disk, a file at a time, and hand back what was read.

        Only the checkpoint files are touched: putting any of it in place, which is what
        needs the planner and the device, is :func:`_apply_fetched`.

        A file that is not there is left to the open to report. Both of them raise
        FileNotFoundError naming the path - safetensors as well as the builtin open - so a
        check ahead of them said no more than they do, and said it with a stat per file, on
        storage where a stat is a round trip to another machine.

        Args:
            items (list[ReadItem]): The items to read, in any order.
            open_files (_OpenFiles): The files this read is holding open.

        Returns:
            list[tuple[ReadItem, Any]]: Each item with what was read for it.

        Raises:
            FileNotFoundError: If a file the items name is not there.
        """
        fetched: list[tuple[ReadItem, Any]] = []
        for path, reqs in self._group_items_by_file(items).items():
            if path.endswith(".bytes"):
                # BYTE_IO: one bytes file per rank with per-item offsets.
                fetched.extend(_fetch_bytes_file(path, reqs, self.storage_data))
            else:
                # TENSOR: one safetensors file per rank
                reader = open_files.reader(path)
                fetched.extend(_fetch_tensor_file(reader, reqs, self.storage_data))
        return fetched

    @dcp_timer_decorator
    def execute_read(
        self,
        plan: LoadPlan,
        planner: LoadPlanner,
        broadcast_groups: Optional[dict[tuple, Any]] = None,
        broadcast_batch_bytes: int = 0,
    ) -> None:
        """
        Read data from storage, overlapping the reads with the copies and sends they feed.

        Shards several ranks hold are dealt with first, in the order the global plan put
        them in, which every rank arrives at the same way. The rank that was chosen to read
        one starts its broadcast the moment it has it and goes straight on to the next shard,
        so a send runs while the one after it is still being read. Only so many sends are
        left going at once; past that the oldest is waited on to make room.

        What no other rank shares comes last. Nobody is waiting on it, so reading it earlier
        would only hold up the broadcasts, and reading it here fills the time the final sends
        are still in flight. Nobody waiting on it also means nothing holds it to the order the
        ranks agreed on, so it is read a file at a time rather than a shard at a time: each
        file is opened once and read for every shard that wants it.

        Args:
            plan (LoadPlan): Load plan containing ReadItems.
            planner (LoadPlanner): Load planner for resolving and committing tensors.
            broadcast_groups (Optional[dict]): Communication group of every shard the plan
                marked, keyed by rank tuple, as :func:`ensure_broadcast_groups` returns them.
                Building one is collective, so it is done before the read rather than here.
            broadcast_batch_bytes (int): Shards smaller than this are gathered and sent
                together instead of one at a time. Zero sends every shard on its own.

        Raises:
            KeyError: If the plan marked a shard whose group is not among the ones given.
        """
        groups = broadcast_groups or {}

        by_shard: dict[MetadataIndex, list] = {}
        for item in plan.items:
            by_shard.setdefault(item.dest_index, []).append(item)
        shared = sorted(
            (index for index, items in by_shard.items() if items[0].source is not None),
            key=lambda index: (index.fqn, index.offset, index.index),
        )
        # What this rank goes to storage for, in the order it will ask for it: the shards it
        # was picked to read for its group, a shard at a time so that a group waiting on this
        # rank waits for its own shard and no more, then what nobody else shares, a file at a
        # time.
        batches = [by_shard[index] for index in shared
                   if by_shard[index][0].source.src_rank == self.rank]
        batches.extend(self._private_batches(by_shard).values())

        in_flight: deque = deque()
        batcher = BroadcastBatcher(_broadcast_batch_bytes(broadcast_batch_bytes), groups)
        open_files = _open_checkpoint_files()
        try:
            read = (self._fetch_from_storage(batch, open_files) for batch in batches)
            for index in shared:
                items = by_shard[index]
                if items[0].source.src_rank == self.rank:
                    _apply_fetched(next(read), planner)
                batcher.add(in_flight, planner.state_dict, items[0])

            # Whatever is still waiting for company goes now, before the shards nobody
            # shares: the ranks holding them are waiting on these, and reading what
            # nobody wants first would keep them waiting through all of it.
            batcher.flush(in_flight)
            for batch in read:
                _apply_fetched(batch, planner)
            wait_broadcasts(in_flight)
        finally:
            open_files.close()

        if shared:
            logger.info(
                "[rank=%d] >>> %d replicated shards sent in %d broadcasts, %d of them gathered",
                self.rank, len(shared), batcher.sent, batcher.batched,
            )
