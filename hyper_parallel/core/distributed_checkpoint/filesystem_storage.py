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
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

from safetensors import safe_open

from hyper_parallel.core.distributed_checkpoint.metadata import (
    Metadata,
    MetadataIndex,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
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
    broadcast_loaded_tensors,
    dcp_timer_decorator,
    platform,
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
        if module in _MetadataUnpickler._FOREIGN_MODULE_MAP:
            module = _MetadataUnpickler._FOREIGN_MODULE_MAP[module]
        return super().find_class(module, name)


class FileSystemWriter(StorageWriter):
    """
    File system storage writer implementation.

    Saves checkpoint data to the local file system, organizing tensors
    into safetensors files and bytes into separate files.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]):
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

    def configure_writer(self, is_coordinator: bool, **kwargs) -> None:
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

    def optimize_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Optimize global plan.

        Args:
            plans (list[SavePlan]): List of local plans from all ranks.

        Returns:
            list[SavePlan]: Optimized global plans.
        """
        return plans


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


def _load_bytes_file(
        path: str,
        reqs: list[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[MetadataIndex, StorageInfo],
) -> None:
    """
    Load bytes from a file.

    Args:
        path (str): Path to the bytes file.
        reqs (list[ReadItem]): List of ReadItems for this file.
        planner (LoadPlanner): Load planner for loading bytes.
    """
    with open(path, "rb") as f:
        for req in reqs:
            storage_info = storage_data.get(req.storage_index)
            if storage_info is None:
                raise KeyError(
                    f"StorageInfo not found for index {req.storage_index}"
                )
            f.seek(storage_info.offset)
            value = f.read(storage_info.length)
            planner.apply_bytes(req, value)


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


def _load_torch_tensor_file(
        path: str,
        reqs: list[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[MetadataIndex, StorageInfo],
) -> None:
    """Load tensor slices from a Torch safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as tensor_file:
        available_keys = set(tensor_file.keys())
        for req in reqs:
            storage_info = _get_storage_info(req, storage_data)
            tensor_key = storage_info.tensor_key or req.storage_index.fqn
            if tensor_key not in available_keys:
                raise KeyError(f"Key {tensor_key} not found in checkpoint file {path}")
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
            _validate_and_copy_tensor(req, tensor, planner)


def _load_ms_tensor_file(
        path: str,
        reqs: list[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[MetadataIndex, StorageInfo],
) -> None:
    """Load tensor slices through the ms platform adapter."""
    param_dict = platform.load_checkpoint(path)
    for req in reqs:
        storage_info = _get_storage_info(req, storage_data)
        tensor_key = storage_info.tensor_key or req.storage_index.fqn
        if tensor_key not in param_dict:
            raise KeyError(f"Key {tensor_key} not found in checkpoint file {path}")
        tensor = narrow_tensor_by_index(
            param_dict[tensor_key],
            req.storage_offsets,
            req.lengths,
        )
        _validate_and_copy_tensor(req, tensor, planner)


def _load_tensor_file(
        path: str,
        reqs: list[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[MetadataIndex, StorageInfo],
) -> None:
    """
    Load and process tensors from a safetensors file.

    Args:
        path (str): Path to the safetensors file.
        reqs (list[ReadItem]): List of ReadItems for this file.
        planner (LoadPlanner): Load planner for resolving and committing tensors.
        storage_data (dict[MetadataIndex, StorageInfo]): Physical storage mapping.
    """
    if platform.platform_type == PlatformType.PYTORCH:
        _load_torch_tensor_file(path, reqs, planner, storage_data)
    else:
        _load_ms_tensor_file(path, reqs, planner, storage_data)


class FileSystemReader(StorageReader):
    """
    File system storage reader implementation.

    Reads checkpoint data from the local file system, loading tensors
    from safetensors files and bytes from separate files.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]):
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        # Cached storage layout: MetadataIndex -> StorageInfo (torch-aligned)
        self.storage_data: dict[MetadataIndex, StorageInfo] = {}
        self.rank: int = 0
        self.is_coordinator: bool = False
        self.broadcast_from_minimum_rank = False

    def initialize_reader(self, checkpoint_id: Optional[Union[Path, str]] = None) -> None:
        """
        Initialize storage reader with new checkpoint directory.

        Args:
            checkpoint_id (Optional[Union[Path, str]]): New checkpoint directory path. Default None.
        """
        if checkpoint_id:
            self.checkpoint_dir = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    @dcp_timer_decorator
    def load_metadata(self, **kwargs) -> Metadata:
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

    def configure_reader(self, metadata: Metadata, is_coordinator: bool, **kwargs) -> None:
        """Configure storage reader."""
        # Cache storage_data separately for quick lookup in execute_read.
        # This mirrors torch.filesystem, where reader keeps a storage_data dict.
        self.storage_data = getattr(metadata, "storage_data", None)
        self.is_coordinator = is_coordinator
        # Do not evaluate get_rank() when an offline caller supplies rank. The
        # default process group is intentionally not initialized by converters.
        self.rank = kwargs["rank"] if "rank" in kwargs else platform.get_rank()
        self.broadcast_from_minimum_rank = kwargs.get("broadcast_from_minimum_rank", self.broadcast_from_minimum_rank)

    def optimize_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Optimize local plan.

        Args:
            plan (LoadPlan): Local load plan.

        Returns:
            LoadPlan: Optimized local plan.
        """
        return plan

    def optimize_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Optimize global plan.

        Args:
            plans (list[LoadPlan]): List of local plans from all ranks.

        Returns:
            list[LoadPlan]: Optimized global plans.
        """
        return plans

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

    def _group_items_by_file(self, plan: LoadPlan) -> dict[str, list]:
        """
        Group ReadItems by storage file path.

        Args:
            plan (LoadPlan): Load plan containing ReadItems.

        Returns:
            dict[str, list[ReadItem]]: Dictionary mapping file paths to lists of ReadItems.
        """
        per_file: dict[str, list] = {}
        for read_item in plan.items:
            path = self._get_storage_path(read_item)
            per_file.setdefault(path, []).append(read_item)
        return per_file

    @dcp_timer_decorator
    def execute_read(
        self,
        plan: LoadPlan,
        planner: LoadPlanner,
        broadcast_groups: Optional[dict[tuple, Any]] = None,
    ) -> None:
        """
        Read data from storage.

        Aligned with torch filesystem read_data: groups ReadItems by file,
        loads each file once, narrows tensors by storage_offsets/lengths for
        resharding, then resolves/copies/commits data.

        Args:
            plan (LoadPlan): Load plan containing ReadItems.
            planner (LoadPlanner): Load planner for resolving and committing tensors.
            broadcast_groups (Optional[dict]): Communication groups for broadcast.
                Only consulted when this reader was configured with
                ``broadcast_from_minimum_rank``; omit it for a plain read.
        """
        # Group ReadItems by storage file path (like torch per_file)
        per_file = self._group_items_by_file(plan)

        # Process each file
        for path, reqs in per_file.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")

            if path.endswith(".bytes"):
                # BYTE_IO: one bytes file per rank with per-item offsets.
                _load_bytes_file(path, reqs, planner, self.storage_data)
            else:
                # TENSOR: one safetensors file per rank
                _load_tensor_file(path, reqs, planner, self.storage_data)

        if self.broadcast_from_minimum_rank:
            broadcast_loaded_tensors(planner.state_dict, broadcast_groups)
