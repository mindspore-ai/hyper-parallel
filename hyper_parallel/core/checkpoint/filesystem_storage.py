# Copyright 2026 Huawei Technologies Co., Ltd
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
from pathlib import Path
from typing import Any, Optional, Union

from hyper_parallel.core.checkpoint.metadata import Metadata, MetadataIndex
from hyper_parallel.core.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from hyper_parallel.core.checkpoint.storage import (
    StorageInfo,
    StorageReader,
    StorageWriter,
    WriteResult,
    _metadata_file_name,
)
from hyper_parallel.core.checkpoint.util import narrow_tensor_by_index
from hyper_parallel.platform import get_platform


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
        self.rank = kwargs.get("rank", get_platform().get_rank())
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

    def _write_bytes_item(self, item: WriteItem) -> WriteResult:
        """
        Write a single bytes item to storage.

        Args:
            item (WriteItem): WriteItem containing bytes data.

        Returns:
            WriteResult: Write result with storage metadata.
        """
        fqn = item.index.fqn
        file_name = f"{fqn}_rank{self.rank}.bytes"
        file_path = self.checkpoint_dir / file_name
        with open(file_path, "wb") as f:
            if isinstance(item.bytes_io_data, bytes):
                f.write(item.bytes_io_data)
            else:
                pickle.dump(item.bytes_io_data, f)
            try:
                length = f.tell()
            except (OSError, IOError):
                length = 0
        storage_info = StorageInfo(
            relative_path=file_name,
            offset=0,
            length=length,
        )
        return WriteResult(
            index=item.index,
            storage_data=storage_info,
        )

    def _collect_tensors(self, plan: SavePlan, planner: SavePlanner) -> dict[str, Any]:
        """
        Collect tensor data from planner cache.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            planner (SavePlanner): Save planner.

        Returns:
            dict[str, Any]: Dictionary mapping FQN to tensor data.

        Raises:
            RuntimeError: If tensor data not found in planner cache.
        """
        tensor_dict: dict[str, Any] = {}
        for item in plan.items:
            if item.type.value == "tensor" and item.tensor_data:
                # Get tensor from planner cache instead of tensor_data
                tensor = planner.get_tensor(item.index)
                if tensor is None:
                    raise RuntimeError(
                        f"Tensor data not found in planner cache for index {item.index}. "
                        f"FQN: {item.index.fqn}"
                    )
                fqn = item.index.fqn
                tensor_dict[fqn] = tensor
        return tensor_dict

    def _write_tensors(self, plan: SavePlan, tensor_dict: dict[str, Any]) -> list[WriteResult]:
        """
        Write all tensors to safetensors file and create WriteResults.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            tensor_dict (dict[str, Any]): Dictionary mapping FQN to tensor data.

        Returns:
            list[WriteResult]: List of write results for tensor items.
        """
        if not tensor_dict:
            return []

        platform = get_platform()
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
                )
                results.append(
                    WriteResult(
                        index=item.index,
                        storage_data=storage_info,
                    )
                )
        return results

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

        # Collect tensors and write bytes objects
        for item in plan.items:
            if item.type.value == "byte_io":
                results.append(self._write_bytes_item(item))

        # Collect and write tensors
        tensor_dict = self._collect_tensors(plan, planner)
        results.extend(self._write_tensors(plan, tensor_dict))

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
        should_save = self.use_collectives and self.is_coordinator or not self.use_collectives

        if should_save:
            # Build storage_data: map MetadataIndex -> StorageInfo
            storage_md: dict[MetadataIndex, StorageInfo] = {}
            for wr_list in results:
                for wr in wr_list:
                    storage_md[wr.index] = wr.storage_data
            metadata.storage_data = storage_md

            # Save metadata file
            if self.use_collectives:
                metadata_file = self.checkpoint_dir / _metadata_file_name
            else:
                metadata_file = self.checkpoint_dir / f".rank{self.rank}_metadata"
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
        target_tensor.copy_(tensor)
        planner.apply_tensor(req, target_tensor)
    else:
        # mindspore or non-tensor: copy via commit path
        planner.apply_tensor(req, tensor)


def _load_bytes_file(path: str, reqs: list[ReadItem], planner: LoadPlanner) -> None:
    """
    Load bytes from a file.

    Args:
        path (str): Path to the bytes file.
        reqs (list[ReadItem]): List of ReadItems for this file.
        planner (LoadPlanner): Load planner for loading bytes.
    """
    for req in reqs:
        with open(path, "rb") as f:
            value = f.read()
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


def _load_tensor_file(
        path: str, reqs: list[ReadItem], planner: LoadPlanner
) -> None:
    """
    Load and process tensors from a safetensors file.

    Args:
        path (str): Path to the safetensors file.
        reqs (list[ReadItem]): List of ReadItems for this file.
        planner (LoadPlanner): Load planner for resolving and committing tensors.
    """
    platform = get_platform()
    param_dict = platform.load_checkpoint(path)

    for req in reqs:
        fqn = req.storage_index.fqn
        if fqn not in param_dict:
            raise KeyError(f"Key {fqn} not found in checkpoint file {path}")

        full_tensor = param_dict[fqn]
        # Narrow by storage_offsets/lengths (resharding)
        tensor = narrow_tensor_by_index(
            full_tensor,
            req.storage_offsets,
            req.lengths,
        )
        target_tensor = planner.acquire_tensor(req)
        if hasattr(target_tensor, "detach"):
            target_tensor = target_tensor.detach()

        # Size check (torch-aligned AssertionError)
        target_size = _get_tensor_size(target_tensor)
        tensor_size = _get_tensor_size(tensor)
        if target_size is not None and tensor_size is not None:
            if target_size != tensor_size:
                raise AssertionError(
                    f"req {req.storage_index} mismatch sizes "
                    f"{target_size} vs {tensor_size}"
                )

        # Copy data to target
        _copy_tensor_to_target(req, tensor, target_tensor, planner)


class FileSystemReader(StorageReader):
    """
    File system storage reader implementation.

    Reads checkpoint data from the local file system, loading tensors
    from safetensors files and bytes from separate files.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]):
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        # Cached storage layout: MetadataIndex -> StorageInfo (torch-aligned)
        self.storage_data: Optional[dict[MetadataIndex, StorageInfo]] = None
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
            metadata_file = self.checkpoint_dir / f".rank{rank}_metadata"
        else:
            metadata_file = self.checkpoint_dir / _metadata_file_name

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        return metadata

    def configure_reader(self, metadata: Metadata, is_coordinator: bool, **kwargs) -> None:
        """Configure storage reader."""
        # Cache storage_data separately for quick lookup in execute_read.
        # This mirrors torch.filesystem, where reader keeps a storage_data dict.
        self.storage_data = getattr(metadata, "storage_data", None)
        self.is_coordinator = is_coordinator
        self.rank = kwargs.get("rank", get_platform().get_rank())

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
        storage_data = self.storage_data

        if storage_data is not None:
            storage_info = storage_data.get(read_item.storage_index)
            if storage_info is None:
                raise KeyError(f"StorageInfo not found for index {read_item.storage_index}")
            return str(self.checkpoint_dir / storage_info.relative_path)
        # Fallback: derive path from rank & fqn (legacy format without storage_data)
        if read_item.type == LoadItemType.TENSOR:
            rank = read_item.storage_index.index or self.rank
            return str(self.checkpoint_dir / f"_rank{rank}_.safetensors")
        fqn = read_item.storage_index.fqn
        rank = read_item.storage_index.index or self.rank
        return str(self.checkpoint_dir / f"{fqn}_rank{rank}.bytes")

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

    def execute_read(self, plan: LoadPlan, planner: LoadPlanner) -> None:
        """
        Read data from storage.

        Aligned with torch filesystem read_data: groups ReadItems by file,
        loads each file once, narrows tensors by storage_offsets/lengths for
        resharding, then resolves/copies/commits data.

        Args:
            plan (LoadPlan): Load plan containing ReadItems.
            planner (LoadPlanner): Load planner for resolving and committing tensors.
        """
        # Group ReadItems by storage file path (like torch per_file)
        per_file = self._group_items_by_file(plan)

        # Process each file
        for path, reqs in per_file.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")

            if path.endswith(".bytes"):
                # BYTE_IO: one file per (fqn, rank)
                _load_bytes_file(path, reqs, planner)
            else:
                # TENSOR: one safetensors file per rank
                _load_tensor_file(path, reqs, planner)
