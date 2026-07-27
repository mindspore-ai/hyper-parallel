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
from collections.abc import Collection
from pathlib import Path
from typing import Any, Optional, Union

from safetensors import safe_open

from hyper_parallel.core.distributed_checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
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
    METADATA_FILE_NAME,
    StorageInfo,
    StorageReader,
    StorageWriter,
    WriteResult,
)
from hyper_parallel.core.distributed_checkpoint.util import narrow_tensor_by_index
from hyper_parallel.core.distributed_checkpoint.versioning import (
    CURRENT_CHECKPOINT_VERSION,
    migrate_metadata,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType


class FileSystemWriter(StorageWriter):
    """
    File system storage writer implementation.

    Saves checkpoint data to the local file system, organizing tensors
    into safetensors files and bytes into separate files.

    When *incremental_from* and *changed_fqns* are provided, only changed
    items are written to disk; unchanged items inherit their storage
    location from the baseline checkpoint (with relative paths relocated).
    """

    def __init__(
        self,
        checkpoint_dir: Union[Path, str],
        incremental_from: Optional[Union[Path, str]] = None,
        changed_fqns: Optional[Collection[str]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank: int = 0
        self.is_coordinator: bool = False
        self.use_collectives: bool = True
        self._incremental_from: Optional[Path] = (
            Path(incremental_from) if isinstance(incremental_from, str) else incremental_from
        )
        self._changed_fqns: Optional[frozenset[str]] = (
            frozenset(changed_fqns) if changed_fqns is not None else None
        )
        self._baseline_metadata: Optional[Metadata] = None

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
        self.rank = kwargs.get("rank") if "rank" in kwargs else get_platform().get_rank()
        self.use_collectives = kwargs.get("use_collectives", True)

    def optimize_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Optimize local plan.

        For incremental saves, embeds the normalized ``changed_fqns`` into
        ``plan.storage_data`` so that :meth:`optimize_global_plan` can verify
        cross-rank consistency.

        Args:
            plan (SavePlan): Local save plan.

        Returns:
            SavePlan: Optimized local plan.
        """
        if self._changed_fqns is not None:
            plan = SavePlan(
                items=plan.items,
                storage_data={"changed_fqns": self._changed_fqns},
                planner_data=plan.planner_data,
            )
        return plan

    def optimize_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Optimize global plan.

        For incremental saves:
        1. Verifies all ranks have identical ``changed_fqns``.
        2. Filters out WriteItems whose FQN is not in ``changed_fqns``.
        3. Loads and migrates the baseline metadata.

        Args:
            plans (list[SavePlan]): List of local plans from all ranks.

        Returns:
            list[SavePlan]: Optimized global plans.

        Raises:
            ValueError: If ranks disagree on ``changed_fqns``, or if an
                unchanged tensor has incompatible storage metadata.
        """
        if self._changed_fqns is None:
            return plans

        all_fqn_sets = []
        for plan in plans:
            plan_fqns = plan.storage_data.get("changed_fqns") if plan.storage_data else None
            if plan_fqns is None:
                raise ValueError(
                    "Incremental save requires all ranks to provide changed_fqns, "
                    "but a rank's plan has no changed_fqns in storage_data."
                )
            all_fqn_sets.append(frozenset(plan_fqns))
        if len(set(all_fqn_sets)) != 1:
            raise ValueError(
                f"All ranks must agree on changed_fqns for incremental save, "
                f"but got {len(set(all_fqn_sets))} distinct sets."
            )

        self._load_baseline_metadata()

        filtered_plans = []
        for plan in plans:
            changed_items = [item for item in plan.items if item.index.fqn in self._changed_fqns]
            filtered_plans.append(SavePlan(
                items=changed_items,
                storage_data=plan.storage_data,
                planner_data=plan.planner_data,
            ))
        return filtered_plans

    def _load_baseline_metadata(self) -> None:
        """Load and migrate baseline checkpoint metadata.

        Raises:
            FileNotFoundError: If the baseline metadata file does not exist.
        """
        if self._baseline_metadata is not None:
            return
        baseline_reader = FileSystemReader(self._incremental_from)
        if self.use_collectives:
            self._baseline_metadata = migrate_metadata(baseline_reader.load_metadata())
        else:
            self._baseline_metadata = migrate_metadata(baseline_reader.load_metadata(rank=self.rank))

    @staticmethod
    def _validate_unchanged_tensor(
        fqn: str,
        current_md: TensorStorageMetadata,
        baseline_md: TensorStorageMetadata,
    ) -> None:
        """Verify that an unchanged FQN has compatible tensor metadata.

        Args:
            fqn: The FQN being validated.
            current_md: Tensor metadata from the current save plan.
            baseline_md: Tensor metadata from the baseline checkpoint.

        Raises:
            ValueError: If dtype, size, chunk offsets/sizes or chunk order differ.
        """
        if current_md.properties.dtype != baseline_md.properties.dtype:
            raise ValueError(
                f"Unchanged FQN {fqn!r} has dtype mismatch: current "
                f"{current_md.properties.dtype!r} vs baseline "
                f"{baseline_md.properties.dtype!r}. Mark this FQN as changed."
            )
        if current_md.size != baseline_md.size:
            raise ValueError(
                f"Unchanged FQN {fqn!r} has size mismatch: current "
                f"{current_md.size} vs baseline {baseline_md.size}. "
                f"Mark this FQN as changed."
            )
        if len(current_md.chunks) != len(baseline_md.chunks):
            raise ValueError(
                f"Unchanged FQN {fqn!r} has different chunk count: current "
                f"{len(current_md.chunks)} vs baseline {len(baseline_md.chunks)}. "
                f"Mark this FQN as changed."
            )
        for i, (cur_chunk, base_chunk) in enumerate(
            zip(current_md.chunks, baseline_md.chunks)
        ):
            if cur_chunk.offsets != base_chunk.offsets or cur_chunk.sizes != base_chunk.sizes:
                raise ValueError(
                    f"Unchanged FQN {fqn!r} chunk {i} differs: current "
                    f"({cur_chunk.offsets}, {cur_chunk.sizes}) vs baseline "
                    f"({base_chunk.offsets}, {base_chunk.sizes}). "
                    f"Mark this FQN as changed."
                )

    def _relocate_baseline_path(self, baseline_relative_path: str) -> str:
        """Relocate a baseline ``StorageInfo.relative_path`` to the current checkpoint dir.

        If the baseline is itself an incremental checkpoint, its relative paths
        are already relative to the baseline dir.  We resolve them against the
        baseline dir first, then compute a new relative path from the current
        checkpoint dir.

        Args:
            baseline_relative_path: The relative path stored in baseline metadata.

        Returns:
            str: A relative path from the current checkpoint dir to the same file.
        """
        baseline_abs = (self._incremental_from / baseline_relative_path).resolve()
        current_abs = self.checkpoint_dir.resolve()
        return os.path.relpath(str(baseline_abs), str(current_abs))


    def _serialize_bytes_item(self, item: WriteItem, planner: SavePlanner) -> bytes:
        """Serialize a BYTE_IO item payload while preserving current behavior."""
        data = planner.get_data(item)
        if isinstance(data, bytes):
            return data
        return pickle.dumps(data)


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

    def _collect_tensors(self, plan: SavePlan, planner: SavePlanner) -> dict[str, Any]:
        """
        Collect tensor data from planner runtime lookup.

        Args:
            plan (SavePlan): Save plan containing WriteItems.
            planner (SavePlanner): Save planner.

        Returns:
            dict[str, Any]: Dictionary mapping FQN to tensor data.

        Raises:
            RuntimeError: If tensor data cannot be resolved for an item.
        """
        tensor_dict: dict[str, Any] = {}
        for item in plan.items:
            if item.type.value == "tensor" and item.tensor_data:
                tensor = planner.get_data(item)
                if tensor is None:
                    raise RuntimeError(
                        f"Tensor data could not be resolved for index {item.index}. "
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

        # Write all BYTE_IO items into one file per rank
        results.extend(self._write_bytes_items(plan, planner))

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

        For incremental saves, this method:
        1. Validates that unchanged FQNs have compatible tensor metadata.
        2. Merges baseline storage_data (with relocated paths) for unchanged items.
        3. Overlays current write results for changed items.
        4. Verifies that the merged storage_data covers all expected indices.

        Args:
            metadata (Metadata): Checkpoint metadata to update.
            results (list[list[WriteResult]]): Write results from all ranks (or single rank when use_collectives=False).

        Raises:
            ValueError: If the merged storage_data is incomplete or if an unchanged
                FQN has incompatible tensor metadata.
        """
        should_save = not self.use_collectives or (self.use_collectives and self.is_coordinator)
        if not should_save:
            return

        storage_md: dict[MetadataIndex, StorageInfo] = {}

        if self._baseline_metadata is not None and self._changed_fqns is not None:
            self._merge_incremental_storage(metadata, results, storage_md)
        else:
            for wr_list in results:
                for wr in wr_list:
                    storage_md[wr.index] = wr.storage_data

        metadata.storage_data = storage_md
        metadata.version = CURRENT_CHECKPOINT_VERSION

        # Save metadata file
        if self.use_collectives:
            metadata_file = self.checkpoint_dir / METADATA_FILE_NAME
        else:
            metadata_file = self.checkpoint_dir / f"{self.rank}{METADATA_FILE_NAME}"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

    def _merge_incremental_storage(
        self,
        metadata: Metadata,
        results: list[list[WriteResult]],
        storage_md: dict[MetadataIndex, StorageInfo],
    ) -> None:
        """Merge baseline and current write results into *storage_md*.

        Args:
            metadata: Current checkpoint metadata (full logical view).
            results: Write results from this save (changed items only).
            storage_md: Destination dict to populate with merged storage_data.

        Raises:
            ValueError: If an unchanged FQN has incompatible metadata, or if
                the merged index is incomplete.
        """
        baseline_storage = self._baseline_metadata.storage_data or {}
        baseline_sdict = self._baseline_metadata.state_dict_metadata

        # 1. Validate unchanged tensor FQNs and inherit baseline storage_data
        for fqn, current_entry in metadata.state_dict_metadata.items():
            if fqn in self._changed_fqns:
                continue
            if fqn not in baseline_sdict:
                raise ValueError(
                    f"New FQN {fqn!r} is not marked as changed but does not "
                    f"exist in the baseline checkpoint."
                )
            baseline_entry = baseline_sdict[fqn]
            if isinstance(current_entry, TensorStorageMetadata):
                if not isinstance(baseline_entry, TensorStorageMetadata):
                    raise ValueError(
                        f"Unchanged FQN {fqn!r} changed type from "
                        f"{type(baseline_entry).__name__} to "
                        f"{type(current_entry).__name__}. Mark as changed."
                    )
                self._validate_unchanged_tensor(fqn, current_entry, baseline_entry)
            elif isinstance(current_entry, BytesStorageMetadata):
                if not isinstance(baseline_entry, BytesStorageMetadata):
                    raise ValueError(
                        f"Unchanged FQN {fqn!r} changed type from "
                        f"{type(baseline_entry).__name__} to "
                        f"{type(current_entry).__name__}. Mark as changed."
                    )
            # Inherit baseline storage indices with relocated paths
            for idx, info in baseline_storage.items():
                if idx.fqn == fqn:
                    relocated_path = self._relocate_baseline_path(info.relative_path)
                    storage_md[idx] = StorageInfo(
                        relative_path=relocated_path,
                        offset=info.offset,
                        length=info.length,
                    )

        # 2. Overlay current write results for changed FQNs
        for wr_list in results:
            for wr in wr_list:
                storage_md[wr.index] = wr.storage_data

        # 3. Verify completeness: every expected MetadataIndex must exist
        expected_indices = set()
        for fqn, entry in metadata.state_dict_metadata.items():
            if isinstance(entry, TensorStorageMetadata):
                for i, chunk in enumerate(entry.chunks):
                    expected_indices.add(MetadataIndex(fqn=fqn, offset=chunk.offsets, index=i))
            elif isinstance(entry, BytesStorageMetadata):
                expected_indices.add(MetadataIndex(fqn=fqn))

        missing = expected_indices - set(storage_md.keys())
        if missing:
            missing_fqns = sorted({idx.fqn for idx in missing})
            raise ValueError(
                f"Incremental checkpoint metadata is incomplete; missing "
                f"storage indices for FQNs: {missing_fqns}. "
                f"Mark missing FQNs as changed or ensure baseline covers them."
            )


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

    if platform.platform_type == PlatformType.PYTORCH:
        with safe_open(path, framework="pt", device="cpu") as tensor_file:
            for req in reqs:
                fqn = req.storage_index.fqn
                if fqn not in tensor_file.keys():
                    raise KeyError(f"Key {fqn} not found in checkpoint file {path}")
                tensor_slices = tuple(
                    slice(int(off), int(off) + int(length))
                    for off, length in zip(req.storage_offsets, req.lengths)
                )
                if tensor_slices:
                    tensor = tensor_file.get_slice(fqn)[tensor_slices]
                else:
                    tensor = narrow_tensor_by_index(
                        tensor_file.get_tensor(fqn),
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
        return

    param_dict = platform.load_checkpoint(path)
    for req in reqs:
        fqn = req.storage_index.fqn
        if fqn not in param_dict:
            raise KeyError(f"Key {fqn} not found in checkpoint file {path}")
        full_tensor = param_dict[fqn]
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
            metadata_file = self.checkpoint_dir / f"{rank}{METADATA_FILE_NAME}"
        else:
            metadata_file = self.checkpoint_dir / METADATA_FILE_NAME

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
        self.rank = kwargs.get("rank") if "rank" in kwargs else get_platform().get_rank()

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
                # BYTE_IO: one bytes file per rank with per-item offsets.
                _load_bytes_file(path, reqs, planner, self.storage_data)
            else:
                # TENSOR: one safetensors file per rank
                _load_tensor_file(path, reqs, planner)
