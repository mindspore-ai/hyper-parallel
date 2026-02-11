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
"""Storage interfaces for checkpoint save and load."""
import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from hyper_parallel.core.checkpoint.metadata import Metadata, MetadataIndex
from hyper_parallel.core.checkpoint.planner import LoadPlan, LoadPlanner, SavePlan, SavePlanner

_metadata_file_name = ".metadata"

@dataclass
class StorageInfo:
    """
    Storage information for a single logical item.

    Torch-aligned: matches torch.distributed.checkpoint.filesystem._StorageInfo.

    Attributes:
        relative_path: Path relative to checkpoint root.
        offset: Byte offset within the file.
        length: Byte length of the data (best-effort, may be -1 for tensors).
    """
    relative_path: str
    offset: int
    length: int


@dataclass
class WriteResult:
    """
    Result of writing a single logical item.

    Torch-aligned: contains the metadata index and storage information.
    """
    index: MetadataIndex
    storage_data: StorageInfo


class StorageWriter(abc.ABC):
    """
    Abstract base class for storage writers.

    Defines the interface for writing checkpoint data to storage backends.
    """

    @abc.abstractmethod
    def initialize_writer(self, checkpoint_id: Optional[Union[Path, str]] = None) -> None:
        """
        Initialize storage writer with optional new checkpoint directory.

        Args:
            checkpoint_id (Optional[Union[Path, str]]): The ID/path of the checkpoint directory.
                If None, uses the previously configured checkpoint directory.
        """

    @abc.abstractmethod
    def configure_writer(self, is_coordinator: bool, **kwargs) -> None:
        """
        Configure storage writer with coordinator and rank information.

        Args:
            is_coordinator (bool): Whether this rank is the coordinator rank.
            **kwargs: Additional keyword arguments (e.g., rank, use_collectives).
        """

    @abc.abstractmethod
    def optimize_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Optimize local plan for storage-specific optimizations.

        Args:
            plan (SavePlan): The local save plan to optimize.

        Returns:
            SavePlan: The optimized local save plan.
        """

    @abc.abstractmethod
    def optimize_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Optimize global plan from all local plans.

        Args:
            plans (list[SavePlan]): List of local save plans from all ranks.

        Returns:
            list[SavePlan]: List of optimized global save plans.
        """

    @abc.abstractmethod
    def execute_write(self, plan: SavePlan, planner: SavePlanner) -> list[WriteResult]:
        """
        Execute write operation to storage and return write results.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): The save planner instance for accessing tensor data.

        Returns:
            list[WriteResult]: List of write results containing storage information for each written item.
        """

    @abc.abstractmethod
    def finalize_checkpoint(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """
        Finalize checkpoint writing and complete metadata.

        Args:
            metadata (Metadata): The checkpoint metadata to finalize.
            results (list[list[WriteResult]]): List of write results from all ranks,
                where each inner list contains WriteResults from one rank.
        """


class StorageReader(abc.ABC):
    """
    Abstract base class for storage readers.

    Defines the interface for reading checkpoint data from storage backends.
    """

    @abc.abstractmethod
    def initialize_reader(self, checkpoint_id: Optional[Union[Path, str]] = None) -> None:
        """
        Initialize storage reader with optional new checkpoint directory.

        Args:
            checkpoint_id (Optional[Union[Path, str]]): The ID/path of the checkpoint directory.
                If None, uses the previously configured checkpoint directory.
        """

    @abc.abstractmethod
    def load_metadata(self, **kwargs) -> Metadata:
        """
        Load checkpoint metadata from storage.

        Args:
            **kwargs: Additional keyword arguments (e.g., rank for rank-local metadata).

        Returns:
            Metadata: The loaded checkpoint metadata.
        """

    @abc.abstractmethod
    def configure_reader(self, metadata: Metadata, is_coordinator: bool, **kwargs) -> None:
        """
        Configure storage reader with metadata and coordinator information.

        Args:
            metadata (Metadata): The checkpoint metadata.
            is_coordinator (bool): Whether this rank is the coordinator rank.
            **kwargs: Additional keyword arguments (e.g., rank, use_collectives).
        """

    @abc.abstractmethod
    def optimize_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Optimize local plan for storage-specific optimizations.

        Args:
            plan (LoadPlan): The local load plan to optimize.

        Returns:
            LoadPlan: The optimized local load plan.
        """

    @abc.abstractmethod
    def optimize_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Optimize global plan from all local plans.

        Args:
            plans (list[LoadPlan]): List of local load plans from all ranks.

        Returns:
            list[LoadPlan]: List of optimized global load plans.
        """

    @abc.abstractmethod
    def execute_read(self, plan: LoadPlan, planner: LoadPlanner) -> None:
        """
        Execute read operation from storage according to the load plan.

        Args:
            plan (LoadPlan): The load plan to execute.
            planner (LoadPlanner): The load planner instance for applying loaded data.
        """
