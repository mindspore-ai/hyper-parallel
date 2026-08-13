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
"""Build pretraining LLM datasets stored in ``.idx/.bin`` files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, TypeAlias

from hyper_models.components.datasets.llm.indexed_data_config import (
    GPTDatasetConfig,
    build_gpt_dataset_config,
    resolve_data_paths,
)
from hyper_models.components.datasets.llm.indexed_pretrain_dataset import (
    GPTDataset,
    GPTFromMRDataset,
    MockGPTDataset,
)
from hyper_models.components.datasets.llm.indexed_split_builder import build_dataset_splits
from hyper_models.components.datasets.parallel import DatasetParallelContext

DataPath: TypeAlias = str | Sequence[str]
DatasetSplits: TypeAlias = tuple[Any | None, Any | None, Any | None]
SplitBuilder: TypeAlias = Callable[..., DatasetSplits]
InstructionSplitBuilder: TypeAlias = Callable[..., Sequence[Any | None]]


class IndexedPretrainDatasetBuilder:
    """Build GPT pretraining datasets from indexed token files.

    Its ``build`` method keeps the indexed provider flow visible while each
    private step can be implemented and reviewed independently.
    """

    def __init__(
            self,
            *,
            data_path: DataPath,
            data_config: Mapping[str, Any],
            train_valid_test_num_samples: Sequence[int],
            parallel_context: DatasetParallelContext | None = None,
    ) -> None:
        """Store the inputs required by the indexed pretraining build.

        Args:
            data_path: Indexed prefix, directory, or weighted paths. May be
                omitted when mock data is enabled.
            data_config: Indexed pretraining build options.
            train_valid_test_num_samples: Trainer-derived target sizes.
            parallel_context: Distributed Dataset construction callbacks.
        """
        self.data_path = data_path
        self.data_config = data_config
        self.train_valid_test_num_samples = train_valid_test_num_samples
        self.parallel_context = parallel_context or DatasetParallelContext(
            data_index_cache=bool(data_config.get("data_index_cache", False))
        )

    def build(self) -> DatasetSplits:
        """Build train, validation, and test indexed datasets.

        Returns:
            Train, validation, and test datasets.
        """
        # Discover indexed files and add their default blend weights.
        # Mock Datasets generate their samples and do not need filesystem
        # discovery. Real indexed Datasets still require an explicit path.
        if bool(self.data_config.get("mock_data", False)):
            data_paths = []
        else:
            if self.data_path is None:
                raise ValueError("data_path is required when mock_data is false")
            data_paths = self._resolve_data_paths()

        # Normalize Dataset options. Sample counts remain separate and are
        # passed directly to the split builder.
        config = self._build_gpt_dataset_config(data_paths)

        # Follow the Provider boundary: instruction and pretraining datasets
        # share one public entry but own separate construction pipelines.
        dataset_splits = self._build_dataset_splits(config)
        return dataset_splits

    def _build_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Select the instruction or indexed-pretraining Dataset pipeline."""
        if bool(self.data_config.get("is_instruction_dataset", False)):
            dataset_splits = self._build_instruction_dataset_splits(config)
            return dataset_splits

        dataset_splits = self._build_pretrain_dataset_splits(config)
        return dataset_splits

    def _build_pretrain_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Build GPT pretraining splits through the selected Dataset and blend types."""
        # Select GPTDataset / MockGPTDataset / GPTFromMRDataset.
        # logger.info(f'Iolan, config {config}')
        dataset_type = self._select_dataset_type(config)

        # Select the standard or simple blended Dataset builder.
        split_builder = self._select_split_builder(config)

        # Build train, validation, and test datasets.
        datasets = split_builder(
            dataset_type=dataset_type,
            train_valid_test_num_samples=self.train_valid_test_num_samples,
            parallel_context=self.parallel_context,
            config=config,
        )
        return datasets

    def _build_instruction_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Build packed instruction splits through the configured implementation.

        The concrete packed-dataset implementation remains outside this provider.

        Raises:
            NotImplementedError: If no instruction Dataset builder is configured.
            ValueError: If the configured builder does not return three splits.
        """
        instruction_builder = self.data_config.get("instruction_dataset_builder")
        if not callable(instruction_builder):
            raise NotImplementedError(
                "is_instruction_dataset=True requires an instruction_dataset_builder"
            )
        instruction_split_builder: InstructionSplitBuilder = instruction_builder

        data_prefix = config.blend if config.blend is not None else self.data_path
        datasets = instruction_split_builder(
            data_prefix=data_prefix,
            splits_string=config.split,
            train_valid_test_num_samples=self.train_valid_test_num_samples,
            seq_length=config.sequence_length,
            seed=config.random_seed,
            tokenizer=config.tokenizer,
        )
        if not isinstance(datasets, Sequence) or len(datasets) != 3:
            raise ValueError("instruction_dataset_builder must return train, validation, and test")
        dataset_splits = (datasets[0], datasets[1], datasets[2])
        return dataset_splits

    def _resolve_data_paths(self) -> list[str]:
        """Delegate indexed path discovery to the config build module."""
        if self.data_path is None:
            raise ValueError("data_path is required when mock_data is false")
        data_paths = resolve_data_paths(
            self.data_path,
            distributed_walk=bool(self.data_config["distributed_walk"]),
        )
        return data_paths

    def _build_gpt_dataset_config(self, data_paths: Sequence[str]) -> GPTDatasetConfig:
        """Delegate GPT Dataset option normalization."""
        config = build_gpt_dataset_config(data_paths, self.data_config)
        return config

    def _select_dataset_type(self, config: GPTDatasetConfig) -> type:
        """Select GPT, Mock GPT, or MR GPT Dataset."""
        if config.mock:
            return MockGPTDataset
        if config.is_dataset_from_mr:
            return GPTFromMRDataset
        return GPTDataset

    @staticmethod
    def _select_split_builder(config: GPTDatasetConfig) -> SplitBuilder:
        """
        Select the standard or simple train/valid/test builder.
            "no": BlendedDataset
            "inter/intra": SimpleBlendedDataset
        """
        if config.simple_blend == "no":
            split_builder = partial(build_dataset_splits, blend_mode="no")
            return split_builder

        if not config.is_dataset_from_mr:
            raise ValueError(
                "simple_blend values other than 'no' require is_dataset_from_mr=True"
            )
        if config.simple_blend not in {"inter", "intra"}:
            raise ValueError(
                "simple_blend must be one of 'no', 'inter', or 'intra'; "
                f"got {config.simple_blend!r}"
            )
        split_builder = partial(build_dataset_splits, blend_mode=config.simple_blend)
        return split_builder


def build_indexed_dataset(
        *,
        data_path: DataPath = None,
        data_config: Mapping[str, Any],
        train_valid_test_num_samples: Sequence[int],
        parallel_context: DatasetParallelContext | None = None,
) -> DatasetSplits:
    """Build indexed pretraining datasets through the dedicated builder.

    Args:
        data_path: Indexed prefix, directory, or weighted paths. May be omitted
            when mock data is enabled.
        data_config: Indexed pretraining build options.
        train_valid_test_num_samples: Trainer-derived target sizes.
        parallel_context: Distributed Dataset construction callbacks assembled
            by the Trainer.

    Returns:
        Train, validation, and test datasets.
    """
    builder = IndexedPretrainDatasetBuilder(
        data_path=data_path,
        data_config=data_config,
        train_valid_test_num_samples=train_valid_test_num_samples,
        parallel_context=parallel_context,
    )
    datasets = builder.build()
    return datasets
