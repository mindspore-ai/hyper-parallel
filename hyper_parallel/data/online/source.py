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
"""Shared Online source loading and multi-source construction policy."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from hyper_parallel.data.constants import ONLINE_FILE_FORMATS, ONLINE_SPLIT_NAMES, ONLINE_STOPPING_STRATEGIES

OnlinePathSpec: TypeAlias = str | Sequence[str]
OnlineDataPath: TypeAlias = OnlinePathSpec | Mapping[str, OnlinePathSpec]


class _OnlineSourceBuilder:
    """Share source parsing and interleave policy across Online adapters."""

    def __init__(self, data_path: OnlineDataPath | None, data_config: Mapping[str, Any]) -> None:
        """Store the source and composition configuration."""
        self.data_path = data_path
        self.data_config = data_config

    @property
    def _random_seed(self) -> int:
        """Return the common source shuffle and blend seed."""
        return int(self.data_config.get("random_seed", 42))

    @property
    def _stopping_strategy(self) -> str:
        """Return the multi-source exhaustion policy."""
        stopping_strategy = str(self.data_config.get("stopping_strategy", "first_exhausted"))
        if stopping_strategy not in ONLINE_STOPPING_STRATEGIES:
            raise ValueError(
                f"Unsupported Online stopping_strategy {stopping_strategy!r}; "
                f"expected one of {ONLINE_STOPPING_STRATEGIES!r}"
            )
        return stopping_strategy

    def _create_source_loaders(self) -> tuple[list[_OnlineLeafSourceLoader], list[float]]:
        """Create one leaf source loader per configured source.

        Example:
            ``data_path="/data/train.jsonl"`` produces one source with weight
            ``1.0``.

            The following configuration produces two sources. Both inherit
            ``split="98, 1, 1"``, while the second source overrides it::

                {
                    "split": "98, 1, 1",
                    "sources": [
                        {"data_path": "/data/a.parquet", "weight": 0.7},
                        {
                            "data_path": "/data/b.parquet",
                            "weight": 0.3,
                            "split": "90, 5, 5",
                        },
                    ],
                }

        Returns:
            Leaf source loaders and their corresponding unnormalized weights.
        """
        sources = self.data_config.get("sources")
        if sources is None:
            if self.data_path is None:
                raise ValueError("Online Dataset requires data_path or data_config.sources")
            source_loader = _OnlineLeafSourceLoader(self.data_path, self.data_config)
            return [source_loader], [1.0]

        if self.data_path is not None:
            raise ValueError("Online Dataset cannot combine data_path with data_config.sources")
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence) or not sources:
            raise ValueError("Online data_config.sources must be a non-empty sequence")

        common_config = dict(self.data_config)
        common_config.pop("sources")
        source_loaders = []
        source_weights = []
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("Each Online source must be a mapping with data_path and weight")
            try:
                source_path = source["data_path"]
                source_weight = float(source["weight"])
            except KeyError as error:
                raise ValueError("Each Online source must define data_path and weight") from error
            if source_weight <= 0:
                raise ValueError("Online source weight must be positive")

            source_config = dict(common_config)
            for option_name, option_value in source.items():
                if option_name not in ("data_path", "weight"):
                    source_config[option_name] = option_value
            source_loaders.append(_OnlineLeafSourceLoader(source_path, source_config))
            source_weights.append(source_weight)
        return source_loaders, source_weights

    def _interleave_sources(self, source_datasets: Sequence[Any], weights: Sequence[float]) -> Any:
        """Interleave Mapping or Iterable sources through the Hugging Face interface."""
        if len(source_datasets) == 1:
            return source_datasets[0]

        try:
            # datasets remains optional for Indexed-only installations.
            from datasets import interleave_datasets  # pylint: disable=C0415
        except ImportError as error:
            raise ImportError("Online Dataset requires the optional 'datasets' package") from error

        weight_sum = sum(weights)
        probabilities = []
        for weight in weights:
            probability = weight / weight_sum
            probabilities.append(probability)

        interleaved_dataset = interleave_datasets(
            list(source_datasets),
            probabilities=probabilities,
            seed=self._random_seed,
            stopping_strategy=self._stopping_strategy,
        )
        return interleaved_dataset


class _OnlineLeafSourceLoader:
    """Load one local or Hub source with ``datasets.load_dataset``."""

    def __init__(self, data_path: OnlineDataPath, data_config: Mapping[str, Any]) -> None:
        """Store one leaf source description."""
        self.data_path = data_path
        self.data_config = data_config

    def load(
        self,
        *,
        streaming: bool,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Load, restore configured splits, and filter one leaf source."""
        try:
            # datasets remains optional for Indexed-only installations.
            from datasets import load_dataset  # pylint: disable=C0415
        except ImportError as error:
            raise ImportError("Online Dataset requires the optional 'datasets' package") from error

        # Examples: "Salesforce/wikitext", "/data/*.parquet", {"train": "/data/train.jsonl"}.
        load_dataset_path, load_options, split_expressions = self._build_load_request(streaming=streaming)
        ratio_split_config = self.data_config.get("split")
        filter_before_ratio_split = ratio_split_config is not None and sample_filter is not None
        selected_split: str | list[str] = "train"
        if split_expressions is not None and not filter_before_ratio_split:
            selected_splits = []
            for split_expression in split_expressions:
                if split_expression is not None:
                    selected_splits.append(split_expression)
            selected_split = selected_splits

        loaded_dataset = load_dataset(
            load_dataset_path,
            **load_options,
            split=selected_split,
            streaming=streaming,
            cache_dir=self.data_config.get("cache_dir"),
        )

        if filter_before_ratio_split:
            filtered_dataset = loaded_dataset.filter(sample_filter)
            split_percentages = self._parse_split_percentages(ratio_split_config)
            filtered_splits = self._split_filtered_dataset(filtered_dataset, split_percentages)
            return filtered_splits

        dataset_result = loaded_dataset
        if split_expressions is not None:
            # Example: ("train", None, "test") keeps the empty valid slot.
            loaded_dataset_iterator = iter(loaded_dataset)
            restored_splits = []
            for split_expression in split_expressions:
                if split_expression is None:
                    restored_splits.append(None)
                    continue
                loaded_split = next(loaded_dataset_iterator)
                restored_splits.append(loaded_split)
            dataset_result = (restored_splits[0], restored_splits[1], restored_splits[2])

        if sample_filter is None:
            return dataset_result

        # Example: (train, None, test) filters available splits and preserves the empty valid slot.
        if isinstance(dataset_result, tuple):
            filtered_splits = []
            for split_dataset in dataset_result:
                if split_dataset is None:
                    filtered_splits.append(None)
                    continue
                filtered_split = split_dataset.filter(sample_filter)
                filtered_splits.append(filtered_split)
            filtered_dataset = (filtered_splits[0], filtered_splits[1], filtered_splits[2])
            return filtered_dataset

        filtered_dataset = dataset_result.filter(sample_filter)
        return filtered_dataset

    def _build_load_request(
        self,
        *,
        streaming: bool,
    ) -> tuple[str, dict[str, Any], tuple[str | None, str | None, str | None] | None]:
        """Build the path, keyword arguments, and split selections for ``load_dataset``."""
        split_config = self.data_config.get("split")
        split_expressions = None
        if split_config is not None and streaming:
            raise ValueError("Online ratio split does not support streaming")

        if split_config is not None and isinstance(self.data_path, Mapping):
            raise ValueError("Online ratio split cannot be combined with pre-split paths")

        if split_config is not None:
            split_expressions = self._parse_split_ratios(split_config)

        load_options: dict[str, Any] = {}
        # Example: {"train": "/data/train.jsonl", "valid": "/data/valid.jsonl"}.
        if isinstance(self.data_path, Mapping):
            data_files, load_dataset_path, split_expressions = self._resolve_split_data_files(self.data_path)
            load_options["data_files"] = data_files
            resolved_request = (load_dataset_path, load_options, split_expressions)
            return resolved_request

        is_local_path = not isinstance(self.data_path, str)
        if isinstance(self.data_path, str):
            file_extension = os.path.splitext(self.data_path)[1].lower()
            is_local_path = os.path.exists(self.data_path) or file_extension in ONLINE_FILE_FORMATS

        # Examples: ["/data/a.jsonl", "/data/b.jsonl"], "/data/*.parquet".
        if is_local_path:
            data_files, load_dataset_path = self._collect_local_data_files(self.data_path)
            load_options["data_files"] = data_files
            resolved_request = (load_dataset_path, load_options, split_expressions)
            return resolved_request

        # Example: "Salesforce/wikitext" is a Hub dataset ID.
        load_dataset_path = self.data_path
        config_name = self.data_config.get("config_name")
        if config_name is not None:
            load_options["name"] = config_name

        resolved_request = (load_dataset_path, load_options, split_expressions)
        return resolved_request

    @staticmethod
    def _split_filtered_dataset(dataset: Any, percentages: Sequence[int]) -> tuple[Any, Any, Any]:
        """Split one filtered Mapping Dataset using deterministic contiguous ranges."""
        dataset_length = len(dataset)
        train_end = int(round(percentages[0] * dataset_length / 100.0))
        valid_end = int(round((percentages[0] + percentages[1]) * dataset_length / 100.0))
        split_ranges = ((0, train_end), (train_end, valid_end), (valid_end, dataset_length))

        split_datasets = []
        for percentage, (split_start, split_end) in zip(percentages, split_ranges):
            split_dataset = None
            if percentage > 0:
                split_dataset = dataset.select(range(split_start, split_end))
            split_datasets.append(split_dataset)
        dataset_splits = (split_datasets[0], split_datasets[1], split_datasets[2])
        return dataset_splits

    def _resolve_split_data_files(
        self,
        data_path: Mapping[str, OnlinePathSpec],
    ) -> tuple[dict[str, list[str]], str, tuple[str | None, str | None, str | None]]:
        """Resolve pre-split local paths into ``load_dataset`` arguments."""
        if not data_path:
            raise ValueError("Online split data_path mapping must not be empty")

        split_data_files: dict[str, list[str]] = {}
        loader_formats = set()
        for split_name, split_path in data_path.items():
            if split_name not in ONLINE_SPLIT_NAMES:
                raise ValueError(
                    f"Unsupported Online data_path split {split_name!r}; "
                    f"expected one of {ONLINE_SPLIT_NAMES!r}"
                )

            data_files, loader_format = self._collect_local_data_files(split_path)
            split_data_files[split_name] = data_files
            loader_formats.add(loader_format)

        if len(loader_formats) != 1:
            raise ValueError("Online train/valid/test paths must use one common file format")

        train_split = None
        if "train" in split_data_files:
            train_split = "train"

        valid_split = None
        if "valid" in split_data_files:
            valid_split = "valid"

        test_split = None
        if "test" in split_data_files:
            test_split = "test"

        selected_splits = (train_split, valid_split, test_split)
        loader_format = loader_formats.pop()
        return split_data_files, loader_format, selected_splits

    @staticmethod
    def _collect_local_data_files(data_path: OnlinePathSpec) -> tuple[list[str], str]:
        """Collect local files and select their ``load_dataset`` format."""
        if isinstance(data_path, str):
            configured_paths = [data_path]
        else:
            configured_paths = list(data_path)

        data_files = []
        for configured_path in configured_paths:
            if not os.path.isdir(configured_path):
                data_files.append(configured_path)
                continue

            for filename in sorted(os.listdir(configured_path)):
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in ONLINE_FILE_FORMATS:
                    data_files.append(os.path.join(configured_path, filename))

        if not data_files:
            raise ValueError("Online files must use one format: JSON/JSONL/Parquet/CSV/Arrow")

        first_extension = os.path.splitext(data_files[0])[1].lower()
        loader_format = ONLINE_FILE_FORMATS.get(first_extension)
        for data_file in data_files[1:]:
            file_extension = os.path.splitext(data_file)[1].lower()
            if ONLINE_FILE_FORMATS.get(file_extension) != loader_format:
                raise ValueError("Online files must use one format: JSON/JSONL/Parquet/CSV/Arrow")

        if loader_format is None:
            raise ValueError("Online files must use one format: JSON/JSONL/Parquet/CSV/Arrow")

        return data_files, loader_format

    @staticmethod
    def _parse_split_percentages(split: Any) -> tuple[int, int, int]:
        """Parse and validate train, validation, and test percentages."""
        try:
            percentages = []
            for split_part in split.split(","):
                percentages.append(int(split_part.strip()))
        except (AttributeError, ValueError) as error:
            raise ValueError("Online split must contain three integers such as '98, 1, 1'") from error

        if len(percentages) != 3:
            raise ValueError("Online split must contain three integers such as '98, 1, 1'")

        if sum(percentages) != 100:
            raise ValueError("Online split percentages must sum to 100")
        if any(percentage < 0 for percentage in percentages):
            raise ValueError("Online split percentages must be non-negative")

        split_percentages = (percentages[0], percentages[1], percentages[2])
        return split_percentages

    @classmethod
    def _parse_split_ratios(cls, split: Any) -> tuple[str | None, str | None, str | None]:
        """Translate train/valid/test percentages into ``load_dataset`` split expressions."""
        train_percentage, valid_percentage, test_percentage = cls._parse_split_percentages(split)

        train_end = train_percentage
        valid_end = train_percentage + valid_percentage
        split_ranges = ((0, train_end), (train_end, valid_end), (valid_end, 100))

        split_expressions: list[str | None] = []
        for split_begin, split_end in split_ranges:
            if split_begin == split_end:
                split_expression = None
            elif split_begin == 0 and split_end == 100:
                split_expression = "train"
            elif split_begin == 0:
                split_expression = f"train[:{split_end}%]"
            elif split_end == 100:
                split_expression = f"train[{split_begin}%:]"
            else:
                split_expression = f"train[{split_begin}%:{split_end}%]"
            split_expressions.append(split_expression)

        online_split_expressions = (split_expressions[0], split_expressions[1], split_expressions[2])
        return online_split_expressions


__all__ = [
    "OnlineDataPath",
    "OnlinePathSpec",
]
