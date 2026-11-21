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
"""Normalize and validate offline Dataset build options."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

from hyper_parallel.data.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration shared by blended indexed datasets."""

    random_seed: int
    sequence_length: int
    blend: list[str] | None = None
    blend_per_split: list[list[str] | None] | None = None
    split: str | None = None
    split_matrix: list[tuple[float, float] | None] | None = field(init=False, default=None)
    path_to_cache: str | None = None
    mmap_bin_files: bool = True
    mock: bool = False
    tokenizer: Any = None

    def __post_init__(self) -> None:
        """Validate the configured data distribution and derive split ranges."""
        if self.mock:
            return

        if self.blend_per_split is not None and any(self.blend_per_split):
            if self.blend is not None:
                raise ValueError("blend and blend_per_split are incompatible")
            if self.split is not None:
                raise ValueError("split and blend_per_split are incompatible")
            if len(self.blend_per_split) != 3:
                raise ValueError("blend_per_split must contain train, validation, and test blends")
            return

        if self.blend is None:
            raise ValueError("one of either blend or blend_per_split must be provided")
        if self.split is None:
            raise ValueError("both blend and split must be provided")

        # Parse at most three train/validation/test ratios, padding omitted
        # trailing splits with zero before normalization.
        split_values = [float(value) for value in re.findall(r"[.0-9]+", self.split)]
        split_values.extend([0.0] * (3 - len(split_values)))
        if len(split_values) != 3:
            raise ValueError(f"split length {len(split_values)} does not match expected 3")

        if not all(value >= 0.0 for value in split_values):
            raise ValueError("all split values must be non-negative")

        normalized_values = (np.asarray(split_values, dtype=np.float64) / np.sum(split_values)).tolist()
        canonical_values = []
        for value in normalized_values:
            rounded_value = round(value, 12)
            precision = 2 if rounded_value == round(rounded_value, 2) else 12
            canonical_values.append(f"{rounded_value:.{precision}f}")
        self.split = ", ".join(canonical_values)

        # Convert normalized ratios into non-overlapping ranges. A zero-sized
        # split is represented by None.
        boundaries = [0.0]
        for value in normalized_values:
            boundaries.append(boundaries[-1] + value)
        self.split_matrix = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            self.split_matrix.append(None if end <= begin else (begin, end))


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration required to build indexed GPT Dataset splits.

    ``inter_document_masking`` preserves packed GPT document boundaries from
    Indexed sequence lengths, including corpora without EOD tokens.
    """

    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    create_ltor_fields: bool = False
    inter_document_masking: bool = False
    drop_last_partial_validation_sequence: bool = True
    add_extra_token_to_sequence: bool = True
    allow_ambiguous_pad_tokens: bool = False
    dataset_margin: float = 1.005
    skip_data_check: bool = False
    reuse_idx: bool = False
    data_lazy_load: bool = False

    low_level_dataset: type | None = None
    simple_blend: str | None = None
    dataset_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the blended Dataset fields followed by GPT-specific fields."""
        super().__post_init__()

        if self.low_level_dataset is None:
            raise ValueError("low_level_dataset must be explicitly configured")
        if self.simple_blend is None:
            raise ValueError("simple_blend must be explicitly configured")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.tokenizer is None and not self.mock:
            raise ValueError("Attribute 'tokenizer' must not be None")


def build_gpt_dataset_config(
    data_paths: Sequence[str],
    data_config: Mapping[str, Any],
) -> GPTDatasetConfig:
    """Build a GPT Dataset config from provider fields.

    Args:
        data_paths: Weighted source paths discovered by the offline Dataset builder.
        data_config: Indexed pretraining configuration supplied by the caller.

    Returns:
        Configuration required by GPT Dataset and split construction.

    Raises:
        ValueError: If a required build field is missing.
    """
    # The source argument parser owns defaults and validation. This stage only
    # renames fields where the Dataset implementation expects another name.
    blend_per_split = [
        data_config.get("train_data_path"),
        data_config.get("valid_data_path"),
        data_config.get("test_data_path"),
    ]
    has_independent_blends = any(blend_per_split)
    try:
        mock_data = bool(data_config.get("mock_data", False))
        tokenizer = data_config.get("tokenizer")

        if tokenizer is None and not mock_data:
            raise ValueError("tokenizer is required when mock_data is false")

        config = GPTDatasetConfig(
            random_seed=data_config["random_seed"],
            sequence_length=data_config["max_seq_len"],
            tokenizer=tokenizer,
            mock=mock_data,
            blend=None if has_independent_blends else list(data_paths),
            blend_per_split=blend_per_split if has_independent_blends else None,
            split=None if has_independent_blends else data_config.get("split", "1, 0, 0"),
            path_to_cache=data_config.get("data_cache_path"),
            mmap_bin_files=data_config.get("mmap_bin_files", True),
            reuse_idx=data_config.get("reuse_idx", False),
            data_lazy_load=data_config.get("data_lazy_load", False),
            reset_position_ids=data_config.get("reset_position_ids", False),
            reset_attention_mask=data_config.get("reset_attention_mask", False),
            eod_mask_loss=data_config.get("eod_mask_loss", False),
            create_ltor_fields=data_config.get("create_ltor_fields_in_dataloader", False),
            inter_document_masking=data_config.get("inter_document_masking", False),
            drop_last_partial_validation_sequence=data_config.get("drop_last_partial_validation_sequence", True),
            add_extra_token_to_sequence=data_config.get("add_extra_token_to_sequence", True),
            allow_ambiguous_pad_tokens=data_config.get("allow_ambiguous_pad_tokens", False),
            dataset_margin=data_config.get("dataset_margin", 1.005),
            skip_data_check=data_config.get("skip_data_check", False),
            low_level_dataset=resolve_dataset_type(data_config["low_level_dataset"]),
            dataset_options=(
                {key: value for key, value in data_config["low_level_dataset"].items() if key != "_target_"}
                if isinstance(data_config["low_level_dataset"], Mapping) else {}
            ),
            simple_blend=data_config["simple_blend"],
        )
    except KeyError as error:
        raise ValueError(f"Missing indexed Dataset build option: {error.args[0]}") from error
    source_paths = list(data_paths) if len(data_paths) <= 1 else list(data_paths[1::2])
    logger.debug("Resolved indexed sources=%d, first four=%s", len(source_paths), source_paths[:4])

    return config


def resolve_dataset_type(spec: Mapping[str, Any] | type) -> type:
    """Resolve an explicitly selected class without constructing a Dataset."""
    if isinstance(spec, type):
        return spec
    if not isinstance(spec, Mapping) or "_target_" not in spec:
        raise ValueError("low_level_dataset requires an explicit Dataset _target_")
    module, _, name = spec["_target_"].rpartition(".")
    dataset_type = getattr(import_module(module), name)
    if not isinstance(dataset_type, type):
        raise ValueError("low_level_dataset._target_ must identify a Dataset class")
    return dataset_type
