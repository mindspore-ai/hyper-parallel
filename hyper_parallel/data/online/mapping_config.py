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
"""Normalize internal Online Mapping build policy independently of callback ownership."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

from hyper_parallel.data.constants import ONLINE_SPLIT_COUNT


@dataclass
class OnlineMappingConfig:
    """Carry explicit reader, filtering, sampling and cache policy into Mapping construction.

    Args:
        source_options: Existing YAML data_config, including file readers, splits and weights.
        sample_filter: Runtime validation callback; never serialized into a cache fingerprint.
        filter_identifiers: Serializable identity of that callback's validation rules.
        use_filtered_cache: Explicitly select persistent valid-record and sampling indexes.
        mode: Abort on invalid records in production, or filter them in debug.
        train_valid_test_num_samples: Requested record accesses before packing, or natural split lengths.

    Example:
        Three valid records, targets=[5, 0, 0] -> mid shuffles complete epochs
        until 5 accesses are available. Enabling a cache requires a stable filter
        identity and path_to_cache; disabling it does not inspect callback attributes.
    """

    source_options: Mapping[str, Any]
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None
    filter_identifiers: Mapping[str, Any] | None = None
    use_filtered_cache: bool = False
    mode: Literal["production", "debug"] = "production"
    train_valid_test_num_samples: list[int] | tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        """Snapshot user options and validate policy before any record scan or file publication."""
        if not isinstance(self.source_options, Mapping):
            raise ValueError("Online Mapping source_options must be a mapping")

        self.source_options = deepcopy(dict(self.source_options))
        if self.mode not in ("production", "debug"):
            raise ValueError(f"Unsupported dataset mode: {self.mode}")

        if not isinstance(self.use_filtered_cache, bool):
            raise ValueError("Online Mapping use_filtered_cache must be a boolean")

        if self.sample_filter is not None and not callable(self.sample_filter):
            raise ValueError("Online Mapping sample_filter must be callable")

        sizes = self.train_valid_test_num_samples
        if sizes is not None:
            if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence) or len(sizes) != ONLINE_SPLIT_COUNT:
                raise ValueError("Online Mapping sample targets must contain train, validation, and test sizes")

            if any(isinstance(size, bool) or not isinstance(size, Integral) for size in sizes):
                raise ValueError("Online Mapping sample targets must be integers")

            if any(size < 0 for size in sizes):
                raise ValueError("Online Mapping sample targets must be non-negative")

            self.train_valid_test_num_samples = tuple(int(size) for size in sizes)
        if self.filter_identifiers is not None:
            if not isinstance(self.filter_identifiers, Mapping):
                raise ValueError("Mapping filter_identifiers must be a mapping")

            try:
                description = json.dumps(dict(self.filter_identifiers), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("Mapping filter_identifiers must be JSON-serializable") from error

            self.filter_identifiers = json.loads(description)
        if self.use_filtered_cache:
            if not isinstance(self.path_to_cache, str) or not self.path_to_cache.strip():
                raise ValueError("Filtered Mapping sources require data_config.path_to_cache")

            if self.sample_filter is not None and not self.filter_identifiers:
                raise ValueError("Filtered Mapping cache requires explicit filter_identifiers")

    @property
    def random_seed(self) -> int:
        """Return the source-local shuffle seed from the existing source options."""
        seed = int(self.source_options.get("random_seed", 42))
        return seed

    @property
    def path_to_cache(self) -> str | None:
        """Return the shared index-cache root without deriving policy from a callback."""
        cache_root = self.source_options.get("path_to_cache")
        return cache_root

    @property
    def reading_options(self) -> dict[str, Any]:
        """Select only file-reading options for low-level valid-index fingerprints.

        Example:
            Changing split, seed or blend weights reuses the same valid-record index;
            changing a reader option creates a different low-level cache.
        """
        reading_options = dict(self.source_options)
        for option in (
            "split", "random_seed", "path_to_cache", "sources", "shuffle", "shuffle_buffer_size",
            "blend_strategy", "stopping_strategy", "split_by_data_parallel",
        ):
            reading_options.pop(option, None)
        return reading_options


def build_online_mapping_config(
    data_config: Mapping[str, Any],
    *,
    transform: Any,
    training_config: Any = None,
    mode: Literal["production", "debug"] = "production",
    train_valid_test_num_samples: list[int] | tuple[int, int, int] | None = None,
) -> OnlineMappingConfig:
    """Assemble Mapping policy at the Omni entry, where the transform is explicitly available.

    Args:
        data_config: Existing YAML source options; the caller's mapping is not modified.
        transform: Omni transform providing is_valid_sample and unique_identifiers.
        training_config: Training plan supplying the seed, without inferring packed batch counts.
        mode: Invalid-record handling policy.
        train_valid_test_num_samples: Optional raw-record access targets for the three splits.

    Returns:
        Normalized internal configuration passed to the Mapping source builder.

    Example:
        YAML path_to_cache + a DeepSeek transform + training seed 43
        -> explicit filter callback, its cache identity, enabled index cache and seed 43.
        A transform with no cache identity keeps the legacy loading policy.
    """
    source_options = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    source_options["random_seed"] = 42 if training_seed is None else int(training_seed)
    filter_identifiers = transform.unique_identifiers
    config = OnlineMappingConfig(
        source_options=source_options,
        sample_filter=transform.is_valid_sample,
        filter_identifiers=filter_identifiers,
        use_filtered_cache=filter_identifiers is not None,
        mode=mode,
        train_valid_test_num_samples=train_valid_test_num_samples,
    )
    return config
