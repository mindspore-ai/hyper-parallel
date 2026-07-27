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
"""Public LLM and Omni collate-function build stage."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from torch.utils.data import default_collate

Feature = Mapping[str, Any]
CollatedBatch = Mapping[str, Any]
InternalDataCollator = Callable[[Sequence[Feature]], CollatedBatch]
_DataCollatorT = TypeVar("_DataCollatorT", bound=InternalDataCollator)


def build_collate_fn(
        internal_data_collator: _DataCollatorT,
) -> _DataCollatorT:
    """Expose one model-specific micro-batch collator to DataLoader.

    Args:
        internal_data_collator: LLM or VLM collator that builds one model
            micro-batch from a sequence of samples.

    Returns:
        The validated model-specific micro-batch collator.

    Raises:
        ValueError: If the model collator is not callable.
    """
    if not callable(internal_data_collator):
        raise ValueError("internal_data_collator must be callable")
    collate_fn = internal_data_collator
    return collate_fn


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}")


def calculate_num_micro_batches(
        global_batch_size: int,
        micro_batch_size: int,
        dp_world_size: int,
) -> int:
    """Calculate the number of micro-batches in one optimizer step.

    Args:
        global_batch_size: Number of samples processed globally by one
            optimizer step.
        micro_batch_size: Number of samples processed by one data-parallel
            rank in one forward/backward pass.
        dp_world_size: Number of data-parallel ranks.

    Returns:
        Number of micro-batches each data-parallel rank must process before
        one optimizer update.

    Raises:
        ValueError: If an input is not positive, or ``global_batch_size`` is
            not divisible by ``micro_batch_size * dp_world_size``.
    """
    _validate_positive_int("global_batch_size", global_batch_size)
    _validate_positive_int("micro_batch_size", micro_batch_size)
    _validate_positive_int("dp_world_size", dp_world_size)

    samples_per_distributed_micro_batch = micro_batch_size * dp_world_size
    if global_batch_size % samples_per_distributed_micro_batch != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size * dp_world_size ({samples_per_distributed_micro_batch})"
        )
    return global_batch_size // samples_per_distributed_micro_batch


@dataclass
class MakeMicroBatchCollator:
    """Split one per-rank optimizer batch into collated micro-batches.

    A DataLoader using this collator must load all samples needed by the local
    rank for one optimizer step. For fixed-size batching its ``batch_size`` is
    therefore ``global_batch_size // dp_world_size``. The resulting iterator
    item is always a non-empty ``list[dict]``, including when
    ``num_micro_batch`` is one.

    The collator accepts ordinary mapping features as well as VeOmni-style
    single-item list/tuple wrappers produced by a one-to-many transform.

    Args:
        num_micro_batch: Number of micro-batches in one optimizer step.
        internal_data_collator: Collator used to turn the samples in each
            micro-batch into model inputs. Defaults to PyTorch
            ``default_collate``.

    Example:
        ``DataLoader(dataset, batch_size=global_batch_size // dp_world_size,
        collate_fn=MakeMicroBatchCollator(num_micro_batch))`` returns one
        ``list[dict]`` per optimizer step.

    Note:
        Adapted from VeOmni's Apache-2.0 ``MakeMicroBatchCollator`` contract.
    """

    num_micro_batch: int
    internal_data_collator: InternalDataCollator = default_collate

    def __post_init__(self) -> None:
        _validate_positive_int("num_micro_batch", self.num_micro_batch)
        if not callable(self.internal_data_collator):
            raise ValueError("internal_data_collator must be callable")

    @staticmethod
    def _normalize_feature(feature: Any, index: int) -> Feature:
        if isinstance(feature, Mapping):
            return feature
        if (
                isinstance(feature, (list, tuple))
                and len(feature) == 1
                and isinstance(feature[0], Mapping)
        ):
            return feature[0]
        raise ValueError(
            "Each feature must be a mapping or a single-item list/tuple "
            f"containing a mapping, but feature {index} has type {type(feature).__name__}"
        )

    def __call__(self, features: Sequence[Any]) -> list[dict[str, Any]]:
        """Collate samples into the Trainer ``list[dict]`` step contract.

        Args:
            features: Samples loaded for the local rank's optimizer step.

        Returns:
            One dictionary of model inputs per micro-batch.

        Raises:
            ValueError: If no samples are provided, the samples cannot be split
                evenly, a feature has an unsupported structure, or the internal
                collator does not return a mapping.
        """
        if not features:
            raise ValueError("features must contain at least one sample")
        if len(features) % self.num_micro_batch != 0:
            raise ValueError(
                f"Feature count ({len(features)}) must be divisible by "
                f"num_micro_batch ({self.num_micro_batch})"
            )

        normalized_features = [
            self._normalize_feature(feature, index)
            for index, feature in enumerate(features)
        ]
        micro_batch_size = len(normalized_features) // self.num_micro_batch
        micro_batches = []
        for start in range(0, len(normalized_features), micro_batch_size):
            collated_batch = self.internal_data_collator(
                normalized_features[start:start + micro_batch_size]
            )
            if not isinstance(collated_batch, Mapping):
                raise ValueError(
                    "internal_data_collator must return a mapping, but got "
                    f"{type(collated_batch).__name__}"
                )
            micro_batches.append(dict(collated_batch))

        return micro_batches


__all__ = [
    "MakeMicroBatchCollator",
    "build_collate_fn",
    "calculate_num_micro_batches",
]
