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
"""Dataset contracts shared by LLM and Omni components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from torch.utils.data import IterableDataset

RawSample: TypeAlias = Mapping[str, Any]
ModelSample: TypeAlias = Mapping[str, Any]


class SampleTransform(Protocol):
    """Convert one source record into one model-facing sample."""

    def __call__(self, sample: RawSample) -> ModelSample | Sequence[ModelSample]:
        """Transform one raw sample.

        Args:
            sample: Record emitted by a dataset source.

        Returns:
            One model-facing sample, or multiple samples when a transform
            splits one source record. Multi-sample results require a Dataset
            or packing stage that can flatten them.
        """
        raise NotImplementedError


def is_iterable_dataset(dataset: Any) -> bool:
    """Return whether a Dataset streams samples without integer indexing.

    Args:
        dataset: Dataset-like runtime object.

    Returns:
        ``True`` when the object is iterable but does not expose
        ``__getitem__`` for mapping-style access.
    """
    if isinstance(dataset, IterableDataset):
        return True

    has_iterator = callable(getattr(dataset, "__iter__", None))
    has_index_access = callable(getattr(dataset, "__getitem__", None))
    return has_iterator and not has_index_access
