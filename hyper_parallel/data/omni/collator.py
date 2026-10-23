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
"""Collate prepared Omni samples without changing text collation."""

from collections.abc import Mapping, Sequence
from typing import Any

# This opt-in path uses the existing Torch StatefulDataLoader contract.
import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import default_collate  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.omni.constants import OMNI_CONCAT_FIELDS


class OmniCollator:
    """Batch already prepared Omni samples without performing sequence packing."""

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack token fields and concatenate variable-length modality fields."""
        if not model_samples:
            raise ValueError("model_samples must contain at least one Omni sample")

        sample_fields = set()
        for model_sample in model_samples:
            sample_fields.update(model_sample)

        batch = {}
        for field in sample_fields:
            values = []
            for model_sample in model_samples:
                if field in model_sample:
                    values.append(model_sample[field])

            if field in OMNI_CONCAT_FIELDS:
                batch[field] = torch.cat(values, dim=0)
            else:
                batch[field] = default_collate(values)

        return batch
