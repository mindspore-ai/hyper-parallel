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
"""Metadata for whole, already-transformed HP image-text conversations."""

from collections.abc import Mapping
from typing import Any

from hyper_parallel.distributed_data.schema import SampleMetadata, WorkloadCost


def vlm_sample_metadata(sample: Mapping[str, Any]) -> SampleMetadata:
    """Estimate visual work without changing native VLM sample fields.

    Args:
        sample: One CPU ``VLMChatTransform`` output, before ``VLMCollator``.

    Returns:
        Physical padded sequence width for capacity and raw image patch count
        for encoder cost. LLM cost uses the padded width: the native dense
        text computation does not disappear at masked padding positions.

    Raises:
        ValueError: If sequence or image tensors do not follow the native schema.

    Note:
        These are workload proxies, not calibrated time estimates. Image patches
        come from the processor's actual grid, not a hard-coded patch/merge size.
        Decode/transform has already run on the Reader and is not balanced here.
    """
    input_ids = sample.get("input_ids")
    grid = sample.get("image_grid_thw")
    pixels = sample.get("pixel_values")
    if input_ids is None or input_ids.ndim != 1 or input_ids.shape[0] == 0:
        raise ValueError("VLM metadata requires one non-empty 1-D input_ids tensor before collation")
    if grid is None or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("VLM metadata requires image_grid_thw with shape [num_images, 3]")
    if grid.device.type != "cpu":
        raise ValueError("VLM metadata must be extracted on CPU before device transfer")
    image_patches = 0
    for dimensions in grid.tolist():
        if any(not isinstance(dimension, int) or isinstance(dimension, bool) for dimension in dimensions):
            raise ValueError("VLM image_grid_thw must contain integer patch dimensions")
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("VLM image_grid_thw dimensions must be positive")
        image_patches += dimensions[0] * dimensions[1] * dimensions[2]
    if pixels is None or pixels.ndim != 2 or pixels.shape[0] != image_patches:
        raise ValueError("VLM pixel_values rows must equal the raw patch count from image_grid_thw")
    seq_len = int(input_ids.shape[0])
    return SampleMetadata(pack_tokens=seq_len, cost=WorkloadCost(encoder=float(image_patches), llm=float(seq_len)))
