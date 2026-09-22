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
"""Build DeepSeek-V4.1 model inputs from generic runtime batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
)
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)


class DeepseekV41Runtime(RuntimeInputAdapter):
    """Build V4.1 packed attention and CP-local image insertion inputs."""

    def __init__(
            self,
            *,
            include_position_ids: bool = True,
            include_image_sequence_start: bool = True,
    ) -> None:
        """Configure fields already supplied by the selected batch runtime.

        Args:
            include_position_ids: Build CP-global position IDs. Omni batches
                require this; ``TextParallelBatch`` already owns them.
            include_image_sequence_start: Build the CP-local image insertion
                offset. Text-only batches do not need this field.
        """
        self.include_position_ids = include_position_ids
        self.include_image_sequence_start = include_image_sequence_start

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the stable set of model inputs produced by this instance."""
        fields = ["packed_seq_params"]
        if self.include_position_ids:
            fields.append("position_ids")
        if self.include_image_sequence_start:
            fields.append("image_sequence_start")
        return tuple(fields)

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build packed attention metadata and Omni image coordinates."""
        local_input_shape = context.local_input_shape
        if len(local_input_shape) != 2 or int(local_input_shape[0]) != 1:
            raise ValueError(
                "DeepSeek-V4.1 compact packing requires local input shape [1, sequence], "
                f"got {tuple(local_input_shape)}"
            )
        if "cu_seq_lens" not in batch:
            raise ValueError("DeepSeek-V4.1 batching requires packed cu_seq_lens")

        local_sequence_length = int(local_input_shape[1])
        cp_rank = context.parallel_ranks.get("cp", 0)
        cp_size = context.parallel_sizes.get("cp", 1)
        runtime_inputs = {
            "packed_seq_params": SharedCompressedPackedSequence(
                cu_seq_lens=batch["cu_seq_lens"],
                local_query_start=cp_rank * local_sequence_length,
                local_query_length=local_sequence_length,
                global_sequence_length=cp_size * local_sequence_length,
            )
        }
        cp_start = runtime_inputs["packed_seq_params"].local_query_start
        if self.include_position_ids:
            input_ids = batch["input_ids"]
            position_ids = torch.arange(
                cp_start,
                cp_start + local_sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            runtime_inputs["position_ids"] = position_ids
        if self.include_image_sequence_start:
            runtime_inputs["image_sequence_start"] = cp_start
        return runtime_inputs


__all__ = ["DeepseekV41Runtime"]
