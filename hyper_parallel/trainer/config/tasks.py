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
"""Trainer-owned configuration for standalone prediction.

The runtime Predictor accepts a protocol rather than importing this task-host
configuration, keeping model execution and request processing independent.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hyper_parallel.infer.config import GenerationOptions
from hyper_parallel.trainer.config.target import _serialize_config_value


@dataclass
class PredictionConfig:
    """Offline text-generation settings; runtime objects are injected in code.

    Args:
        input: Input TXT, JSON, or JSONL request file.
        output_file: Destination JSONL file, restored to input request order.
        batch_size: Maximum requests per batch on each independent DP lane.
        max_batch_tokens: Optional bound on padded prompt tokens per batch:
            request count times the longest encoded prompt length. Generated
            tokens are not included. A single prompt over the bound is rejected.
        bucket_by_length: Sort each lane's requests by encoded prompt length
            before batching to reduce left padding. This does not enable
            continuous batching or change the final output order.
        overwrite: Allow replacing an existing output file.
        generation: Generation options, including the new-token limit and
            whether to reuse KV cache within each generation call. Cache is not
            preserved across separate prediction calls.
    """

    input: Optional[str] = None
    output_file: Optional[str] = None
    batch_size: int = 1
    max_batch_tokens: Optional[int] = None
    bucket_by_length: bool = False
    overwrite: bool = False
    generation: GenerationOptions = field(default_factory=GenerationOptions)

    def __post_init__(self) -> None:
        """Reject batch sizes the static P1 request planner cannot shard."""
        if (not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool)) or self.batch_size < 1:
            raise ValueError(f"prediction.batch_size must be a positive integer, got {self.batch_size}")
        if self.max_batch_tokens is not None and (
                not isinstance(self.max_batch_tokens, int)
                or isinstance(self.max_batch_tokens, bool) or self.max_batch_tokens < 1):
            raise ValueError("prediction.max_batch_tokens must be a positive integer or null")
        if not isinstance(self.overwrite, bool) or not isinstance(self.bucket_by_length, bool):
            raise ValueError("prediction.overwrite and bucket_by_length must be bools")
        if isinstance(self.generation, dict):
            self.generation = GenerationOptions(**self.generation)
        elif self.generation is None:
            self.generation = GenerationOptions()
        elif not isinstance(self.generation, GenerationOptions):
            raise ValueError("prediction.generation must contain GenerationOptions")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction options in their compact YAML shape."""
        return {
            "input": self.input,
            "output_file": self.output_file,
            "batch_size": self.batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "bucket_by_length": self.bucket_by_length,
            "overwrite": self.overwrite,
            "generation": _serialize_config_value(self.generation),
        }
