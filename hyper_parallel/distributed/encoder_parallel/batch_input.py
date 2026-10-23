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
"""Adapt encoder planning to the existing Omni batch-input protocol."""

from typing import Any, Mapping

from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime


class EncoderParallelInputAdapter:
    """Compose per-microbatch encoder plans with an optional model input adapter."""

    def __init__(self, runtime: EncoderParallelRuntime, inner: Any = None) -> None:
        """Keep the visual runtime and the existing Omni input provider."""
        self.runtime = runtime
        self.inner = inner

    def build(self, *, batch: Mapping[str, Any], parallel_context: Any) -> dict[str, Any]:
        """Validate model metadata, then build the collective work plan."""
        prepared = self.runtime.preflight_batch(batch)
        generated = {} if self.inner is None else self.inner.build(batch=batch, parallel_context=parallel_context)
        if not isinstance(generated, Mapping):
            raise TypeError("Runtime input adapter must return a mapping")
        if "encoder_parallel_plan" in generated or "encoder_parallel_plan" in batch:
            raise ValueError("encoder_parallel_plan has more than one owner")
        return {**generated, "encoder_parallel_plan": self.runtime.finalize_plan(*prepared)}
