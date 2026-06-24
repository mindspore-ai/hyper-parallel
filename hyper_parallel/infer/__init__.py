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
"""Autoregressive generation utilities."""

from hyper_parallel.infer.generation import generate
from hyper_parallel.infer.kv_cache import KVCache
from hyper_parallel.infer.mixin import GenerateMixin
from hyper_parallel.infer.sampler import (
    apply_repetition_penalty,
    greedy_sample,
    sample_next_token,
    top_k_sample,
    top_p_sample,
)
from hyper_parallel.infer.utils import (
    GenerationConfig,
    apply_logits_processors,
    build_causal_mask,
    build_position_ids,
    gather_context_parallel_logits,
    gather_tensor_parallel_logits,
    prepare_logits_for_sampling,
    should_stop_generation,
)

__all__ = [
    "GenerationConfig",
    "GenerateMixin",
    "KVCache",
    "apply_logits_processors",
    "apply_repetition_penalty",
    "build_causal_mask",
    "build_position_ids",
    "gather_context_parallel_logits",
    "gather_tensor_parallel_logits",
    "generate",
    "greedy_sample",
    "prepare_logits_for_sampling",
    "sample_next_token",
    "should_stop_generation",
    "top_k_sample",
    "top_p_sample",
]
