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
"""kimi_k26.adapter.distributed: CP input sharding and the EP compute archetype.

The family's attention-side CP is the shared registry wrapper
``sdpa_hf`` (K/V all-gather plus the D-04 offset mask), declared per model
config through a ``plan_overrides`` ``when: cp`` entry. This package owns the
VLM-specific piece that no generic wrapper can express: slicing the text
tower's ``inputs_embeds`` (after the full-sequence vision scatter) to this
CP rank's window and building the offset-aware causal+padding mask.

It also owns the family's expert-parallel archetype factory
(``ep_compute.kimi_k26_ep_compute_fn``), which composes the public
``build_ep_compute`` skeleton with the sigmoid-group router and offers the
benchmark-only balanced router behind its ``fix_router`` switch.
"""

from hyper_parallel.models.kimi_k26.adapter.distributed.context_parallel import (
    bind_context_parallel,
)
from hyper_parallel.models.kimi_k26.adapter.distributed.ep_compute import (
    kimi_k26_ep_compute_fn,
)

__all__ = ["bind_context_parallel", "kimi_k26_ep_compute_fn"]
