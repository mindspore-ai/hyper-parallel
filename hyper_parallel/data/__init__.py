# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""data: dataset building, batching, and parallel batch primitives.

Stage 6 (05 §15.10) final layout: ``text`` (LLM builders/transforms/chat
templates and online sources), ``indexed`` (``.idx``/``.bin`` datasets and
the native helpers), ``batching`` (collators, dataloaders, get-batch),
``parallel`` (DP samplers, CP/TP batch distribution, build barrier),
``vlm`` (transitional vision-language boundary with its own facade), and
``tools`` (offline preparation). Shared constants live in
``data.constants``; dataset logging in ``data.dataset_logging``.

The package root deliberately has no flat re-exports — import the owning
subpackage (VLM symbols only via ``hyper_parallel.data.vlm``).
"""
