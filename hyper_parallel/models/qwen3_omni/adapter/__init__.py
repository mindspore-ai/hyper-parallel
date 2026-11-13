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
"""Adapters for the Qwen3-Omni model family (registered as ``qwen3_omni``)."""

#: ``config.model_type`` spellings this family accepts. HyperParallel registers the
#: family directory as ``qwen3_omni``, while the HF checkpoint reports
#: ``qwen3_omni_moe`` (architecture ``Qwen3OmniMoeForConditionalGeneration``);
#: ``registry._FAMILY_DIR_ALIASES`` maps the HF spelling to the family directory,
#: so runtime guards accept both.
QWEN3_OMNI_MODEL_TYPES = ("qwen3_omni", "qwen3_omni_moe")
