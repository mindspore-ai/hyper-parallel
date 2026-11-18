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
"""Compatibility entry for the canonical V4.1 model specification."""

from hyper_parallel.models.deepseek_v41.deepseek_v41_spec import MODEL_SPEC
from hyper_parallel.models.registry import register_model_adapter

DEEPSEEK_V41_ADAPTER_SPEC = MODEL_SPEC

register_model_adapter(MODEL_SPEC)

__all__ = ["DEEPSEEK_V41_ADAPTER_SPEC"]
