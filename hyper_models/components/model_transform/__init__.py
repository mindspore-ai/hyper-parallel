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
"""Structure-preserving model transforms."""

from hyper_models.components.model_transform.replacement import (
    ModuleReplacementPlan,
    ModuleReplacementSpec,
    ModuleReplacementTarget,
    apply_module_replacements,
    compile_module_replacements,
    module_replacement,
)

__all__ = [
    "ModuleReplacementPlan",
    "ModuleReplacementSpec",
    "ModuleReplacementTarget",
    "apply_module_replacements",
    "compile_module_replacements",
    "module_replacement",
]
