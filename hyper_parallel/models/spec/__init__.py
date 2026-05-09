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
"""ModelSpec and registry for model registration."""
from hyper_parallel.models.spec.model_spec import ModelSpec
from hyper_parallel.models.spec.registry import register_spec, get_spec
from hyper_parallel.models.spec.state_dict_adapter import StateDictAdapter

__all__ = ["ModelSpec", "register_spec", "get_spec", "StateDictAdapter"]
