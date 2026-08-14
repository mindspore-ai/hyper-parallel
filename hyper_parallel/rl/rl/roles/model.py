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
"""Model identity shared by training and rollout runtimes."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ModelRegistration:
    """Resolved logical model and its local artifacts."""
    name: str
    hyper_model_name: str
    weights_path: str
    tokenizer_path: str
__all__ = ["ModelRegistration"]
