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
"""Keep HP native expert outputs in the activation dtype on empty source ranks."""
from types import MethodType

from torch import nn

from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule


def _native_unpermute(self, values, unpermute_indices, topk_weights, output_shape):
    # HP's empty branch multiplies these tensors to keep the router gradient
    # connected. Match their dtypes so the empty result has the normal output dtype.
    if values.shape[0] == 0:
        topk_weights = topk_weights.to(values.dtype)
    return GroupedExpertsModule.unpermute(self, values, unpermute_indices, topk_weights, output_shape)


def bind_native_unpermute(model: nn.Module) -> None:
    """Apply the empty-row dtype fix to the HP expert modules used by this RL role."""
    for experts in model.modules():
        if isinstance(experts, GroupedExpertsModule):
            experts.unpermute = MethodType(_native_unpermute, experts)
