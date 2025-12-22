# Copyright 2025 Huawei Technologies Co., Ltd
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
"""common net"""
import torch
from torch import nn


class SimpleModel(nn.Module):
    """simple model"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 8).npu())

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        x = torch.sum(x)
        return x
