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
"""Torch HSDP cell state"""
from dist_parallel.hsdp.hsdp_state import HSDPState
from dist_parallel.hsdp.platform.torch.param import TorchHSDPParam
from dist_parallel.hsdp.platform.torch.platform import TorchPlatform

class TorchHSDPState(HSDPState):
    """Torch HSDP cell state"""

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        params = self.cell.named_parameters()
        for param_name, param in params:
            if hasattr(param, "has_hsdp_param"):
                continue
            hsdp_param = TorchHSDPParam(self.cell, param_name, param, self.config, self.platform)
            param.has_hsdp_param = True
            self.hsdp_params.append(hsdp_param)
            if hsdp_param.sharded:
                self.sharded_hsdp_params.append(hsdp_param)
