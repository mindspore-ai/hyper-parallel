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
"""MindSpore HSDP cell state"""
from mindspore.common.api import _no_grad
from dist_parallel.hsdp.hsdp_state import HSDPState
from dist_parallel.hsdp.platform.mindspore.param import MindSporeHSDPParam


class MindSporeHSDPState(HSDPState):
    """MindSpore HSDP cell state"""

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        cells = self.cell.cells_and_names()
        for _, sub_cell in cells:
            params = sub_cell._params.items() #pylint: disable=W0212
            for param_name, param in params:
                if hasattr(param, "has_hsdp_param"):
                    continue
                hsdp_param = MindSporeHSDPParam(sub_cell, param_name, param, self.config, self.platform)
                param.has_hsdp_param = True
                self.hsdp_params.append(hsdp_param)
                if hsdp_param.sharded:
                    self.sharded_hsdp_params.append(hsdp_param)

    @_no_grad()
    def shard(self):
        """change parameters to sharded state"""
        super().shard()

    @_no_grad()
    def unshard(self):
        """change parameters to unsharded state"""
        super().unshard()

    @_no_grad()
    def prefetch(self):
        """prefetch unsharded parameters"""
        super().prefetch()

    @_no_grad()
    def zero_grads(self):
        """zero grad or grad buffer"""
        super().zero_grads()

    @_no_grad()
    def set_grad_ready(self, hsdp_param):
        """set grad ready"""
        super().set_grad_ready(hsdp_param)

    @_no_grad()
    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        super().set_requires_grad_sync(requires_grad_sync)
