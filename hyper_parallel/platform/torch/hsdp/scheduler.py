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
"""HSDP scheduler"""
from hyper_parallel.core.hsdp.hsdp_scheduler import HSDPScheduler
from hyper_parallel.platform.torch.hsdp.state import TorchHSDPState
from hyper_parallel.platform import get_platform


class TorchHSDPScheduler(HSDPScheduler):
    """TorchHSDPScheduler is used to implement optimizer level."""

    def _init_platform(self):
        """init platform"""
        self.platform = get_platform()

    def _new_cell_state(self):
        """new cell state"""
        self.hsdp_state = TorchHSDPState(self.cell, self.config, self.platform)
