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
"""Regression coverage for single-rank HSDP parameter storage."""

import unittest

import torch

from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2


class TestHSDPParamStorage(unittest.TestCase):
    """Check parameter identity across shard storage replacement."""

    def test_single_rank_unsharded_alias_refreshes_replaced_storage(self):
        """Stable unsharded parameters must follow a replaced shard storage."""
        param = object.__new__(HSDPParamV2)
        param.shard_world_size = 1
        param.sharded_param = torch.nn.Parameter(torch.ones(2, 3))
        param._sharded_param_data = param.sharded_param.detach().view(-1)
        param.unsharded_param_buffers = [param._sharded_param_data]
        param.init_unsharded_param()
        stable = param.unsharded_param
        param.sharded_param.data = torch.full((2, 3), 7.0)
        param._sharded_param_data = param.sharded_param.detach().view(-1)
        param.unsharded_param_buffers = [param._sharded_param_data]
        param.init_unsharded_param()
        self.assertIs(param.unsharded_param, stable)
        self.assertEqual(stable.data_ptr(), param.sharded_param.data_ptr())
        self.assertTrue(torch.equal(stable, torch.full((2, 3), 7.0)))
