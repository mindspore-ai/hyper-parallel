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
"""CPU validation of the Xing40 sequencewise router CP binding contract."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.models.xing40.adapter.distributed.moe_context_parallel import xing40_moe_cp_wrapper
from hyper_parallel.models.xing40.modeling_xing40 import Xing40Config, Xing40MoE


class TestXing40MoECPBinding(unittest.TestCase):
    """Reject invalid or repeated group setup without requiring distributed devices."""

    def test_moe_cp_binding_rejects_invalid_or_repeated_setup(self):
        """
        Feature: Explicit sequencewise MoE CP group binding.
        Description: Bind a real MoE module shell without creating process groups.
        Expectation: A single CP2 binding succeeds; unsupported or repeated setup fails.
        """
        module = Xing40MoE(Xing40Config(
            hidden_size=8, moe_intermediate_size=4, n_routed_experts=4,
            num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
            use_pad_tokens=True, moe_aux_loss_coeff=0.0006,
            moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001,
        ))
        group = object()
        cp_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: group)
        with self.assertRaisesRegex(TypeError, 'Xing40 MoE'):
            xing40_moe_cp_wrapper(torch.nn.Linear(2, 2), None, None, cp_mesh, None)
        with self.assertRaisesRegex(ValueError, 'CP2'):
            xing40_moe_cp_wrapper(module, None, None, None, None)
        self.assertEqual(xing40_moe_cp_wrapper(module, None, None, cp_mesh, None), [])
        self.assertIs(module.aux_loss_cp_group, group)
        with self.assertRaisesRegex(ValueError, 'already configured'):
            xing40_moe_cp_wrapper(module, None, None, cp_mesh, None)
