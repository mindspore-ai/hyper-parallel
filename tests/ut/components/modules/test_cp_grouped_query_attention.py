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
"""CPU contracts for the GQA family CP member and Ring selector."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.modules import GroupedQueryAttention
from hyper_parallel.components.modules.module_family import get_family_build_info


class TestCPGroupedQueryAttention(unittest.TestCase):
    """Verify family selection and CP attention preparation without an NPU."""

    @staticmethod
    def _config():
        return SimpleNamespace(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            v_head_dim=4,
            attention_bias=False,
            o_proj_bias=False,
            qk_norm=False,
            rms_norm_eps=1e-6,
            dtype=torch.float32,
            sliding_window=None,
            is_causal=True,
            attention_dropout=0.0,
        )

    def test_family_selects_cp_member_for_ring_algorithm(self):
        """The shared GQA family passes the Ring selector to its CP member."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring",
        )

        self.assertEqual(type(module).__name__, "CPGroupedQueryAttentionModule")
        self.assertEqual(get_family_build_info(module).resolved.variant, "cp")
        self.assertEqual(module.parallel_module_spec().binding_kind, "cp")

        from hyper_parallel.components.modules.module_family import FamilyShardingContext
        from hyper_parallel.distributed._builder.family_layout import _module_parallel
        from hyper_parallel.distributed.module_binding import validate_parallel_module

        family = get_family_build_info(module).resolved
        layout_context = FamilyShardingContext(("cp",), {"cp": 4, "tp": 1, "ep": 1, "pp": 1}, {})
        sharding = family.sharding(module=module, module_fqn="self_attn", member=family,
                                   context=layout_context)
        validate_parallel_module(module, _module_parallel(module, family), sharding, layout_context)

    def test_ring_dense_and_packed_forward_contracts(self):
        """Dense and packed inputs reach Ring with the corresponding layout and options."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring", load_balance="zigzag",
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=0, mesh=object())))
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "cp_grouped_query_attention.ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            dense = torch.randn(1, 2, 16)
            dense_output, dense_weights = module(dense)
            packed = torch.randn(1, 2, 16)
            packed_output, packed_weights = module(packed, actual_seq_len=[4, 8])

        self.assertEqual(dense_output.shape, dense.shape)
        self.assertEqual(packed_output.shape, packed.shape)
        self.assertIsNone(dense_weights)
        self.assertIsNone(packed_weights)
        self.assertEqual(ring.call_args_list[0].kwargs["input_layout"], "BSND")
        self.assertEqual(ring.call_args_list[1].kwargs["input_layout"], "TND")
        self.assertEqual(ring.call_args_list[1].kwargs["document_lengths"], (4, 4))
        self.assertEqual(ring.call_args_list[1].kwargs["load_balance"], "zigzag")
        self.assertTrue(ring.call_args_list[0].kwargs["prune_documents"])
        self.assertTrue(ring.call_args_list[1].kwargs["prune_documents"])
