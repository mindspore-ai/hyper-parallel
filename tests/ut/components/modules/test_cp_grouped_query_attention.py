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
from torch import nn

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

    def test_ring_dense_forward_keeps_bsnd_contract(self):
        """Dense BSH input returns the ordinary output-projection shape."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 2}, cp_algorithm="ring", prune_documents=False,
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=2, rank=0, mesh=object())))
        hidden_states = torch.randn(1, 4, 16)
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "cp_grouped_query_attention.ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            output, weights = module(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNone(weights)
        self.assertEqual(ring.call_args.kwargs["input_layout"], "BSND")

    def test_packed_forward_converts_cumulative_lengths(self):
        """Packed metadata is converted to global document lengths and TND."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring",
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=0, mesh=object())))
        hidden_states = torch.randn(1, 2, 16)
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "cp_grouped_query_attention.ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            output, _ = module(hidden_states, actual_seq_len=[4, 8])

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(ring.call_args.kwargs["input_layout"], "TND")
        self.assertEqual(ring.call_args.kwargs["document_lengths"], (4, 4))

    def test_pruning_and_balance_reach_ring_kernel(self):
        """Packed module options enable both planners on the Ring kernel."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring",
            load_balance="zigzag", prune_documents=True,
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=0, mesh=object())))
        hidden_states = torch.randn(1, 2, 16)
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "cp_grouped_query_attention.ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            module(hidden_states, actual_seq_len=[4, 8])

        self.assertEqual(ring.call_args.kwargs["load_balance"], "zigzag")
        self.assertTrue(ring.call_args.kwargs["prune_documents"])

    def test_default_pruning_is_a_dense_noop_without_document_metadata(self):
        """The default flag is forwarded but only prunes when packed metadata exists."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 2}, cp_algorithm="ring",
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=2, rank=0, mesh=object())))
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "cp_grouped_query_attention.ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            module(torch.randn(1, 2, 16))
        self.assertTrue(ring.call_args.kwargs["prune_documents"])

    def test_global_positions_are_sliced_to_local_cp_tokens(self):
        """Global rotary tables use the source rank's contiguous token slice."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring",
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=2, mesh=object())))
        cosine = torch.arange(8).view(1, 8, 1).expand(1, 8, 4)
        sine = -cosine

        positions = module.prepare_positions((1, 2), (cosine, sine))

        torch.testing.assert_close(positions[0], cosine[:, 4:6])
        torch.testing.assert_close(positions[1], sine[:, 4:6])

    def test_source_module_uses_shared_gqa_conversion_contract(self):
        """Ring construction retains output/norm modules and exposes QKV transforms."""
        source = SimpleNamespace(
            config=self._config(),
            q_proj=nn.Linear(16, 16),
            k_proj=nn.Linear(16, 8),
            v_proj=nn.Linear(16, 8),
            o_proj=nn.Linear(16, 16),
            q_norm=nn.Identity(),
            k_norm=nn.Identity(),
            scaling=0.5,
            attention_dropout=0.0,
            is_causal=True,
            sliding_window=None,
        )
        source_module = nn.Module()
        for name, value in vars(source).items():
            setattr(source_module, name, value)

        module = GroupedQueryAttention(module=source_module, context={"cp_size": 4}, cp_algorithm="ring")

        self.assertIs(module.o_proj, source.o_proj)
        self.assertIs(module.q_norm, source.q_norm)
        self.assertTrue(module.make_transforms())

    def test_explicit_mask_is_rejected(self):
        """Ring does not silently reinterpret an explicit visibility mask."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 2}, cp_algorithm="ring", prune_documents=False,
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=2, rank=0, mesh=object())))
        with self.assertRaisesRegex(ValueError, "explicit attention mask"):
            module(torch.randn(1, 4, 16), attention_mask=torch.ones(1, 4, dtype=torch.bool))

    def test_sliding_window_is_rejected_during_forward(self):
        """A source's windowed attention semantics cannot be silently dropped."""
        config = self._config()
        config.sliding_window = 128
        module = GroupedQueryAttention(
            config=config, context={"cp_size": 2}, cp_algorithm="ring", prune_documents=False,
        )
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=2, rank=0, mesh=object())))
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            module(torch.randn(1, 4, 16))

    def test_ring_member_rejects_ulysses_degree(self):
        """Ring uses the complete CP group without an extra Ulysses split."""
        module = GroupedQueryAttention(
            config=self._config(), context={"cp_size": 4}, cp_algorithm="ring", ulysses_degree=2,
        )
        with self.assertRaisesRegex(ValueError, "requires ulysses_degree=1"):
            module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=0, mesh=object())))
