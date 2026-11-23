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
"""CPU contracts for the Ring GQA module-family member."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import RingGroupedQueryAttention
from hyper_parallel.components.modules.module_family import get_family_build_info


class TestRingGroupedQueryAttention(unittest.TestCase):
    """Verify family selection and runtime layout translation without an NPU."""

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

    def test_family_builds_ring_member(self):
        """The explicit Ring factory resolves to its concrete family member."""
        module = RingGroupedQueryAttention(config=self._config(), context={"cp_size": 4})

        self.assertEqual(type(module).__name__, "RingGroupedQueryAttentionModule")
        self.assertEqual(get_family_build_info(module).resolved.variant, "ring")
        self.assertEqual(module.parallel_module_spec().binding_kind, "cp")

        from hyper_parallel.components.modules.module_family import FamilyShardingContext
        from hyper_parallel.distributed._builder.family_layout import _module_parallel
        from hyper_parallel.distributed.module_binding import validate_parallel_module

        family = get_family_build_info(module).resolved
        layout_context = FamilyShardingContext(("cp",), {"cp": 4, "tp": 1, "ep": 1, "pp": 1}, {})
        sharding = family.sharding(module=module, module_fqn="self_attn", member=family,
                                   context=layout_context)
        validate_parallel_module(module, _module_parallel(module, family), sharding, layout_context)

    def test_dense_forward_keeps_bsnd_contract(self):
        """Dense BSH input returns the ordinary output-projection shape."""
        module = RingGroupedQueryAttention(config=self._config())
        hidden_states = torch.randn(1, 4, 16)
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "ring_grouped_query_attention.ulysses_ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            output, weights = module(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNone(weights)
        self.assertEqual(ring.call_args.kwargs["input_layout"], "BSND")

    def test_packed_forward_converts_cumulative_lengths(self):
        """Packed metadata is converted to global document lengths and TND."""
        module = RingGroupedQueryAttention(config=self._config(), context={"cp_size": 4})
        module.bind_parallel(SimpleNamespace(cp_context=Mock(size=4, rank=0, mesh=object())))
        hidden_states = torch.randn(1, 2, 16)
        with patch(
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "ring_grouped_query_attention.ulysses_ring_attention",
            side_effect=lambda query, key, value, *args, **kwargs: query,
        ) as ring:
            output, _ = module(hidden_states, actual_seq_len=[4, 8])

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(ring.call_args.kwargs["input_layout"], "TND")
        self.assertEqual(ring.call_args.kwargs["document_lengths"], (4, 4))

    def test_global_positions_are_sliced_to_local_cp_tokens(self):
        """Global rotary tables use the source rank's contiguous token slice."""
        module = RingGroupedQueryAttention(config=self._config(), context={"cp_size": 4})
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

        module = RingGroupedQueryAttention(module=source_module, context={"cp_size": 4})

        self.assertIs(module.o_proj, source.o_proj)
        self.assertIs(module.q_norm, source.q_norm)
        self.assertTrue(module.make_transforms())

    def test_explicit_mask_is_rejected(self):
        """Ring does not silently reinterpret an explicit visibility mask."""
        module = RingGroupedQueryAttention(config=self._config())
        with self.assertRaisesRegex(ValueError, "explicit attention mask"):
            module(torch.randn(1, 4, 16), attention_mask=torch.ones(1, 4, dtype=torch.bool))

    def test_sliding_window_is_rejected_at_construction(self):
        """A source's windowed attention semantics cannot be silently dropped."""
        config = self._config()
        config.sliding_window = 128
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            RingGroupedQueryAttention(config=config)

    def test_ulysses_degree_must_divide_configured_cp_degree(self):
        """Reject unsupported decomposition before mesh binding or execution."""
        with self.assertRaisesRegex(ValueError, "divide the CP degree"):
            RingGroupedQueryAttention(config=self._config(), ulysses_degree=2)
