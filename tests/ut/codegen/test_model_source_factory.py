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
"""Keep model-owned source views equivalent to native compute factories."""

import inspect
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

from hyper_parallel.codegen.plan.methods import _source_spec
from hyper_parallel.models.deepseek_v41.adapter import codegen
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_megamoe_compute_fn,
)
from hyper_parallel.trainer.config.target import Target


def _text_forward(self, hidden_states, input_ids=None):
    del self, input_ids
    return hidden_states


def _multimodal_forward(self, hidden_states, input_ids=None, image_mask=None,
                        router_token_mask=None, router_sequence_ids=None, router_num_sequences=None):
    del self, input_ids, image_mask, router_token_mask, router_sequence_ids, router_num_sequences
    return hidden_states


class TestModelSourceFactory(unittest.TestCase):
    """Validate native initialization order and the shared computation body."""

    def test_source_and_native_share_compute_without_source_resource_initialization(self):
        for forward in (_text_forward, _multimodal_forward):
            with self.subTest(contract=forward.__name__):
                experts = Mock(spec=DeepseekV41MegaMoeExperts, swiglu_limit=10.0)
                module = SimpleNamespace(is_hash=False, experts=experts, gate=SimpleNamespace(top_k=2),
                                         shared_experts=SimpleNamespace(limit=10.0))
                module.forward = MethodType(forward, module)
                arguments = dict(module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None)
                source_factory = codegen.source_compute_factory(deepseek_v41_megamoe_compute_fn)
                source = source_factory(**arguments)
                experts.configure.assert_not_called()
                self.assertFalse(hasattr(module.gate, "expert_bias_update_groups"))
                events = []
                experts.configure.side_effect = lambda *_args: events.append(module.gate.expert_bias_update_groups)
                native = deepseek_v41_megamoe_compute_fn(**arguments)
                experts.configure.assert_called_once_with(None, 2)
                self.assertEqual(events, [()])
                self.assertIs(native.__code__, source.__code__)
                self.assertEqual(inspect.signature(native), inspect.signature(source))
                for name, expected in inspect.getclosurevars(native).nonlocals.items():
                    actual = inspect.getclosurevars(source).nonlocals[name]
                    if inspect.isfunction(expected):
                        self.assertIs(expected.__code__, actual.__code__)
                        self.assertEqual(inspect.getclosurevars(expected).nonlocals,
                                         inspect.getclosurevars(actual).nonlocals)
                    else:
                        self.assertEqual(expected, actual)
                module.shared_experts.limit = 1.0
                for factory in (source_factory, deepseek_v41_megamoe_compute_fn):
                    with self.assertRaisesRegex(ValueError, "same swiglu_limit"):
                        factory(**arguments)
                experts.configure.assert_called_once()

    def test_adapter_source_view_preserves_target_arguments_and_original_spec(self):
        native = Target(deepseek_v41_megamoe_compute_fn, target_path="native.megamoe", fix_router=True)
        spec = SimpleNamespace(local_compute_fn=native, region_dispatch=False)
        exported = _source_spec(spec, codegen)
        self.assertIs(spec.local_compute_fn, native)
        self.assertIs(native.callable, deepseek_v41_megamoe_compute_fn)
        self.assertTrue(exported.local_compute_fn.fix_router)
        self.assertIs(exported.local_compute_fn.callable,
                      codegen.source_compute_factory(deepseek_v41_megamoe_compute_fn))
        self.assertIs(_source_spec(spec, None), spec)
        untouched = SimpleNamespace(local_compute_fn=_text_forward)
        self.assertIs(_source_spec(untouched, codegen), untouched)
