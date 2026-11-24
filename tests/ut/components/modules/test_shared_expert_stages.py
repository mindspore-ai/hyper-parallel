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
"""Shared FC1/FC2 stages retain the original full-forward mathematics and module hooks."""

import copy
from itertools import product
import unittest

import torch

from examples.torch.module_family.config import SharedExpertConfig
from hyper_parallel.components.modules.moe.shared_expert import SharedExpertModule
from tests.common.mark_utils import arg_mark


def _original_forward(module, hidden):
    """Run the pre-split implementation as an independent serial oracle."""
    if module._preserve_source_layout:  # pylint: disable=protected-access
        intermediate = torch.cat((module.gate_proj(hidden), module.up_proj(hidden)), dim=-1)
        return module.down_proj(module.activation_func(intermediate).to(hidden.dtype))
    intermediate, bias = module.linear_fc1(hidden)
    if bias is not None:
        intermediate = intermediate + bias
    intermediate = module.activation_func(intermediate)
    output, output_bias = module.linear_fc2(intermediate)
    if not module._return_bias_tuple:  # pylint: disable=protected-access
        if output_bias is not None:
            output = output + output_bias
        return output
    return output, output_bias


def _combined_output(result):
    """Keep output and separately returned bias in the scalar gradient objective."""
    if isinstance(result, tuple):
        output, bias = result
        return output if bias is None else output + bias
    return result


class TestSharedExpertStages(unittest.TestCase):
    """Check projection layouts, activation policies, returned bias and hook coverage."""

    def setUp(self) -> None:
        """Isolate CPU cases from callers' random streams."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(65)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_split_forward_matches_original_outputs_and_all_gradients(self) -> None:
        """Feature: Shared expert two-phase numerical contract.
        Description: Compare split and full forward against the original implementation across layouts and dtype.
        Expectation: Outputs, output bias, hidden gradients and every projection gradient match bitwise.
        """
        layouts = ((False, False), (False, True), (True, False))
        for dtype, limit, (separate, bias_tuple) in product((torch.float64, torch.float32, torch.bfloat16),
                                                         (0., .7), layouts):
            with self.subTest(dtype=dtype, limit=limit, separate=separate, bias_tuple=bias_tuple):
                config = SharedExpertConfig(8, 12, bias=True, return_bias_tuple=bias_tuple, swiglu_limit=limit)
                shared = SharedExpertModule(config=config, preserve_source_layout=separate).to(dtype=dtype)
                reference = copy.deepcopy(shared)
                hidden = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                other = hidden.detach().clone().requires_grad_()
                identities = {name: id(value) for name, value in shared.named_parameters()}
                intermediate = shared(hidden, stage="fc1")
                self.assertEqual(intermediate.shape, (2, 3, 12))
                actual = shared(intermediate, stage="fc2")
                expected = _original_forward(reference, other)
                actual_values = actual if isinstance(actual, tuple) else (actual,)
                expected_values = expected if isinstance(expected, tuple) else (expected,)
                for value, oracle in zip(actual_values, expected_values):
                    if value is None:
                        self.assertIsNone(oracle)
                    else:
                        torch.testing.assert_close(value, oracle, rtol=0, atol=0)
                _combined_output(actual).float().square().sum().backward()
                _combined_output(expected).float().square().sum().backward()
                torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                for value, oracle in zip(shared.parameters(), reference.parameters()):
                    torch.testing.assert_close(value.grad, oracle.grad, rtol=0, atol=0)
                full = shared(hidden.detach())
                torch.testing.assert_close(_combined_output(full), _combined_output(actual), rtol=0, atol=0)
                self.assertEqual(identities, {name: id(value) for name, value in shared.named_parameters()})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_split_phases_enter_module_hooks_and_reject_invalid_stage(self) -> None:
        """Feature: Split shared expert module lifecycle.
        Description: Observe pre/post hooks around each phase using the normal module call interface.
        Expectation: Both phases enter hooks, retain backward connectivity, and reject unknown stages.
        """
        shared = SharedExpertModule(config=SharedExpertConfig(8, 12))
        events = []

        def _pre(module, args, kwargs):
            """Record the stage entering the shared expert."""
            del module, args
            events.append(("pre", kwargs["stage"]))

        def _post(module, args, kwargs, output):
            """Record the stage leaving the shared expert."""
            del module, args, output
            events.append(("post", kwargs["stage"]))

        handles = (shared.register_forward_pre_hook(_pre, with_kwargs=True),
                   shared.register_forward_hook(_post, with_kwargs=True))
        hidden = torch.randn(2, 8, requires_grad=True)
        intermediate = shared(hidden, stage="fc1")
        shared(intermediate, stage="fc2").sum().backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(events, [("pre", "fc1"), ("post", "fc1"), ("pre", "fc2"), ("post", "fc2")])
        self.assertIsNotNone(hidden.grad)
        with self.assertRaisesRegex(ValueError, "stage must be"):
            shared(hidden, stage="unknown")
