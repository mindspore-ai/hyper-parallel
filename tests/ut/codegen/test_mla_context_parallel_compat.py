# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Keep source-visible MLA dispatch aligned with native overlap semantics."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.models.deepseek_v3.adapter.distributed import context_parallel


class TestMLAContextParallelCompatibility(unittest.TestCase):
    """Exercise the shared forward without devices or collectives."""

    def test_output_projection_and_overlap_follow_runtime_plan(self):
        """Retain native projection hooks and supported overlap combinations."""
        hidden = torch.ones(1, 2, 4)
        for overlap, tp_degree, ulysses_degree, gather_degree, grad_enabled in (
            (False, 1, 2, 1, True),
            (True, 1, 2, 1, True),
            (True, 1, 2, 2, True),
            (True, 2, 2, 1, True),
            (True, 1, 1, 2, True),
            (True, 1, 2, 1, False),
        ):
            with self.subTest(overlap=overlap, tp=tp_degree, ulysses=ulysses_degree,
                              gather=gather_degree, grad=grad_enabled):
                runtime = Mock(return_value=(hidden, None))
                runtime.plan = SimpleNamespace(
                    overlap=overlap, tp_degree=tp_degree,
                    ulysses_degree=ulysses_degree, all_gather_degree=gather_degree,
                )
                module = SimpleNamespace(
                    mla_cp_runtime=runtime, v_head_dim=4,
                    project_latent_inputs=Mock(return_value=(hidden,)),
                    project_output=Mock(return_value=hidden),
                    o_proj=Mock(return_value=hidden),
                )
                output_overlap = overlap and tp_degree == 1 and ulysses_degree > 1 and grad_enabled
                with torch.set_grad_enabled(grad_enabled), patch.object(
                    context_parallel, "MLAOutputProjectionState",
                ) as projection_state:
                    output, weights = context_parallel.mla_cp_forward(module, hidden)
                self.assertIs(output, hidden)
                self.assertIsNone(weights)
                self.assertEqual(runtime.call_args.kwargs["restore_output"], not output_overlap)
                self.assertEqual(runtime.call_args.kwargs["overlap_input_backward"],
                                 output_overlap and gather_degree == 1)
                if output_overlap:
                    projection_state.assert_called_once_with(runtime.ulysses_mesh, 4)
                    module.o_proj.assert_called_once_with(hidden, mla_output_state=projection_state.return_value)
                    module.project_output.assert_not_called()
                else:
                    module.project_output.assert_called_once()
                    projected, residual = module.project_output.call_args.args
                    torch.testing.assert_close(projected, hidden)
                    self.assertIs(residual, hidden)
                    module.o_proj.assert_not_called()
                    projection_state.assert_not_called()

    def test_kv_cache_is_rejected_before_runtime_execution(self):
        """A cache must fail before runtime state or projections are accessed."""
        with self.assertRaisesRegex(ValueError, "does not support a KV cache"):
            context_parallel.mla_cp_forward(SimpleNamespace(), torch.ones(1, 2, 4), past_key_values=object())
