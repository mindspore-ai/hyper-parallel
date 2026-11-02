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
"""CPU configuration contracts for sparse and legacy MoonEP prefetch ownership."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep import runtime
from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as sparse
from hyper_parallel.models.qwen3_moe.adapter.distributed import expert_parallel as qwen
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestSparseConfiguration(unittest.TestCase):
    """Reject inactive sparse options before binding runtime or FSDP extensions."""

    @staticmethod
    def _model(fsdp: bool = False) -> torch.nn.Module:
        model = torch.nn.Module()
        with patch.object(sparse.dist, "get_rank", return_value=0):
            model._moonep_executor = sparse.SparseMoonEPExecutor(
                model, Mock(), ep_group=object(), ep_size=2, num_experts=4, top_k=2)
        if fsdp:
            model.hsdp_scheduler = SimpleNamespace(
                modules=[model], cell=model, enable_param_grad_gate=Mock(),
                register_all_gather_extension=Mock(), register_gradient_extension=Mock())
        return model


    def test_sparse_fsdp_default_binds_notification_extensions(self) -> None:
        """Default configuration retains FSDP notification hooks without depth state."""
        model = self._model(fsdp=True)
        configured = runtime.configure_moonep(model)
        self.assertIsInstance(configured, sparse.SparseMoonEPRuntime)
        self.assertIs(model._moonep_executor.runtime, configured)
        self.assertIs(model._moonep_runtime, configured)
        model.hsdp_scheduler.enable_param_grad_gate.assert_called_once_with()
        model.hsdp_scheduler.register_all_gather_extension.assert_called_once()
        model.hsdp_scheduler.register_gradient_extension.assert_called_once()
        self.assertIsNone(configured.workspace)


    def test_qwen_passes_compact_thresholds_to_the_selected_executor(self) -> None:
        """Model configuration passes each threshold to the path that consumes it."""
        for use_megamoe in (False, True):
            with self.subTest(sparse=use_megamoe):
                model = SimpleNamespace(experts=torch.nn.Module(),
                                        gate=SimpleNamespace(num_experts=4, top_k=2))
                mesh = MagicMock()
                mesh.__getitem__.return_value.size.return_value = 2
                name = "SparseMoonEPExecutor" if use_megamoe else "MoonEPExecutor"
                with patch.object(qwen, name, autospec=True) as create, \
                        patch.object(qwen, "_install_bound_forward"):
                    qwen.qwen3moe_ep_compute_fn(
                        module=model, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
                        use_moonep=True, use_megamoe=use_megamoe,
                        moonep_token_dedup=True,
                        moonep_fwd_ratio=0.6, moonep_bwd_ratio=0.9)
                options = create.call_args.kwargs
                self.assertTrue(options["token_dedup"])
                self.assertEqual(options["forward_ratio"], 0.6)
                self.assertEqual(options["backward_ratio"], 0.9)
