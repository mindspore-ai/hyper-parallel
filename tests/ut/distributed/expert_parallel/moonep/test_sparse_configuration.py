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

import inspect
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

    def test_sparse_constructor_has_no_unused_prefetch_depth(self) -> None:
        """An obsolete keyword is rejected by the signature instead of stored unused."""
        self.assertNotIn("prefetch_depth", inspect.signature(sparse.SparseMoonEPExecutor).parameters)
        model = self._model()
        self.assertFalse(hasattr(model._moonep_executor, "prefetch_depth"))
        with self.assertRaisesRegex(TypeError, "prefetch_depth"):
            sparse.SparseMoonEPExecutor(model, Mock(), ep_group=object(), ep_size=2,
                                       num_experts=4, top_k=2, prefetch_depth=0)

    def test_compact_thresholds_are_user_configurable(self) -> None:
        """Sparse and nonfused modes retain compression unless the user selects another threshold."""
        model = self._model()
        self.assertFalse(model._moonep_executor.token_dedup)
        self.assertEqual(model._moonep_executor.forward_ratio, 0.75)
        self.assertEqual(model._moonep_executor.backward_ratio, 1.0)
        with patch.object(sparse.dist, "get_rank", return_value=0):
            configured = sparse.SparseMoonEPExecutor(
                model, Mock(), ep_group=object(), ep_size=2, num_experts=4, top_k=2,
                token_dedup=True,
                forward_ratio=0.6, backward_ratio=0.9)
            self.assertTrue(configured.token_dedup)
            self.assertEqual(configured.forward_ratio, 0.6)
            self.assertEqual(configured.backward_ratio, 0.9)
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                sparse.SparseMoonEPExecutor(
                    model, Mock(), ep_group=object(), ep_size=2, num_experts=4, top_k=2,
                    backward_ratio=-0.1)
        nonfused = runtime.MoonEPExecutor(
            model, Mock(), ep_group=None, ep_size=1, num_experts=2, top_k=1,
            token_dedup=True, forward_ratio=0.5, backward_ratio=0.9)
        self.assertTrue(nonfused.token_dedup)
        self.assertEqual(nonfused.forward_ratio, 0.5)
        self.assertEqual(nonfused.backward_ratio, 0.9)

    def test_small_experts_do_not_reserve_large_replica_staging(self) -> None:
        """Keep the original 64 MiB quota for production shapes and shrink tiny crops."""
        small_gate = torch.empty((4, 128, 512), device="meta")
        small_down = torch.empty((4, 256, 128), device="meta")
        large_gate = torch.empty((8, 2048, 1536), device="meta")
        large_down = torch.empty((8, 768, 2048), device="meta")
        self.assertEqual(sparse._transfer_quota_bytes(small_gate, small_down, 4), 2 * 1024 * 1024)
        self.assertEqual(sparse._transfer_quota_bytes(large_gate, large_down, 8), 64 * 1024 * 1024)

    def test_sparse_explicit_depth_fails_before_runtime_or_hook_binding(self) -> None:
        """Zero, positive and invalid values cannot be silently accepted in either mode."""
        for fsdp in (False, True):
            for depth in (0, 1, 3, -1, False, 1.5, "1"):
                with self.subTest(fsdp=fsdp, depth=depth):
                    model = self._model(fsdp)
                    with patch.object(runtime, "SparseMoonEPRuntime") as create:
                        with self.assertRaisesRegex(ValueError, "FSDP 通知决定预取"):
                            runtime.configure_moonep(model, prefetch_depth=depth)
                    create.assert_not_called()
                    self.assertIsNone(model._moonep_executor.runtime)
                    self.assertEqual(model._moonep_executor.layer, -1)
                    self.assertFalse(hasattr(model, "_moonep_runtime"))
                    if fsdp:
                        model.hsdp_scheduler.enable_param_grad_gate.assert_not_called()
                        model.hsdp_scheduler.register_all_gather_extension.assert_not_called()
                        model.hsdp_scheduler.register_gradient_extension.assert_not_called()

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

    def test_sparse_standalone_none_keeps_lazy_on_demand_runtime(self) -> None:
        """None retains its public meaning of an unspecified legacy-only option."""
        model = self._model()
        configured = runtime.configure_moonep(model, prefetch_depth=None)
        self.assertIsInstance(configured, sparse.SparseMoonEPRuntime)
        self.assertIsNone(configured.workspace)
        self.assertEqual(configured.tickets, {})
        self.assertEqual(configured.groups, ())

    def test_legacy_standalone_depth_remains_functional(self) -> None:
        """The compatibility option still selects the original script prefetch mode."""
        for depth in (0, 1, 3):
            with self.subTest(depth=depth):
                model = torch.nn.Module()
                model._moonep_executor = runtime.MoonEPExecutor(
                    model, Mock(), ep_group=None, ep_size=1, num_experts=2, top_k=1)
                with patch.object(runtime, "MoonEPRuntime") as create:
                    configured = runtime.configure_moonep(model, prefetch_depth=depth)
                create.assert_called_once_with([model._moonep_executor], standalone_prefetch=bool(depth))
                self.assertIs(configured, create.return_value)

    def test_qwen_passes_prefetch_depth_only_to_legacy_executor(self) -> None:
        """The adapter's preserved legacy option never reaches the sparse constructor."""
        for use_megamoe in (False, True):
            for depth in (0, 1):
                with self.subTest(sparse=use_megamoe, depth=depth):
                    model = SimpleNamespace(experts=torch.nn.Module(),
                                            gate=SimpleNamespace(num_experts=4, top_k=2))
                    mesh = MagicMock()
                    mesh.__getitem__.return_value.size.return_value = 2
                    name = "SparseMoonEPExecutor" if use_megamoe else "MoonEPExecutor"
                    with patch.object(qwen, name, autospec=True) as create, \
                            patch.object(qwen, "_install_bound_forward"):
                        result = qwen.qwen3moe_ep_compute_fn(
                            module=model, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
                            use_moonep=True, use_megamoe=use_megamoe, moonep_prefetch_depth=depth)
                    self.assertIs(result, qwen._qwen_moonep_compute)
                    options = create.call_args.kwargs
                    self.assertEqual(options["ep_size"], 2)
                    if use_megamoe:
                        self.assertNotIn("prefetch_depth", options)
                        self.assertNotIn("activation", options)
                    else:
                        self.assertEqual(options["prefetch_depth"], depth)
                        self.assertEqual(options["activation"], "swiglu")

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
