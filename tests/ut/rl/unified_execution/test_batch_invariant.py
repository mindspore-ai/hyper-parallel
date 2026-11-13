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
"""CPU tests for process policy ownership, immutable verification and fail-closed setup."""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rl.unified_execution import batch_invariant as bi
from rl.unified_execution.batch_invariant import normalize_policy
from rl.unified_execution.contract import build_unified_payload
from rl.unified_execution.runtimes.hyper_parallel.training import guard_training_numerics
from rl.unified_execution.runtimes.vllm.model_adapter import HyperParallelQwen3ForCausalLM
from rl.unified_execution.runtimes.vllm.parallel_context import InferenceParallelContext
from rl.weight_sync import vllm_worker
from rl.workers.engine.model import UNIFIED_QWEN3_ARCHITECTURE


class TestBatchInvariant(unittest.TestCase):
    """Exercise a fake NPU without mutating real Torch deterministic state."""

    def setUp(self) -> None:
        """Install isolated fake process state and register its cleanup."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.events = []
        self.level = 0
        self.deterministic = False
        self.fill = SimpleNamespace(fill_uninitialized_memory=False)
        self.torch = Mock(
            __version__="2.10.0+cpu",
            utils=SimpleNamespace(deterministic=self.fill),
            are_deterministic_algorithms_enabled=Mock(side_effect=lambda: self.deterministic),
            no_grad=Mock(side_effect=nullcontext),
            mm=Mock(side_effect=lambda *_args: self.events.append("mm")),
        )
        self.backend = SimpleNamespace(
            __version__="2.10.0.post7",
            npu=SimpleNamespace(
                is_available=Mock(return_value=True), is_initialized=Mock(return_value=True),
                current_device=Mock(return_value=2),
                set_deterministic_level=Mock(side_effect=self._set_level),
                synchronize=Mock(side_effect=lambda: self.events.append("synchronize")),
                _get_deterministic_level=Mock(side_effect=AssertionError("mutating Python getter")),
            ),
            _C=SimpleNamespace(_npu_get_deterministic_level=Mock(side_effect=lambda: self.level)),
        )
        self.stack.enter_context(patch.object(bi, "_PROCESS_STATE", None))
        self.stack.enter_context(patch.object(bi, "torch", self.torch))
        self.stack.enter_context(patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}))
        self.stack.enter_context(patch.dict(sys.modules, {
            "torch_npu": self.backend,
            "vllm_ascend": SimpleNamespace(),
            "vllm_ascend.batch_invariant": SimpleNamespace(_batch_invariant_LIB=None),
        }))

    def _set_level(self, level: int) -> None:
        self.events.append("set_level")
        self.level = level
        self.deterministic = level > 0

    def test_normalization_is_versioned_without_importing_backend(self) -> None:
        """CPU configuration fixes provider identity without loading optional dependencies."""
        with patch.object(bi.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            policy = bi.normalize_policy({})
            self.assertEqual(bi.normalize_policy(policy), policy)
        self.assertEqual(policy["torch_version"], "2.10.0")
        self.assertEqual(policy["torch_npu_version"], "2.10.0.post7")
        self.assertIs(policy["fill_uninitialized_memory"], False)
        policy["level"] = 1
        self.assertEqual(bi.normalize_policy({})["level"], 3)

    def test_unsupported_policy_fields_and_values_are_rejected(self) -> None:
        """Alternate providers and non-boolean fill values fail before installation."""
        cases = [None, [], {"provider": "ascendc_v2"}, {"fill_uninitialized_memory": "false"},
                 {"fill_uninitialized_memory": 1}, {"level": True}, {"level": 1},
                 {"torch_version": "2.11.0"}, {"torch_npu_version": "2.10.0"}, {"unknown": True}]
        for policy in cases:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                bi.normalize_policy(policy)

    def test_installation_applies_options_before_one_startup_gemm(self) -> None:
        """Initialization disables unused-memory filling and forces CANN submission once."""
        snapshot = bi.initialize_batch_invariant({})
        self.assertEqual(self.events, ["set_level", "mm", "synchronize"])
        self.assertEqual(snapshot["device"], "npu:2")
        self.assertEqual(snapshot["level"], 3)
        self.assertIs(snapshot["deterministic_algorithms"], True)
        self.assertIs(snapshot["fill_uninitialized_memory"], False)
        self.assertEqual(snapshot["torch_version"], "2.10.0+cpu")
        self.assertEqual(self.torch.ones.call_count, 2)
        self.assertEqual(bi.initialize_batch_invariant({}), snapshot)
        self.assertEqual(self.events, ["set_level", "mm", "synchronize"])

    def test_explicit_fill_on_remains_available(self) -> None:
        """An explicit filled-memory baseline overrides the optimized default."""
        policy = {"fill_uninitialized_memory": True}
        snapshot = bi.initialize_batch_invariant(policy)
        self.assertIs(snapshot["fill_uninitialized_memory"], True)
        self.assertEqual(bi.verify_batch_invariant(policy), snapshot)
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            bi.initialize_batch_invariant({})

    def test_verify_never_installs_or_changes_state(self) -> None:
        """Missing installation fails before any backend import or setter call."""
        with patch.object(bi.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            with self.assertRaisesRegex(RuntimeError, "not completed initialization"):
                bi.verify_batch_invariant({})
        self.backend.npu.set_deterministic_level.assert_not_called()
        self.assertIs(self.fill.fill_uninitialized_memory, False)
        bi.initialize_batch_invariant({})
        events = list(self.events)
        with patch.object(bi.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            bi.verify_batch_invariant({})
        self.assertEqual(self.events, events)
        self.backend.npu._get_deterministic_level.assert_not_called()

    def test_state_drift_fails_without_repair_or_another_gemm(self) -> None:
        """Raw level, Torch flag, fill and selected device must all remain fixed."""
        bi.initialize_batch_invariant({})
        for name, target, value in (
            ("level", self, 1), ("deterministic", self, False),
            ("fill_uninitialized_memory", self.fill, True),
            ("return_value", self.backend.npu.current_device, 3),
        ):
            with self.subTest(name=name), patch.object(target, name, value):
                for operation in (bi.verify_batch_invariant, bi.initialize_batch_invariant):
                    with self.assertRaisesRegex(RuntimeError, "state drift"):
                        operation({})
                self.assertEqual(getattr(target, name), value)
        self.assertEqual(self.events, ["set_level", "mm", "synchronize"])

    def test_global_vllm_conflicts_are_rejected_before_and_after_install(self) -> None:
        """Neither an inherited environment flag nor an existing aten patch may coexist."""
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "VLLM_BATCH_INVARIANT"):
                bi.initialize_batch_invariant({})
        self.backend.npu.set_deterministic_level.assert_not_called()
        bi.initialize_batch_invariant({})
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "VLLM_BATCH_INVARIANT"):
                bi.verify_batch_invariant({})
        for name in ("vllm_ascend", "vllm_ascend.batch_invariant"):
            with self.subTest(name=name), patch.dict(sys.modules, {
                name: SimpleNamespace(_batch_invariant_LIB=object()),
            }):
                with self.assertRaisesRegex(RuntimeError, "aten patches"):
                    bi.initialize_batch_invariant({})
                with self.assertRaisesRegex(RuntimeError, "aten patches"):
                    bi.verify_batch_invariant({})
        self.assertEqual(self.backend.npu.set_deterministic_level.call_count, 1)

    def test_forked_child_does_not_accept_parent_installation(self) -> None:
        """The child must initialize its own selected NPU despite inherited Python globals."""
        parent = bi.initialize_batch_invariant({})
        with patch.object(bi.os, "getpid", return_value=parent["pid"] + 1):
            with self.assertRaisesRegex(RuntimeError, "not completed initialization"):
                bi.verify_batch_invariant({})
            child = bi.initialize_batch_invariant({})
        self.assertNotEqual(parent["pid"], child["pid"])
        self.assertEqual(self.backend.npu.set_deterministic_level.call_count, 2)

    def test_failed_cann_submission_does_not_produce_a_ready_policy(self) -> None:
        """A getter value alone cannot hide a failure applying options on the device."""
        self.backend.npu.synchronize.side_effect = RuntimeError("CANN submit failed")
        with self.assertRaisesRegex(RuntimeError, "CANN submit failed"):
            bi.initialize_batch_invariant({})
        with self.assertRaisesRegex(RuntimeError, "not completed initialization"):
            bi.verify_batch_invariant({})


def _policy() -> dict:
    return normalize_policy({"provider": "torch_npu_level3", "fill_uninitialized_memory": True})


class TestTrainingNumericalGuard(unittest.TestCase):
    """Training hooks must preserve model ownership and fail before computation."""

    def test_guard_preserves_parameters_and_installs_only_one_forward_check(self) -> None:
        """Repeated installation leaves Parameter and optimizer references intact."""
        model = nn.Linear(3, 2)
        parameters = tuple(model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        with patch("rl.unified_execution.runtimes.hyper_parallel.training.verify_batch_invariant") as verify:
            guard_training_numerics(model, _policy())
            guard_training_numerics(model, _policy())
            verify.reset_mock()
            model(torch.ones(2, 3)).sum().backward()
        verify.assert_called_once_with(_policy())
        for original, current, optimized in zip(parameters, model.parameters(), optimizer.param_groups[0]["params"]):
            self.assertIs(current, original)
            self.assertIs(optimized, original)
            self.assertIsNotNone(original.grad)

    def test_drift_rejects_forward_before_model_body(self) -> None:
        """An altered process policy cannot execute the guarded training model."""
        model = nn.Linear(3, 2)
        with patch("rl.unified_execution.runtimes.hyper_parallel.training.verify_batch_invariant") as verify:
            guard_training_numerics(model, _policy())
            verify.side_effect = RuntimeError("policy drift")
            with patch.object(model, "forward", wraps=model.forward) as forward:
                with self.assertRaisesRegex(RuntimeError, "policy drift"):
                    model(torch.ones(2, 3))
                forward.assert_not_called()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


class TestInferenceNumericalBoundaries(unittest.TestCase):
    """Use real Torch modules and the production wrapper with mocked device boundaries."""

    def setUp(self) -> None:
        """Keep model construction, attention and policy state on CPU test doubles."""
        self.events = []
        self.hp_model = nn.Module()
        self.hp_model.config = SimpleNamespace(tie_word_embeddings=False)
        self.hp_model.model = nn.Module()
        self.hp_model.model.layers = nn.ModuleList()
        self.hp_model.model.embed_tokens = nn.Embedding(8, 3)
        self.hp_model.model.norm = nn.Identity()
        self.hp_model.lm_head = nn.Linear(3, 8, bias=False)
        payload = build_unified_payload({
            "consistency": {"schema_version": 1, "mode": "batch_invariant",
                            "attention_backend": "flash_attention_npu_v4",
                            "batch_invariant": {"provider": "torch_npu_level3",
                                                "fill_uninitialized_memory": True}},
            "model": {"name": "qwen3"}, "train": {"plan_overrides": []},
            "rollout": {"vllm": {"enforce_eager": False}},
        })
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3", hyper_rl_unified_execution=payload), enforce_eager=False,
            ),
            additional_config={}, quant_config=None,
        )
        self.initialize = self._patch("initialize_batch_invariant", "initialize", {})
        self.verify = self._patch("verify_batch_invariant", "verify", {})
        self._patch("build_parallel_context", "parallel", InferenceParallelContext())
        self.build = self._patch("build_inference_model", "build", self.hp_model)
        self._patch("capture_training_structure", "structure", {"fingerprint": "structure"})
        self._patch("install_paged_attention", "attention", None)

    def _patch(self, name: str, event: str, result: object) -> Mock:
        def record(*_args: object, **_kwargs: object) -> object:
            """Record the dependency call and return its CPU stand-in."""
            self.events.append(event)
            return result

        patcher = patch(f"rl.unified_execution.runtimes.vllm.model_adapter.{name}", side_effect=record)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def test_initialize_precedes_model_and_verify_follows_attention_installation(self) -> None:
        """The wrapper applies policy before construction and checks completed setup."""
        wrapper = HyperParallelQwen3ForCausalLM(vllm_config=self.config)
        self.assertEqual(self.events, ["initialize", "parallel", "build", "structure", "attention", "verify"])
        self.initialize.assert_called_once_with(_policy())
        self.verify.assert_called_once_with(_policy())
        self.assertIs(wrapper.lm_head.weight, self.hp_model.lm_head.weight)
        self.assertIs(wrapper.model.embed_tokens.weight, self.hp_model.model.embed_tokens.weight)

    def test_publication_drift_preserves_weight_loader_transaction(self) -> None:
        """A failed policy check cannot start or finish a weight transaction."""
        for boundary, method, active in (("begin_weight_update", "begin", False),
                                         ("finish_weight_update", "finish", True)):
            with self.subTest(boundary=boundary):
                self.verify.side_effect = None
                wrapper = HyperParallelQwen3ForCausalLM(vllm_config=self.config)
                loader = wrapper.weight_loader
                if active:
                    loader.begin()
                seen = set(loader.seen)
                self.verify.side_effect = RuntimeError("policy drift")
                with patch.object(loader, method, wraps=getattr(loader, method)) as update:
                    with self.assertRaisesRegex(RuntimeError, "policy drift"):
                        getattr(wrapper, boundary)()
                    update.assert_not_called()
                self.assertEqual(loader.active, active)
                self.assertEqual(loader.seen, seen)


class TestWakeNumericalBoundary(unittest.TestCase):
    """Ascend wake must verify restored workers before refreshing decode graphs."""

    def test_wake_verifies_before_graph_refresh_and_propagates_drift(self) -> None:
        """Wake restores memory first, but drift blocks all later graph refresh work."""
        for tags in (None, ["weights"], ["kv_cache"]):
            for drift in (False, True):
                with self.subTest(tags=tags, drift=drift):
                    self._assert_wake_boundary(tags, drift)

    def _assert_wake_boundary(self, tags: list[str] | None, drift: bool) -> None:
        """Give each wake case an independent closure and isolated worker class."""
        events = []
        original_wake = Mock(side_effect=lambda *_: events.append("wake"))
        worker_type = type("NPUWorker", (), {
            "wake_up": original_wake, "start_weight_update": Mock(), "finish_weight_update": Mock(),
        })
        model = Mock()

        def verify() -> None:
            """Record the worker check and optionally report simulated policy drift."""
            events.append("verify")
            if drift:
                raise RuntimeError("policy drift")

        model.verify_numerical_policy.side_effect = verify
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(get_model=lambda: model),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(
                architectures=[UNIFIED_QWEN3_ARCHITECTURE])),
        )
        module = SimpleNamespace(NPUWorker=worker_type)
        with patch.dict("sys.modules", {"vllm_ascend.worker.worker": module}), \
                patch.object(vllm_worker, "_patch_state", SimpleNamespace(ascend_lifecycle=False)), \
                patch.object(vllm_worker, "_refresh_ascend_decode_graphs",
                             side_effect=lambda _: events.append("refresh")) as refresh:
            vllm_worker._patch_ascend_weight_update_lifecycle()
            if drift:
                with self.assertRaisesRegex(RuntimeError, "policy drift"):
                    worker_type.wake_up(worker, tags)
                refresh.assert_not_called()
            else:
                worker_type.wake_up(worker, tags)
        expected = ["wake", "verify"]
        if not drift and (tags is None or "kv_cache" in tags):
            expected.append("refresh")
        self.assertEqual(events, expected)
        original_wake.assert_called_once_with(worker, tags)
