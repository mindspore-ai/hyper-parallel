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
"""Shared-plan transport through production vLLM launch and publication hooks."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from rl.config.resolve import resolve_unified_execution_config
from rl.rollout import vllm_plugin
from rl.rollout.vllm import VLLMGenerationEngine
from rl.unified_execution.contract import build_unified_payload
from rl.weight_sync import vllm_worker
from rl.weight_sync.vllm_worker import _finish_custom_weight_update, _load_packed_weights
from rl.workers.engine.model import UNIFIED_MODEL_ARCHITECTURES, UNIFIED_QWEN3_ARCHITECTURE, ModelRegistration


def _configuration():
    return {
        "consistency": {"schema_version": 1, "mode": "architecture"},
        "model": {"name": "qwen3", "attn_implementation": "sdpa"},
        "train": {"accelerator": {"tp": 1}, "plan_overrides": [{
            "match": "*.input_layernorm",
            "replace": {"kind": "module", "implementation": {"_target_":
                "hyper_parallel.models.qwen3.adapter.conversion.module_replacement.replace_qwen3_rms_norm"}},
        }]},
        "rollout": {"engine": "vllm", "vllm": {
            "enforce_eager": True, "tensor_parallel_size": 1, "data_parallel_size": 1,
            "deployment": "colocated",
        }},
    }


def _model():
    return ModelRegistration("qwen", "qwen3", "/model", "/tokenizer", "Qwen3ForCausalLM",
                             "qwen3", "qwen3", True)


class TestUnifiedRuntime(unittest.TestCase):
    """Exercise configuration flow without a device or inference process."""

    def test_same_plan_reaches_cli_and_worker_environment(self):
        """The launch payload comes from the Actor plan and preserves sleep settings."""
        original = _configuration()
        resolved = resolve_unified_execution_config(original)
        self.assertNotIn("model_implementation", original["rollout"]["vllm"])
        for deployment in ("colocated", "disjoint"):
            with self.subTest(deployment=deployment):
                resolved["rollout"]["vllm"]["deployment"] = deployment
                engine = VLLMGenerationEngine(_model(), resolved["rollout"])
                command = engine._server_command("127.0.0.1", 8100)
                extra = json.loads(command[command.index("--additional-config") + 1])
                architecture = json.loads(command[command.index("--hf-overrides") + 1])
                self.assertEqual(architecture["architectures"], [UNIFIED_QWEN3_ARCHITECTURE])
                self.assertEqual(extra["weight_nz_mode"], 0)
                self.assertNotIn("hyper_rl_unified_execution", extra)
                self.assertEqual(architecture["hyper_rl_unified_execution"]["plan_overrides"],
                                 original["train"]["plan_overrides"])
                self.assertEqual("--enable-sleep-mode" in command, deployment == "colocated")
                self.assertEqual(engine._server_environment("3")["HYPER_RL_UNIFIED_EXECUTION"], "1")

    def test_level3_policy_reaches_worker_without_global_vllm_patch(self):
        """Shared numerical policy retains prefix, chunked and decode graph settings."""
        original = _configuration()
        original["consistency"].update({
            "mode": "batch_invariant", "attention_backend": "flash_attention_npu_v4",
            "batch_invariant": {"provider": "torch_npu_level3", "fill_uninitialized_memory": True},
        })
        original["rollout"]["vllm"].update({
            "enforce_eager": False, "batch_invariant": False,
            "enable_prefix_caching": True, "enable_chunked_prefill": True,
            "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        })
        resolved = resolve_unified_execution_config(original)
        engine = VLLMGenerationEngine(_model(), resolved["rollout"])
        command = engine._server_command("127.0.0.1", 8100)
        overrides = json.loads(command[command.index("--hf-overrides") + 1])
        policy = overrides["hyper_rl_unified_execution"]["batch_invariant"]
        self.assertEqual(policy["provider"], "torch_npu_level3")
        self.assertIs(policy["fill_uninitialized_memory"], True)
        self.assertEqual(policy["torch_npu_version"], "2.10.0.post7")
        self.assertEqual(overrides["hyper_rl_unified_execution"], resolved["rollout"]["vllm"]["unified_execution"])
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertNotIn("--enforce-eager", command)
        self.assertEqual(json.loads(command[command.index("--compilation-config") + 1]),
                         {"cudagraph_mode": "FULL_DECODE_ONLY"})
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}):
            self.assertEqual(engine._server_environment("3")["VLLM_BATCH_INVARIANT"], "0")
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                engine._server_environment("3")

    def test_plugin_registers_lazy_architecture_in_worker(self):
        """The existing general plugin registers the exact import path in each process."""
        registry = Mock()
        versions = {"vllm": "0.23.0+empty", "vllm-ascend": "0.23.0.post1"}
        with patch.dict(os.environ, {"HYPER_RL_UNIFIED_EXECUTION": "1"}, clear=True), \
                patch.dict("sys.modules", {"vllm": SimpleNamespace(ModelRegistry=registry)}), \
                patch.object(vllm_plugin, "package_version", side_effect=versions.__getitem__), \
                patch.object(vllm_plugin, "install_vllm_weight_sync_hooks"), \
                patch.object(vllm_plugin, "install_tool_evidence"), \
                patch.object(vllm_plugin, "install_partial_prefill_rng_fix"):
            vllm_plugin.install_runtime_hooks()
        self.assertEqual(registry.register_model.call_args_list, [
            call(name, f"rl.unified_execution.runtimes.vllm.model_entrypoints:{name}")
            for name in ("HyperParallelQwen3ForCausalLM", "HyperParallelQwen3MoeForCausalLM")
        ])

    def test_incomplete_model_update_cannot_commit_version(self):
        """Missing model weights leave the worker transaction active and unpublished."""
        for architecture in UNIFIED_MODEL_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = Mock()
                model.finish_weight_update.side_effect = ValueError("missing QKV slice")
                worker = SimpleNamespace(
                    _check_weight_transfer_engine=Mock(), _weight_update_active=True,
                    _rl_pending_policy_version=2, _rl_loaded_policy_version=1,
                    model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
                    model_runner=SimpleNamespace(get_model=lambda model=model: model),
                )
                with self.assertRaisesRegex(ValueError, "missing QKV slice"):
                    _finish_custom_weight_update(worker)
                self.assertTrue(worker._weight_update_active)
                self.assertEqual(worker._rl_loaded_policy_version, 1)
                model.finish_weight_update.side_effect = None
                _finish_custom_weight_update(worker)
                self.assertFalse(worker._weight_update_active)
                self.assertEqual(worker._rl_loaded_policy_version, 2)

    def test_unified_worker_rejects_unaccepted_nonempty_bucket(self):
        """A unified TP worker must accept weights instead of acknowledging an empty load."""
        for architecture in UNIFIED_MODEL_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = Mock()
                model.load_weights.return_value = set()
                worker = SimpleNamespace(
                    model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
                    model_runner=SimpleNamespace(get_model=lambda model=model: model),
                )
                weights = [("model.norm.weight", object())]
                with patch("rl.weight_sync.vllm_worker.unpack_packed_weights", return_value=weights):
                    with self.assertRaisesRegex(RuntimeError, "did not accept any parameter"):
                        _load_packed_weights(worker, object(), [])


def _engine(mode: str | None = "batch_invariant") -> VLLMGenerationEngine:
    """Build an isolated rollout engine with an explicit numerical policy."""
    model = ModelRegistration("qwen", "qwen3", "/model", "/tokenizer", "Qwen3ForCausalLM",
                              "qwen3", "qwen3", True)
    config = {
        "consistency": {"schema_version": 1, "mode": mode},
        "model": {"name": "qwen3"}, "train": {"plan_overrides": []},
        "rollout": {"vllm": {"deployment": "disjoint", "enforce_eager": True,
                              "data_parallel_size": 2, "tensor_parallel_size": 2}},
    }
    if mode == "batch_invariant":
        config["consistency"].update({
            "attention_backend": "flash_attention_npu_v4",
            "batch_invariant": {"provider": "torch_npu_level3", "fill_uninitialized_memory": True},
        })
    if mode is not None:
        payload = build_unified_payload(config)
        config["rollout"]["vllm"].update(model_implementation="unified", unified_execution=payload)
    return VLLMGenerationEngine(model, config["rollout"])


def _snapshot(engine: VLLMGenerationEngine) -> dict:
    policy = engine._unified_execution["batch_invariant"]
    return {
        "policy": dict(policy),
        "state": {"provider": policy["provider"], "level": 3, "deterministic_algorithms": True,
                  "fill_uninitialized_memory": True, "torch_version": "2.10.0+cpu",
                  "torch_npu_version": "2.10.0.post7", "pid": 100, "device": "npu:0"},
    }


class TestNumericalPolicyWorkerRPC(unittest.TestCase):
    """State RPCs expose verified model state and preserve all DP worker responses."""

    def test_rpc_delegates_unified_and_does_not_touch_native_models(self) -> None:
        """The model verifier owns state readback and its errors propagate unchanged."""
        model = Mock()
        report = {"policy": {}, "state": {}}
        model.verify_numerical_policy.return_value = report
        worker = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[UNIFIED_QWEN3_ARCHITECTURE])),
            model_runner=SimpleNamespace(get_model=Mock(return_value=model)),
        )
        self.assertIs(vllm_worker.get_unified_numerical_policy(worker), report)
        model.verify_numerical_policy.assert_called_once_with()
        model.verify_numerical_policy.side_effect = RuntimeError("policy drift")
        with self.assertRaisesRegex(RuntimeError, "policy drift"):
            vllm_worker.get_unified_numerical_policy(worker)
        worker.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
        worker.model_runner.get_model.reset_mock()
        self.assertIsNone(vllm_worker.get_unified_numerical_policy(worker))
        worker.model_runner.get_model.assert_not_called()

    def test_dp_rpc_collects_every_engine_and_keeps_other_methods_unchanged(self) -> None:
        """The numerical RPC keeps both TP replies from each of two DP engines."""
        async def original(unused_client: object, method: str, timeout: float | None = None,
                           args: tuple = (), kwargs: dict | None = None) -> list:
            """Stand in for the unpatched vLLM route for unrelated methods."""
            del unused_client
            del method, timeout, args, kwargs
            return ["original"]

        client_type = type("DPLBAsyncMPClient", (), {"collective_rpc_async": original})
        client = client_type()
        client.core_engines = (0, 1)
        utility_call = AsyncMock(side_effect=[["dp0tp0", "dp0tp1"], ["dp1tp0", "dp1tp1"]])
        client._call_utility_async = utility_call
        with patch.dict("sys.modules", {
            "vllm.v1.engine.core_client": SimpleNamespace(DPLBAsyncMPClient=client_type),
        }), patch.object(vllm_worker, "_patch_state", SimpleNamespace(dp_rpc=False)):
            vllm_worker._patch_dp_weight_rpc()
            results = asyncio.run(client.collective_rpc_async("get_unified_numerical_policy"))
            self.assertEqual(results, ["dp0tp0", "dp0tp1", "dp1tp0", "dp1tp1"])
            self.assertEqual(asyncio.run(client.collective_rpc_async("other_method")), ["original"])
        self.assertEqual(utility_call.await_count, 2)


class TestNumericalPolicyStartupHandshake(unittest.TestCase):
    """Startup accepts only complete, matching requested and actual worker policies."""

    def test_matching_workers_are_saved_with_process_local_identity(self) -> None:
        """PID and device differ legitimately, while provider state and versions agree."""
        engine = _engine()
        local = _snapshot(engine)
        reports = [deepcopy(local) for _ in range(4)]
        for index, report in enumerate(reports):
            report["state"].update(pid=200 + index, device=f"npu:{index}")
        client = Mock()
        client.collective_rpc.return_value = reports
        with patch("rl.rollout.vllm.verify_batch_invariant", return_value=local["state"]) as verify:
            engine._verify_worker_numerical_policies(client)
        verify.assert_called_once_with(local["policy"])
        client.collective_rpc.assert_called_once_with("get_unified_numerical_policy")
        self.assertEqual(engine._unified_numerical_policy_snapshots, reports)

    def test_missing_or_mismatched_workers_fail_closed(self) -> None:
        """Partial DP results, tampered policies, booleans and version drift all fail."""
        engine = _engine()
        local = _snapshot(engine)
        valid = [deepcopy(local) for _ in range(4)]
        cases = [[], valid[:2], None, [None] * 4]
        for section, field, value in (
                ("policy", "fill_uninitialized_memory", 1), ("policy", "provider", "other"),
                ("state", "level", 1), ("state", "level", 3.0),
                ("state", "deterministic_algorithms", False), ("state", "deterministic_algorithms", 1),
                ("state", "fill_uninitialized_memory", False), ("state", "provider", "other"),
                ("state", "torch_version", "2.10.0+other"), ("state", "torch_npu_version", "2.10.0.post8")):
            reports = deepcopy(valid)
            reports[2][section][field] = value
            cases.append(reports)
        for index, reports in enumerate(cases):
            with self.subTest(case=index):
                client = Mock()
                client.collective_rpc.return_value = reports
                with patch("rl.rollout.vllm.verify_batch_invariant", return_value=local["state"]):
                    with self.assertRaisesRegex(RuntimeError, "numerical"):
                        engine._verify_worker_numerical_policies(client)
                self.assertIsNone(engine._unified_numerical_policy_snapshots)

    def test_owned_launch_checks_after_readiness_and_closes_on_mismatch(self) -> None:
        """Numerical handshake failure uses the existing owned-process startup cleanup."""
        engine = _engine()
        local = _snapshot(engine)
        client = Mock()
        events = []
        client.wait_ready.side_effect = lambda _: events.append("ready")
        client.collective_rpc.side_effect = lambda _: events.append("rpc") or []
        with patch("rl.rollout.vllm.socket.create_connection", side_effect=OSError), \
                patch("rl.rollout.vllm.subprocess.Popen"), \
                patch("rl.rollout.vllm._VLLMHTTPClient", return_value=client), \
                patch.object(engine, "_server_command", return_value=["vllm"]), \
                patch.object(engine, "_server_environment", return_value={}), \
                patch("rl.rollout.vllm.verify_batch_invariant", return_value=local["state"]):
            with self.assertRaisesRegex(RuntimeError, "worker snapshots"):
                engine._launch_client("0,1,2,3", "127.0.0.1", 8100)
        self.assertEqual(events, ["ready", "rpc"])
        client.close.assert_called_once_with()
