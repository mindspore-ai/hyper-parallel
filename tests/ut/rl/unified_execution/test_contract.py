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
"""Shared execution plan, worker identity and numerical policy."""
from __future__ import annotations

import copy
import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn
from transformers import Qwen3Config

from hyper_parallel.components.parallelism_invariant.utils import batch_invariant as level3
from hyper_parallel.models.replacement import module_replacement
from rl.unified_execution import batch_invariant as bi
from rl.unified_execution.contract import build_unified_payload, fingerprint, parse_unified_payload
from rl.unified_execution.execution_plan import EXPERTS_TARGET
from rl.unified_execution.vllm.model_builder import build_inference_model
from rl.unified_execution.models.structure import capture_training_structure
from rl.unified_execution.vllm.parallel_context import build_parallel_context


def _expert_plan():
    return [{"match": "*.mlp.experts", "replace": {"kind": "module",
             "implementation": {"_target_": EXPERTS_TARGET}}}]


class MarkerLinear(nn.Linear):
    """Observable replacement with identical linear arithmetic and state names."""


@module_replacement
def replace_linear(*, module: nn.Module, module_fqn: str, context: dict) -> nn.Module:
    """Exercise HP's actual decorated factory with no optional device dependency."""
    del module_fqn, context
    result = MarkerLinear(module.in_features, module.out_features, bias=module.bias is not None)
    result.weight = module.weight
    result.bias = module.bias
    return result


def _config() -> dict:
    return {
        "consistency": {"schema_version": 1, "mode": "architecture"},
        "model": {"name": "qwen3"},
        "train": {"plan_overrides": []},
        "rollout": {"vllm": {"enforce_eager": True}},
    }


def _replacement() -> dict:
    return {"match": "lm_head", "replace": {
        "kind": "module", "implementation": {"_target_": f"{__name__}.replace_linear"},
    }}


def _batch_invariant_config() -> dict:
    config = _config()
    config["consistency"].update({
        "mode": "batch_invariant", "attention_backend": "flash_attention_npu_v4",
        "batch_invariant": {"provider": "torch_npu_level3", "fill_uninitialized_memory": True},
    })
    return config


def _tiny_config() -> Qwen3Config:
    return Qwen3Config.from_dict({
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "tie_word_embeddings": True,
    })


class TestUnifiedExecutionContract(unittest.TestCase):
    """Serialization cannot silently alter the Actor's computation selection."""

    def test_v4_selection_is_shared_and_fingerprinted(self) -> None:
        config = _config()
        config["consistency"]["attention_backend"] = "flash_attention_npu_v4"
        config["rollout"]["vllm"]["enforce_eager"] = False
        payload = build_unified_payload(config)
        self.assertEqual(payload["attention_backend"], "flash_attention_npu_v4")
        self.assertEqual(parse_unified_payload(payload), payload)
        payload["attention_backend"] = "native"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            parse_unified_payload(payload)


class TestBatchInvariantContract(unittest.TestCase):
    """The numerical policy must survive transport without weakening existing guards."""

    def test_worker_rejects_policy_tampering_even_with_recomputed_digest(self) -> None:
        """Unsupported versions and incomplete derived policies are never trusted."""
        mutations = (("provider", "ascendc_v2"),
                     ("fill_uninitialized_memory", 1), ("level", 1),
                     ("torch_version", "2.9.0"), ("torch_npu_version", "2.10.0.post8"),
                     ("extra", True))
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                payload = build_unified_payload(_batch_invariant_config())
                payload["batch_invariant"][field] = value
                payload.pop("plan_fingerprint")
                payload["plan_fingerprint"] = fingerprint(payload)
                with self.assertRaises(ValueError):
                    parse_unified_payload(payload)
        for field in build_unified_payload(_batch_invariant_config())["batch_invariant"]:
            with self.subTest(missing=field):
                payload = build_unified_payload(_batch_invariant_config())
                del payload["batch_invariant"][field]
                with self.assertRaises(ValueError):
                    parse_unified_payload(payload)


class TestUnifiedQwen3Build(unittest.TestCase):
    """Real shared model construction retains computation and tied parameter identities."""

    def setUp(self) -> None:
        """Keep public builder tests on CPU regardless of host accelerator availability."""
        device = patch(
            "hyper_parallel.models._transformers.auto_model._current_device", return_value=torch.device("cpu"),
        )
        device.start()
        self.addCleanup(device.stop)

    def test_real_replacement_runs_and_fingerprint_detects_missing_side(self) -> None:
        """Check real module replacement and detect a mismatched execution plan."""
        config = _config()
        config["train"]["plan_overrides"] = [_replacement()]
        payload = build_unified_payload(config)
        model = build_inference_model(_tiny_config(), payload["plan_overrides"])
        self.assertIsInstance(model.lm_head, MarkerLinear)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        report = capture_training_structure(model, payload)
        payload["training_structure"] = report
        self.assertEqual(parse_unified_payload(payload)["training_structure"], report)
        output = model(torch.tensor([[1, 2, 3]]), use_cache=False)
        self.assertEqual(tuple(output.logits.shape), (1, 3, 32))
        unconverted = build_inference_model(_tiny_config(), [])
        with self.assertRaisesRegex(ValueError, "did not apply"):
            capture_training_structure(unconverted, payload)


class TestParallelContract(unittest.TestCase):
    """Keep complete training declarations without inventing inference CP or PP."""

    @staticmethod
    def _config() -> dict:
        return {
            "consistency": {"schema_version": 1, "mode": "architecture"},
            "model": {"name": "qwen3"},
            "train": {"accelerator": {"tp": 2, "cp": 1, "pp": 1, "sequence_parallel": True},
                      "plan_overrides": [{"match": "*.mlp.down_proj", "params": {"weight": [None, "tp"]}}]},
            "rollout": {"vllm": {"enforce_eager": True}},
        }

    def test_explicit_inference_topology_does_not_mutate_training_topology(self) -> None:
        """Explicit CP/PP=1 permits a different training topology without claiming parity."""
        config = self._config()
        config["train"]["accelerator"].update(pp=2)
        config["rollout"]["vllm"].update(tensor_parallel_size=4, context_parallel_size=1,
                                       pipeline_parallel_size=1, sequence_parallel=False)
        payload = parse_unified_payload(build_unified_payload(config))
        self.assertEqual(payload["training_parallelism"], config["train"]["accelerator"])
        self.assertEqual(payload["parallelism"], {"tp": 4, "cp": 1, "pp": 1, "sequence_parallel": False})
        tampered = copy.deepcopy(payload)
        tampered["parallelism"]["tp"] = 8
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            parse_unified_payload(tampered)


class TestParallelWorkerMesh(unittest.TestCase):
    """HyperParallel adopts the process group initialized by vLLM."""

    @staticmethod
    def _config(tp: int = 4) -> SimpleNamespace:
        return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
                               device_config=SimpleNamespace(device="cpu"))

    def test_ep_mesh_uses_synchronized_dp_tp_domain(self) -> None:
        """HP EP spans the actual MoE ranks without enabling a second expert implementation."""
        config = self._config(tp=1)
        config.parallel_config.data_parallel_size = 2
        config.parallel_config.data_parallel_rank = 1
        config.model_config = SimpleNamespace(enforce_eager=True, is_moe=True)
        payload = {"parallelism": {"tp": 1, "cp": 1, "pp": 1},
                   "expert_parallelism": {"training": 2, "inference": 2}, "plan_overrides": _expert_plan()}
        fake_vllm = ModuleType("vllm.distributed")
        tp_group, dp_group = object(), object()
        fake_vllm.get_tp_group = lambda: SimpleNamespace(ranks=[1], device_group=tp_group)
        fake_vllm.get_dp_group = lambda: SimpleNamespace(ranks=[0, 1], device_group=dp_group)
        fake_vllm.get_ep_group = lambda: SimpleNamespace(ranks=[0, 1])
        tp_mesh, dense_mesh, expert_mesh = MagicMock(), MagicMock(), MagicMock()
        tp_mesh.__getitem__.return_value.get_local_rank.return_value = 0
        dense_mesh.__getitem__.return_value.get_local_rank.return_value = 1
        expert_mesh.__getitem__.return_value.get_local_rank.return_value = 1
        with patch.dict("sys.modules", {"vllm.distributed": fake_vllm}), \
                patch("rl.unified_execution.vllm.parallel_context.dist.is_initialized", return_value=True), \
                patch("rl.unified_execution.vllm.parallel_context.DeviceMesh.from_group",
                      side_effect=[tp_mesh, dense_mesh]) as from_group, \
                patch("rl.unified_execution.vllm.parallel_context.build_expert_mesh",
                      return_value=expert_mesh) as build_mesh:
            result = build_parallel_context(config, payload)
        from_group.assert_called_with([dp_group, tp_group], "cpu", mesh=[[0], [1]], mesh_dim_names=("dp", "tp"))
        build_mesh.assert_called_once_with(dense_mesh, 2)
        self.assertEqual((result.dp_rank, result.ep_rank, result.tp_rank), (1, 1, 0))
        self.assertIs(result.mesh_context.fsdp_moe_mesh, expert_mesh)


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
        self.stack.enter_context(patch.object(level3, "_PROCESS_STATE", None))
        self.stack.enter_context(patch.object(level3, "torch", self.torch))
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
