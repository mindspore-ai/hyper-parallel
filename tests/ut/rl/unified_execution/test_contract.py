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
"""CPU contract and real HP replacement tests for architecture-mode inference."""

import copy
import unittest
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from rl.unified_execution.contract import build_unified_payload, fingerprint, parse_unified_payload
from rl.unified_execution.models import get_model_family
from rl.unified_execution.runtimes.hyper_parallel.builder import build_inference_model
from rl.unified_execution.runtimes.vllm.parallel_context import build_parallel_context, payload_from_vllm_config
from rl.unified_execution.validation import capture_training_structure
from rl.workers.engine.model import architecture_for_implementation
from torch import nn
from transformers import Qwen3Config

from hyper_parallel.models.replacement import module_replacement


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

    def test_v4_rollout_only_override_is_rejected(self) -> None:
        config = _config()
        config["consistency"]["attention_backend"] = "flash_attention_npu_v4"
        config["rollout"]["vllm"]["attention_backend"] = "FLASH_ATTN"
        with self.assertRaisesRegex(ValueError, "overridden"):
            build_unified_payload(config)

    def test_hf_transport_preserves_contract_and_rejects_obsolete_platform_payload(self) -> None:
        """Preserve the model-owned contract and reject the obsolete transport location."""
        payload = build_unified_payload(_config())
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(hyper_rl_unified_execution=payload)),
            additional_config={},
        )
        self.assertEqual(payload_from_vllm_config(config), payload)
        config.additional_config["hyper_rl_unified_execution"] = {**payload, "attention_backend": "other"}
        with self.assertRaisesRegex(ValueError, "hf_overrides"):
            payload_from_vllm_config(config)
        config.additional_config["hyper_rl_unified_execution"] = payload
        with self.assertRaisesRegex(ValueError, "hf_overrides"):
            payload_from_vllm_config(config)
        config.additional_config.clear()
        del config.model_config.hf_config.hyper_rl_unified_execution
        with self.assertRaisesRegex(ValueError, "payload"):
            payload_from_vllm_config(config)

    def test_full_plan_preserves_layout_declarations(self) -> None:
        config = _config()
        config["train"]["plan_overrides"] = [_replacement(), {"match": "*.self_attn", "region_dispatch": False}]
        payload = build_unified_payload(config)
        self.assertEqual(payload["plan_overrides"], config["train"]["plan_overrides"])
        self.assertEqual(payload["compute_plan_overrides"], [_replacement()])
        self.assertEqual(payload["excluded_plan_overrides"], [])
        self.assertEqual(parse_unified_payload(payload), payload)
        config["train"]["plan_overrides"].clear()
        self.assertEqual(len(payload["plan_overrides"]), 2)

    def test_training_precision_must_match_rollout(self) -> None:
        config = _config()
        config["train"]["mixed_precision"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "BF16"):
            build_unified_payload(config)


class TestBatchInvariantContract(unittest.TestCase):
    """The numerical policy must survive transport without weakening existing guards."""

    def test_level3_policy_is_normalized_fingerprinted_and_detached(self) -> None:
        """Both workers receive the requested policy and fixed provider identity."""
        config = _batch_invariant_config()
        payload = build_unified_payload(config)
        self.assertEqual(payload["batch_invariant"], {
            "provider": "torch_npu_level3", "fill_uninitialized_memory": True,
            "level": 3, "torch_version": "2.10.0", "torch_npu_version": "2.10.0.post7",
        })
        self.assertEqual(parse_unified_payload(payload), payload)
        architecture = _config()
        architecture["consistency"]["attention_backend"] = "flash_attention_npu_v4"
        self.assertNotEqual(payload["plan_fingerprint"], build_unified_payload(architecture)["plan_fingerprint"])
        config["consistency"]["batch_invariant"]["fill_uninitialized_memory"] = False
        self.assertIs(payload["batch_invariant"]["fill_uninitialized_memory"], True)

    def test_batch_invariant_requires_v4_and_explicit_policy(self) -> None:
        """Mode selection alone cannot opt into an unspecified numerical path."""
        for field in ("attention_backend", "batch_invariant"):
            with self.subTest(field=field):
                config = _batch_invariant_config()
                del config["consistency"][field]
                with self.assertRaises(ValueError):
                    build_unified_payload(config)

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

    def test_batch_invariant_retains_parallel_and_graph_options(self) -> None:
        """Existing TP/SP, prefix cache, chunked prefill and decode graph remain configurable."""
        config = _batch_invariant_config()
        config["train"]["accelerator"] = {"tp": 2, "sequence_parallel": True}
        config["rollout"]["vllm"].update({
            "enforce_eager": False, "enable_prefix_caching": True, "enable_chunked_prefill": True,
            "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        })
        payload = build_unified_payload(config)
        self.assertEqual(payload["parallelism"], {
            "tp": 2, "cp": 1, "pp": 1, "sequence_parallel": True,
        })
        self.assertEqual(parse_unified_payload(payload), payload)


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
        config["train"]["accelerator"].update(cp=2, pp=2)
        config["rollout"]["vllm"].update(tensor_parallel_size=4, context_parallel_size=1,
                                       pipeline_parallel_size=1, sequence_parallel=False)
        payload = parse_unified_payload(build_unified_payload(config))
        self.assertEqual(payload["training_parallelism"], config["train"]["accelerator"])
        self.assertEqual(payload["parallelism"], {"tp": 4, "cp": 1, "pp": 1, "sequence_parallel": False})
        tampered = copy.deepcopy(payload)
        tampered["parallelism"]["tp"] = 8
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            parse_unified_payload(tampered)

    def test_explicit_or_inherited_inference_cp_pp_fail_before_worker_launch(self) -> None:
        """Unsupported axes fail instead of selecting a different parallel algorithm."""
        for axis, field in (("cp", "context_parallel_size"), ("pp", "pipeline_parallel_size")):
            for inherited in (False, True):
                with self.subTest(axis=axis, inherited=inherited):
                    config = self._config()
                    if inherited:
                        config["train"]["accelerator"][axis] = 2
                    else:
                        config["rollout"]["vllm"][field] = 2
                    with self.assertRaisesRegex(ValueError, "supports HP TP/SP only"):
                        build_unified_payload(config)


class TestParallelWorkerMesh(unittest.TestCase):
    """HyperParallel adopts the process group initialized by vLLM."""

    @staticmethod
    def _config(tp: int = 4) -> SimpleNamespace:
        return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
                               device_config=SimpleNamespace(device="cpu"))

    def test_tp_mesh_reuses_existing_group_with_nonzero_global_ranks(self) -> None:
        """The actual engine group owns ranks; no new TP or PP group is constructed."""
        payload = {"parallelism": {"tp": 4, "cp": 1, "pp": 1, "sequence_parallel": True}}
        fake_vllm = ModuleType("vllm.distributed")
        group = SimpleNamespace(ranks=[4, 5, 6, 7], device_group=object(), rank_in_group=3, world_size=4)
        fake_vllm.get_tp_group = MagicMock(return_value=group)
        mesh = MagicMock()
        mesh.get_local_rank.return_value = 3
        mesh.__getitem__.return_value.get_local_rank.return_value = 3
        with patch.dict("sys.modules", {"vllm.distributed": fake_vllm}), \
                patch("rl.unified_execution.runtimes.vllm.parallel_context.dist") as distributed, \
                patch("rl.unified_execution.runtimes.vllm.parallel_context.DeviceMesh.from_group",
                      return_value=mesh) as from_group, \
                patch("rl.unified_execution.runtimes.vllm.parallel_context.MeshContext") as mesh_context:
            distributed.is_initialized.return_value = True
            result = build_parallel_context(self._config(), payload)
        distributed.new_group.assert_not_called()
        distributed.init_process_group.assert_not_called()
        from_group.assert_called_once_with(group.device_group, "cpu", mesh_dim_names=("tp",))
        self.assertEqual(result.tp_rank, 3)
        self.assertTrue(result.sequence_parallel)
        self.assertIs(result.mesh_context, mesh_context.return_value)

    def test_ep_mesh_uses_synchronized_dp_tp_domain(self) -> None:
        """HP EP spans the actual MoE ranks without enabling a second expert implementation."""
        config = self._config(tp=1)
        config.parallel_config.data_parallel_size = 2
        config.parallel_config.data_parallel_rank = 1
        config.model_config = SimpleNamespace(enforce_eager=True, is_moe=True)
        payload = {"parallelism": {"tp": 1, "cp": 1, "pp": 1},
                   "expert_parallelism": {"training": 2, "inference": 2}}
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
                patch("rl.unified_execution.runtimes.vllm.parallel_context.dist.is_initialized", return_value=True), \
                patch("rl.unified_execution.runtimes.vllm.parallel_context.DeviceMesh.from_group",
                      side_effect=[tp_mesh, dense_mesh]) as from_group, \
                patch("rl.unified_execution.runtimes.vllm.parallel_context.build_expert_mesh",
                      return_value=expert_mesh) as build_mesh:
            result = build_parallel_context(config, payload)
        from_group.assert_called_with([dp_group, tp_group], "cpu", mesh=[[0], [1]], mesh_dim_names=("dp", "tp"))
        build_mesh.assert_called_once_with(dense_mesh, 2)
        self.assertEqual((result.dp_rank, result.ep_rank, result.tp_rank), (1, 1, 0))
        self.assertIs(result.mesh_context.fsdp_moe_mesh, expert_mesh)

    def test_ep_requires_eager_and_valid_rank_domain(self) -> None:
        """Do not silently change graph mode or create incomplete expert groups."""
        config = self._config(tp=2)
        config.model_config = SimpleNamespace(enforce_eager=False)
        payload = {"parallelism": {"tp": 2, "cp": 1, "pp": 1},
                   "expert_parallelism": {"inference": 2}}
        with self.assertRaisesRegex(ValueError, "shared graph EP recipe"):
            build_parallel_context(config, payload)
        payload["expert_parallelism"]["inference"] = 3
        with self.assertRaisesRegex(ValueError, "must divide"):
            build_parallel_context(config, payload)


class TestSharedPlanNormalization(unittest.TestCase):
    """Inference normalizes the original declaration without rewriting its algorithm."""


class TestParallelPublicationTopology(unittest.TestCase):
    """Publication distinguishes each TP worker and each DP engine."""


if __name__ == "__main__":
    unittest.main()


def _moe_config() -> dict:
    return {
        "model": {"name": "qwen3_moe"}, "train": {"plan_overrides": []},
        "rollout": {"engine": "vllm", "vllm": {"enforce_eager": True}},
        "consistency": {"schema_version": 1, "mode": "architecture"},
    }


class TestMoEContract(unittest.TestCase):
    """Shared expert identity and publication contracts."""

    def test_model_identity_roundtrip_and_tampering(self) -> None:
        """A MoE contract cannot be relabeled as Dense with a recomputed digest."""
        payload = build_unified_payload(_moe_config())
        self.assertEqual(payload["model_family"], "qwen3_moe")
        self.assertEqual(parse_unified_payload(payload), payload)
        changed = deepcopy(payload)
        changed["model_family"] = "unsupported_moe"
        changed.pop("plan_fingerprint")
        changed["plan_fingerprint"] = fingerprint(changed)
        with self.assertRaisesRegex(ValueError, "Unsupported unified"):
            parse_unified_payload(changed)
        family = get_model_family("qwen3_moe")
        self.assertTrue(family.moe)
        self.assertEqual(architecture_for_implementation("unified", model_family="qwen3_moe"), family.vllm_architecture)

    def test_hp_tp_sp_and_expert_contract(self) -> None:
        """Expert degree is distinct from dense TP and preserved at the worker boundary."""
        config = _moe_config()
        config["train"]["accelerator"] = {"tp": 2, "ep": 2, "sequence_parallel": True}
        payload = build_unified_payload(config)
        self.assertEqual(payload["parallelism"]["tp"], 2)
        self.assertEqual(payload["expert_parallelism"], {"training": 2, "inference": 2})
        self.assertEqual(parse_unified_payload(payload), payload)
        config["rollout"]["vllm"]["enforce_eager"] = False
        with self.assertRaisesRegex(ValueError, "EP requires"):
            build_unified_payload(config)

    def test_ep_can_span_synchronized_data_parallel_workers(self) -> None:
        """TP1/DP2/EP2 uses two workers, not four workers or vLLM fused experts."""
        config = _moe_config()
        config["train"]["accelerator"] = {"dp_replicate": 2, "ep": 2}
        config["rollout"]["vllm"].update(data_parallel_size=2, expert_parallel_size=2)
        payload = build_unified_payload(config)
        self.assertEqual(payload["parallelism"]["tp"], 1)
        self.assertEqual(payload["expert_parallelism"]["inference"], 2)

    def test_vllm_cannot_enable_an_independent_expert_strategy(self) -> None:
        """The scheduler must not override HP's expert ownership."""
        for flag in ("enable_expert_parallel", "enable_eplb"):
            config = _moe_config()
            config["rollout"]["vllm"][flag] = True
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "does not use vLLM"):
                build_unified_payload(config)
