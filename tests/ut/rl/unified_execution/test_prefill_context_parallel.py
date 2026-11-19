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
"""PCP configuration, worker ownership and CP1 preservation without devices."""

import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import yaml

from rl.config.resolve import resolve_unified_execution_config
from rl.rollout.topology import model_parallel_size, resolve_vllm_rollout_topology
from rl.rollout.vllm import VLLMGenerationEngine, build_vllm_engine
from rl.unified_execution.core.contract import build_unified_payload, parse_unified_payload
from rl.unified_execution.runtimes.vllm.parallel import build_parallel_context
from rl.weight_sync.vllm_worker import _rollout_worker_topology
from rl.workers.engine.model import ModelRegistration


def configuration() -> dict:
    """Use the same explicit shared recipe as the device consistency example."""
    root = Path(__file__).resolve().parents[4]
    path = root / "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_rl.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestPrefillContextParallel(unittest.TestCase):
    """Validate independent training CP, inference PCP and publication domains."""

    def test_pcp_expert_domain_keeps_training_topology(self) -> None:
        """Only the EP degree selects groups within the full DP x PCP x TP domain."""
        config = configuration()
        config["rollout"]["vllm"].update(context_parallel_size=2, data_parallel_size=1)
        for ep_size in (2, 4):
            config["rollout"]["vllm"]["expert_parallel_size"] = ep_size
            payload = parse_unified_payload(build_unified_payload(config))
            self.assertEqual(payload["parallelism"]["cp"], 2)
            self.assertEqual(payload["training_parallelism"]["cp"], 1)
            self.assertEqual(payload["expert_parallelism"], {"training": 4, "inference": ep_size})

    def test_pcp_requires_shared_arithmetic_and_dcp_one(self) -> None:
        """An unrelated numerical backend must not acquire PCP implicitly."""
        for change in ("policy", "dcp", "capacity"):
            config = configuration()
            config["rollout"]["vllm"]["context_parallel_size"] = 2
            if change == "policy":
                config["consistency"] = {"schema_version": 1, "mode": "architecture"}
            elif change == "dcp":
                config["rollout"]["vllm"]["decode_context_parallel_size"] = 2
            else:
                config["train"]["plan_overrides"] = [entry for entry in config["train"]["plan_overrides"]
                                                      if entry.get("when") != "ep"]
                config["rollout"]["vllm"]["enforce_eager"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_unified_payload(config)

    def test_expert_degree_is_authenticated_by_the_payload(self) -> None:
        """A worker must reject a changed EP degree with the old plan fingerprint."""
        config = configuration()
        config["rollout"]["vllm"].update(context_parallel_size=2, data_parallel_size=1, expert_parallel_size=4)
        payload = build_unified_payload(config)
        payload["expert_parallelism"]["inference"] = 2
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            parse_unified_payload(payload)

    def test_cross_pcp_mesh_extends_hp_planning_without_changing_native_dp(self) -> None:
        """Native DP1/PCP1/TP1 coordinates become HP DP3/TP1 in an eight-rank domain."""
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,
                                            prefill_context_parallel_size=2, decode_context_parallel_size=1,
                                            data_parallel_size=2, data_parallel_rank=1),
            device_config=SimpleNamespace(device="cpu"),
            model_config=SimpleNamespace(enforce_eager=True, is_moe=True),
        )
        payload = {"parallelism": {"tp": 2, "cp": 2, "pp": 1}, "expert_parallelism": {"inference": 4},
                   "plan_overrides": configuration()["train"]["plan_overrides"]}
        native = ModuleType("vllm.distributed")
        tp_group = object()
        native.get_tp_group = lambda: SimpleNamespace(ranks=[6, 7], device_group=tp_group)
        get_native_dp = MagicMock()
        native.get_dp_group = get_native_dp
        native.get_ep_group = lambda: SimpleNamespace(ranks=list(range(8)))
        native.get_pcp_group = lambda: SimpleNamespace(world_size=2, rank_in_group=1)
        tp_mesh, dense_mesh, expert_mesh, root = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        tp_mesh.__getitem__.return_value.get_local_rank.return_value = 1
        dense_mesh.__getitem__.return_value.get_local_rank.return_value = 3
        expert_mesh.__getitem__.return_value.get_local_rank.return_value = 3
        module = "rl.unified_execution.runtimes.vllm.parallel"
        with patch.dict("sys.modules", {"vllm.distributed": native}), \
                patch(module + ".dist.is_initialized", return_value=True), \
                patch(module + ".DeviceMesh.from_group", side_effect=[tp_mesh, dense_mesh]) as from_group, \
                patch(module + ".init_device_mesh", return_value=root) as build_mesh, \
                patch(module + ".build_expert_mesh", return_value=expert_mesh) as build_experts, \
                patch("rl.unified_execution.runtimes.vllm.parallel.build_expert_runtime") as build_runtime:
            context = build_parallel_context(config, payload)
        build_mesh.assert_called_once_with("cpu", (4, 2), mesh_dim_names=("dp", "tp"), rank_list=tuple(range(8)))
        build_experts.assert_called_once_with(root, 4)
        build_runtime.assert_called_once_with(config, payload, expert_mesh["ep"].get_group(), list(range(8)))
        self.assertIs(context.expert_runtime, build_runtime.return_value)
        from_group.assert_called_with([root["dp"].get_group(), tp_group], "cpu",
                                      mesh=[[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=("dp", "tp"))
        get_native_dp.assert_not_called()
        self.assertEqual((context.dp_size, context.dp_rank), (2, 1))
        self.assertEqual((context.mesh_context.dp_size, context.mesh_context.dp_rank), (4, 3))
        self.assertEqual(context.mesh_context.cp_size, 1)

    def test_expert_contract_rejects_invalid_domains_and_payload_types(self) -> None:
        """Expert degrees and payload types are validated before mesh construction."""
        config = configuration()
        config["rollout"]["vllm"].update(context_parallel_size=2, data_parallel_size=1,
                                       expert_parallel_size=8)
        with self.assertRaisesRegex(ValueError, "DP x PCP x TP"):
            build_unified_payload(config)
        config["rollout"]["vllm"]["expert_parallel_size"] = 4
        payload = build_unified_payload(config)
        payload["expert_parallelism"] = "invalid"
        with self.assertRaisesRegex(ValueError, "expert_parallelism"):
            parse_unified_payload(payload)

    def test_cp_one_has_identical_payload_and_launch_selection(self) -> None:
        """Explicit PCP1 must not activate a new runner or change the old launch recipe."""
        config = configuration()
        old = build_unified_payload(config)
        config["rollout"]["vllm"]["context_parallel_size"] = 1
        self.assertEqual(build_unified_payload(config), old)
        engine = self.engine(config)
        command = engine._server_command("127.0.0.1", 8100)
        self.assertNotIn("--prefill-context-parallel-size", command)
        self.assertNotIn("enable_pcp_embedding_lmhead_weight_sharding",
                         json.loads(command[command.index("--additional-config") + 1]))
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            self.assertEqual(engine._server_environment("0,1,2,3")["VLLM_USE_V2_MODEL_RUNNER"], "0")

    def test_pcp_launch_uses_native_runner_and_retains_hp_weight_ownership(self) -> None:
        """PCP alone activates MRV2; native embedding sharding stays disabled."""
        config = configuration()
        config["rollout"]["vllm"].update(context_parallel_size=2, data_parallel_size=1, expert_parallel_size=2)
        engine = self.engine(config)
        command = engine._server_command("127.0.0.1", 8100)
        self.assertEqual(command[command.index("--prefill-context-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--decode-context-parallel-size") + 1], "1")
        extra = json.loads(command[command.index("--additional-config") + 1])
        self.assertIs(extra["enable_pcp_embedding_lmhead_weight_sharding"], False)
        self.assertEqual(engine._server_environment("0,1,2,3")["VLLM_USE_V2_MODEL_RUNNER"], "1")

    @staticmethod
    def engine(config: dict) -> VLLMGenerationEngine:
        """Construct a CPU-only controller without starting a server."""
        config = resolve_unified_execution_config(config)
        model = ModelRegistration("qwen", "qwen3_moe", "/model", "/tokenizer", "Qwen3MoeForCausalLM",
                                  "qwen3_moe", "qwen3_moe", False)
        return VLLMGenerationEngine(model, config["rollout"])

    def test_visible_devices_include_pcp_without_changing_engine_count(self) -> None:
        """Schedulers still count DP replicas while worker allocation includes PCP."""
        for pcp in (1, 2, 4):
            config = {"deployment": "colocated", "model_implementation": "unified",
                      "data_parallel_size": 1, "tensor_parallel_size": 2,
                      "context_parallel_size": pcp, "port": 8100}
            devices = ",".join(str(index) for index in range(2 * pcp))
            topology = resolve_vllm_rollout_topology(config, {
                "LOCAL_WORLD_SIZE": str(2 * pcp), "ASCEND_RT_VISIBLE_DEVICES": devices,
            })
            self.assertEqual(model_parallel_size(config), 2 * pcp)
            self.assertEqual(topology.engine_count, 1)
            self.assertEqual(topology.tensor_parallel_size, 2)
            self.assertEqual(len(topology.visible_devices), 2 * pcp)
        with self.assertRaisesRegex(ValueError, "must match"):
            resolve_vllm_rollout_topology(config, {"LOCAL_WORLD_SIZE": "2"})

    def test_worker_rejects_moe_pcp_without_unified_ep(self) -> None:
        """Direct worker payloads must obey the same MoE PCP boundary as the controller."""
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,
                                            prefill_context_parallel_size=2, data_parallel_size=1),
            model_config=SimpleNamespace(enforce_eager=True, is_moe=True),
        )
        for ep in (1, 2):
            payload = {"parallelism": {"tp": 2, "cp": 2, "pp": 1}, "expert_parallelism": {"inference": ep}}
            with self.subTest(ep=ep), self.assertRaisesRegex(ValueError, "graph-capable"):
                build_parallel_context(config, payload)

    def test_publication_addresses_every_pcp_replica_without_changing_tp(self) -> None:
        """Distinct PCP owners must never collapse onto one weight-update recipient."""
        native = ModuleType("vllm.distributed")
        ipc = ModuleType("vllm_ascend.distributed.weight_transfer.npu_ipc_engine")
        coordinates = set()
        for rank in range(8):
            native.get_tp_group = lambda: SimpleNamespace(rank_in_group=rank % 2, world_size=2)
            native.get_pcp_group = lambda: SimpleNamespace(rank_in_group=(rank // 2) % 2)
            ipc.npu_generate_uuid = lambda: f"host-{rank}"
            worker = SimpleNamespace(parallel_config=SimpleNamespace(
                data_parallel_size=2, data_parallel_index=rank // 4, prefill_context_parallel_size=2,
            ))
            for visible in ("", "0,1,2,3,4,5,6,7"):
                with patch.dict("sys.modules", {"vllm.distributed": native,
                                                "vllm_ascend.distributed.weight_transfer.npu_ipc_engine": ipc}), \
                        patch.dict(os.environ, {"HYPER_RL_ROLLOUT_VISIBLE_DEVICES": visible}):
                    topology = _rollout_worker_topology(worker)
                self.assertEqual((topology["dp_rank"], topology["tp_rank"]), (rank // 2, rank % 2))
                self.assertEqual((topology["dp_size"], topology["tp_size"]), (4, 2))
                coordinates.add((topology["dp_rank"], topology["tp_rank"]))
        self.assertEqual(len(coordinates), 8)

    def test_publisher_counts_pcp_replicas_and_retains_actual_tp_width(self) -> None:
        """Transport fanout includes PCP, while weight conversion still sees TP2."""
        config = configuration()
        config["rollout"]["vllm"].update(context_parallel_size=2)
        config = resolve_unified_execution_config(config)
        model = ModelRegistration("qwen", "qwen3_moe", "/model", "/tokenizer", "Qwen3MoeForCausalLM",
                                  "qwen3_moe", "qwen3_moe", False)
        with patch("rl.rollout.vllm.build_weight_publisher") as publisher, \
                patch("rl.rollout.vllm.VLLMGenerationEngine"):
            build_vllm_engine(config["rollout"], model)
        self.assertEqual(publisher.call_args.kwargs["tensor_parallel_size"], 2)
        self.assertEqual(publisher.call_args.kwargs["data_parallel_size"], 4)
