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
"""Both hybrid checkpoints use the same short-budget SWE Code Agent contract."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import rl
from rl.algorithm import build_algorithm
from rl.config import build_model_registration, build_runtime_config, validate_config
from rl.roles.rollout.vllm import VLLMGenerationEngine

from examples.swe_smith.prepare_data import prepare_data


class TestCodeAgentRecipes(unittest.TestCase):
    """Check real serving recipes without loading weights or starting services."""

    def setUp(self) -> None:
        """Create only the checkpoint identity and public SWE data needed by validation."""
        directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        (self.path / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False,
                            "layer_types": ["linear_attention", "full_attention"]},
        }), encoding="utf-8")
        instance = {"instance_id": "train-task", "problem_statement": "Repair the issue.",
                    "repo": "fixture/repo", "image_name": "fixture/image",
                    "FAIL_TO_PASS": ["tests/test_x.py::test_bug"], "PASS_TO_PASS": []}
        train, validation = self.path / "train.jsonl", self.path / "validation.jsonl"
        train.write_text(json.dumps(instance))
        validation.write_text(json.dumps({**instance, "instance_id": "validation-task"}))
        prepare_data({"train": train, "validation": validation}, self.path / "prepared", repeats=4)

    def _config(self, model: str = "qwen3_8_27b") -> dict:
        recipes = Path(rl.__file__).resolve().parent.parent / "examples/swe_smith/configs"
        config = yaml.safe_load((recipes / f"{model}.yaml").read_text(encoding="utf-8"))
        config["model"].update(weights_path=str(self.path), tokenizer_path=str(self.path))
        config["data"].update(train_path=str(self.path / "prepared/train.parquet"),
                              test_path=str(self.path / "prepared/validation.parquet"))
        config["agentic"]["codex"]["task"]["data_dir"] = str(self.path / "prepared")
        return config

    def test_both_models_build_public_training_and_native_serving(self) -> None:
        """Keep model-specific sizes with shared text loading, tool parser and raw evidence."""
        for model, size in (("qwen3_5_9b", 2), ("qwen3_8_27b", 4)):
            with self.subTest(model=model):
                config = self._config(model)
                validate_config(config, build_algorithm(config["algorithm"]))
                runtime = build_runtime_config(config)
                self.assertEqual(runtime.model.to_dict()["_target_"],
                                 "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained")
                self.assertEqual(runtime.accelerator.tp_size, 1)
                self.assertEqual(runtime.fsdp_config.dp_shard_size, size)
                self.assertTrue(runtime.fsdp_config.enable_offload)
                engine = VLLMGenerationEngine(build_model_registration(config), config["rollout"])
                command = engine._server_command("127.0.0.1", config["rollout"]["vllm"]["port"])
                for flag in ("--language-model-only", "--enforce-eager", "--no-enable-prefix-caching",
                             "--enable-auto-tool-choice"):
                    self.assertIn(flag, command)
                for flag, expected in (("--tensor-parallel-size", str(size)), ("--data-parallel-size", "1"),
                                       ("--tool-call-parser", "qwen3_coder"), ("--logprobs-mode", "raw_logprobs"),
                                       ("--reasoning-parser", "qwen3")):
                    self.assertEqual(command[command.index(flag) + 1], expected)

    def test_colocated_serving_does_not_inherit_trainer_hccl_ports(self) -> None:
        """Trainer collectives must not bind the live rollout process's NPU socket ports."""
        trainer_ports = {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63200-63299"}
        for model in ("qwen3_5_9b", "qwen3_8_27b"):
            with self.subTest(model=model), patch.dict(os.environ, trainer_ports):
                config = self._config(model)
                engine = VLLMGenerationEngine(build_model_registration(config), config["rollout"])
                environment = engine._server_environment("0,1")
                self.assertNotEqual(environment["HCCL_IF_BASE_PORT"], trainer_ports["HCCL_IF_BASE_PORT"])
                start, end = map(int, environment["HCCL_NPU_SOCKET_PORT_RANGE"].split("-"))
                self.assertTrue(end < 63200 or start > 63299)

    def test_invalid_context_evidence_and_accumulation_are_rejected(self) -> None:
        """Reject inconsistent budgets and probability semantics before creating a service."""
        for keys, value in ((("agentic", "codex", "model_context_window"), 16385),
                            (("rollout", "vllm", "logprobs_mode"), "processed_logprobs"),
                            (("rollout", "vllm", "tool_call_parser"), "hermes"),
                            (("rollout", "vllm", "enforce_eager"), False),
                            (("train", "response_mini_batch_size"), 5)):
            invalid = self._config()
            section = invalid
            for key in keys[:-1]:
                section = section[key]
            section[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                validate_config(invalid, build_algorithm(invalid["algorithm"]))

    def test_explicit_single_replica_decode_graph_recipe(self) -> None:
        """Only the reviewed capture mode and complete size list permit graph publication."""
        config = self._config()
        vllm = config["rollout"]["vllm"]
        vllm.update(enforce_eager=False, compilation_config={
            "mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4],
            "max_cudagraph_capture_size": 4,
        })
        validate_config(config, build_algorithm(config["algorithm"]))
        command = VLLMGenerationEngine(build_model_registration(config), config["rollout"])._server_command(
            "127.0.0.1", vllm["port"])
        self.assertNotIn("--enforce-eager", command)
        self.assertEqual(json.loads(command[command.index("--compilation-config") + 1]), vllm["compilation_config"])
        for key, value in (("mode", 2), ("cudagraph_mode", "FULL"), ("cudagraph_capture_sizes", [2, 4]),
                           ("cudagraph_capture_sizes", [1, 8]), ("max_cudagraph_capture_size", 8)):
            invalid = deepcopy(config)
            invalid["rollout"]["vllm"]["compilation_config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid, build_algorithm(invalid["algorithm"]))
