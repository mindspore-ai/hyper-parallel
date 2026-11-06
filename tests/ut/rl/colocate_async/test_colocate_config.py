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
"""Colocated recipe admission must not change existing execution modes."""
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rl.algorithm import build_algorithm
from rl.config.loader import load_config
from rl.config.schema import ColocateAsyncConfig
from rl.config.validate import validate_colocate_async_config, validate_config
from rl.async_runtime.protocol import configuration_fingerprint
from rl.trainer.colocate_async import ColocateAsyncTrainer
from tests.ut.rl.colocate_async.support import trainer_fixture


def recipe() -> dict:
    """Provide the minimal logical scope without constructing models."""
    return {
        "model": {"name": "qwen3"}, "algorithm": {"name": "grpo"},
        "data": {"num_workers": 0, "pin_memory": False},
        "rollout": {"engine": "vllm", "num_return_sequences": 2,
                    "vllm": {"deployment": "colocated", "data_parallel_size": 1,
                             "tensor_parallel_size": 2, "enforce_eager": True}},
        "agentic": {"runner": "internal", "max_turns": 1, "interaction_mode": "single_turn"},
        "train": {"response_mini_batch_size": 2, "micro_batch_size": 1,
                  "accelerator": {"dp_shard": 2, "cpu_offload": True}},
        "colocate_async": {"groups_per_dp_rank": 1},
    }


class TestColocateConfig(unittest.TestCase):
    """Check capacity derivation, explicit rejection and existing common validation."""

    def test_disabled_correction_keeps_legacy_checkpoint_fingerprint(self):
        """Verify disabled correction keeps legacy checkpoint fingerprint."""
        roles, _, _ = trainer_fixture()
        config = recipe()
        roles._tracker = roles.tracker
        roles.codex_runtime = None
        roles.device_handle = Mock()
        with patch("rl.trainer.colocate_async.build_resources", return_value=roles), patch(
                "rl.trainer.colocate_async.dist.get_rank", return_value=0):
            legacy = ColocateAsyncTrainer(config)
            self.assertNotIn("off_policy", legacy.config["colocate_async"])
            config["colocate_async"]["off_policy"] = {"mode": "off"}
            disabled = ColocateAsyncTrainer(config)
            self.assertEqual(configuration_fingerprint(legacy.config), configuration_fingerprint(disabled.config))
            config["colocate_async"]["off_policy"]["mode"] = "metrics_only"
            monitored = ColocateAsyncTrainer(config)
            self.assertNotEqual(configuration_fingerprint(legacy.config), configuration_fingerprint(monitored.config))

    def test_codex_reuses_multi_call_sampling_with_independent_evaluation(self):
        """Verify codex reuses multi call sampling with independent evaluation."""
        config = recipe()
        config["agentic"] = {"runner": "codex", "max_turns": 10, "codex": {"gateway_port": 8200}}
        config["evaluation"] = {"enabled": True}
        # Calls per episode vary; the shared DP alignment handles call-row padding.
        config["train"]["response_mini_batch_size"] = 4
        self.assertEqual(validate_colocate_async_config(config).groups_per_dp_rank, 1)
        for turns in (True, 0, -1):
            config["agentic"]["max_turns"] = turns
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                validate_colocate_async_config(config)
        config["agentic"]["max_turns"] = 10
        del config["agentic"]["codex"]
        with self.assertRaises(ValueError):
            validate_colocate_async_config(config)

    def test_warmup_uses_global_prompt_groups(self):
        for warmup in (0, 1, 2):
            config = recipe()
            config["colocate_async"]["warmup_batches"] = warmup
            settings = validate_colocate_async_config(config).sampling_config(2)
            self.assertEqual(settings.max_inflight_groups, 2 * (1 + warmup))
            self.assertEqual(settings.max_buffer_groups, settings.max_inflight_groups)
            self.assertEqual(settings.publish_every_updates, 1)
            self.assertNotIn("async_training", config)

    def test_invalid_controls_are_rejected_before_startup(self):
        for key, value in (("warmup_batches", True), ("warmup_batches", 1.5),
                           ("groups_per_dp_rank", "2"), ("operation_timeout", float("inf")),
                           ("max_group_bytes", 0), ("max_buffer_bytes", 1),
                           ("max_candidate_retries", -1), ("unknown", 1)):
            config = recipe()
            config["colocate_async"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_colocate_async_config(config)

    def test_unsupported_combinations_are_rejected(self):
        """Verify unsupported combinations are rejected."""
        for section, key, value in (("algorithm", "name", "ppo"), ("model", "name", "qwen3_moe"),
                                    ("agentic", "runner", "codex"), ("agentic", "max_turns", 2),
                                    ("rollout", "temperature", .8), ("data", "pin_memory", True)):
            config = recipe()
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_colocate_async_config(config)
        for key, value in (("sleep_level", 2), ("sleep_level", True), ("enforce_eager", False),
                           ("deployment", "disjoint"), ("data_parallel_size", True),
                           ("tensor_parallel_size", 1), ("visible_devices", "0,1"),
                           ("batch_invariant", True), ("weight_sync", {"strategy": "direct_reshard"})):
            config = recipe()
            config["rollout"]["vllm"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_colocate_async_config(config)

    def test_modes_topology_and_batch_sizes_are_explicit(self):
        """Verify modes topology and batch sizes are explicit."""
        config = recipe()
        config["async_training"] = {}
        with self.assertRaisesRegex(ValueError, "either"):
            validate_colocate_async_config(config)
        config = recipe()
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "single-node"):
                validate_colocate_async_config(config)
        config["train"]["response_mini_batch_size"] = 3
        with self.assertRaisesRegex(ValueError, "divide"):
            validate_colocate_async_config(config)

    def test_shipped_recipe_passes_role_configuration_validation(self):
        """Verify shipped recipe passes role configuration validation."""
        root = Path(find_spec("hyper_parallel").origin).parent
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            }), encoding="utf-8")
            (directory / "data.parquet").touch()
            for task in ("gsm8k", "search_r1"):
                with self.subTest(task=task):
                    name = f"rl/examples/colocate_async/configs/qwen3_4b_{task}.yaml"
                    config = load_config(str(root / name), [])
                    config["model"].update(weights_path=str(directory), tokenizer_path=str(directory))
                    config["data"].update(train_path=str(directory / "data.parquet"),
                                          test_path=str(directory / "data.parquet"))
                    world = config["train"]["accelerator"]["dp_shard"]
                    environment = {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(world))),
                                   "WORLD_SIZE": str(world), "LOCAL_WORLD_SIZE": str(world)}
                    with patch.dict(os.environ, environment), patch(
                            "rl.config.validate.torch.accelerator.current_accelerator",
                            return_value=SimpleNamespace(type="npu")):
                        validate_colocate_async_config(config)
                        del config["colocate_async"]
                        validate_config(config, build_algorithm(config["algorithm"]))

    def test_normal_sync_import_remains_isolated(self):
        script = (
            "import sys, rl.trainer; "
            "assert not any(name.startswith('rl.trainer.colocate_async') for name in sys.modules); "
            "assert 'rl.async_runtime.producer' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script],
                                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_sampling_settings_still_validate_storage(self):
        with self.assertRaises(ValueError):
            ColocateAsyncConfig(max_buffer_bytes=1).sampling_config(2)
