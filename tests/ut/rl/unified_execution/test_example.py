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
"""CPU regression for the example's strict pre-update consistency gate."""

import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hyper_parallel.rl.examples.consistency.scripts.prepare_checkpoint import prepare_checkpoint, verify_checkpoint
from hyper_parallel.rl.examples.consistency.scripts.run_consistency import (
    check_metrics,
    check_metrics_on_all_ranks,
    run,
)


def _metrics():
    return {f"training/{key}": value for key, value in {
        "rollout_probs_diff_valid": 1, "log_probs_valid_tokens": 16,
        "rollout_log_probs_diff_max": 0, "rollout_log_probs_diff_mean": 0,
        "actor_log_probs_mean": -2, "rollout_log_probs_mean": -2,
    }.items()}


class TestConsistencyExample(unittest.TestCase):
    """Prevent empty or nonfinite comparisons and incomplete RL runs from passing."""

    def test_reject_missing_empty_nonfinite_and_unequal_metrics(self):
        """Reject missing or invalid evidence of train and rollout equality."""
        self.assertEqual(check_metrics(_metrics())["log_probs_valid_tokens"], 16)
        for key, value in (("rollout_probs_diff_valid", 0), ("log_probs_valid_tokens", 0),
                           ("rollout_log_probs_diff_max", 1e-7),
                           ("actor_log_probs_mean", float("nan")),
                           ("rollout_log_probs_mean", float("inf"))):
            metrics = _metrics()
            metrics[f"training/{key}"] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_metrics(metrics)
        with self.assertRaises(KeyError):
            check_metrics({})

    def test_report_requires_completed_updates_and_retained_rollout(self):
        """Write success only after both verified updates and publications complete."""
        config = {"train": {"max_steps": 2, "accelerator": {}},
                  "logging": {"log_steps": 1}, "rollout": {"vllm": {}}}
        for complete, retained in ((True, True), (False, True), (True, False)):
            trainer = SimpleNamespace(state=SimpleNamespace(global_step=0),
                                      targets=SimpleNamespace(prepare=lambda *_: (None, _metrics())))

            def _train(complete=complete, retained=retained, trainer=trainer):
                for step in range(1, 3):
                    rollout = SimpleNamespace(rollout_log_probs=object() if retained else None)
                    trainer.targets.prepare(rollout, True, {})
                    trainer.state.global_step = step
                if not complete:
                    raise RuntimeError("Publication failed")

            trainer.train = _train
            with self.subTest(complete=complete, retained=retained), TemporaryDirectory() as directory:
                report = Path(directory) / "consistency.json"
                report.write_text("stale pass")
                with patch.dict("os.environ", {"RANK": "0"}), patch(
                    "rl.trainer.distributed.dist.get_world_size", return_value=1,
                ), patch(
                    "hyper_parallel.rl.examples.consistency.scripts.run_consistency.build_trainer",
                    return_value=trainer,
                ):
                    if complete and retained:
                        run(config, report)
                        self.assertIn('"passed": true', report.read_text())
                    else:
                        with self.assertRaises(RuntimeError):
                            run(config, report)
                        self.assertFalse(report.exists(), "A failed run must not leave a success report")

    def test_nonzero_rank_uses_global_verdict_without_local_metrics(self):
        """Nonzero ranks receive empty metrics but must share rank zero's failure."""
        with patch.dict("os.environ", {"RANK": "1"}), patch("rl.trainer.distributed.dist") as distributed:
            distributed.get_world_size.return_value = 2

            def _gather_success(records, local_error):
                records[:] = [None, local_error]

            distributed.all_gather_object.side_effect = _gather_success
            self.assertEqual(check_metrics_on_all_ranks({}), {})

            def _gather_failure(records, local_error):
                records[:] = ["rank zero: log probabilities differ", local_error]

            distributed.all_gather_object.side_effect = _gather_failure
            with self.assertRaisesRegex(RuntimeError, "log probabilities differ"):
                check_metrics_on_all_ranks({})

    def test_root_validation_failure_is_synchronized_before_update(self):
        """Rank zero publishes its validation error through the existing collective."""
        with patch.dict("os.environ", {"RANK": "0"}), patch("rl.trainer.distributed.dist") as distributed:
            distributed.get_world_size.return_value = 2

            def _gather(records, local_error):
                records[:] = [local_error, None]

            distributed.all_gather_object.side_effect = _gather
            metrics = _metrics()
            metrics["training/rollout_log_probs_diff_max"] = 1e-6
            with self.assertRaisesRegex(RuntimeError, "consistency failed"):
                check_metrics_on_all_ranks(metrics)
            distributed.all_gather_object.assert_called_once()


class TestConsistencyCheckpoint(unittest.TestCase):
    """Only decoder depth changes; original checkpoints and expert counts remain intact."""

    def test_crop_preserves_tensors_and_refuses_overwrite(self) -> None:
        """Export indexed shards without renumbering layers or dropping global weights."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "cropped"
            source.mkdir()
            config = {"model_type": "qwen3_moe", "num_hidden_layers": 3, "max_window_layers": 3,
                      "num_experts": 128, "num_experts_per_tok": 8, "mlp_only_layers": [2]}
            (source / "config.json").write_text(json.dumps(config))
            weights = {name: torch.arange(8, dtype=torch.float32).reshape(2, 4) + index
                       for index, name in enumerate(("model.embed_tokens.weight", "model.layers.0.test.weight",
                                                     "model.layers.1.test.weight", "model.layers.2.test.weight",
                                                     "model.norm.weight", "lm_head.weight"))}
            save_file(weights, source / "source.safetensors")
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": dict.fromkeys(weights, "source.safetensors"),
            }))
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            manifest = prepare_checkpoint(source, output, 1)
            self.assertEqual(manifest["layers"], 1)
            exported = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            expected = set(weights) - {"model.layers.1.test.weight", "model.layers.2.test.weight"}
            self.assertEqual(set(exported), expected)
            for name, filename in exported.items():
                with safe_open(output / filename, framework="pt", device="cpu") as checkpoint:
                    torch.testing.assert_close(checkpoint.get_tensor(name), weights[name], atol=0, rtol=0)
            actual_config = json.loads((output / "config.json").read_text())
            self.assertEqual(actual_config["num_experts"], 128)
            self.assertEqual(actual_config["num_experts_per_tok"], 8)
            self.assertEqual(actual_config["mlp_only_layers"], [])
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
            with self.assertRaises(FileExistsError):
                prepare_checkpoint(source, output, 1)
            with self.assertRaises(ValueError):
                prepare_checkpoint(source, root / "invalid", 0)
            self.assertFalse((root / "invalid").exists())
            verify_checkpoint(source, output, 1)
            with self.assertRaisesRegex(ValueError, "decoder depth"):
                verify_checkpoint(source, output, 2)
            shard = output / manifest["shards"][0]["output_file"]
            shard.write_bytes(shard.read_bytes() + b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_checkpoint(source, output, 1)
