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
"""Unit tests for the tiny Qwen3-MoE training demo."""

import unittest
from pathlib import Path

import torch
import yaml

from examples.training_demo.data import TinyCausalDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
CP_CONFIG_DIR = REPO_ROOT / "examples" / "training_demo" / "cp_configs"
MOE_REPLACEMENT_TARGET = (
    "hyper_parallel.auto_models.components.models.qwen3_moe_fusions."
    "replace_qwen3_moe_sparse_moe"
)
CP_VARIANTS = {
    "sync_colossal": (
        "hyper_parallel.auto_models.components.models.qwen3_moe_fusions."
        "qwen3_moe_flash_attention_cp_wrapper",
        2,
        None,
    ),
    "sync_load_balance": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "sdpa_hf_load_balance_cp_wrapper",
        2,
        None,
    ),
    "sync_ulysses": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "sdpa_hf_ulysses_cp_wrapper",
        2,
        None,
    ),
    "sync_hybrid": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "sdpa_hf_hybrid_cp_wrapper",
        4,
        2,
    ),
    "async_colossal": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "qwen3_moe_async_colossal_cp_wrapper",
        2,
        None,
    ),
    "async_ulysses": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "qwen3_moe_async_ulysses_cp_wrapper",
        2,
        None,
    ),
    "async_hybrid": (
        "hyper_parallel.auto_models.components.distributed.cp_wrappers."
        "qwen3_moe_async_hybrid_cp_wrapper",
        4,
        2,
    ),
}


class TestTinyQwen3TrainingDemo(unittest.TestCase):
    """Validate deterministic labels and the seven CP wrapper configurations."""

    def test_example_dataset_emits_pre_shifted_causal_labels(self) -> None:
        """Preserve cross-shard next-token targets before CP slicing."""
        dataset = TinyCausalDataset(
            num_samples=2,
            seq_len=6,
            vocab_size=19,
            seed=23,
        )

        self.assertEqual(
            tuple(dataset.input_ids.shape),
            (2, 6),
            f"Unexpected input shape: expected={(2, 6)}, got={tuple(dataset.input_ids.shape)}",
        )
        self.assertEqual(
            tuple(dataset.labels.shape),
            (2, 6),
            f"Unexpected label shape: expected={(2, 6)}, got={tuple(dataset.labels.shape)}",
        )
        self.assertTrue(
            torch.equal(dataset.input_ids[:, 1:], dataset.labels[:, :-1]),
            f"Labels are not next-token shifted: input_ids={dataset.input_ids}, labels={dataset.labels}",
        )

    def test_cp_wrapper_yaml_matrix(self) -> None:
        """Declare exactly one expected wrapper and topology per CP variant."""
        config_files = {path.stem for path in CP_CONFIG_DIR.glob("*.yaml")}
        self.assertEqual(
            config_files,
            set(CP_VARIANTS),
            f"CP config set mismatch: expected={set(CP_VARIANTS)}, got={config_files}",
        )

        for variant, (expected_target, expected_cp_size, expected_degree) in CP_VARIANTS.items():
            with self.subTest(variant=variant):
                config_path = CP_CONFIG_DIR / f"{variant}.yaml"
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                wrapper_specs = [
                    override["inner_wrapper"]
                    for override in config["plan_overrides"]
                    if "inner_wrapper" in override
                ]
                self.assertEqual(
                    len(wrapper_specs),
                    1,
                    f"Wrapper count mismatch for {variant}: expected=1, got={len(wrapper_specs)}",
                )
                wrapper_spec = wrapper_specs[0]
                self.assertEqual(
                    wrapper_spec["_target_"],
                    expected_target,
                    f"Wrapper target mismatch for {variant}: expected={expected_target}, "
                    f"got={wrapper_spec['_target_']}",
                )
                self.assertEqual(
                    config["accelerator"]["cp_size"],
                    expected_cp_size,
                    f"CP size mismatch for {variant}: expected={expected_cp_size}, "
                    f"got={config['accelerator']['cp_size']}",
                )
                self.assertEqual(
                    wrapper_spec.get("ulysses_degree"),
                    expected_degree,
                    f"Ulysses degree mismatch for {variant}: expected={expected_degree}, "
                    f"got={wrapper_spec.get('ulysses_degree')}",
                )
                replacement_targets = [
                    override["replace_module"]["_target_"]
                    for override in config["plan_overrides"]
                    if override.get("match") == "*.mlp"
                ]
                self.assertEqual(
                    replacement_targets,
                    [MOE_REPLACEMENT_TARGET],
                    f"MoE replacement mismatch: expected={[MOE_REPLACEMENT_TARGET]}, "
                    f"got={replacement_targets}",
                )
                self.assertEqual(
                    config["dataset"]["_target_"],
                    "examples.training_demo.data.TinyCausalDataset",
                    f"Dataset target mismatch: expected=examples.training_demo.data.TinyCausalDataset, "
                    f"got={config['dataset']['_target_']}",
                )
                self.assertEqual(
                    config["dataloader"]["_target_"],
                    "hyper_parallel.auto_models.components.datasets.FixedBatchDataLoader",
                    "DataLoader target mismatch: "
                    "expected=hyper_parallel.auto_models.components.datasets.FixedBatchDataLoader, "
                    f"got={config['dataloader']['_target_']}",
                )
                self.assertEqual(
                    config["dataloader"]["collate_fn"]["_target_"],
                    "hyper_parallel.auto_models.components.datasets.build_indexed_collate_fn",
                    "Collator target mismatch: "
                    "expected=hyper_parallel.auto_models.components.datasets.build_indexed_collate_fn, "
                    f"got={config['dataloader']['collate_fn']['_target_']}",
                )
                expected_get_batch = {
                    "_target_": "hyper_parallel.auto_models.components.datasets.ParallelBatch",
                    "source_type": "indexed",
                }
                self.assertEqual(
                    config["dataloader"]["get_batch"],
                    expected_get_batch,
                    f"Get-batch target mismatch: expected={expected_get_batch}, "
                    f"got={config['dataloader']['get_batch']}",
                )


if __name__ == "__main__":
    unittest.main()
