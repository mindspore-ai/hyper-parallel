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
"""CPU contracts for the Qwen3 experiment launcher."""

import os
import subprocess
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "examples" / "scripts" / "run_qwen3_consistency_docker.sh"


def _run_launcher(
    deployment: str = "colocated",
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    """Run launcher configuration resolution without Docker or NPU access."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HYPER_QWEN3_")
    }
    environment.update(
        {
            "HYPER_QWEN3_DRY_RUN": "true",
            "HYPER_QWEN3_VISIBLE_DEVICES": "0,1",
            **overrides,
        }
    )
    return subprocess.run(
        ["/usr/bin/bash", str(_SCRIPT), deployment],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _resolved_values(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    """Parse the launcher's stable key-value dry-run output."""
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_dp2_defaults_match_the_selected_performance_baseline() -> None:
    """The zero-override launcher reproduces the selected DP2 benchmark setup."""
    resolved = _resolved_values(_run_launcher())

    assert resolved["image"] == (
        "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/"
        "hyper-rl:v0.22.1rc1-arm64"
    )
    assert resolved["device_count"] == "2"
    assert resolved["trainer_dp_shard"] == "2"
    assert "vllm_topology" not in resolved
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_port"] == "8100"
    assert resolved["api_server_count"] == "auto"
    assert resolved["global_prompt_count"] == "4"
    assert resolved["global_child_count"] == "16"
    assert resolved["response_mini_batch_size"] == "8"
    assert resolved["max_steps"] == "2"
    assert resolved["learning_rate"] == "0"
    assert resolved["learning_gate_enabled"] == "false"
    assert resolved["max_num_batched_tokens"] == "2048"
    assert resolved["max_num_seqs"] == "auto"
    assert resolved["config_name"] == "qwen3_4b_gsm8k_vllm_production.yaml"


def test_dp4_internal_dp_weak_scaling_is_derived_from_visible_devices() -> None:
    """Four visible devices define trainer, rollout, and weak-scaling batch sizes."""
    result = _run_launcher(
        HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
    )

    resolved = _resolved_values(result)

    assert "vllm_topology" not in resolved
    assert resolved["device_count"] == "4"
    assert resolved["trainer_world_size"] == "4"
    assert resolved["trainer_dp_shard"] == "4"
    assert resolved["rollout_tensor_parallel_size"] == "1"
    assert resolved["rollout_data_parallel_size"] == "4"
    assert resolved["api_server_count"] == "auto"
    assert resolved["prompt_batch_size"] == "2"
    assert resolved["global_prompt_count"] == "8"
    assert resolved["global_child_count"] == "32"
    assert resolved["response_mini_batch_size"] == "8"
    assert resolved["max_train_samples"] == "8"
    assert resolved["config_name"] == "qwen3_4b_gsm8k_vllm_production.yaml"
    assert resolved["torchrun_nproc_per_node"] == "4"
    assert resolved["override_train_accelerator_dp_shard"] == "4"
    assert resolved["override_rollout_data_parallel_size"] == "4"
    assert resolved["override_max_num_seqs"] == "omitted"


def test_four_device_tp2_selects_the_consistency_recipe() -> None:
    """Matched TP2 resolves the validated FSDP-shard2 and rollout DP2 topology."""
    result = _run_launcher(
        HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
        HYPER_QWEN3_TP="2",
    )

    resolved = _resolved_values(result)

    assert resolved["device_count"] == "4"
    assert resolved["trainer_dp_shard"] == "2"
    assert resolved["trainer_tensor_parallel_size"] == "2"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["weight_sync_strategy"] == "full_gather"
    assert resolved["weight_sync_fallback"] == "none"
    assert resolved["prompt_batch_size"] == "1"
    assert resolved["global_prompt_count"] == "2"
    assert resolved["max_train_samples"] == "2"
    assert resolved["max_new_tokens"] == "32"
    assert resolved["response_mini_batch_size"] == "4"
    assert resolved["learning_rate"] == "1e-6"
    assert resolved["rollout_port"] == "8422"
    assert resolved["config_name"] == "qwen3_4b_gsm8k_vllm_tp2_consistency.yaml"
    assert resolved["override_train_accelerator_dp_shard"] == "2"
    assert resolved["override_train_accelerator_tp"] == "2"
    assert resolved["override_rollout_data_parallel_size"] == "2"
    assert resolved["override_rollout_tensor_parallel_size"] == "2"


def test_four_device_disjoint_tp1_splits_trainer_and_rollout_dp() -> None:
    """Two Trainer NPUs and two rollout NPUs retain the proven TP1 topology."""
    resolved = _resolved_values(
        _run_launcher(
            "disjoint",
            HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
        )
    )

    assert resolved["deployment"] == "disjoint"
    assert resolved["trainer_visible_devices"] == "0,1"
    assert resolved["rollout_visible_devices"] == "2,3"
    assert resolved["trainer_world_size"] == "2"
    assert resolved["trainer_dp_shard"] == "2"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_tensor_parallel_size"] == "1"
    assert resolved["torchrun_nproc_per_node"] == "2"


def test_four_device_disjoint_tp2_resolves_minimal_matched_topology() -> None:
    """Pure Trainer TP2 and rollout DP1 x TP2 use disjoint two-card sets."""
    resolved = _resolved_values(
        _run_launcher(
            "disjoint",
            HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
            HYPER_QWEN3_TP="2",
        )
    )

    assert resolved["trainer_visible_devices"] == "0,1"
    assert resolved["rollout_visible_devices"] == "2,3"
    assert resolved["trainer_world_size"] == "2"
    assert resolved["trainer_dp_shard"] == "1"
    assert resolved["rollout_data_parallel_size"] == "1"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["torchrun_nproc_per_node"] == "2"


def test_response_group_size_can_be_increased_for_learning_acceptance() -> None:
    """An explicit response count scales child work and the local mini-batch."""
    resolved = _resolved_values(
        _run_launcher(
            "disjoint",
            HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
            HYPER_QWEN3_TP="2",
            HYPER_QWEN3_NUM_RETURN_SEQUENCES="8",
        )
    )

    assert resolved["num_return_sequences"] == "8"
    assert resolved["global_child_count"] == "8"
    assert resolved["response_mini_batch_size"] == "8"


def test_eight_device_disjoint_tp2_resolves_full_dp2_tp2_topology() -> None:
    """Four Trainer and four rollout NPUs express the final disjoint gate."""
    resolved = _resolved_values(
        _run_launcher(
            "disjoint",
            HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            HYPER_QWEN3_TP="2",
            HYPER_QWEN3_TRAINER_COUNT="4",
            HYPER_QWEN3_ROLLOUT_DP="2",
        )
    )

    assert resolved["trainer_visible_devices"] == "0,1,2,3"
    assert resolved["rollout_visible_devices"] == "4,5,6,7"
    assert resolved["trainer_world_size"] == "4"
    assert resolved["trainer_dp_shard"] == "2"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["torchrun_nproc_per_node"] == "4"


def test_launcher_uses_mounted_source_without_shadow_distribution() -> None:
    """The fixed image entry point remains visible while source code is mounted."""
    script = _SCRIPT.read_text(encoding="utf-8")

    assert "pip install" not in script
    assert "source_dir=$(mktemp" not in script
    assert (
        "PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel"
        in script
    )


@pytest.mark.parametrize(
    ("base_port", "socket_range"),
    [
        ("1023", "1023-1030"),
        ("65521", "65500-65521"),
        ("65000", "65001-65010"),
        ("65000", "invalid"),
    ],
)
def test_launcher_rejects_invalid_hccl_ports(
    base_port: str,
    socket_range: str,
) -> None:
    """Invalid CANN HCCL ports fail before Docker or NPU startup."""
    result = _run_launcher(
        HCCL_IF_BASE_PORT=base_port,
        HCCL_NPU_SOCKET_PORT_RANGE=socket_range,
    )

    assert result.returncode != 0
    assert "HCCL ports must use START-END in [1024, 65520]" in result.stderr


def test_tp2_acceptance_workload_and_direct_fallback_are_explicit() -> None:
    """The strong numerical gate can select fresh prompts and fallback."""
    result = _run_launcher(
        HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3",
        HYPER_QWEN3_TP="2",
        HYPER_QWEN3_WEIGHT_SYNC_STRATEGY="direct_reshard",
        HYPER_QWEN3_WEIGHT_SYNC_FALLBACK="full_gather",
        HYPER_QWEN3_MAX_TRAIN_SAMPLES="4",
        HYPER_QWEN3_MAX_NEW_TOKENS="256",
    )

    resolved = _resolved_values(result)

    assert resolved["weight_sync_strategy"] == "direct_reshard"
    assert resolved["weight_sync_fallback"] == "full_gather"
    assert resolved["max_train_samples"] == "4"
    assert resolved["max_new_tokens"] == "256"


def test_dp8_internal_dp_weak_scaling_is_derived_from_visible_devices() -> None:
    """Eight visible devices scale topology and global work without fixed card constants."""
    result = _run_launcher(
        HYPER_QWEN3_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )

    resolved = _resolved_values(result)

    assert resolved["device_count"] == "8"
    assert resolved["trainer_world_size"] == "8"
    assert resolved["trainer_dp_shard"] == "8"
    assert resolved["rollout_data_parallel_size"] == "8"
    assert resolved["global_prompt_count"] == "16"
    assert resolved["global_child_count"] == "64"
    assert resolved["response_mini_batch_size"] == "8"
    assert resolved["max_train_samples"] == "16"
    assert resolved["torchrun_nproc_per_node"] == "8"
    assert resolved["override_rollout_data_parallel_size"] == "8"
    assert resolved["override_max_num_seqs"] == "omitted"


def test_dp4_shared_strong_scaling_preserves_global_workload() -> None:
    """An explicit local prompt batch defines the DP4 strong-scaling workload."""
    result = _run_launcher(
        HYPER_QWEN3_VISIBLE_DEVICES="4,5,6,7",
        HYPER_QWEN3_PROMPT_BATCH_SIZE="1",
    )

    resolved = _resolved_values(result)

    assert resolved["rollout_data_parallel_size"] == "4"
    assert resolved["prompt_batch_size"] == "1"
    assert resolved["global_prompt_count"] == "4"
    assert resolved["global_child_count"] == "16"
    assert resolved["response_mini_batch_size"] == "4"
    assert resolved["max_train_samples"] == "4"
    assert resolved["override_rollout_data_parallel_size"] == "4"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"HYPER_QWEN3_VISIBLE_DEVICES": ""}, "comma-separated NPU list"),
        ({"HYPER_QWEN3_VISIBLE_DEVICES": "0,"}, "comma-separated NPU list"),
        ({"HYPER_QWEN3_VISIBLE_DEVICES": "0,a"}, "comma-separated NPU list"),
        ({"HYPER_QWEN3_VISIBLE_DEVICES": "0,00"}, "distinct NPUs"),
        (
            {
                "HYPER_QWEN3_VISIBLE_DEVICES": "0",
            },
            "requires at least two visible NPUs",
        ),
        ({"HYPER_QWEN3_VLLM_TOPOLOGY": "rank_local"}, "topology option was removed"),
        ({"HYPER_QWEN3_PROMPT_BATCH_SIZE": "0"}, "must be positive"),
        ({"HYPER_QWEN3_MAX_NEW_TOKENS": "0"}, "must be positive"),
        ({"HYPER_QWEN3_NUM_RETURN_SEQUENCES": "1"}, "at least 2"),
        ({"HYPER_QWEN3_MAX_TRAIN_SAMPLES": "0"}, "must be positive"),
        (
            {
                "HYPER_QWEN3_VISIBLE_DEVICES": "0,1,2,3",
                "HYPER_QWEN3_TP": "2",
                "HYPER_QWEN3_MAX_TRAIN_SAMPLES": "1",
            },
            "at least the global prompt count",
        ),
        ({"HYPER_QWEN3_ROLLOUT_PORT": "65536"}, "integer from 1 to 65535"),
        ({"HYPER_QWEN3_DRY_RUN": "maybe"}, "must be true or false"),
        ({"HYPER_QWEN3_TP": "3"}, "must be 1 or 2"),
        ({"HYPER_QWEN3_TP": "2"}, "validated four-device topology"),
        (
            {"HYPER_QWEN3_WEIGHT_SYNC_STRATEGY": "invalid"},
            "must be full_gather or direct_reshard",
        ),
        (
            {"HYPER_QWEN3_WEIGHT_SYNC_FALLBACK": "invalid"},
            "must be none or full_gather",
        ),
        (
            {"HYPER_QWEN3_WEIGHT_SYNC_FALLBACK": "full_gather"},
            "requires direct_reshard",
        ),
    ],
)
def test_launcher_rejects_invalid_automatic_configuration(
    overrides: dict[str, str],
    message: str,
) -> None:
    """Invalid device and workload inputs fail before file, Docker, or NPU access."""
    result = _run_launcher(**overrides)

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"HYPER_QWEN3_VISIBLE_DEVICES": "0,1"},
            "requires rollout NPUs",
        ),
        (
            {
                "HYPER_QWEN3_VISIBLE_DEVICES": "0,1,2,3",
                "HYPER_QWEN3_TP": "2",
                "HYPER_QWEN3_TRAINER_COUNT": "3",
            },
            "Trainer count 3 must be divisible by TP2",
        ),
        (
            {
                "HYPER_QWEN3_VISIBLE_DEVICES": "0,1,2,3",
                "HYPER_QWEN3_TP": "2",
                "HYPER_QWEN3_ROLLOUT_DP": "2",
            },
            "rollout DP2 requires 4 rollout devices",
        ),
    ],
)
def test_disjoint_launcher_rejects_invalid_resource_split(
    overrides: dict[str, str],
    message: str,
) -> None:
    """Disjoint role sizes must consume the selected devices exactly once."""
    result = _run_launcher("disjoint", **overrides)

    assert result.returncode != 0
    assert message in result.stderr
