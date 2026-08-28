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
"""CPU contracts for the Qwen3 TP weight-sync launcher."""

import os
from pathlib import Path
import subprocess

import pytest


_SCRIPT = Path(__file__).parents[1] / "examples" / "scripts" / "run_qwen3_tp_docker.sh"


def _dry_run(tmp_path: Path, deployment: str, **overrides: str) -> dict[str, str]:
    """Resolve one launcher configuration without touching Docker or NPU state."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HYPER_QWEN3_TP_")
    }
    environment.update({"HYPER_QWEN3_TP_DRY_RUN": "true", **overrides})
    result = subprocess.run(
        ["/usr/bin/bash", str(_SCRIPT), deployment],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_colocated_defaults_to_shared_dp1_tp2(tmp_path: Path) -> None:
    """The two-card colocated smoke no longer selects rank-local ownership."""
    resolved = _dry_run(tmp_path, "colocated")

    assert resolved["image"] == (
        "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/"
        "hyper-rl:v0.22.1rc1-arm64"
    )
    assert "runtime_topology" not in resolved
    assert resolved["trainer_count"] == "2"
    assert resolved["rollout_data_parallel_size"] == "1"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["rollout_device_count"] == "2"
    assert resolved["rollout_port"] == "8500"
    assert resolved["required_devices"] == "2"


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
    tmp_path: Path,
    base_port: str,
    socket_range: str,
) -> None:
    """Invalid CANN HCCL ports fail during dry-run resolution."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HYPER_QWEN3_TP_")
    }
    environment.update(
        {
            "HYPER_QWEN3_TP_DRY_RUN": "true",
            "HCCL_IF_BASE_PORT": base_port,
            "HCCL_NPU_SOCKET_PORT_RANGE": socket_range,
        }
    )
    result = subprocess.run(
        ["/usr/bin/bash", str(_SCRIPT), "colocated"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "HCCL ports must use START-END in [1024, 65520]" in result.stderr


def test_four_colocated_devices_scale_shared_runtime_to_dp2_tp2(
    tmp_path: Path,
) -> None:
    """An explicit four-card allocation retains the DP2 x TP2 acceptance path."""
    resolved = _dry_run(
        tmp_path,
        "colocated",
        HYPER_QWEN3_TP_VISIBLE_DEVICES="0,1,2,3",
    )

    assert "runtime_topology" not in resolved
    assert resolved["trainer_count"] == "4"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_device_count"] == "4"
    assert resolved["required_devices"] == "4"


def test_four_colocated_devices_resolve_independent_trainer_and_rollout_tp(
    tmp_path: Path,
) -> None:
    """One launcher covers matched TP2 and both normal-mode mismatched directions."""
    for trainer_tp, rollout_tp, trainer_dp, rollout_dp in (
        (2, 2, 2, 2),
        (1, 2, 4, 2),
        (2, 1, 2, 4),
    ):
        resolved = _dry_run(
            tmp_path,
            "colocated",
            HYPER_QWEN3_TP_VISIBLE_DEVICES="0,1,2,3",
            HYPER_QWEN3_TP_TRAINER_TP=str(trainer_tp),
            HYPER_QWEN3_TP_ROLLOUT_TP=str(rollout_tp),
        )

        assert resolved["trainer_tensor_parallel_size"] == str(trainer_tp)
        assert resolved["trainer_dp_shard_size"] == str(trainer_dp)
        assert resolved["rollout_tensor_parallel_size"] == str(rollout_tp)
        assert resolved["rollout_data_parallel_size"] == str(rollout_dp)


def test_disjoint_uses_one_shared_dp2_tp2_rollout(tmp_path: Path) -> None:
    """D3 gives one endpoint the full four-device rollout deployment."""
    resolved = _dry_run(
        tmp_path,
        "disjoint",
        HYPER_QWEN3_TP_VISIBLE_DEVICES="0,1,2,3,4,5",
    )

    assert "runtime_topology" not in resolved
    assert resolved["trainer_count"] == "2"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["rollout_device_count"] == "4"
    assert resolved["rollout_visible_devices"] == "2,3,4,5"
    assert resolved["rollout_port"] == "8400"
    assert resolved["required_devices"] == "6"


def test_four_disjoint_devices_resolve_pure_tp2_and_dp1_tp2(
    tmp_path: Path,
) -> None:
    """The minimal matched TP2 disjoint topology uses two NPUs per role."""
    resolved = _dry_run(
        tmp_path,
        "disjoint",
        HYPER_QWEN3_TP_VISIBLE_DEVICES="0,1,2,3",
        HYPER_QWEN3_TP_TRAINER_TP="2",
        HYPER_QWEN3_TP_ROLLOUT_TP="2",
        HYPER_QWEN3_TP_ROLLOUT_DP="1",
        HYPER_QWEN3_TP_ROLLOUT_PORT="8460",
    )

    assert resolved["trainer_count"] == "2"
    assert resolved["trainer_dp_shard_size"] == "1"
    assert resolved["rollout_data_parallel_size"] == "1"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["rollout_visible_devices"] == "2,3"
    assert resolved["rollout_port"] == "8460"
    assert resolved["required_devices"] == "4"


def test_eight_disjoint_devices_resolve_full_dp2_tp2_topology(
    tmp_path: Path,
) -> None:
    """The final gate allocates four Trainer and four rollout NPUs."""
    resolved = _dry_run(
        tmp_path,
        "disjoint",
        HYPER_QWEN3_TP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        HYPER_QWEN3_TP_TRAINER_COUNT="4",
        HYPER_QWEN3_TP_ROLLOUT_DP="2",
        HYPER_QWEN3_TP_TRAINER_TP="2",
        HYPER_QWEN3_TP_ROLLOUT_TP="2",
    )

    assert resolved["trainer_count"] == "4"
    assert resolved["trainer_dp_shard_size"] == "2"
    assert resolved["rollout_data_parallel_size"] == "2"
    assert resolved["rollout_tensor_parallel_size"] == "2"
    assert resolved["rollout_visible_devices"] == "4,5,6,7"
    assert resolved["required_devices"] == "8"


def test_removed_topology_override_fails_explicitly(tmp_path: Path) -> None:
    """An old topology override is not silently reinterpreted after migration."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HYPER_QWEN3_TP_")
    }
    environment.update(
        {
            "HYPER_QWEN3_TP_DRY_RUN": "true",
            "HYPER_QWEN3_TP_TOPOLOGY": "rank_local",
        }
    )
    result = subprocess.run(
        ["/usr/bin/bash", str(_SCRIPT), "colocated"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "HYPER_QWEN3_TP_TOPOLOGY topology option was removed" in result.stderr


def test_relative_result_root_is_normalized_before_docker(tmp_path: Path) -> None:
    """Docker bind sources must be absolute even when users pass a relative root."""
    relative_root = "phase5a-results"
    resolved = _dry_run(
        tmp_path,
        "colocated",
        HYPER_QWEN3_TP_RESULT_ROOT=relative_root,
    )

    assert resolved["result_root"] == str(tmp_path / relative_root)
    assert resolved["num_return_sequences"] == "2"
    assert resolved["max_new_tokens"] == "32"
    assert resolved["rollout_seed"] == ""
    assert resolved["learning_rate"] == "1e-6"


def test_rollout_seed_override_is_resolved_for_reproducible_workloads(
    tmp_path: Path,
) -> None:
    """A D2 arm can reproduce a prior workload without coupling it to consistency."""
    resolved = _dry_run(
        tmp_path,
        "colocated",
        HYPER_QWEN3_TP_ROLLOUT_SEED="20260825",
    )

    assert resolved["rollout_seed"] == "20260825"
