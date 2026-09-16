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

"""Lightweight launchers for public MegaMoe layer system tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe.py")
_PRECISION_WORLD_SIZE = 2
_PERFORMANCE_WORLD_SIZE = 4


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for the Torch framework."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError(
            "MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with "
            "--multicore on"
        )


def _run_precision_worker(monkeypatch) -> None:
    """Run the two-rank precision worker with an isolated SHMEM endpoint."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_level0_balanced",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_level0_precision(monkeypatch) -> None:
    """Compare output, dX, route dW, W1 dW, and W2 dW with common MoE."""
    _run_precision_worker(monkeypatch)


def _run_performance_worker(monkeypatch, result_dir: Path) -> dict:
    """Run common/MegaMoe A/B in one four-rank process group."""
    result_path = result_dir / "routed_expert_ab.json"
    monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PERF_WARMUP", "3")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_MEASURED", "5")
    monkeypatch.setenv("HP_MEGA_MOE_MAX_PERF_RATIO", "1.10")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_RESULT", str(result_path))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "62000-62063")
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_fwd_bwd_performance",
                    num_proc=_PERFORMANCE_WORLD_SIZE,
                )
            ],
            global_num_proc=_PERFORMANCE_WORLD_SIZE,
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_representative_performance(monkeypatch, tmp_path: Path) -> None:
    """Enforce one steady-state common/MegaMoe routed-expert A/B threshold."""
    _prepare_torch_multicore_test_environment()
    comparison = _run_performance_worker(monkeypatch, tmp_path)
    results = comparison["results"]
    reference = results["common"]
    for tag, backend_result in results.items():
        assert backend_result["shape"] == reference["shape"], (
            f"performance shape drifted for {tag}: "
            f"expected={reference['shape']}, got={backend_result['shape']}"
        )
        assert backend_result["topology"] == reference["topology"], (
            f"performance topology drifted for {tag}: "
            f"expected={reference['topology']}, got={backend_result['topology']}"
        )
        assert backend_result["device_healthy"], (
            f"performance device health failed for {tag}: result={backend_result}"
        )
        assert all(backend_result["validation"].values()), (
            f"performance validation failed for {tag}: "
            f"validation={backend_result['validation']}"
        )

    common_median = results["common"]["rank_max_fwd_bwd_ms"]["median"]
    mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
    assert common_median > 0.0, (
        f"common median latency must be positive, got {common_median}."
    )
    assert mega_median > 0.0, f"MegaMoe median latency must be positive, got {mega_median}."
    ratio = mega_median / common_median
    assert ratio <= comparison["maximum_allowed_ratio"], (
        "MegaMoe routed-expert performance exceeded the A/B threshold: "
        f"ratio={ratio}, maximum={comparison['maximum_allowed_ratio']}."
    )
    print(
        "MegaMoe representative A/B: "
        f"common_ms={common_median}, mega_ms={mega_median}, ratio={ratio:.6f}"
    )


def _run_acceptance_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int, worker: str | None = None,
    heap_bytes: int = 64 * 1024 * 1024,
) -> dict:
    """Launch one real-device acceptance worker and require its rank evidence."""
    _prepare_torch_multicore_test_environment()
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path)))
    result_path = result_dir / f"{case}.json"
    if worker is None:
        worker = "_test_mega_moe_resources.py" if cards == 2 else "_test_mega_moe_defaults.py"
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(cards))
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(result_path))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    if cards == 2:
        monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(heap_bytes))
    else:
        monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    with without_inherited_rank_environment():
        parallel_run(
            [TorchCase(str(Path(__file__).with_name(worker)), case, num_proc=cards)],
            global_num_proc=cards,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["ranks"]) == cards, f"expected {cards} rank records, got {len(result['ranks'])}."
    return result


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_local_capacity_lifetime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Preserve all gradients across odd receive tails and two outstanding routes."""
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_local_capacity_lifetime", 2, "_test_mega_moe_memory.py",
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_push_memory_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validate real receive/dX aliases and fallback with a minimally page-aligned heap."""
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_push_memory_reuse", 2, "_test_mega_moe_push_memory.py",
        heap_bytes=2 * 1024 * 1024,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_poisoned_buffers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Require complete writes after NaN and stale-value poisoning across route changes."""
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_poisoned_buffers", 2, "_test_mega_moe_zeroing.py",
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_device_ready_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validate peer readiness under checkpoint replay, rank skew and stream reuse."""
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_device_ready_lifecycle", 2, "_test_mega_moe_ready.py",
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_shared_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Shared execution resources and private SHMEM ownership.

    Description: Compare real serial stacks and alternate streams, then mix private/managed close orders.
    Expectation: Outputs, gradients and updates match while memory remains stable.
    """
    _run_acceptance_worker(
        monkeypatch,
        tmp_path,
        "test_mega_moe_shared_resource_acceptance",
        2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_default_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Default and explicit capacity without Router-provided counts.

    Description: Compare identical representative shapes in ABBA order, including Router timing.
    Expectation: Outputs and gradients agree; stable latency, peak memory and software identity are recorded.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_default_capacity_acceptance", 4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_large_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Execute compute tasks beyond 25,600 and compare all gradients with common MoE."""
    # This graph needs a larger symmetric heap than the small acceptance cases.
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_large_runtime", 2, "_test_mega_moe_runtime.py",
        heap_bytes=2 * 1024**3,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_group_list_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Graph-sized grouped-matmul scratch beyond 16 experts per rank.

    Description: Repeat balanced, empty-expert, skew and single-destination routes with 10 to 128 local experts.
    Expectation: Forward, all gradients and SGD updates match common MoE over eight steps per shape.
    """
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_group_list_isolation", 2, "_test_mega_moe_runtime.py",
        heap_bytes=128 * 1024 * 1024,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_moe_token_permute_grad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validate the metadata-only permutation backward on each device."""
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_moe_token_permute_grad", 2, "_test_moe_token_permute_grad.py",
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_mega_moe_transport_coexistence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: mega moe transport coexistence.

    Description: Alternate push and pull with hotspot routes, retained graphs, checkpoint and profiling.
    Expectation: Compare both transports, retained gradients, checkpoint and instrumented execution.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_push_pull_coexistence", 4,
                           "_test_mega_moe_transport.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_mega_moe_pull_hotspot_small_heap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: mega moe pull hotspot small heap.

    Description: Move the full receive load between destinations on four ranks.
    Expectation: Validate a four-rank hotspot in a heap smaller than push receive storage.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_pull_hotspot_small_heap", 4,
                           "_test_mega_moe_transport.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_mega_moe_transport_qwen_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: mega moe transport qwen shape.

    Description: Compare both modes with common MoE at 1x, 1.25x and 2.5x receive loads.
    Expectation: Validate H5120/I1792/E48/seq4096 on EP4 at 1x, 1.25x and 2.5x receive loads.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_push_pull_qwen_shape", 4,
                           "_test_mega_moe_transport.py")
