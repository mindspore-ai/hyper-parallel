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
from tests.torch.multicore._evidence_helpers import (
    capture_worker_logs,
    run_acceptance_worker,
)
from tests.common.port_utils import allocate_port
from tests.torch.multicore._case_launcher import run_multicore_case
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe.py")
_PRECISION_WORLD_SIZE = 2
_PERFORMANCE_WORLD_SIZE = 4


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_dynamic_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str) -> None:
    """Feature: Automatic MegaMoe tokens.
    Description: Run growth, ragged/empty lengths, full gradients and retained graphs in each transport.
    Expectation: Independent numerical oracles pass while cached plans and pending backward survive growth.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    # Preserve the total deadline of the two original 420-second workers after consolidation.
    _run_acceptance_worker(monkeypatch, tmp_path, "test_dynamic_tokens", 8,
                           "_test_mega_moe_dynamic_tokens.py", timeout=840)


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for the Torch framework."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError(
            "MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with "
            "--multicore on"
        )


def _run_precision_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the two-rank precision worker with an isolated SHMEM endpoint."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment(), capture_worker_logs(
        _WORKER, "test_mega_moe_level0_balanced", Path("logs/mega_moe_precision"),
    ):
        run_multicore_case(_WORKER, "test_mega_moe_level0_balanced", num_proc=_PRECISION_WORLD_SIZE)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_level0_precision(monkeypatch) -> None:
    """Compare EIO/EOI output, input/route/expert gradients and SGD with common MoE."""
    _run_precision_worker(monkeypatch)


def _run_performance_worker(monkeypatch, result_dir: Path) -> dict:
    """Run common/MegaMoe A/B in one four-rank process group."""
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(result_dir))) / "performance"
    result_path = result_dir / "routed_expert_ab.json"
    monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PERF_WARMUP", "3")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_MEASURED", "5")
    monkeypatch.setenv("HP_MEGA_MOE_MAX_PERF_RATIO", "1.10")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_RESULT", str(result_path))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "62000-62063")
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment(), capture_worker_logs(
        _WORKER, "test_mega_moe_fwd_bwd_performance", result_dir / "worker_logs",
    ):
        run_multicore_case(_WORKER, "test_mega_moe_fwd_bwd_performance", num_proc=_PERFORMANCE_WORLD_SIZE)
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int, worker: str,
    heap_bytes: int = 64 * 1024 * 1024,
    variant: str | None = None, timeout: float = 420,
) -> dict:
    """Launch one real-device acceptance worker and require its rank evidence."""
    _prepare_torch_multicore_test_environment()
    print(
        f"MegaMoe worker {case}: cards={cards}, assigned_devices={os.getenv('ASCEND_RT_VISIBLE_DEVICES')}, "
        f"hccl_base={os.getenv('HCCL_IF_BASE_PORT')}, hccl_ports={os.getenv('HCCL_NPU_SOCKET_PORT_RANGE')}", flush=True,
    )
    worker_path = str(Path(__file__).with_name(worker))

    def launch(result_dir: Path) -> None:
        """Keep device setup and distributed execution outside the CPU helper."""
        with without_inherited_rank_environment(), capture_worker_logs(worker_path, case, result_dir / "worker_logs"):
            run_multicore_case(worker_path, case, num_proc=cards, timeout=timeout)

    return run_acceptance_worker(monkeypatch, tmp_path, case, cards, launch, heap_bytes, variant)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_local_capacity_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
) -> None:
    """Feature: Transport storage lifetime.
    Description: Run retained backward and profiling through the shared memory worker in both modes.
    Expectation: Outputs and all gradients agree across hotspots, tails and receive-buffer reuse.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_local_capacity_lifetime", 2, "_test_mega_moe_memory.py",
        heap_bytes=(34 if dispatch_mode == "pull" else 64) * 1024**2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_poisoned_buffers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
) -> None:
    """Require complete writes after NaN and stale-value poisoning across both transports."""
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_poisoned_buffers", 2, "_test_mega_moe_zeroing.py",
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_resource_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
) -> None:
    """Feature: Shared-resource training and device-owned ready generations.
    Description: Compare serial, alternating-stream and replay stacks in both transports.
    Expectation: SGD, memory, ready generations and launch-error recovery remain correct.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_resource_lifecycle", 2, "_test_mega_moe_lifecycle.py",
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_router_interface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
) -> None:
    """Feature: Router graph without caller-provided counts.
    Description: Run one common/MegaMoe SGD step at the representative shape in each transport.
    Expectation: Output, Router/input/route/expert gradients and parameter updates agree.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_router_interface", 4,
                           "_test_mega_moe_router_interface.py")


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
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_mega_moe_group_list_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
) -> None:
    """Feature: Graph-sized grouped-matmul scratch beyond 16 experts per rank.

    Description: Exercise odd tails, 16/17 and 32/33 boundaries and 128 local experts in both transports.
    Expectation: Forward, all gradients and SGD updates match common MoE across route transitions per shape.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    _run_acceptance_worker(
        monkeypatch, tmp_path, "test_mega_moe_group_list_isolation", 2, "_test_mega_moe_runtime.py",
        heap_bytes=128 * 1024 * 1024,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_mega_moe_subgroups(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Noncontiguous EP subgroups.

    Description: Run two independent groups with external weights in push and pull modes.
    Expectation: Outputs and all gradients match an unsharded reference.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_subgroups", 4, "_test_mega_moe_runtime.py")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential",
)
@pytest.mark.parametrize("variant", ["single", "checkpoint"])
def test_mega_moe_heap_growth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant: str) -> None:
    """Feature: Collective push heap growth.
    Description: Rebuild after one forward while keeping both autograd graphs live.
    Expectation: Repeated reverse backwards match common EP and the old heap is invalidated.
    """
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    monkeypatch.setenv("HP_MEGA_MOE_GROWTH_CASE", variant)
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_heap_growth", 2, "_test_mega_moe_memory.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_mega_moe_model_replacement_and_layouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Model replacement and tensor layout contracts.

    Description: Replace a collected model and exercise strided inputs and external weights in both modes.
    Expectation: The physical heap grows and all outputs, original-parameter gradients and updates match common EP.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_model_replacement_and_layouts", 2,
                           "_test_mega_moe_memory.py")
