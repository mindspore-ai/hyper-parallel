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

"""Import-light launchers for DeepSeek-V4.1 MegaMoe block precision."""

import json
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).with_name("_test_deepseek_v41_megamoe.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
@pytest.mark.parametrize("route,vision,ep_size", [("learned", False, 2), ("hotspot", False, 2), ("learned", True, 1)])
def test_deepseek_v41_megamoe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str, route: str, vision: bool, ep_size: int,
) -> None:
    """
    Feature: DeepSeek-V4.1 native MegaMoe execution.
    Description: Compare full block gradients and updates using two NPU processes.
    Expectation: The selected routing and transport satisfy the reported accuracy gate.
    """
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("DeepSeek-V4.1 MegaMoe ST requires an activated Torch multicore payload")
    result_path = tmp_path / "deepseek_v41_megamoe.json"
    arguments = ["--acceptance", "fp32", "--dispatch-mode", dispatch_mode, "--route", route, "--ep-size", str(ep_size),
                 "--output", str(result_path)]
    if vision:
        arguments.append("--vision")
    monkeypatch.setenv("HP_DSV41_MEGAMOE_ARGUMENTS", json.dumps(arguments))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    with without_inherited_rank_environment():
        torchrun_case(_WORKER, "test_deepseek_v41_megamoe", num_proc=2)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["config"]["synchronize_step_weights"], "Expected identical weights before each step"
    assert all(report["fp32_passed"].values()), "Expected both HF BF16 and MegaMoe to satisfy FP32 bounds"
    assert report["passed"], f"Expected precision pass, got report={report}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
@pytest.mark.parametrize("world_size,dispatch_mode,init_dtype",
                         [(2, "push", "float32"), (4, "pull", "bfloat16")])
def test_deepseek_v41_megamoe_training(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world_size: int, dispatch_mode: str, init_dtype: str,
) -> None:
    """
    Feature: DeepSeek-V4.1 MegaMoe Trainer integration.
    Description: Run real EP/FSDP Trainer steps, gradient accumulation and checkpoint restore.
    Expectation: Expert updates and exact checkpoint restoration pass with EDP-only reductions.
    """
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("DSV4.1 Trainer ST requires an activated Torch multicore payload")
    monkeypatch.setenv("HP_DSV41_TRAINING_OUTPUT", str(tmp_path))
    monkeypatch.setenv("HP_DSV41_TRAINING_EP", "2")
    monkeypatch.setenv("HP_DSV41_TRAINING_MODE", dispatch_mode)
    monkeypatch.setenv("HP_DSV41_MODEL_INIT_DTYPE", init_dtype)
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    worker = str(Path(__file__).with_name("_test_deepseek_v41_megamoe_training.py"))
    with without_inherited_rank_environment():
        torchrun_case(worker, "test_deepseek_v41_megamoe_training", num_proc=world_size)
    for rank in range(world_size):
        report = json.loads((tmp_path / f"rank{rank}" / "result.json").read_text(encoding="utf-8"))
        assert report["passed"] and report["model_checkpoint_exact"], f"Rank {rank} did not pass Trainer checks"
