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
"""Launch single-card precision checks for the MegaGate module entry."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_gate.py")


def _run_worker(case: str) -> None:
    """Run one MegaGate module worker on one NPU."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available("hyper_parallel_mega_gate_torch"):
        raise RuntimeError("MegaGate ST requires a wheel or PYTHONPATH payload built with --multicore on")
    with without_inherited_rank_environment():
        parallel_run([TorchCase(_WORKER, case, num_proc=1)], global_num_proc=1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_basic_parity() -> None:
    """Protect one representative native forward and backward path."""
    _run_worker("test_mega_gate_basic_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_forward_parity() -> None:
    """Cover text, vision, empty, dynamic, and padded forward cases."""
    _run_worker("test_mega_gate_forward_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_backward_parity() -> None:
    """Cover text, vision, dynamic, and retained backward cases."""
    _run_worker("test_mega_gate_backward_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_route_math_parity() -> None:
    """Cover exact operation order, extreme values, tails, and dynamic TopK."""
    _run_worker("test_mega_gate_route_math_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_pipeline_profile() -> None:
    """Validate forward and backward mega-kernel traces."""
    _run_worker("test_mega_gate_pipeline_profile")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_training_graphs() -> None:
    """Cover dynamic graphs and asynchronous optimizer steps."""
    _run_worker("test_mega_gate_training_graphs")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_checkpoint_training() -> None:
    """Cover both checkpoint implementations at the model Gate shape."""
    _run_worker("test_mega_gate_checkpoint_training")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_memory_stability() -> None:
    """Check allocation release across standard training lengths."""
    _run_worker("test_mega_gate_memory_stability")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_symbol_resolution_concurrency() -> None:
    """Cover concurrent first-use Route and RouteGrad symbol resolution."""
    _run_worker("test_mega_gate_symbol_resolution_concurrency")
