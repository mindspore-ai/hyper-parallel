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
"""Launcher for the MegaMoe SwiGLU clamp numerical NPU regression."""

from __future__ import annotations

import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


_WORKER = str(Path(__file__).resolve().parent / "_test_swiglu_clamp_st.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_swiglu_clamp_numerics() -> None:
    """Validate clamp boundary numerics and the default unclamped path.

    Feature:
        MegaMoe exposes optional clipped SwiGLU semantics.
    Description:
        Run the boundary-value and no-clamp NPU numerical cases.
    Expectation:
        MegaMoe outputs and checked gradients match their reference paths.
    """
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError(
            "MegaMoe SwiGLU clamp ST requires a payload built with --multicore on"
        )
    os.environ["HP_MEGA_MOE_WORLD_SIZE"] = "1"
    os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"] = str(64 * 1024 * 1024)
    os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = (
        f"tcp://127.0.0.1:{allocate_port()}"
    )
    with without_inherited_rank_environment():
        parallel_run(
            [TorchCase(_WORKER, "test_swiglu_clamp_numerics", num_proc=1)],
            global_num_proc=1,
        )
        parallel_run(
            [TorchCase(_WORKER, "test_swiglu_clamp_input_gradient", num_proc=1)],
            global_num_proc=1,
        )
        parallel_run(
            [TorchCase(_WORKER, "test_swiglu_clamp_ub_scratch", num_proc=1)],
            global_num_proc=1,
        )
