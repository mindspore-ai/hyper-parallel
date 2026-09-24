# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test vpp schedule"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

# Absolute path so torchrun+pytest finds the worker regardless of cwd.
_VPP_SCHEDULE = str(Path(__file__).resolve().parent / "vpp_schedule.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_vpp_schedule_group1():
    """
    Feature: parallel run case in pipeline_parallel
    Description:
        Runs ``test_vpp`` / ``run_parallel``. Before the VPP schedule, each **PP domain**
        (two ranks per domain when world size is 4) builds its own ``DeviceMesh``,
        calls ``manual_seed(parallel_seed, domain_mesh)`` with **different seeds per domain**,
        then runs ``torch.randn_like`` on a **sharded** ``DTensor`` on that mesh.

        ``simple_mlp.MLP`` still includes ``Dropout(p=0)`` for a no-op random-style module in the
        main network without affecting ``allclose`` vs standalone.

    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp", 12346, 4)
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vpp_dxdw_split():
    """
    Feature: PyTorch VPP dx/dw split under concurrent backward/forward execution.
    Description: Run four ranks with split input/weight backward and compare every local gradient.
    Expectation: Pipeline outputs and gradients match the full-model reference.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_dxdw_split", num_proc=4)
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vpp_pipeline_swap():
    """
    Feature: PyTorch pipeline activation swap under Interleaved 1F1B.
    Description: Run paired no-swap/swap phases on the existing four-rank,
        two-local-chunk VPP layout and compare outputs, gradients, and peak memory.
    Expectation: Accuracy matches and swap reduces peak NPU memory on every rank.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_pipeline_swap", num_proc=4)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_vpp_schedule_group1_gloo():
    """
    Feature: parallel run case in pipeline_parallel
    Description:
        Runs ``test_vpp`` / ``run_parallel``. Before the VPP schedule, each **PP domain**
        (two ranks per domain when world size is 4) builds its own ``DeviceMesh``,
        calls ``manual_seed(parallel_seed, domain_mesh)`` with **different seeds per domain**,
        then runs ``torch.randn_like`` on a **sharded** ``DTensor`` on that mesh.

        ``simple_mlp.MLP`` still includes ``Dropout(p=0)`` for a no-op random-style module in the
        main network without affecting ``allclose`` vs standalone.

    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp", num_proc=4)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_vpp_schedule_deep_warmup():
    """
    Feature: Interleaved 1F1B deep-warmup (vpp=2, M=3*PP) correctness.
    Description: Launch ``test_vpp_deep_warmup`` on 4 ranks (pp=4, vpp=2,
        micro_batch_num=12). Regression test for the zero-width DATA_LOAD
        splice: when per-FWD DATA_LOAD slots widen each rank's schedule columns
        by its warmup depth, the last rank's BWD_RECV lands after the BWD that
        consumes it (garbage grads, then irecv(None)).
    Expectation: Worker exits 0 on every rank; losses and weights match the
        standalone reference.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_deep_warmup", 12348, 4)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_vpp_schedule_deep_warmup_gloo():
    """
    Feature: Interleaved 1F1B deep-warmup (vpp=2, M=3*PP) correctness on gloo.
    Description: The ``test_vpp_deep_warmup`` worker on 4 CPU/gloo ranks — same
        regression shape as :func:`test_vpp_schedule_deep_warmup`.
    Expectation: Worker exits 0 on every rank; losses and weights match the
        standalone reference.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_deep_warmup", num_proc=4)
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vpp_dynamic_batch_p2p_cold_start():
    """
    Feature: Dynamic-shape VPP with a cold batched-P2P process group.
    Description: Launch eight ranks as two four-rank PP groups, with two virtual stages per rank.
    Expectation: The first batched peer operation does not hang during communicator initialization.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_dynamic_batch_p2p_cold_start", num_proc=8)
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vpp_dynamic_multi_stream_p2p_cold_start():
    """
    Feature: Dynamic-shape VPP with peer-specific P2P groups.
    Description: Launch two four-rank PP domains with an independent group for each PP edge.
    Expectation: Group initialization and the first batched peer operations complete without deadlock.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp_dynamic_multi_stream_p2p_cold_start", num_proc=8)
    ])
