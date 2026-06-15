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
"""msrun entry for TP + fully_shard end-to-end tests."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_TP_FSDP = os.path.join(os.path.dirname(__file__), "_tp_fully_shard_e2e.py")
_MAX_PARALLEL_DEVICE_NUM = 8


def _build_tp_fully_shard_case(case_name: str, worker_num: int) -> MindSporeCase:
    """Build one TP + fully_shard e2e case for the shared msrun wrapper."""
    return MindSporeCase(
        _TEST_TP_FSDP,
        case_name,
        worker_num=worker_num,
        local_worker_num=worker_num,
    )


def _run_tp_fully_shard_group(case_names: list[str], worker_num: int) -> None:
    """Launch TP + fully_shard cases in batches that fit the local device budget."""
    batch_size = max(1, _MAX_PARALLEL_DEVICE_NUM // worker_num)
    for start in range(0, len(case_names), batch_size):
        case_batch = case_names[start: start + batch_size]
        parallel_run([
            _build_tp_fully_shard_case(case_name, worker_num)
            for case_name in case_batch
        ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_four_card_suite() -> None:
    """
    Feature: TP + fully_shard on 4-card meshes.
    Description:
        1. Basic TP + fully_shard local loss/grad parity with standalone.
        2. TP + fully_shard fp32-reduce main_grad local loss/grad parity and optimizer-step coverage.
        3. Same tensor dim TP + fully_shard non-dim0 sharding parity.
        4. Empty-weight initialization and load-state path parity.
    Expectation: 4-card cases run in parallel batches and match standalone references.
    """
    _run_tp_fully_shard_group([
        "test_tp_plus_fully_shard_loss_and_grad_match_standalone",
        "test_tp_plus_fully_shard_fp32_main_grad_match_standalone",
        "test_tp_plus_fully_shard_same_dim_non_dim0_match_standalone",
        "test_tp_plus_fully_shard_empty_weight_match_standalone",
    ], worker_num=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_mesh_none_suite() -> None:
    """
    Feature: TP + fully_shard(mesh=None) compatibility mode.
    Description:
        1. TP-sharded DTensor compatibility mode local loss/grad parity.
        2. Pure-TP replicated DTensor gradients managed by fully_shard(mesh=None).
    Expectation: 2-card cases run in parallel and match standalone references.
    """
    _run_tp_fully_shard_group([
        "test_tp_plus_fully_shard_mesh_none_compat_match_standalone",
        "test_pure_tp_replicate_grad_managed_by_fully_shard_matches_standalone",
    ], worker_num=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_plus_tp_on_3d_root_mesh_match_standalone() -> None:
    """
    Feature: HSDP + TP on a 3D root mesh.
    Description: Run a `(dp, fsdp, tp)` mixed-parallel HSDP + TP training case and compare loss/grad.
    Expectation: Distributed local loss and local shard gradients match standalone slices.
    """
    _run_tp_fully_shard_group([
        "test_hsdp_plus_tp_on_3d_root_mesh_match_standalone",
    ], worker_num=8)
