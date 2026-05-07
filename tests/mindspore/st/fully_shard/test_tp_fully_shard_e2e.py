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

import pytest
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_TP_FSDP = os.path.join(os.path.dirname(__file__), "_tp_fully_shard_e2e.py")
pytestmark = pytest.mark.skip(reason="MindSpore ST gate issue, tracked separately")


def _build_tp_fully_shard_case(case_name: str, port: int, worker_num: int, device_num: int, rank_size: int):
    """Build one TP + fully_shard e2e case for the shared msrun wrapper."""
    return MindSporeCase(
        _TEST_TP_FSDP,
        case_name,
        port,
        worker_num,
        device_num,
        rank_size,
    )


def _run_tp_fully_shard_cases(*cases: MindSporeCase):
    """Launch compatible TP + fully_shard e2e cases together."""
    parallel_run(list(cases))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_basic_shape_suite():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run the basic TP + fully_shard training case and compare loss/grad with standalone.
    Expectation: Distributed local losses and local shard gradients match standalone slices.
    """
    _run_tp_fully_shard_cases(
        _build_tp_fully_shard_case("test_tp_plus_fully_shard_loss_and_grad_match_standalone", 18199, 4, 4, 2)
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_setup_and_compat_suite():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run the empty-weight and mesh=None compatibility TP + fully_shard training cases together and
                 compare loss/grad with standalone.
    Expectation: Distributed local losses and local shard gradients match standalone slices.
    """
    _run_tp_fully_shard_cases(
        _build_tp_fully_shard_case("test_tp_plus_fully_shard_empty_weight_match_standalone", 18203, 4, 4, 2),
        _build_tp_fully_shard_case(
            "test_tp_plus_fully_shard_mesh_none_compat_match_standalone",
            18206,
            2,
            2,
            2,
        ),
        _build_tp_fully_shard_case(
            "test_pure_tp_replicate_grad_managed_by_fully_shard_matches_standalone",
            18208,
            2,
            2,
            2,
        ),
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_same_dim_non_dim0_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters on the same tensor dim.
    Description: Run a TP + fully_shard case where TP and fully_shard both shard weight dim 1 and compare
                 local loss/grad with standalone.
    Expectation: Distributed local loss and local shard gradients match the TP-first then fully_shard slices.
    """
    _run_tp_fully_shard_cases(
        _build_tp_fully_shard_case("test_tp_plus_fully_shard_same_dim_non_dim0_match_standalone", 18207, 4, 4, 2)
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_plus_tp_on_3d_root_mesh_match_standalone():
    """
    Feature: HSDP + TP on a 3D root mesh.
    Description: Run a `(dp, fsdp, tp)` mixed-parallel HSDP + TP training case and compare loss/grad.
    Expectation: Distributed local loss and local shard gradients match standalone slices.
    """
    _run_tp_fully_shard_cases(
        _build_tp_fully_shard_case("test_hsdp_plus_tp_on_3d_root_mesh_match_standalone", 18205, 8, 8, 2)
    )
