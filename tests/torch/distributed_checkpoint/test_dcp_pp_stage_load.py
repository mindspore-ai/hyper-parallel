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
"""test DCP broadcast load under pipeline parallelism, on a (pp=2, dp=2, tp=2) mesh"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_PP_STAGE_LOAD = "dcp_pp_stage_load.py"

# The mesh these cases need is (pp=2, dp=2, tp=2), so each of them takes the whole budget
# and they run one after another rather than side by side.
_WORLD_SIZE = 8


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pp_stage_load_different_parameters():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp_stages_hold_different_parameters
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_stages_hold_different_parameters",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pp_stage_load_without_replicated_shards():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp_stage_without_replicated_shards
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_stage_without_replicated_shards",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_dcp_pp_stage_load_interleaved_stages():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp_stages_interleave_in_the_shard_order
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_stages_interleave_in_the_shard_order",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_dcp_pp_stage_load_tied_parameter():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp_tied_parameter_spans_both_stages
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_tied_parameter_spans_both_stages",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pp4_stage_load_different_parameters():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp4_stages_hold_different_parameters
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp4_stages_hold_different_parameters",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pp4_stage_load_tied_across_distant_stages():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp4_tied_parameter_skips_the_middle_stages
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp4_tied_parameter_skips_the_middle_stages",
                  num_proc=_WORLD_SIZE),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_pp_stage_load_gloo():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pp_stages_hold_different_parameters
        2.test_dcp_load_pp_stage_without_replicated_shards
        3.test_dcp_load_pp4_tied_parameter_skips_the_middle_stages
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_stages_hold_different_parameters",
                  num_proc=_WORLD_SIZE),
    ])
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp_stage_without_replicated_shards",
                  num_proc=_WORLD_SIZE),
    ])
    parallel_run([
        TorchCase(DCP_PP_STAGE_LOAD, "test_dcp_load_pp4_tied_parameter_skips_the_middle_stages",
                  num_proc=_WORLD_SIZE),
    ])
