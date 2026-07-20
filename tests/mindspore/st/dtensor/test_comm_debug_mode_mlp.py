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
"""Pytest launchers for CommDebugMode MLP distributed tests (2 ranks, hccl).

Three parallel_run waves, each case uses 2 ranks (sum ranks per wave <= 8):

  Wave 1 (test_comm_debug_mode_wave_one):   4 cases × 2 ranks
  Wave 2 (test_comm_debug_mode_wave_two):   4 cases × 2 ranks
  Wave 3 (test_comm_debug_mode_wave_three): 1 case  × 2 ranks

Coverage:
  1. Collective communication counting (get_comm_counts / get_total_counts)
  2. Tracing table non-empty output (generate_comm_debug_tracing_table)
  3. noise_level filtering (level 0 vs level 1 line count comparison)
  4. Module boundary annotation (noise_level=2 with module tracker)
  5. Platform method restoration after __exit__ (identity check)
  6. Collective count accumulation across multiple forward passes
  7. Parameter and sharding info collection (get_parameter_info / get_sharding_info)
  8. Tracing table file export (log_comm_debug_tracing_table_to_file)
  9. JSON export (generate_json_dump)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_WORKER = str(Path(__file__).resolve().parent / "_test_comm_debug_mode_mlp.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        MindSporeCase(_WORKER, case_name, master_port, 2, 2)
        for case_name, master_port in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_comm_debug_mode_wave_one():
    """
    Feature: CommDebugMode basic tracing — 4 cases × 2 ranks in one parallel_run
    Description:
        1. test_comm_debug_mode_captures_collectives:
           Verify collective operations are counted via get_comm_counts() / get_total_counts().
        2. test_comm_debug_mode_debug_string:
           Verify generate_comm_debug_tracing_table() returns non-empty content with group info.
        3. test_comm_debug_mode_tracing_table:
           Verify noise_level=1 table has >= lines than noise_level=0 table.
        4. test_comm_debug_mode_with_module_tracker:
           Verify CommDebugMode(module=model) captures collectives correctly.
    Expectation: All 4 worker cases run successfully.
    """
    _run_group(
        ("test_comm_debug_mode_captures_collectives", 18501),
        ("test_comm_debug_mode_debug_string", 18502),
        ("test_comm_debug_mode_tracing_table", 18503),
        ("test_comm_debug_mode_with_module_tracker", 18504),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_comm_debug_mode_wave_two():
    """
    Feature: CommDebugMode lifecycle and info APIs — 4 cases × 2 ranks in one parallel_run
    Description:
        1. test_comm_debug_mode_restores_platform:
           Verify platform collective methods are exactly restored after __exit__ (identity check).
        2. test_comm_debug_mode_multiple_forwards:
           Verify collective counts accumulate additively across 3 consecutive forward passes.
        3. test_comm_debug_mode_parameter_and_sharding_info:
           Verify get_parameter_info() is non-empty and get_sharding_info() contains DTensor placements.
        4. test_comm_debug_mode_log_to_file:
           Verify log_comm_debug_tracing_table_to_file() writes valid content to disk.
    Expectation: All 4 worker cases run successfully.
    """
    _run_group(
        ("test_comm_debug_mode_restores_platform", 18505),
        ("test_comm_debug_mode_multiple_forwards", 18506),
        ("test_comm_debug_mode_parameter_and_sharding_info", 18507),
        ("test_comm_debug_mode_log_to_file", 18508),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_comm_debug_mode_wave_three():
    """
    Feature: CommDebugMode JSON export — 1 case × 2 ranks in one parallel_run
    Description:
        1. test_comm_debug_mode_json_dump:
           Verify generate_json_dump() produces a valid JSON file containing comm_counts,
           total_counts, and records fields with correct collective count values.
    Expectation: Worker case runs successfully.
    """
    _run_group(
        ("test_comm_debug_mode_json_dump", 18509),
    )
