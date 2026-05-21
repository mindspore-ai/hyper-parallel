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
"""Unit tests for the SAPP-PPB ILP solver.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/test_sapp_solver.py
"""
import os
import pytest

# pylint: disable=C0413
from sapp_ppb.utils.layer import generate_layers_list
from sapp_ppb.utils.config import parse_training_config
from sapp_ppb.sapp.sapp_pipeline import SappPipeline


class TestSappSolver:
    """A test class for testing sapp solver."""

    def test_sapp_solver_vpp1(self):
        """Solve ``test.json`` at pp=16 / mb=16 / vpp=1 and verify time + memory layout."""
        layers = generate_layers_list(
            os.path.dirname(os.path.abspath(__file__)), "test"
        )
        pipe = SappPipeline(
            model_name="test",
            num_of_stage=16,
            num_of_micro_batch=16,
            max_memory=53000,
            layers=layers,
            num_of_interleave=1,
            vpp_less_memory=True,
        )
        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=20)
        total_time = pipe.simulate(show=False, file_name=None)
        mem_par = pipe.get_memory_parameter()
        mem_act = pipe.get_memory_activation()
        expected_mem_par = [
            18915.0, 7915.0, 9498.0, 9498.0, 9498.0, 9498.0, 9498.0, 9498.0,
            9498.0, 9498.0, 11081.0, 11081.0, 9498.0, 9498.0, 9498.0, 11971.0,
        ]
        expected_mem_act = [
            1748.0, 2542.0, 2574.0, 2574.0, 3368.0, 3368.0, 4162.0, 4162.0,
            4956.0, 4956.0, 5782.0, 5782.0, 4956.0, 4956.0, 4956.0, 4956.0,
        ]

        # Different CBC/PuLP builds may select a near-equivalent feasible stage
        # assignment. Keep the check focused on solver quality and memory
        # feasibility instead of one exact stage ordering.
        assert total_time == pytest.approx(110070, abs=1200)
        assert len(mem_par[0]) == 16
        assert len(mem_act[0]) == 16
        assert sorted(mem_par[0]) == pytest.approx(sorted(expected_mem_par), abs=200)
        assert sorted(mem_act[0]) == pytest.approx(sorted(expected_mem_act), abs=200)
        assert all(mem_p + mem_a <= 53000 for mem_p, mem_a in zip(mem_par[0], mem_act[0]))

    def test_sapp_simulate_vpp3(self):
        """Use ``simulate_yaml`` with a fixed offset/recompute configuration at vpp=3."""
        layers = generate_layers_list(os.path.dirname(os.path.abspath(__file__)), "sim")
        pipe = SappPipeline(
            model_name="sim",
            num_of_stage=16,
            num_of_micro_batch=16,
            max_memory=53000,
            layers=layers,
            num_of_interleave=3,
            vpp_less_memory=True,
        )
        pipe.construct_problem(solver="pulp")
        total_time = pipe.simulate_yaml(
            {
                "offset": [
                    [-2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
                ],
                "recompute": [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                ],
                "select_recompute": [
                    [1, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 1, 0, 0, 0, 1, 0, 0],
                    [3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 1, 0, 0, 1, 1, 0, 0],
                ],
                "select_comm_recompute": [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                ],
            },
            show=False,
            interleave_num=3,
            file_name=None,
        )

        assert total_time == pytest.approx(154184.4, abs=0.1)

    def test_sapp_seq_pp(self):
        """Solve a small problem with sequence pipeline (``seq_split_num=2``)."""
        cur_path = os.path.dirname(os.path.abspath(__file__))
        work_path = os.path.join(cur_path, "seqpp")
        config_file_path = os.path.join(work_path, "seq_config.yaml")
        extracted_training_params = parse_training_config(config_file_path)
        layers = generate_layers_list(work_path, "seq")
        pipe = SappPipeline(
            model_name="seq",
            num_of_stage=8,
            num_of_micro_batch=32,
            max_memory=40000,
            layers=layers,
            num_of_interleave=1,
            seq_split_num=2,
            extracted_training_params=extracted_training_params,
        )
        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=90, dump_folder=None)
        pipe.simulate(show=False, file_name=None)

        activation_nums = [[9, 8, 7, 6, 5, 4, 3, 2]]
        peak_mems = []
        expected_peak_mems = [33407.0, 36592.0, 33155.0, 28670.0, 24185.0, 29550.0, 22822.5, 13395.0]
        for s in range(8):
            param_mem = pipe.problem_.stage_param_memory(
                pipe.problem_.variables_,
                pipe.problem_.layers_sorted_,
                s,
                pipe.problem_.num_of_stage_,
                pipe.problem_.num_of_interleave_
            ).value()

            act_mem = pipe.problem_.stage_active_memory(
                pipe.problem_.variables_,
                pipe.problem_.layers_sorted_,
                s,
                pipe.problem_.num_of_interleave_,
                activation_nums
            ).value()
            peak_mem = act_mem + param_mem
            peak_mems.append(peak_mem)

        bias = [(x - y) / y for x, y in zip(peak_mems, expected_peak_mems)]
        assert all(x < 0.1 for x in bias)
        assert all(x < 40000 for x in peak_mems)
