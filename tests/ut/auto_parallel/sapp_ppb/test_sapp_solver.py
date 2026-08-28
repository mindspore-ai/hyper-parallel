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
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

# pylint: disable=C0413,W0212,R0903,E1120
# pylint: disable=invalid-name
from hyper_parallel.auto_parallel.sapp_ppb.sapp import sapp_pipeline as pipeline_module
from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline, choose_interleave, flatten
from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import generate_layers_list
from hyper_parallel.auto_parallel.sapp_ppb.utils.config import parse_training_config
from hyper_parallel.auto_parallel.sapp_ppb.simulator.pipeline_builder import PipelineBuilder
from hyper_parallel.auto_parallel.sapp_ppb.simulator.plot_manager import PlotMgr
from hyper_parallel.auto_parallel.sapp_ppb.simulator.pp_simulator import PipelineSimulator
from hyper_parallel.auto_parallel.sapp_ppb.simulator.causal_error import CausalCommError, CausalError
from hyper_parallel.auto_parallel.sapp_ppb.simulator.sim_block import (
    HeadBlockSim,
    MicroBlockSim,
    RecBlockSim,
    SendBlockSim,
)
from hyper_parallel.auto_parallel.sapp_ppb.simulator import utils as simulator_utils
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer
import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute


class _Var:
    """Object exposing varValue for yaml_from_internal."""

    def __init__(self, value: Any) -> None:
        """Store a fake solver variable value."""
        self.varValue = value


class _FakeProblem:
    """Small problem facade for SappPipeline wrapper methods."""

    def __init__(self, body_name: str = "body", has_memory: bool = True) -> None:
        """Build fake problem state."""
        self.body_name = body_name
        self.has_memory = has_memory
        self.recompute_considered_ = {rec: True for rec in Recompute.TYPE}
        self.variables_ = {
            body_name: {
                rec: [[_Var(1), _Var(1)]]
                for rec in Recompute.TYPE
            }
        }

    def has_some_memory_info(self) -> bool:
        """Return whether memory APIs should be used."""
        return self.has_memory

    def solve(self, time_limit: int, dump_folder: str) -> None:
        """Record-compatible solve placeholder."""
        self.solved = (time_limit, dump_folder)

    def result(self) -> dict[str, list[list[str]]]:
        """Return a deterministic compact result."""
        return {"body": [["stage0", "stage1"]]}

    def get_simulator_memory_activation(self) -> list[list[int]]:
        """Return activation memory for two stages."""
        return [[2, 3]]

    def get_simulator_memory_parameter(self) -> list[list[int]]:
        """Return parameter memory for two stages."""
        return [[5, 7]]

    def get_simulator_forward_time(self) -> list[list[int]]:
        """Return forward time for two stages."""
        return [[1, 1]]

    def get_simulator_recompute_time(self) -> list[list[int]]:
        """Return recompute overhead for two stages."""
        return [[0, 0]]

    def get_simulator_time(self) -> list[list[int]]:
        """Return total time for two stages."""
        return [[3, 3]]

    def compute_activation_nums(self, pp: int, interleave: int, micro_batch: int) -> list[list[int]]:
        """Return simple activation multipliers."""
        del micro_batch
        return [[pp - stage for stage in range(pp)] for _ in range(interleave)]

    def compute_less_activation_nums(self, pp: int, interleave: int) -> list[list[int]]:
        """Return simple less-memory activation multipliers."""
        return [[pp - stage for stage in range(pp)] for _ in range(interleave)]

    def compute_activation_nums_dual(self, pp: int, interleave: int, micro_batch: int) -> list[list[int]]:
        """Return simple dual-pipe activation multipliers."""
        del micro_batch
        return [[pp - stage for stage in range(pp)] for _ in range(interleave)]

    def compute_activation_seq_nums(
            self, pp: int, interleave: int, seq_split_num: int,
            micro_batch: int, less_memory: bool) -> list[list[int]]:
        """Return simple sequence-pipeline activation multipliers."""
        del seq_split_num, micro_batch, less_memory
        return [[pp - stage for stage in range(pp)] for _ in range(interleave)]


class _ChoicePipe:
    """Fake pipeline used to test choose_interleave."""

    times = {1: 40, 2: 20, 3: 30, 4: 25}

    def __init__(self, **kwargs: Any) -> None:
        """Record the interleave being evaluated."""
        self.interleave = kwargs["num_of_interleave"]

    def construct_problem(self, solver: str = "pulp") -> None:
        """No-op construct."""
        del solver

    def solve_problem(self) -> None:
        """No-op solve."""

    def simulate(self, show: bool = False) -> int:
        """Return a deterministic time by interleave."""
        del show
        return self.times[self.interleave]

    def get_result(self) -> dict[str, list[list[str]]]:
        """Return the chosen interleave in the result."""
        return {"interleave": [[str(self.interleave)]]}


def _make_layers(body_layers=4):
    """Build HEAD/BODY/TAIL layers for direct SappPipeline tests."""
    activation = {rec: rec.value + 1 for rec in Recompute.TYPE}
    return [
        Layer("unit", "head", Layer.type_enum.HEAD, 1, time=1, forward_time=None, memory_parameter=3),
        Layer(
            "unit",
            "body",
            Layer.type_enum.BODY,
            body_layers,
            time=2,
            forward_time=None,
            memory_parameter=5,
            memory_activation_rec=activation,
        ),
        Layer("unit", "tail", Layer.type_enum.TAIL, 1, time=1, forward_time=None, memory_parameter=7),
    ]


def _make_pipeline(has_memory=True):
    """Build a two-stage SappPipeline with a fake problem attached."""
    layers = _make_layers()
    pipe = SappPipeline("unit", 2, 2, 128, layers, num_of_interleave=1)
    pipe.problem_ = _FakeProblem("body", has_memory=has_memory)
    return pipe, layers[1]


def _manual_strategy():
    """Return a legal manual recompute strategy for one BODY layer."""
    return {
        Recompute.TYPE.NONE: [[1, 1]],
        Recompute.TYPE.SLCT: [[0, 0]],
        Recompute.TYPE.COMM: [[0, 0]],
        Recompute.TYPE.BOTH: [[0, 0]],
        Recompute.TYPE.FULL: [[1, 1]],
    }


class TestSappSolver:
    """A test class for testing sapp solver."""

    def test_compute_activation_seq_nums_single_interleave(self):
        """compute_activation_seq_nums produces correct counts for a single interleave chunk."""
        nums = SappSolver.compute_activation_seq_nums(
            num_of_stage=4, num_of_interleave=1,
            seq_split_num=2, micro_batch=8, less_memory=False,
        )
        assert len(nums) == 1
        assert len(nums[0]) == 4
        # With interleave=1: nums[0][s] = num_of_stage - s + seq_split_num - 1
        assert nums[0][0] == 5
        assert nums[0][3] == 2

    def test_compute_activation_seq_nums_multi_interleave_normal(self):
        """compute_activation_seq_nums fills 2-D result capped by micro_batch."""
        nums = SappSolver.compute_activation_seq_nums(
            num_of_stage=4, num_of_interleave=2,
            seq_split_num=2, micro_batch=16, less_memory=False,
        )
        assert len(nums) == 2
        for chunk in nums:
            assert len(chunk) == 4
            for v in chunk:
                assert v <= 16

    def test_compute_activation_seq_nums_multi_interleave_less_memory(self):
        """compute_activation_seq_nums uses gap=1 when less_memory is True."""
        nums = SappSolver.compute_activation_seq_nums(
            num_of_stage=4, num_of_interleave=2,
            seq_split_num=2, micro_batch=16, less_memory=True,
        )
        assert len(nums) == 2
        assert all(v <= 16 for chunk in nums for v in chunk)

    def test_seq_mem_static_formulas(self):
        """Seq-PP memory-formula static methods return finite float values."""
        params = {
            "batch_size": 1,
            "num_heads": 16,
            "seq_length": 2048,
            "head_dim": 128,
            "model_parallel": 2,
            "hidden_size": 4096,
            "vocab_size": 32000,
        }
        act = SappSolver.compute_seq_mem_activation(1000.0, params, 2)
        assert isinstance(act, float)

        prm = SappSolver.compute_seq_mem_parameter(1000.0, params)
        assert isinstance(prm, float)

        # mp > 1 branch
        head_mp2 = SappSolver.compute_seq_mem_head_cost(1000.0, params, 2)
        assert isinstance(head_mp2, float)

        # mp == 1 branch
        params_mp1 = dict(params)
        params_mp1["model_parallel"] = 1
        head_mp1 = SappSolver.compute_seq_mem_head_cost(1000.0, params_mp1, 2)
        assert isinstance(head_mp1, float)

        tail = SappSolver.compute_seq_mem_tail_cost(1000.0, params, 2)
        assert isinstance(tail, float)

    def test_stage_param_and_active_memory_no_solver(self):
        """stage_param_memory and stage_active_memory evaluate with injected varValues.

        The LP problem is constructed normally but ``solve_problem`` is never called.
        Instead every binary variable is fixed to ``varValue=1`` so that the LP
        expressions can be evaluated without invoking the CBC solver.
        """
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
        # Inject varValue=1 into every body-layer LP variable so expressions can
        # be evaluated without running the CBC solver.
        prob = pipe.problem_
        for body_layer in prob.layers_sorted_[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                for inter_id in range(prob.num_of_interleave_):
                    for stage_id in range(prob.num_of_stage_):
                        prob.variables_[body_layer.name_][rec][inter_id][stage_id].varValue = 1

        activation_nums = [[9, 8, 7, 6, 5, 4, 3, 2]]
        for s in range(8):
            param_val = prob.stage_param_memory(
                prob.variables_, prob.layers_sorted_,
                s, prob.num_of_stage_, prob.num_of_interleave_,
            ).value()
            act_val = prob.stage_active_memory(
                prob.variables_, prob.layers_sorted_,
                s, prob.num_of_interleave_, activation_nums,
            ).value()
            assert param_val is not None and param_val >= 0
            assert act_val is not None and act_val >= 0

    def test_sapp_pipeline_wrappers_and_manual_strategy(self, monkeypatch, tmp_path):
        """SappPipeline wrapper methods handle fake problem outputs and manual strategies."""
        pipe, body = _make_pipeline()
        monkeypatch.setattr(SappPipeline, "_construct_problem_pulp_", lambda self: _FakeProblem("body"))
        pipe.construct_problem("other")
        assert pipe.problem_.body_name == "body"
        pipe.construct_problem("unknown")
        pipe.solve_problem(time_limit=5, dump_folder=str(tmp_path))
        assert pipe.problem_.solved == (5, str(tmp_path))
        assert pipe.get_result() == {"body": [["stage0", "stage1"]]}
        assert pipe.get_memory_activation() == [[2, 3]]
        assert pipe.get_memory_parameter() == [[5, 7]]
        assert pipe.get_fw_time() == [[1, 1]]
        assert pipe.get_recompute_time() == [[0, 0]]
        assert pipe.get_time() == [[3, 3]]
        assert pipe.naive_layer_per_stage(4) == [[2, 2]]
        pipe.print_yaml_results()

        aggregate_strategy = _manual_strategy()
        split_strategy = pipe.split_layer_per_recompute({
            rec: [values[0][:]]
            for rec, values in aggregate_strategy.items()
        })
        assert pipe.fuse_layer_per_recompute(split_strategy) == aggregate_strategy
        assert pipe.get_manual_memory_parameter(split_strategy) == [[13, 17]]
        assert pipe.get_manual_memory_activation(split_strategy) == [[6, 6]]
        assert pipe.get_manual_time(split_strategy) == [[15, 15]]
        assert pipe.get_manual_fw_time(split_strategy) == [[5, 5]]
        assert pipe.get_manual_recompute_time(split_strategy) == [[0.0, 0.0]]
        pipe.debug_print_manual_theoretical_memory(split_strategy)

        end_time = pipe.simulate_manual(split_strategy, show=False)
        assert end_time > 0
        yaml_time = pipe.simulate_yaml({Recompute.OFFSET: 0}, show=False)
        assert yaml_time > 0

        broken = {body: {rec: [[0, 0]] for rec in Recompute.TYPE}}
        assert pipe.simulate_manual(broken, show=False, interleave_num=2) == sys.maxsize
        broken[body][Recompute.TYPE.NONE][0][0] = -1
        with pytest.raises(ValueError):
            pipe.simulate_manual(broken, show=False)

    def test_sapp_pipeline_simulation_branches_and_manual_files(self, tmp_path, monkeypatch):
        """Naive/manual simulation branches normalize YAML and dispatch to simulation helpers."""
        pipe, _ = _make_pipeline()
        calls = []
        monkeypatch.setattr(pipe, "simulate_yaml", lambda *args, **kwargs: calls.append((args, kwargs)) or 9)
        pipe.simulate_naive(_make_layers(body_layers=4), str(tmp_path))
        assert len(calls) == 2

        calls.clear()
        pipe.simulate_naive(_make_layers(body_layers=5), str(tmp_path))
        assert len(calls) == 1

        manual_file = tmp_path / "manual.yaml"
        manual_file.write_text(
            """
manual:
  offset: [0, 0]
  recompute: [0, 0]
  select_recompute: [0, 0]
  select_comm_recompute: [0, 0]
  interleave_num: 1
  show: true
  file_name: manual.svg
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(pipe, "simulate", lambda *args, **kwargs: 7)
        monkeypatch.setattr(pipeline_module.plt, "savefig", lambda file_name: calls.append(("save", file_name)))
        monkeypatch.setattr(pipeline_module.plt, "show", lambda: calls.append(("show",)))
        pipe.simulate_comparison(str(manual_file), str(tmp_path))
        pipe.simulate_only_manual(str(manual_file), str(tmp_path))
        assert any(item == ("show",) for item in calls)
        assert any(isinstance(item, tuple) and item[0] == "save" for item in calls)

        no_memory_pipe, _ = _make_pipeline(has_memory=False)
        assert no_memory_pipe.simulation([[1, 1]], show=False) > 0

    def test_choose_interleave_flatten_and_layer_distribution(self, monkeypatch):
        """Top-level helpers choose the best fake interleave and flatten VPP matrices."""
        monkeypatch.setattr(pipeline_module, "SappPipeline", _ChoicePipe)
        best_interleave, best_time, distribution = choose_interleave("unit", 2, 2, 128, _make_layers())
        assert best_interleave == 2
        assert best_time == 20
        assert distribution == {"interleave": [["2"]]}
        assert flatten([[1, 2], [3, 4], [5, 6]]) == [9, 12]

    def test_pipeline_simulator_small_runs_and_drawing(self, tmp_path):
        """PipelineSimulator runs with and without communication and renders non-interactively."""
        simulator = PipelineSimulator(
            [1, 1],
            2,
            comm_time=0.1,
            block_mem=2,
            block_mem_par=3,
            layer_recompute=True,
        ).run(comm=False, print_info=False)
        assert "comm" not in simulator.bubbles
        assert len(simulator.peak_memory) == 2
        simulator.print_info()
        simulator.draw(comm=False, connect=False)
        output_file = tmp_path / "pipeline.svg"
        simulator.save(str(output_file), comm=False, connect=False)
        assert output_file.exists()

        vpp_simulator = PipelineSimulator(
            [[1, 1], [1, 2]],
            4,
            comm_time=0.1,
            layer_recompute=[[0, 0], [1, 0]],
            block_mem=[[1, 1], [2, 2]],
            block_mem_par=[[1, 1], [1, 1]],
            method="vpp",
        ).run(comm=False, print_info=False)
        assert "comm" not in vpp_simulator.bubbles
        assert vpp_simulator.end_time > 0
        plt.close("all")

    def test_pipeline_builder_blocks_and_simulator_utils(self):
        """Simulator primitives expose expected labels, geometry and error paths."""
        assert simulator_utils.format_2d_inputs(3, 2, 2).shape == (2, 2)
        assert simulator_utils.format_2d_inputs([1, 2], 1, 2).tolist() == [[1, 2]]
        assert simulator_utils.format_2d_inputs([[1, 2]], 1, 2).tolist() == [[1, 2]]
        with pytest.raises(ValueError):
            simulator_utils.format_2d_inputs(["bad"], 1, 1)
        assert simulator_utils.apply_format(["real", "ideal", "imba"]).strip().startswith("real")
        colored = simulator_utils.apply_color([1.25, "x"], ["31", "32"])
        assert str(colored[0]).startswith("\033[31m")
        assert len(simulator_utils.color_mix("red", "blue")) == 4

        @simulator_utils.timer
        def _timed(value: int) -> int:
            """Small timed function."""
            return value + 1

        assert _timed(2) == 3

        with pytest.raises(NotImplementedError):
            MicroBlockSim(0, "f", 0, 0, 1).build_without_comm()
        loop_block = MicroBlockSim(0, "f", 0, 0, 1)
        loop_block.pre = loop_block
        assert loop_block.loop()
        loop_block.in_queue = True
        loop_block.left = loop_block
        with pytest.raises(ValueError):
            loop_block.build_without_comm()

        line = PipelineBuilder.build_1f1b(2, 2, 1, 0, [1], [2], [3], [4])
        assert line[0].pre.label == ("h", 0)
        assert PipelineBuilder.get_builder("1f1b") is PipelineBuilder.build_1f1b
        assert PipelineBuilder.get_builder("vpp") is PipelineBuilder.build_virtualpipeline
        assert PipelineBuilder.get_builder("vpp2") is PipelineBuilder.build_virtualpipeline2
        with pytest.raises(ValueError):
            PipelineBuilder.get_builder("bad")
        assert PipelineBuilder._inter_merge([MicroBlockSim(0, "f", 0, 0, 1)], [], delta=1)[0].phase is None
        merged = PipelineBuilder._inter_merge(
            [MicroBlockSim(0, "f", 0, 0, 1), MicroBlockSim(0, "f", 1, 0, 1)],
            [MicroBlockSim(0, "b", 0, 0, 1)],
        )
        assert merged[0].phase == "stable"
        assert merged[-1].phase == "cooldown"
        assert PipelineBuilder._inter_merge([], [MicroBlockSim(0, "b", 0, 0, 1)], delta=-1)[0].phase is None

        head = HeadBlockSim(0)
        block = MicroBlockSim(0, "f", 0, 0, 2, pre=head, left=head)
        block.build_without_comm()
        assert block.finish
        block.reset_time_recursive()
        assert block.end is None
        assert head.label == ("h", 0)
        head.right = block
        assert "MicroBlockSim" in head.repr

        send_host = MicroBlockSim(0, "f", 1, 0, 2)
        rec_host = MicroBlockSim(1, "f", 1, 0, 2)
        send = SendBlockSim(0, "f", 1, 0, 0.25, host=send_host)
        rec = RecBlockSim(1, "f", 1, 0, 0.25, host=rec_host)
        send.dual = rec
        rec.dual = send
        assert send.loc_size(0, False, "compact")[2] == 0.25
        assert rec.loc_size(0, False, "compact")[2] == -0.25
        assert send.get_triangle(0, 0, 1, 1)[-1] == [1, 0]
        assert rec.get_triangle(0, 0, 1, 1)[-1] == [-1, -0.5]
        assert not send.comm_loop()

    def test_plot_manager_indices_and_errors(self):
        """PlotMgr validates draw modes and computes compact/joint indices."""
        manager = PlotMgr(num_plots=1)
        blocks = [[MicroBlockSim(0, "f", 0, 0, 1), SendBlockSim(0, "f", 0, 0, 0.5)]]
        assert manager._get_block_indices(blocks, mode="compact")[0].tolist() == [0, 1, 1]
        assert manager._get_block_indices(blocks, mode="joint")[0].tolist() == [0, 1, 1.5]
        with pytest.raises(ValueError):
            manager._get_block_indices(blocks, mode="bad")
        with pytest.raises(ValueError):
            manager._get_block_indices(blocks, mode="timeline")
        with pytest.raises(ValueError):
            PlotMgr(num_plots=2, subplot_args=[111])
        plt.close("all")

    def test_solver_pure_helpers_without_solving(self, tmp_path):
        """Exercise solver result and timing helpers with assigned variables, never CBC."""
        layers = _make_layers(body_layers=4)
        pipe = SappPipeline("unit", 2, 2, 128, layers, num_of_interleave=1)
        pipe.construct_problem(solver="pulp")
        solver = pipe.problem_
        body = layers[1]

        for rec in Recompute.TYPE:
            for stage in range(solver.num_of_stage_):
                solver.variables_[body.name_][rec][0][stage].varValue = 1
        for name in (
                solver.TOTAL_SUM,
                solver.NEXT_DIFF,
                solver.MAX_STAGE_TIME,
                solver.MAX_LAST_CHUNK,
        ):
            solver.variables_[name].varValue = 0
        for name in (solver.CHUNKS_SUM, solver.PREV_DIFF, solver.MEM_OVERHEAD_NAME):
            for variable in solver.variables_[name]:
                variable.varValue = 0

        assert SappSolver.compute_forward_in_backward(4, 8) == [3, 1, 1, 3]
        assert SappSolver.compute_forward_in_backward(4, 2)[:2] == [0, 0]
        assert SappSolver.compute_lm_forward_in_backward(3) == [0, 1, 2]
        assert SappSolver.compute_activation_nums(3, 2, 2) == [[2, 2, 2], [2, 2, 1]]
        assert SappSolver.compute_activation_nums_dual(2, 2, 3) == [[3, 3], [1, 2]]
        assert SappSolver.compute_less_activation_nums(3, 2) == [[3, 3, 3], [3, 2, 1]]
        assert SappSolver.compute_less_activation_nums(3, 1) == [[3, 2, 1]]

        assert solver._reserved_stage_positions() == {(0, 0), (0, 1)}
        solver.dual_ = True
        assert solver._reserved_stage_positions() == {(0, 0), (1, 0)}
        assert solver.stage_param_memory(
            solver.variables_, solver.layers_sorted_, 0, solver.num_of_stage_, solver.num_of_interleave_
        ).value() > 0
        solver.dual_ = False

        assert solver.stage_active_memory_per_micro(
            solver.variables_, solver.layers_sorted_, 0, 0
        ).value() > 0
        assert solver.get_simulator_memory_activation()[0][0] > 0
        assert solver.get_simulator_memory_parameter()[0][0] > 0
        assert solver.get_simulator_time()[0][0] > 0
        assert solver.get_simulator_forward_time()[0][0] > 0
        assert len(solver.get_simulator_recompute_time()[0]) == 2
        assert "body" in solver.result()
        assert solver.has_some_memory_info()

        assert solver._max_stage_bound_i_fp(solver.layers_sorted_, 0, 0).value() > 0
        assert solver._max_stage_bound_i_bp(solver.layers_sorted_, 0, 0).value() > 0
        assert solver._max_stage_bound_head_tail(solver.layers_sorted_, 0, 0, 0).value() > 0
        assert solver._total_sum(solver.layers_sorted_).value() > 0
        assert solver.body_layer_time(solver.PROP_PHASE.FW, body, 0, 0).value() > 0
        assert solver.body_layer_time(solver.PROP_PHASE.BW, body, 0, 0).value() > 0
        assert solver.micro_batch_time(solver.PROP_PHASE.FW, solver.layers_sorted_, 0, 0).value() > 0
        assert solver.micro_batch_time(solver.PROP_PHASE.BW, solver.layers_sorted_, 0, 0).value() > 0
        assert solver._chunks_sum(solver.layers_sorted_, 0).value() > 0
        assert "diff_with_prev_stages" in str(
            solver._prev_diff_sum(solver.layers_sorted_, solver.problem_, 0)
        )
        assert "diff_with_next_stages" in str(
            solver._next_diff_sum(solver.layers_sorted_, solver.problem_)
        )

        solver.print_results()
        solver.debug_print_solver_theoretical_memory()
        solver.dump_problem(str(tmp_path))
        assert list(tmp_path.glob("problem_unit_*.lp"))

        unused_rec = Recompute.TYPE.SLCT
        saved = solver.recompute_considered_[unused_rec]
        solver.recompute_considered_[unused_rec] = False
        solver.add_optional_recompute_constraint(solver.problem_, solver.variables_, solver.layers_sorted_)
        solver.recompute_considered_[unused_rec] = saved

    def test_sequence_updates_and_comm_simulation_without_solver(self, tmp_path, monkeypatch):
        """Cover sequence-pipeline mutation and communication simulation with tiny inputs."""
        params = {
            "batch_size": 1,
            "num_heads": 4,
            "seq_length": 32,
            "head_dim": 8,
            "model_parallel": 2,
            "hidden_size": 32,
            "vocab_size": 64,
        }
        time_updates = []
        body = SimpleNamespace(
            memory_parameter_=10.0,
            memory_activation_rec_={rec: float(rec.value + 1) for rec in Recompute.TYPE},
            time_=12.0,
            forward_time_=4.0,
            backward_time_rec_={rec: 8.0 for rec in Recompute.TYPE},
            recompute_considered_={rec: True for rec in Recompute.TYPE},
            name_="layer_group_1",
            update_internal_time_for_seqpp=lambda: time_updates.append("body"),
        )
        head = SimpleNamespace(
            memory_parameter_=3.0,
            time_=3.0,
            forward_time_=1.0,
            backward_time_rec_={rec: 2.0 for rec in Recompute.TYPE},
            update_internal_time_for_seqpp=lambda: time_updates.append("head"),
        )
        tail = SimpleNamespace(
            memory_parameter_=5.0,
            time_=6.0,
            forward_time_=2.0,
            backward_time_rec_={rec: 4.0 for rec in Recompute.TYPE},
            update_internal_time_for_seqpp=lambda: time_updates.append("tail"),
        )
        seq_solver = object.__new__(SappSolver)
        seq_solver.layers_sorted_ = {
            Layer.type_enum.BODY: [body],
            Layer.type_enum.HEAD: [head],
            Layer.type_enum.TAIL: [tail],
        }
        seq_solver.recompute_considered_ = {rec: True for rec in Recompute.TYPE}
        seq_solver.extracted_training_params_ = params
        seq_solver.seq_split_num_ = 2
        seq_solver.num_of_micro_batch_ = 4
        seq_solver._initialize_seq_pipe_layers()
        assert seq_solver.num_of_micro_batch_ == 8
        assert body.memory_parameter_ > 10
        assert body.memory_activation_rec_[Recompute.TYPE.FULL] is None
        assert head.memory_parameter_ < 3
        assert tail.memory_parameter_ < 5
        assert time_updates == ["body", "head", "tail"]

        simulator = PipelineSimulator(
            [1, 1],
            2,
            comm_time=0.05,
            block_mem=1,
            block_mem_par=1,
        )

        def fake_statistic_info() -> None:
            """Keep communication scheduling coverage independent of memory aggregation."""
            simulator.end_time = max(block.end for line in simulator.lines for block in line)
            simulator.peak_memory = [0, 0]
            simulator.states["block_mem_list"] = [
                np.array([[0, 0], [simulator.end_time, 0]]) for _ in range(simulator.pp)
            ]

        monkeypatch.setattr(simulator, "_statistic_info", fake_statistic_info)
        simulator.run(comm=True, print_info=False)
        assert simulator.end_time > 0
        assert len(simulator.lines) == 2
        simulator.draw(comm=True, connect=True)
        output_file = tmp_path / "comm_pipeline.svg"
        simulator.save(str(output_file), comm=True, connect=True)
        assert output_file.exists()

        monkeypatch.setattr(plt, "show", lambda: None)
        simulator.show(comm=True, connect=False)
        normal_error = CausalError("normal-loop", simulator.blocks, [simulator.blocks[0][0]])
        comm_error = CausalCommError("comm-loop", simulator.lines, [simulator.lines[0][0]])
        assert str(normal_error) == "normal-loop"
        assert str(comm_error) == "comm-loop"
        plt.close("all")
