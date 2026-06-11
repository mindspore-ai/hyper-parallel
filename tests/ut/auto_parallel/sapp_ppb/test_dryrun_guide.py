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
"""Unit tests for the SAPP-PPB dryrun guide configuration generator.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/test_dryrun_guide.py
"""
import argparse
import sys
from typing import Any

import numpy as np

# pylint: disable=C0413,R0903
from hyper_parallel.auto_parallel.sapp_ppb import run_pipeline_balance as run_cli
from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver
from hyper_parallel.auto_parallel.sapp_ppb.utils import config as config_utils
from hyper_parallel.auto_parallel.sapp_ppb.utils import interactive
from hyper_parallel.auto_parallel.sapp_ppb.utils.config import Recompute
from hyper_parallel.auto_parallel.sapp_ppb.utils.config import generate_solvable_config
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer


class _FakeLayer:
    """Minimal printable layer for the CLI runner."""

    def __str__(self) -> str:
        """Return a printable layer description."""
        return "fake-layer"

    def dump(self) -> None:
        """Accept the CLI dump call."""


class _FakePipe:
    """Record SappPipeline workflow calls without solving an ILP."""

    instances = []

    def __init__(self, **kwargs: Any) -> None:
        """Record constructor keyword arguments."""
        self.kwargs = kwargs
        self.calls = []
        _FakePipe.instances.append(self)

    def construct_problem(self, solver: str = "pulp") -> None:
        """Record solver construction."""
        self.calls.append(("construct", solver))

    def solve_problem(self, time_limit: int = 90, dump_folder: str = None) -> None:
        """Record solve arguments."""
        self.calls.append(("solve", time_limit, dump_folder))

    def print_yaml_results(self) -> None:
        """Record YAML printing."""
        self.calls.append(("yaml",))

    def simulate(self, show: bool = True, file_name: str = None) -> int:
        """Record automatic simulation."""
        self.calls.append(("simulate", show, file_name))
        return 123

    def simulate_comparison(self, manual_config: str, output_folder: str) -> None:
        """Record comparison simulation."""
        self.calls.append(("compare", manual_config, output_folder))

    def simulate_naive(self, layers: list, output_folder: str) -> None:
        """Record naive simulation."""
        self.calls.append(("naive", len(layers), output_folder))

    def simulate_only_manual(self, manual_config: str, output_folder: str) -> None:
        """Record manual-only simulation."""
        self.calls.append(("manual", manual_config, output_folder))

    def get_time(self) -> list[list[int]]:
        """Return deterministic stage times."""
        return [[1, 2]]

    def get_memory_parameter(self) -> list[list[int]]:
        """Return deterministic parameter memory."""
        return [[3, 4]]

    def get_memory_activation(self) -> list[list[int]]:
        """Return deterministic activation memory."""
        return [[5, 6]]


def _input_from(values):
    """Return an input() replacement that consumes values in order."""
    iterator = iter(values)
    return lambda _prompt="": next(iterator)


class TestDryunGuide:
    """A test class for testing dryrun guide."""

    def test_dryrun_guide_one_round(self):
        """A single-round dryrun config must identify each recompute coefficient."""
        considered_rec = [Recompute.TYPE.BOTH, Recompute.TYPE.SLCT, Recompute.TYPE.COMM]
        offset_config_list, rec_config_list = generate_solvable_config(16, 17, considered_rec)
        activation_nums = SappSolver.compute_activation_nums(16, 1, 0)[0]
        layer_per_stage = 1
        coef_matrix = []
        rounds = len(offset_config_list)
        for round_ in range(rounds):
            for stage in range(16):
                if stage not in [0, 16 - 1]:
                    coef_matrix.append(
                        [1, layer_per_stage + offset_config_list[round_][stage]]
                        + Recompute.to_list(
                            {
                                rec: rec_config_list[round_][rec][stage]
                                     * activation_nums[stage]
                                for rec in considered_rec
                            }
                        )
                    )
                if len(coef_matrix) == 2 + len(considered_rec):
                    coef_rank = np.linalg.matrix_rank(coef_matrix)
                    assert coef_rank == len(considered_rec) + 2

    def test_generate_solvable_config_rejects_pp2_and_prints_yaml(self, monkeypatch):
        """Dry-run helpers reject unsupported pp=2 and log YAML-compatible configs."""
        assert generate_solvable_config(2, 8, []) is None

        outputs = []
        monkeypatch.setattr(
            config_utils.logger,
            "output",
            lambda msg, *args: outputs.append(msg % args if args else msg),
        )
        config_utils.print_dryrun_config(
            [[0, 1, -1, 0]],
            [
                {
                    Recompute.TYPE.FULL: [1, 0, 0, 0],
                    Recompute.TYPE.SLCT: [0, 1, 0, 0],
                    Recompute.TYPE.COMM: [0, 0, 1, 0],
                    Recompute.TYPE.BOTH: [0, 0, 0, 1],
                }
            ],
        )
        assert any("round 1" in item for item in outputs)
        assert any("select_comm_recompute" in item for item in outputs)

    def test_parse_training_config_success_and_error(self, tmp_path):
        """Training YAML parsing extracts seqpp fields and tolerates missing files."""
        training_file = tmp_path / "training.yaml"
        training_file.write_text(
            """
model:
  model_config:
    num_heads: 8
    hidden_size: 1024
    seq_length: 2048
    vocab_size: 32000
parallel_config:
  data_parallel: 2
  model_parallel: 4
runner_config:
  batch_size: 16
""",
            encoding="utf-8",
        )

        parsed = config_utils.parse_training_config(str(training_file))
        assert parsed["head_dim"] == 128
        assert parsed["context_parallel"] == 1
        assert config_utils.parse_training_config(str(tmp_path / "missing.yaml")) is None

    def test_interactive_global_args_make_layer_and_dryrun(self, monkeypatch):
        """Interactive prompts accept explicit values and dispatch dry-run generation."""
        monkeypatch.setattr("builtins.input", _input_from(["6", "12", "2", "64000"]))
        global_args = interactive.global_arguments()
        assert global_args.stage_num == 6
        assert global_args.micro_batch == 12
        assert global_args.interleave == 2
        assert global_args.max_memory == 64000

        body_values = ["body", "10", "4", "99", "1", "2", "3", "4", "5"]
        monkeypatch.setattr("builtins.input", _input_from(body_values))
        layer = interactive.make_layer(Layer.type_enum.BODY, "toy")
        assert layer.name_ == "body"
        assert layer.nb_layer_ == 4
        assert layer.memory_parameter_ == 99
        assert layer.memory_activation_rec_[Recompute.TYPE.FULL] == 5

        received = {}
        monkeypatch.setattr("builtins.input", _input_from(["4", "8", "y", "n", "y", "n"]))

        def _fake_generate(stage, layers, rec):
            """Capture dry-run arguments and return a deterministic guide."""
            received["args"] = (stage, layers, list(rec))
            return [[0] * stage], [{}]

        monkeypatch.setattr(interactive, "generate_solvable_config", _fake_generate)
        monkeypatch.setattr(interactive, "print_dryrun_config", lambda offsets, recs: received.update(
            {"offsets": offsets, "recs": recs}
        ))
        interactive.dryrun_guide()
        assert received["args"] == (4, 8, [Recompute.TYPE.FULL, Recompute.TYPE.BOTH])
        assert received["offsets"] == [[0, 0, 0, 0]]

    def test_interactive_main_runs_pipeline_with_fake_solver(self, monkeypatch):
        """The no-argument interactive flow builds layers and invokes the pipeline methods."""
        _FakePipe.instances = []
        responses = [
            "y",
            "", "", "", "",
            "toy",
            "head", "1", "2",
            "body", "3", "4", "5", "6", "7", "8", "9", "10",
            "tail", "11", "12",
        ]
        monkeypatch.setattr("builtins.input", _input_from(responses))
        monkeypatch.setattr(interactive, "SappPipeline", _FakePipe)

        interactive.main()

        assert _FakePipe.instances[-1].kwargs["model_name"] == "toy"
        assert ("construct", "pulp") in _FakePipe.instances[-1].calls
        assert ("solve", 40, "output") in _FakePipe.instances[-1].calls

    def test_cli_parser_run_and_main_branches(self, tmp_path, monkeypatch):
        """CLI parser and runner execute init, solve, comparison and manual-only branches."""
        parser = run_cli.build_arg_parser()
        parsed = parser.parse_args(["-s", "8", "-mb", "16", "-lm", "yes", "-dual", "1", "-exec", "false"])
        assert parsed.stage == 8
        assert parsed.micro_batch == 16
        assert parsed.less_memory
        assert parsed.dualpipe_v
        assert not parsed.exec

        manual_config = tmp_path / "manual.yaml"
        manual_config.write_text("manual: {offset: [0, 0], file_name: manual.svg}\n", encoding="utf-8")
        init_file = tmp_path / "init.yaml"
        init_file.write_text("pipeline_config: {}\n", encoding="utf-8")
        calls = []
        _FakePipe.instances = []
        monkeypatch.setattr(
            run_cli,
            "initialize_layer_json",
            lambda model, file_name: calls.append(("init", model, file_name)),
        )
        monkeypatch.setattr(run_cli, "generate_layers_list", lambda folder, model: [_FakeLayer(), _FakeLayer()])
        monkeypatch.setattr(run_cli, "compute_memories", lambda layers, memory_folder, number_of_stage: layers)
        monkeypatch.setattr(run_cli, "SappPipeline", _FakePipe)

        args = argparse.Namespace(
            init="init.yaml",
            model_name="toy",
            output_folder="out",
            manual_config="manual.yaml",
            layer_folder="layers",
            compute_memory=True,
            memory_folder="memory",
            stage=2,
            dump_layer=True,
            micro_batch=2,
            max_memory=100,
            interleave_degree=1,
            less_memory=False,
            dualpipe_v=False,
            constant_memory=0,
            optimization_level=1,
            exec=True,
            time_limit=7,
            simulate_naive=True,
        )
        run_cli.run(args, str(tmp_path))
        assert calls == [("init", "toy", str(init_file))]
        assert ("solve", 7, str(tmp_path / "out")) in _FakePipe.instances[-1].calls
        assert any(call[0] == "compare" for call in _FakePipe.instances[-1].calls)
        assert any(call[0] == "naive" for call in _FakePipe.instances[-1].calls)

        args.exec = False
        run_cli.run(args, str(tmp_path))
        assert ("manual", str(manual_config), str(tmp_path / "out")) in _FakePipe.instances[-1].calls

        no_arg_calls = []
        monkeypatch.setattr(run_cli.interactive, "main", lambda: no_arg_calls.append("interactive"))
        monkeypatch.setattr(sys, "argv", ["run_pipeline_balance.py"])
        run_cli.main()
        assert no_arg_calls == ["interactive"]

        main_calls = []
        monkeypatch.setattr(run_cli, "run", lambda args, base_dir: main_calls.append((args.stage, base_dir)))
        monkeypatch.setattr(sys, "argv", ["run_pipeline_balance.py", "-s", "3"])
        run_cli.main()
        assert main_calls[-1][0] == 3

    def test_interactive_decline_and_empty_dryrun_return(self, monkeypatch):
        """Early-return interactive branches remain harmless."""
        monkeypatch.setattr("builtins.input", _input_from(["n"]))
        interactive.main()

        monkeypatch.setattr("builtins.input", _input_from([""]))
        interactive.dryrun_guide()

    def test_dryrun_guide_multi_rounds(self):
        """Same property must hold when ``generate_solvable_config`` returns multiple rounds."""
        considered_rec = [Recompute.TYPE.BOTH, Recompute.TYPE.SLCT, Recompute.TYPE.COMM]
        offset_config_list, rec_config_list = generate_solvable_config(4, 5, considered_rec)
        activation_nums = SappSolver.compute_activation_nums(4, 1, 0)[0]
        layer_per_stage = 1
        coef_matrix = []
        rounds = len(offset_config_list)
        for round_ in range(rounds):
            for stage in range(4):
                if stage not in [0, 4 - 1]:
                    coef_matrix.append(
                        [1, layer_per_stage + offset_config_list[round_][stage]]
                        + Recompute.to_list(
                            {
                                rec: rec_config_list[round_][rec][stage]
                                     * activation_nums[stage]
                                for rec in considered_rec
                            }
                        )
                    )
                if len(coef_matrix) == 2 + len(considered_rec):
                    coef_rank = np.linalg.matrix_rank(coef_matrix)
                    assert coef_rank == len(considered_rec) + 2
