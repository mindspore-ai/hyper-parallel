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
"""Unit tests for sapp_ppb memory computation.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/test_compute_memory.py
"""
import json
import os
from typing import Any

# pylint: disable=C0413,W0212,R0903
# pylint: disable=invalid-name
import pytest

from hyper_parallel.auto_parallel.sapp_ppb.utils import compute_memory as compute_memory_module
from hyper_parallel.auto_parallel.sapp_ppb.utils import computation_analyzer as analyzer_module
from hyper_parallel.auto_parallel.sapp_ppb.utils import config as config_utils
from hyper_parallel.auto_parallel.sapp_ppb.utils import layer as layer_module
from hyper_parallel.auto_parallel.sapp_ppb.utils.computation_analyzer import ComputationAnalyzer
from hyper_parallel.auto_parallel.sapp_ppb.utils.compute_memory import ComputeMemory
from hyper_parallel.auto_parallel.sapp_ppb.utils.config import memory_parser
from hyper_parallel.auto_parallel.sapp_ppb.utils.error import SAPPError
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer
from hyper_parallel.auto_parallel.sapp_ppb.utils.stage import Stage
import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute


class _FakeComputationAnalyzer:
    """Small stand-in for profiling timeline parsing in Layer.compute_timer."""

    def __init__(self, timeline_folder: str, model_name: str,
                 num_of_micro_batch: int, layer_list: dict) -> None:
        """Expose deterministic layer costs."""
        del timeline_folder, num_of_micro_batch, layer_list
        self.layer_with_cost_list = {model_name: 12.5, "timed": 7.5}


class _FakeComputeMemory:
    """Deterministic memory solver used by compute_memories."""

    recompute_considered_ = {rec: True for rec in Recompute.TYPE}

    def __init__(self, number_of_stage: int, stages_a: Any = None, stages_b: Any = None) -> None:
        """Accept the same constructor shape as ComputeMemory."""
        del number_of_stage, stages_a, stages_b

    def get_memory_head(self) -> int:
        """Return fake HEAD memory."""
        return 11

    def get_memory_tail(self) -> int:
        """Return fake TAIL memory."""
        return 13

    def get_memory_parameter(self) -> int:
        """Return fake BODY parameter memory."""
        return 17

    def get_memory_activation(self, rec: Recompute.TYPE) -> int:
        """Return fake BODY activation memory."""
        return rec.value + 20


def _stage(stage_id, nb_stage=4, nb_layer=2, rec_layers=None, memory_usage=100):
    """Build a Stage with complete recomputation defaults."""
    if rec_layers is None:
        rec_layers = {}
    return Stage(
        sid=stage_id,
        nb_stage=nb_stage,
        nb_layer=nb_layer,
        nb_layer_rec=rec_layers,
        memory_usage=memory_usage,
    )


def _layer(layer_type, name, nb_layer=1, memory_parameter=0, memory_activation=None):
    """Build a Layer with explicit memory defaults."""
    if memory_activation is None:
        memory_activation = {rec: 0 for rec in Recompute.TYPE}
    return Layer(
        model_name="unit",
        name=name,
        ltype=layer_type,
        nb_layer=nb_layer,
        time=10,
        memory_parameter=memory_parameter,
        memory_activation_rec=memory_activation,
    )


class TestComputeMemory:
    """A test class for testing compute memory."""

    def test_compute_memory(self):
        """Six handcrafted stages reproduce the canonical decomposition."""
        num_stage = 16
        per_stage_layer_num = 6
        stage_head = Stage(
            sid=0,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 1, Recompute.TYPE.FULL: 1},
            memory_usage=80267,
        )
        stage_1 = Stage(
            sid=1,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=71519,
        )
        stage_2 = Stage(
            sid=2,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=67376,
        )
        stage_3 = Stage(
            sid=3,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 2},
            memory_usage=52962,
        )
        stage_4 = Stage(
            sid=9,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 2, Recompute.TYPE.FULL: 0},
            memory_usage=39373,
        )
        stage_tail = Stage(
            sid=num_stage - 1,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=16386,
        )

        stages_a = [stage_1, stage_2, stage_3, stage_4, stage_head, stage_tail]

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_a=stages_a)
        memory_head = int(comp_mem.get_memory_head())
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_select_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_recompute = int(comp_mem.get_memory_activation(Recompute.TYPE.FULL))
        assert memory_head == 9785, "memory_head: wrong answer"
        assert memory_parameter == 1562, "memory_parameter: wrong answer"
        assert memory_activation == 822, "memory_activation: wrong answer"
        assert memory_tail == 2868, "memory_tail: wrong answer"
        assert memory_recompute == 32, "memory_recompute: wrong answer"
        assert memory_select_comm == 498, "memory_select_comm: wrong answer"

    def test_compute_memory_with_const(self):
        """Reading ``mem.yaml`` and computing the memory decomposition."""
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "mem.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_a=stages_a)
        memory_head = int(comp_mem.get_memory_head())
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_both = int(comp_mem.get_memory_activation(Recompute.TYPE.BOTH))
        memory_slct = int(comp_mem.get_memory_activation(Recompute.TYPE.SLCT))
        memory_const = int(comp_mem.get_memory_const())

        assert memory_head == 4245, "memory_head: wrong answer"
        assert memory_parameter == 2141, "memory_parameter: wrong answer"
        assert memory_activation == 852, "memory_activation: wrong answer"
        assert memory_tail == 1203, "memory_tail: wrong answer"
        assert memory_both == 511, "memory_both: wrong answer"
        assert memory_comm == 535, "memory_comm: wrong answer"
        assert memory_slct == 767, "memory_slct: wrong answer"
        assert memory_const == -1896, "memory_const: wrong answer"

    def test_compute_memory_pp4(self):
        """Reading ``mem_pp4.yaml`` and computing the memory decomposition."""
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "mem_pp4.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_a=stages_a)
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_full = int(comp_mem.get_memory_activation(Recompute.TYPE.FULL))
        memory_head = int(comp_mem.get_memory_head())

        assert memory_head == 1459, "memory_head: wrong answer"
        assert memory_parameter == 400, "memory_parameter: wrong answer"
        assert memory_activation == 743, "memory_activation: wrong answer"
        assert memory_full == 131, "memory_full: wrong answer"
        assert memory_tail == 3329, "memory_tail: wrong answer"

    def test_compute_memory_lstsq(self):
        """Reading ``mem_lstsq.yaml`` and computing the memory decomposition."""
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "mem_lstsq.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_a=stages_a)
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_slct = int(comp_mem.get_memory_activation(Recompute.TYPE.SLCT))
        memory_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_both = int(comp_mem.get_memory_activation(Recompute.TYPE.BOTH))

        memory_head = int(comp_mem.get_memory_head())

        assert memory_head == 4614, "memory_head: wrong answer"
        assert memory_parameter == 2217, "memory_parameter: wrong answer"
        assert memory_activation == 1002, "memory_activation: wrong answer"
        assert memory_both == 423, "memory_both: wrong answer"
        assert memory_slct == 783, "memory_slct: wrong answer"
        assert memory_comm == 695, "memory_comm: wrong answer"
        assert memory_tail == 5885, "memory_tail: wrong answer"

    def test_stage_helpers_complete_recompute_and_validate(self):
        """Stage completes omitted recompute counters and rejects inconsistent totals."""
        stage = _stage(
            1,
            nb_layer=4,
            rec_layers={Recompute.TYPE.FULL: 1, Recompute.TYPE.COMM: 1},
            memory_usage=64,
        )
        same = _stage(
            2,
            nb_layer=4,
            rec_layers={Recompute.TYPE.FULL: 1, Recompute.TYPE.COMM: 1},
            memory_usage=32,
        )
        different = _stage(3, nb_layer=4, rec_layers={Recompute.TYPE.SLCT: 1})

        assert stage.nb_layer_rec_[Recompute.TYPE.NONE] == 2
        assert stage.same_config(same)
        assert not stage.same_config(different)
        assert stage.same_global_config(different)
        assert [item.id_ for item in compute_memory_module.filter_stage_id([stage, same], 1)] == [1]

        with pytest.raises(SAPPError):
            _stage(4, nb_stage=4)
        with pytest.raises(SAPPError):
            _stage(1, nb_layer=1, rec_layers={Recompute.TYPE.NONE: 0, Recompute.TYPE.FULL: 2})

    def test_recompute_conversion_and_collection_helpers(self):
        """Recompute utilities round-trip YAML data and enumerate considered types."""

        class _Var:
            """Object exposing the varValue attribute used by yaml_from_internal."""

            def __init__(self, value: Any) -> None:
                """Store a fake solver value."""
                self.varValue = value

        lp_variables = {
            Recompute.TYPE.NONE: [[_Var(1), _Var(2)]],
            Recompute.TYPE.SLCT: [[_Var(3), _Var(4)]],
            Recompute.TYPE.COMM: [[_Var(5), _Var(6)]],
            Recompute.TYPE.BOTH: [[_Var(7), _Var(8)]],
            Recompute.TYPE.FULL: [[_Var(9), _Var(10)]],
        }

        yaml_out = Recompute.yaml_from_internal(1, 2, lp_variables, [[20, 30]])
        assert yaml_out[Recompute.OFFSET] == [[5, 0]]
        assert yaml_out[Recompute.YAML_NAME[Recompute.TYPE.FULL]] == [[9, 10]]
        assert yaml_out[Recompute.YAML_NAME[Recompute.TYPE.SLCT]] == [[19, 22]]
        assert yaml_out[Recompute.YAML_NAME[Recompute.TYPE.COMM]] == [[21, 24]]
        assert Recompute.zero_if_none_var([[_Var(None)]], 0, 0) == 0
        assert Recompute.zero_if_none(None, 0, 0) == 0

        parsed = Recompute.internal_from_yaml(
            1,
            2,
            {
                Recompute.OFFSET: 0,
                Recompute.YAML_NAME[Recompute.TYPE.FULL]: True,
                Recompute.YAML_NAME[Recompute.TYPE.SLCT]: True,
                Recompute.YAML_NAME[Recompute.TYPE.COMM]: False,
            },
            [[2, 3]],
        )
        assert parsed[Recompute.TYPE.FULL] == [[2, 3]]
        assert parsed[Recompute.TYPE.NONE] == [[0, 0]]

        used = {
            Recompute.TYPE.NONE: True,
            Recompute.TYPE.SLCT: False,
            Recompute.TYPE.COMM: True,
            Recompute.TYPE.BOTH: False,
            Recompute.TYPE.FULL: True,
        }
        indexes = Recompute.make_all_indexes(used, 2)
        assert len(indexes) == 8
        decoded = Recompute.recomputes_from_indexes(used, [item[:] for item in indexes])
        assert decoded[0][Recompute.TYPE.SLCT] is None
        assert Recompute.assign_used([1, 2], [Recompute.TYPE.SLCT, Recompute.TYPE.BOTH, Recompute.TYPE.FULL])[
            Recompute.TYPE.COMM
        ] == 2
        assert Recompute.get_used_list(used) == [Recompute.TYPE.NONE, Recompute.TYPE.COMM, Recompute.TYPE.FULL]
        assert Recompute.get_unused_list({Recompute.TYPE.NONE: True}) == [
            Recompute.TYPE.SLCT,
            Recompute.TYPE.COMM,
            Recompute.TYPE.BOTH,
            Recompute.TYPE.FULL,
        ]
        assert Recompute.least_recomputed(used) == Recompute.TYPE.NONE
        assert Recompute.most_recomputed(used) == Recompute.TYPE.FULL
        assert Recompute.average([]) == []
        assert Recompute.average([
            {rec: float(rec.value) for rec in Recompute.TYPE},
            {rec: float(rec.value + 2) for rec in Recompute.TYPE},
        ])[Recompute.TYPE.COMM] == 3

    def test_config_parsers_and_model_info_dump(self, tmp_path, monkeypatch):
        """Config helpers validate offsets, parse times and dump derived layer JSON."""
        assert config_utils.process_offset(0, 3) == ([0, 0, 0], 1)
        assert config_utils.process_offset([0, [1, -1, 0]], 3) == ([[0, 0, 0], [1, -1, 0]], 2)
        with pytest.raises(ValueError):
            config_utils.process_offset([1, 2], 3)

        assert config_utils.process_rec_config(2, 3, [0, 0, 0], False) == [[0, 0, 0]]
        assert config_utils.process_rec_config(2, 3, [0, 0, 0], [1, 2, 3]) == [[1, 2, 3]]
        with pytest.raises(ValueError):
            config_utils.process_rec_config(2, 3, [0, 0, 0], [1, 2])

        time_yaml = tmp_path / "time.yaml"
        time_yaml.write_text(
            "time_config:\n  head: 1.5\n  body: 2.5\n  tail: 3.5\n",
            encoding="utf-8",
        )
        assert config_utils.time_parser(str(time_yaml), "unit") == (1.5, 2.5, 3.5)
        with pytest.raises(ValueError):
            config_utils.time_parser(str(tmp_path / "time.txt"), "unit")
        with pytest.raises(ValueError):
            config_utils.memory_parser(None)

        model_info = config_utils.ModelInfo("unit", 1, 2, 3, 4)
        model_info.set_stage_const_mem(9)
        model_info.layer_memory_update({Recompute.TYPE.NONE: 7, Recompute.TYPE.FULL: None}, 11, 13, 17)
        dump_file = tmp_path / "unit.json"
        model_info.dump_json(str(dump_file))
        dumped = json.loads(dump_file.read_text(encoding="utf-8"))
        assert dumped["stage_const_mem"] == 9
        assert dumped["layers_description"][1]["memory_activation"] == 7

        layer_folder = tmp_path / "layers"
        layer_folder.mkdir()
        (layer_folder / "with_const.json").write_text('{"stage_const_mem": 21}', encoding="utf-8")
        (layer_folder / "without_const.json").write_text("{}", encoding="utf-8")
        assert config_utils.get_stage_const_mem(str(layer_folder), "with_const") == 21
        assert config_utils.get_stage_const_mem(str(layer_folder), "without_const") == 0

        monkeypatch.chdir(tmp_path)
        os.makedirs("layers", exist_ok=True)
        work_path = os.path.dirname(os.path.abspath(__file__))
        config_utils.initialize_layer_json("derived", os.path.join(work_path, "mem.yaml"))
        derived = json.loads((tmp_path / "layers" / "derived.json").read_text(encoding="utf-8"))
        assert derived["name"] == "derived"
        assert len(derived["layers_description"]) == 3

    def test_computation_analyzer_parses_msprof_timeline(self, tmp_path, monkeypatch):
        """ComputationAnalyzer derives per-layer costs from a compact fake timeline."""
        cfg_dir = tmp_path / "cfgs"
        cfg_dir.mkdir()
        (cfg_dir / "model_layers.json").write_text(
            json.dumps(
                [
                    {
                        "name": "toy",
                        "pre_defined_layer": {"Head": 0, "Tail": -1},
                        "auto_partition_layer": {"Body": 2},
                    }
                ]
            ),
            encoding="utf-8",
        )
        timeline_dir = tmp_path / "timeline"
        timeline_dir.mkdir()
        timeline = [
            {"name": "Scope Layer", "pid": 3, "tid": 0, "ts": 0, "dur": 0},
            {"name": "MatMul-op0", "pid": 1, "tid": 0, "ts": 0, "dur": 1},
            {"name": "MatMul-op0", "pid": 1, "tid": 0, "ts": 1000, "dur": 1},
            {"name": "MatMul-op0", "pid": 1, "tid": 0, "ts": 2000, "dur": 1},
            {"name": "MatMul-op0", "pid": 1, "tid": 0, "ts": 3000, "dur": 1},
            {"name": "Default", "pid": 3, "tid": 0, "ts": 2000, "dur": 1000},
            {"name": "Head", "pid": 3, "tid": 1, "ts": 2100, "dur": 100},
            {"name": "0-Body", "pid": 3, "tid": 1, "ts": 2200, "dur": 200},
            {"name": "1-Body", "pid": 3, "tid": 1, "ts": 2400, "dur": 300},
            {"name": "Tail", "pid": 3, "tid": 1, "ts": 2700, "dur": 100},
        ]
        trace_file = timeline_dir / "trace_view_0.json"
        trace_file.write_text(json.dumps(timeline), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(analyzer_module, "tqdm", lambda data: data)

        analyzer = ComputationAnalyzer(str(timeline_dir), "toy", 2)

        assert analyzer.layer_list["name"] == "toy"
        assert analyzer.auto_partition_layer_name_list == ["0-Body", "1-Body"]
        assert analyzer.layer_with_computation_time_list["Head"] == pytest.approx(0.1)
        assert analyzer.layer_with_cost_list["Head"] == pytest.approx(0.05)
        assert analyzer.layer_with_cost_list["Body"] == pytest.approx(0.125)
        assert analyzer._load_json_data(str(trace_file)) == timeline
        assert analyzer._parse_step_duration(timeline) == (2000.0, 3000.0)
        assert analyzer._initialize_step_duration(timeline, 0, 0) == (2000.0, 3000.0)
        assert not analyzer._is_counted([], 2000, 3000, {"ts": 1000, "dur": 1})

        with pytest.raises(ValueError):
            analyzer._forward_parser([{"name": "Scope Layer", "pid": 3, "tid": 0, "ts": 0, "dur": 1}])

    def test_layer_generation_aggregation_and_timing(self, tmp_path, monkeypatch):
        """Layer helpers parse JSON, aggregate values and update profiling-derived time."""
        layer_file = tmp_path / "toy.json"
        layer_file.write_text(
            json.dumps(
                {
                    "layers_description": [
                        {"name": "head", "type": "HEAD", "time": 3, "nb_layer": 1, "memory_parameter": 5},
                        {
                            "name": "body",
                            "type": "BODY",
                            "time": 6,
                            "nb_layer": 2,
                            "memory_parameter": 7,
                            "memory_activation": 11,
                            "recompute_coef": 0.5,
                        },
                        {"name": "tail", "type": "TAIL", "time": 4, "nb_layer": 1, "memory_parameter": 13},
                    ]
                }
            ),
            encoding="utf-8",
        )
        layers = layer_module.generate_layers_list(str(tmp_path), "toy")
        assert [layer.name_ for layer in layers] == ["head", "body", "tail"]
        assert len(layer_module.filter_layer_type(layers, Layer.type_enum.BODY)) == 1
        assert "Layer Description" in str(layers[1])

        aggregated = layer_module.aggregate([
            _layer(Layer.type_enum.BODY, "first", nb_layer=1, memory_parameter=None),
            _layer(Layer.type_enum.BODY, "second", nb_layer=2, memory_parameter=10),
        ])
        assert aggregated.nb_layer_ == 3
        assert aggregated.memory_parameter_ == 10

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{}", encoding="utf-8")
        assert not layer_module.generate_layers_list(str(tmp_path), "bad")

        timed_layer = _layer(Layer.type_enum.BODY, "timed")
        monkeypatch.setattr(layer_module, "ComputationAnalyzer", _FakeComputationAnalyzer)
        timed_layer.compute_timer("timeline", {"auto_partition_layer": {"timed": 1}, "pre_defined_layer": {}})
        assert timed_layer.time_ == 7.5
        timed_layer.update_internal_time_for_seqpp(back_ratio=0.25, force_fb=True)
        assert timed_layer.forward_time_ == pytest.approx(5.625)
        timed_layer.dump("unused.json")
        timed_layer.to_json()
        timed_layer.compute_memory("unused")

    def test_compute_memory_edge_paths_and_layer_update(self, tmp_path, monkeypatch):
        """ComputeMemory handles invalid stage sets and compute_memories updates Layer objects."""
        comp_mem = ComputeMemory(number_of_stage=4)
        assert comp_mem.stages_a == []
        assert comp_mem.stages_b == []
        assert comp_mem.recompute_considered_[Recompute.TYPE.NONE]

        invalid_a = ComputeMemory(4, [_stage(1, nb_stage=4), _stage(1, nb_stage=5)])
        assert invalid_a.stages_a == []

        comp_mem.set_stages_b([_stage(1, nb_stage=4), _stage(1, nb_stage=5)])
        assert comp_mem.stages_b == []
        comp_mem.set_stages_b([_stage(1, nb_stage=4)])
        assert len(comp_mem.stages_b) == 1
        comp_mem.set_stages_b([_stage(2, nb_stage=4)])
        assert comp_mem.stages_b == []

        stage_one = _stage(1, nb_stage=4, nb_layer=2, memory_usage=20)
        stage_two = _stage(2, nb_stage=4, nb_layer=2, memory_usage=20)
        assert comp_mem._compute_memory_parameter_local_(stage_one, stage_two) == 10
        assert comp_mem._compute_memory_parameter_local_(stage_one, stage_one) == 0
        assert comp_mem._compute_memory_parameter_local_(stage_one, _stage(2, nb_layer=3)) == 0

        zero_offset_stages = [_stage(stage_id, nb_stage=5, nb_layer=2) for stage_id in range(5)]
        zero_offset = ComputeMemory(5, zero_offset_stages)
        assert not zero_offset._compute_memories_layers_()

        assert comp_mem._average_if_needed(2, {rec: 1 for rec in Recompute.TYPE}, 4, {rec: 3 for rec in Recompute.TYPE})
        assert comp_mem.memory_parameter_ == 3
        assert comp_mem._average_if_needed_fix(
            1,
            2,
            {rec: 1 for rec in Recompute.TYPE},
            3,
            4,
            {rec: 3 for rec in Recompute.TYPE},
        )
        assert comp_mem.memory_const_ == 2

        memory_file = tmp_path / "memory_input"
        memory_file.write_text("", encoding="utf-8")
        layers = [
            _layer(Layer.type_enum.HEAD, "head"),
            _layer(Layer.type_enum.BODY, "body"),
            _layer(Layer.type_enum.TAIL, "tail"),
        ]
        monkeypatch.setattr(compute_memory_module, "ComputeMemory", _FakeComputeMemory)
        updated = compute_memory_module.compute_memories(layers, str(memory_file), 2)
        assert updated[0].memory_parameter_ == 11
        assert updated[1].memory_parameter_ == 17
        assert updated[1].memory_activation_rec_[Recompute.TYPE.FULL] == 24
        assert updated[2].memory_parameter_ == 13
