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
"""Tests for yaml_parser.py — YAML configuration parsing.

Covers:
- parse_yaml_for_optimization() with real YAML fixture
- YamlOptimizationConfig dataclass construction
- New fields: memory_limit, constant_memory, enable_simulation,
  sim_comm_time, vpp_less_memory, optimization_level
- Error cases: missing fields, invalid values, bad types
"""

import os
import tempfile

import pytest
import yaml

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
    parse_yaml_for_optimization,
)

_DEMO_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixture_pp8_32layers.yaml",
)


def _write_yaml(tmpdir: str, data: dict) -> str:
    path = os.path.join(tmpdir, "test.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


def _base_config(**overrides: object) -> dict:
    cfg: dict = {
        "pipeline_config": {
            "pipeline_num": 4,
            "micro_batch_num": 8,
            "num_layer": 16,
        }
    }
    cfg["pipeline_config"].update(overrides)
    return cfg


class TestParseYamlForOptimization:
    """Test parse_yaml_for_optimization with real YAML fixture."""

    def test_demo_yaml_parses(self) -> None:
        config = parse_yaml_for_optimization(_DEMO_YAML)
        assert isinstance(config.pp_degree, int) and config.pp_degree > 0
        assert config.num_layer is None or (isinstance(config.num_layer, int) and config.num_layer > 0)
        assert isinstance(config.micro_batch_num, int) and config.micro_batch_num > 0
        assert isinstance(config.num_of_interleave, int) and config.num_of_interleave >= 1

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_yaml_for_optimization("/nonexistent/path.yaml")

    def test_missing_pipeline_num_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_config()
            del cfg["pipeline_config"]["pipeline_num"]
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="pipeline_num"):
                parse_yaml_for_optimization(path)

    def test_missing_num_layer_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_config()
            del cfg["pipeline_config"]["num_layer"]
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.num_layer is None

    def test_missing_micro_batch_num_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_config()
            del cfg["pipeline_config"]["micro_batch_num"]
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="micro_batch_num"):
                parse_yaml_for_optimization(path)

    def test_zero_pipeline_num_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(pipeline_num=0))
            with pytest.raises(ValueError, match="positive"):
                parse_yaml_for_optimization(path)

    def test_zero_num_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(num_layer=0))
            with pytest.raises(ValueError, match="positive"):
                parse_yaml_for_optimization(path)

    def test_zero_micro_batch_num_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(micro_batch_num=0))
            with pytest.raises(ValueError, match="positive"):
                parse_yaml_for_optimization(path)

    def test_non_dict_top_level_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, [1, 2, 3])
            with pytest.raises(ValueError, match="top-level mapping"):
                parse_yaml_for_optimization(path)

    def test_missing_pipeline_config_section_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, {"other": {}})
            with pytest.raises(ValueError, match="pipeline_config"):
                parse_yaml_for_optimization(path)

    def test_num_of_interleave_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config())
            config = parse_yaml_for_optimization(path)
            assert config.num_of_interleave == 1

    def test_custom_num_of_interleave(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(num_of_interleave=2))
            config = parse_yaml_for_optimization(path)
            assert config.num_of_interleave == 2

    def test_zero_num_of_interleave_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(num_of_interleave=0))
            with pytest.raises(ValueError, match="positive"):
                parse_yaml_for_optimization(path)


class TestYamlNewFields:
    """Test new fields: memory_limit, constant_memory, enable_simulation, sim_comm_time."""

    def test_demo_yaml_memory_limit(self) -> None:
        config = parse_yaml_for_optimization(_DEMO_YAML)
        assert config.memory_limit == 80000

    def test_demo_yaml_constant_memory(self) -> None:
        config = parse_yaml_for_optimization(_DEMO_YAML)
        assert config.constant_memory == 500

    def test_demo_yaml_enable_simulation(self) -> None:
        config = parse_yaml_for_optimization(_DEMO_YAML)
        assert config.enable_simulation is True

    def test_demo_yaml_sim_comm_time(self) -> None:
        config = parse_yaml_for_optimization(_DEMO_YAML)
        assert config.sim_comm_time == pytest.approx(0.1)

    def test_memory_limit_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config())
            config = parse_yaml_for_optimization(path)
            assert config.memory_limit == 0

    def test_constant_memory_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config())
            config = parse_yaml_for_optimization(path)
            assert config.constant_memory == 0

    def test_enable_simulation_defaults_to_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config())
            config = parse_yaml_for_optimization(path)
            assert config.enable_simulation is True

    def test_sim_comm_time_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config())
            config = parse_yaml_for_optimization(path)
            assert config.sim_comm_time == 0.0

    def test_negative_memory_limit_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(memory_limit=-1))
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)

    def test_negative_constant_memory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(constant_memory=-1))
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)

    def test_sim_comm_time_nan_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_config()
            cfg["pipeline_config"]["sim_comm_time"] = float("nan")
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="finite"):
                parse_yaml_for_optimization(path)

    def test_negative_sim_comm_time_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(sim_comm_time=-0.1))
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)

    def test_vpp_less_memory_string_yes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(vpp_less_memory="yes"))
            config = parse_yaml_for_optimization(path)
            assert config.vpp_less_memory is True

    def test_vpp_less_memory_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(vpp_less_memory=True))
            config = parse_yaml_for_optimization(path)
            assert config.vpp_less_memory is True

    def test_optimization_level_valid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for level in (0, 1, 2):
                path = _write_yaml(tmpdir, _base_config(optimization_level=level))
                config = parse_yaml_for_optimization(path)
                assert config.optimization_level == level

    def test_optimization_level_invalid_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(optimization_level=3))
            with pytest.raises(ValueError, match="0, 1, or 2"):
                parse_yaml_for_optimization(path)

    def test_enable_simulation_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, _base_config(enable_simulation=False))
            config = parse_yaml_for_optimization(path)
            assert config.enable_simulation is False


class TestYamlOptimizationConfigDataclass:
    """Test YamlOptimizationConfig direct construction."""

    def test_direct_construction(self) -> None:
        """Test YamlOptimizationConfig with required fields only; defaults fill the rest."""
        config = YamlOptimizationConfig(
            pp_degree=8,
            micro_batch_num=8,
        )
        assert config.pp_degree == 8
        assert config.num_layer is None
        assert config.micro_batch_num == 8
        assert config.num_of_interleave == 1
        assert config.memory_limit == 0
        assert config.constant_memory == 0
        assert config.enable_simulation is True
        assert config.sim_comm_time == 0.0
        assert config.vpp_less_memory is False
        assert config.optimization_level == 1

    def test_direct_construction_with_all_fields(self) -> None:
        """Test YamlOptimizationConfig with every field explicitly provided."""
        config = YamlOptimizationConfig(
            pp_degree=4,
            num_layer=16,
            micro_batch_num=8,
            num_of_interleave=2,
            vpp_less_memory=True,
            optimization_level=2,
            memory_limit=80000,
            constant_memory=500,
            enable_simulation=False,
            sim_comm_time=0.5,
        )
        assert config.num_of_interleave == 2
        assert config.vpp_less_memory is True
        assert config.optimization_level == 2
        assert config.memory_limit == 80000
        assert config.constant_memory == 500
        assert config.enable_simulation is False
        assert config.sim_comm_time == pytest.approx(0.5)

    def test_validate_pp_degree_zero_raises(self) -> None:
        config = YamlOptimizationConfig(pp_degree=0, micro_batch_num=4)
        with pytest.raises(ValueError, match="positive"):
            config.validate()

    def test_validate_micro_batch_zero_raises(self) -> None:
        config = YamlOptimizationConfig(pp_degree=2, micro_batch_num=0)
        with pytest.raises(ValueError, match="positive"):
            config.validate()

    def test_validate_num_layer_none_passes(self) -> None:
        config = YamlOptimizationConfig(pp_degree=2, micro_batch_num=4, num_layer=None)
        config.validate()

    def test_validate_num_layer_zero_raises(self) -> None:
        config = YamlOptimizationConfig(pp_degree=2, micro_batch_num=4, num_layer=0)
        with pytest.raises(ValueError, match="positive"):
            config.validate()
