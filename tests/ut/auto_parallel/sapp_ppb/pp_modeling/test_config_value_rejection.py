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
"""Tests for strict configuration validation.

Covers:
- YAML _validate_int rejects float inputs for integer fields
- YAML _to_bool rejects ambiguous strings
- YAML float rejection for pipeline_num, num_layer, micro_batch_num,
  num_of_interleave, memory_limit, constant_memory, optimization_level
- YAML sim_comm_time nan/negative rejection
- YAML memory_limit negative rejection
- YamlOptimizationConfig.validate() catches invalid values
"""

import os
import tempfile

import pytest
import yaml

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    _to_bool,
    _validate_int,
    parse_yaml_for_optimization,
)


def _write_yaml(tmpdir: str, data: dict) -> str:
    path = os.path.join(tmpdir, "test.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


def _base_yaml_config(**overrides: object) -> dict:
    cfg: dict = {
        "pipeline_config": {
            "pipeline_num": 4,
            "num_layer": 16,
            "micro_batch_num": 8,
        }
    }
    cfg["pipeline_config"].update(overrides)
    return cfg


class TestValidateInt:
    """_validate_int must reject float inputs."""

    def test_validate_int_rejects_float_with_fraction(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_int(1.9, "test_field")

    def test_validate_int_rejects_integer_float(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_int(3.0, "test_field")

    def test_validate_int_accepts_int(self) -> None:
        assert _validate_int(3, "test_field") == 3

    def test_validate_int_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_int(True, "test_field")
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_int(False, "test_field")


class TestYamlIntegerFieldRejection:
    """YAML integer fields must reject float inputs."""

    def test_yaml_pipeline_num_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(pipeline_num=2.5)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_num_layer_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(num_layer=1.9)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_missing_num_layer_defaults_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config()
            del cfg["pipeline_config"]["num_layer"]
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.num_layer is None

    def test_yaml_micro_batch_num_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(micro_batch_num=4.0)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_num_of_interleave_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(num_of_interleave=2.0)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_memory_limit_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(memory_limit=3.5)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_constant_memory_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(constant_memory=500.0)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)

    def test_yaml_optimization_level_float_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(optimization_level=1.0)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="must be an integer"):
                parse_yaml_for_optimization(path)


class TestYamlSimCommTimeRejection:
    """YAML sim_comm_time must be finite and non-negative."""

    def test_sim_comm_time_nan_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config()
            cfg["pipeline_config"]["sim_comm_time"] = float("nan")
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="finite"):
                parse_yaml_for_optimization(path)

    def test_sim_comm_time_inf_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config()
            cfg["pipeline_config"]["sim_comm_time"] = float("inf")
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="finite"):
                parse_yaml_for_optimization(path)

    def test_sim_comm_time_negative_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(sim_comm_time=-0.1)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)

    def test_sim_comm_time_zero_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(sim_comm_time=0.0)
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.sim_comm_time == 0.0


class TestYamlMemoryLimitRejection:
    """YAML memory_limit must be non-negative integer."""

    def test_negative_memory_limit_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(memory_limit=-1)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)

    def test_zero_memory_limit_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(memory_limit=0)
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.memory_limit == 0

    def test_negative_constant_memory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(constant_memory=-1)
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="non-negative"):
                parse_yaml_for_optimization(path)


class TestToBoolStrict:
    """_to_bool must reject ambiguous strings, accept only whitelist."""

    def test_maybe_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret"):
            _to_bool("maybe", "vpp_less_memory")

    def test_true_string_accepted(self) -> None:
        assert _to_bool("true", "field") is True

    def test_yes_string_accepted(self) -> None:
        assert _to_bool("yes", "field") is True

    def test_false_string_accepted(self) -> None:
        assert _to_bool("false", "field") is False

    def test_no_string_accepted(self) -> None:
        assert _to_bool("no", "field") is False

    def test_string_one_accepted(self) -> None:
        assert _to_bool("1", "field") is True

    def test_string_zero_accepted(self) -> None:
        assert _to_bool("0", "field") is False

    def test_bool_input_accepted(self) -> None:
        assert _to_bool(True, "field") is True
        assert _to_bool(False, "field") is False

    def test_int_input_accepted(self) -> None:
        assert _to_bool(1, "field") is True
        assert _to_bool(0, "field") is False

    def test_integer_float_accepted(self) -> None:
        assert _to_bool(1.0, "field") is True
        assert _to_bool(0.0, "field") is False

    def test_non_integer_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret"):
            _to_bool(0.5, "field")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret"):
            _to_bool("", "field")

    def test_yaml_vpp_less_memory_ambiguous_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(vpp_less_memory="maybe")
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="cannot interpret"):
                parse_yaml_for_optimization(path)

    def test_yaml_vpp_less_memory_yes_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(vpp_less_memory="yes")
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.vpp_less_memory is True

    def test_yaml_vpp_less_memory_no_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(vpp_less_memory="no")
            path = _write_yaml(tmpdir, cfg)
            config = parse_yaml_for_optimization(path)
            assert config.vpp_less_memory is False

    def test_yaml_enable_simulation_ambiguous_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _base_yaml_config(enable_simulation="maybe")
            path = _write_yaml(tmpdir, cfg)
            with pytest.raises(ValueError, match="cannot interpret"):
                parse_yaml_for_optimization(path)


class TestYamlOptimizationConfigValidation:
    """YamlOptimizationConfig.validate() catches invalid values."""

    def test_pp_degree_zero(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=0, micro_batch_num=4)
        with pytest.raises(ValueError, match="positive"):
            cfg.validate()

    def test_pp_degree_negative(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=-1, micro_batch_num=4)
        with pytest.raises(ValueError, match="positive"):
            cfg.validate()

    def test_micro_batch_zero(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=2, micro_batch_num=0)
        with pytest.raises(ValueError, match="positive"):
            cfg.validate()

    def test_valid_config_passes(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=2, micro_batch_num=4)
        cfg.validate()

    def test_num_layer_none_passes(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=2, micro_batch_num=4, num_layer=None)
        cfg.validate()

    def test_num_layer_zero_fails(self) -> None:
        cfg = YamlOptimizationConfig(pp_degree=2, micro_batch_num=4, num_layer=0)
        with pytest.raises(ValueError, match="positive"):
            cfg.validate()
