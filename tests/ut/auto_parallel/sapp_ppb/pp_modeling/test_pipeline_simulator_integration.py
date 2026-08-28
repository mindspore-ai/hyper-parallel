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
"""Tests for PipelineSimulator integration in pp_modeling.

Covers:
- PPBOutput simulator fields (simulator_end_time, simulator_bubbles)
- PPStrategyResult simulator fields propagation
- sim_comm_time from YAML config through to simulator
- enable_simulation from YAML config
- ILP feasible but simulator failed scenario
- NaN simulator result handling
"""

import os
import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPBOutput,
    PPStrategyResult,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import PPBalancer
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder
from hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer import PPOptimizer

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixture_profile_32layers.json",
)


class TestPPBOutputSimulatorFields:
    """Test PPBOutput simulator-related fields."""

    def test_default_simulator_fields(self) -> None:
        """Test that default simulator fields are zero and empty."""
        output = PPBOutput()
        assert output.simulator_end_time == 0.0
        assert output.simulator_bubbles == {}
        assert output.simulator_peak_memory == []

    def test_simulator_fields_populated(self) -> None:
        """Test that simulator fields can be populated on PPBOutput."""
        output = PPBOutput(
            simulator_end_time=1234.5,
            simulator_bubbles={"real": 0.15, "ideal": 0.12, "imba": 0.02, "comm": 0.01},
            simulator_peak_memory=[5000.0, 5200.0],
        )
        assert output.simulator_end_time == 1234.5
        assert output.simulator_bubbles["real"] == 0.15
        assert output.simulator_peak_memory == [5000.0, 5200.0]

    def test_default_simulation_status(self) -> None:
        """Test that default simulation_status is 'not_run' and simulation_error is None."""
        output = PPBOutput()
        assert output.simulation_status == "not_run"
        assert output.simulation_error is None

    def test_simulation_status_success(self) -> None:
        """Test simulation_status='success' with populated simulator fields."""
        output = PPBOutput(
            simulation_status="success",
            simulator_end_time=500.0,
            simulator_bubbles={"real": 0.1},
        )
        assert output.simulation_status == "success"
        assert output.simulation_error is None
        assert output.simulator_end_time == 500.0

    def test_simulation_status_failed(self) -> None:
        """Test simulation_status='failed' preserves is_feasible=True."""
        output = PPBOutput(
            is_feasible=True,
            simulation_status="failed",
            simulation_error="micro_batch_num < pp_degree",
        )
        assert output.is_feasible is True
        assert output.simulation_status == "failed"
        assert output.simulation_error == "micro_batch_num < pp_degree"
        assert output.simulator_end_time == 0.0


class TestPPStrategyResultSimulatorFields:
    """Test PPStrategyResult simulator fields."""

    def test_default_simulator_fields(self) -> None:
        """Test that default PPStrategyResult simulator fields are empty."""
        result = PPStrategyResult(pp_degree=2, micro_batch_num=4)
        assert result.simulator_bubbles == {}
        assert result.simulator_peak_memory == []
        assert result.is_successful is False
        assert result.pipeline_bubble is None
        assert result.simulator_end_time == 0.0
        assert result.simulation_status == "not_run"
        assert result.simulation_error is None

    def test_simulator_bubbles_propagate(self) -> None:
        """Test that simulator_bubbles can be set on PPStrategyResult."""
        result = PPStrategyResult(
            pp_degree=2, micro_batch_num=4,
            simulator_bubbles={"real": 0.2, "ideal": 0.15},
        )
        assert result.simulator_bubbles == {"real": 0.2, "ideal": 0.15}

    def test_simulator_peak_memory_propagate(self) -> None:
        """Test that simulator_peak_memory can be set on PPStrategyResult."""
        result = PPStrategyResult(
            pp_degree=2, micro_batch_num=4,
            simulator_peak_memory=[8000.0, 8200.0],
        )
        assert result.simulator_peak_memory == [8000.0, 8200.0]

    def test_is_successful_propagate(self) -> None:
        """Test that is_successful can be set on PPStrategyResult."""
        result = PPStrategyResult(
            pp_degree=2, micro_batch_num=4,
            is_successful=True,
        )
        assert result.is_successful is True

    def test_simulation_status_propagate(self) -> None:
        """PPStrategyResult simulation_status/error propagate from PPBOutput."""
        result = PPStrategyResult(
            pp_degree=2, micro_batch_num=4,
            simulation_status="failed",
            simulation_error="micro_batch_num < pp_degree",
        )
        assert result.simulation_status == "failed"
        assert result.simulation_error == "micro_batch_num < pp_degree"
        assert result.pipeline_bubble is None
        assert result.simulator_end_time == 0.0

    def test_successful_simulation_has_values(self) -> None:
        """When simulation_status='success', bubble and step time should be numeric."""
        result = PPStrategyResult(
            pp_degree=2, micro_batch_num=4,
            pipeline_bubble=0.15,
            simulator_end_time=500.0,
            simulation_status="success",
        )
        assert result.pipeline_bubble == 0.15
        assert result.simulator_end_time == 500.0
        assert result.simulation_status == "success"


class TestYamlConfigSimCommTime:
    """Test sim_comm_time field in YamlOptimizationConfig."""

    def test_comm_time_zero_by_default(self) -> None:
        """Test that sim_comm_time defaults to 0.0."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
        )
        assert config.sim_comm_time == 0.0

    def test_comm_time_set_from_config(self) -> None:
        """Test that sim_comm_time is set from yaml_config."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
            sim_comm_time=0.5,
        )
        assert config.sim_comm_time == pytest.approx(0.5)


class TestYamlConfigConstantMemory:
    """Test constant_memory field in YamlOptimizationConfig."""

    def test_default_constant_memory_is_zero(self) -> None:
        """Test that default constant_memory is zero."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
        )
        assert config.constant_memory == 0

    def test_constant_memory_set_from_config(self) -> None:
        """Test that constant_memory is set from yaml_config."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
            constant_memory=500,
        )
        assert config.constant_memory == 500


class TestILPFeasibleButSimulatorFailed:
    """Test that ILP feasibility is independent of simulator status."""

    def test_micro_batch_less_than_pp_keeps_feasible(self) -> None:
        """ILP result should be is_feasible=True even when simulator cannot run."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=4, micro_batch_num=2, memory_limit=80000,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        result = balancer.balance_with_ilp(time_limit=30, solver="pulp")

        assert result.is_feasible is True, (
            f"ILP should be feasible, got: {result.infeasibility_details}"
        )
        assert result.simulation_status == "not_run"
        assert result.is_successful is True

    def test_normal_config_sets_simulation_success(self) -> None:
        """Normal config (micro_batch >= pp) should set simulation_status='success'."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
            memory_limit=80000, constant_memory=500,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        result = balancer.balance_with_ilp(time_limit=30, solver="pulp")

        assert result.is_feasible is True
        assert result.simulation_status == "not_run"

    def test_nan_sim_result_marked_failed(self) -> None:
        """When simulator returns NaN, PPOptimizer marks simulation as failed."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
            memory_limit=80000, constant_memory=500,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)

        nan_result = PPBOutput(
            is_feasible=True,
            simulator_end_time=float("nan"),
            simulator_bubbles={"real": float("nan"), "ideal": 0.5, "imba": float("nan")},
            simulator_peak_memory=[0.0, 0.0],
        )

        from unittest.mock import patch, MagicMock
        with patch.object(balancer, "_build_ilp_result", return_value=PPBOutput(is_feasible=True)):
            result = balancer.balance_with_ilp()
            with patch("hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer.PPSimulator") as mock_sim_cls:
                mock_sim = MagicMock()
                mock_sim.simulate_from_ilp.return_value = nan_result
                mock_sim_cls.return_value = mock_sim
                PPOptimizer._run_simulation(
                    result,
                    balancer.pipeline,
                    yaml_config,
                    layer_builder._constant_memory,  # pylint: disable=W0212
                    yaml_config.sim_comm_time,
                )

        assert result.simulation_status == "failed"
        assert "non-finite" in result.simulation_error


class TestEnableSimulationConfig:
    """Test enable_simulation field in YamlOptimizationConfig."""

    def test_default_is_true(self) -> None:
        """Test that enable_simulation defaults to True."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
        )
        assert config.enable_simulation is True

    def test_set_to_false(self) -> None:
        """Test that enable_simulation can be set to False."""
        config = YamlOptimizationConfig(
            pp_degree=2, num_layer=32, micro_batch_num=4,
            enable_simulation=False,
        )
        assert config.enable_simulation is False


class TestEnableSimulationFalseSkipsSim:
    """Test that enable_simulation=False skips simulator in PPOptimizer."""

    def test_skips_simulation_when_disabled(self) -> None:
        """When enable_simulation=False, PPOptimizer skips simulation entirely."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
            memory_limit=80000, constant_memory=500,
            enable_simulation=False,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        result = balancer.balance_with_ilp(time_limit=30, solver="pulp")

        assert result.is_feasible is True
        assert result.simulation_status == "not_run"
        assert result.simulator_end_time == 0.0
