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
"""Tests for ILP constraint extraction and PPOptimizer integration.

Tests PPBalancer._extract_layer_offset_from_ilp using YAML+JSON fixtures,
and PPOptimizer end-to-end with real fixture files.
"""

import os

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer import PPOptimizer
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import PPBalancer
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder

_DEMO_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_pp8_32layers.yaml"
)
_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_32layers.json",
)

_PP = 8
_NUM_BODY_LAYERS = 32
_MICRO_BATCH = 8
_MEMORY_LIMIT = 80000

pytestmark = pytest.mark.pp_modeling


class TestExtractLayerOffsetFromILP:
    """Tests for PPBalancer._extract_layer_offset_from_ilp."""

    def test_offset_extracted_after_ilp(self) -> None:
        """After balance_with_ilp, layer_offset is extracted from the ILP solution."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        assert output.is_feasible, f"Infeasible: {output.infeasibility_details}"

        assert isinstance(output.layer_offset, dict)
        assert len(output.layer_offset) > 0, (
            "Expected at least one group in layer_offset, got empty"
        )
        for group_name, group_offset in output.layer_offset.items():
            assert isinstance(group_offset, list), (
                f"group '{group_name}' offset should be List[List[int]]"
            )
            assert len(group_offset) > 0, (
                f"group '{group_name}' expected at least one VPP chunk, got empty"
            )
            assert len(group_offset[0]) == _PP, (
                f"group '{group_name}' expected {_PP} offset values per VPP chunk, "
                f"got {len(group_offset[0])}"
            )

    def test_offset_matches_direct_extraction(self) -> None:
        """PPBOutput.layer_offset matches _extract_layer_offset_from_ilp result."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        assert output.is_feasible

        direct_offset = balancer._extract_layer_offset_from_ilp(output.stage_partition)
        assert output.layer_offset == direct_offset

    def test_uniform_partition_zero_offset(self) -> None:
        """With PP=2 and divisible num_layer, offset should be reasonable."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2,
            micro_batch_num=4, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        assert output.is_feasible

        assert isinstance(output.layer_offset, dict)
        assert len(output.layer_offset) > 0, (
            "Expected at least one group in layer_offset, got empty"
        )
        for group_name, group_offset in output.layer_offset.items():
            assert isinstance(group_offset, list), (
                f"group '{group_name}' offset should be List[List[int]]"
            )
            assert len(group_offset) > 0, (
                f"group '{group_name}' expected at least one VPP chunk, got empty"
            )
            assert len(group_offset[0]) == 2, (
                f"group '{group_name}' expected 2 offset values per VPP chunk for PP=2, "
                f"got {len(group_offset[0])}"
            )
            for row in group_offset:
                for val in row:
                    assert abs(val) <= 8, (
                        f"group '{group_name}' offset should not exceed total layers, got {val}"
                    )


class TestPPOptimizerWithFixtures:
    """Test PPOptimizer produces feasible results with YAML+JSON fixtures."""

    def test_optimizer_with_demo_fixtures(self) -> None:
        """PPOptimizer with demo YAML + demo JSON should produce feasible result."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        optimizer = PPOptimizer()
        result = optimizer.optimize(
            yaml_path=_DEMO_YAML,
            json_path=_DEMO_JSON,
        )

        assert result.pp_degree == _PP
        assert result.micro_batch_num == _MICRO_BATCH
        assert len(result.stage_partition) > 0

    def test_optimizer_result_has_simulator_data(self) -> None:
        """PPOptimizer result should include simulator end time and bubbles."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        optimizer = PPOptimizer()
        result = optimizer.optimize(
            yaml_path=_DEMO_YAML,
            json_path=_DEMO_JSON,
        )

        assert result.simulator_end_time > 0.0
        assert result.simulation_status == "success"
        assert isinstance(result.simulator_bubbles, dict)

    def test_stage_partition_has_recompute_annotations(self) -> None:
        """stage_partition entries should be (layer_id, RecomputeType) tuples."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        assert output.is_feasible

        for stage in output.stage_partition:
            for entry in stage:
                assert isinstance(entry, tuple) and len(entry) == 2, (
                    f"Each entry should be (layer_id, RecomputeType) tuple, got {entry}"
                )
                assert isinstance(entry[0], int), f"layer_id should be int, got {type(entry[0])}"
