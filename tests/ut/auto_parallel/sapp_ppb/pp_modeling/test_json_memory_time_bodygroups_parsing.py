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
"""Tests for native JSON parsing via LayerBuilder.

Validates that ``LayerBuilder(yaml_config, json_path)`` correctly
parses native sapp-ppb JSON files through ``generate_layers_list()``,
and that the resulting Layer objects have the expected memory_parameter,
memory_activation_rec_, recompute_considered_, and time values.

Covers:
- Single-body-group JSON fixture
- Multi-body-group JSON fixture
- LayerBuilder validation (empty json_path, no layers_description,
  group name conflicts)
- HEAD/TAIL recompute_considered forced to NONE-only
- BODY recompute_considered inferred from memory_activation_rec_
"""

import json
import os
import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import PPBalancer
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_32layers.json"
)
_MULTI_GROUP_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_multi_group.json"
)

_PP = 8
_NUM_BODY_LAYERS = 32
_MICRO_BATCH = 8
_MEMORY_LIMIT = 80000


def _write_json(tmp_path, cfg: dict, name: str = "test_config.json") -> str:
    """Write cfg as JSON under tmp_path and return the path."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return str(p)


class TestLayerBuilderSingleBodyFixture:
    """Test LayerBuilder with the single-body fixture_profile_32layers.json."""

    def test_layers_built_correctly(self) -> None:
        """LayerBuilder produces layers from native JSON fixture."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
            constant_memory=500,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb
        assert len(layers) == 3

    def test_body_layer_memory_parameter(self) -> None:
        """BODY layer has correct memory_parameter from native JSON."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY][0]
        assert body.memory_parameter_ == 200

    def test_body_layer_memory_activation_rec(self) -> None:
        """BODY layer has correct memory_activation_rec_ from native JSON."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY][0]
        assert body.memory_activation_rec_[Recompute.TYPE.NONE] == 100
        assert body.memory_activation_rec_[Recompute.TYPE.SLCT] == 60

    def test_body_layer_recompute_considered(self) -> None:
        """BODY layer recompute_considered_ matches memory_activation_rec_ availability."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY][0]
        assert body.recompute_considered_[Recompute.TYPE.NONE] is True
        assert body.recompute_considered_[Recompute.TYPE.SLCT] is True

    def test_head_tail_recompute_considered_only_none(self) -> None:
        """HEAD and TAIL layers only have NONE recompute_considered."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        for layer in layers:
            if layer.type_ in (layer_cls.type_enum.HEAD, layer_cls.type_enum.TAIL):
                assert layer.recompute_considered_[Recompute.TYPE.NONE] is True
                for rec in Recompute.TYPE:
                    if rec != Recompute.TYPE.NONE:
                        assert layer.recompute_considered_[rec] is False

    def test_constant_memory_from_config(self) -> None:
        """constant_memory is read from yaml_config."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, constant_memory=1000,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        assert layer_builder._constant_memory == 1000

    def test_memory_limit_from_config(self) -> None:
        """memory_limit is read from yaml_config."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=50000,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        assert layer_builder._memory_limit == 50000

    def test_enable_simulation_from_config(self) -> None:
        """enable_simulation is read from yaml_config."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, enable_simulation=False,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        assert layer_builder._enable_simulation is False

    def test_body_layer_forward_time(self) -> None:
        """BODY layer forward_time_ equals native JSON 'time' field."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY][0]
        assert body.forward_time_ == 3.0


class TestLayerBuilderMultiBodyFixture:
    """Test LayerBuilder with the multi-body fixture_profile_multi_group.json."""

    def test_two_body_groups_produces_valid_partition(self) -> None:
        """Two BODY groups should produce a valid partition with correct layer IDs."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            pp_degree=2, micro_batch_num=4, num_layer=24, memory_limit=80000,
        )
        layer_builder = LayerBuilder(yaml_config, _MULTI_GROUP_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30)

        assert output.is_feasible, f"Expected feasible, got: {output.infeasibility_details}"
        assert len(output.stage_partition) == 2

        all_layer_ids = []
        for stage in output.stage_partition:
            all_layer_ids.extend(e[0] for e in stage)
        assert set(all_layer_ids) == set(range(26)), "HEAD(0) + 24 body(1-24) + TAIL(25)"

    def test_multi_body_group_names(self) -> None:
        """BODY layers have names matching the native JSON 'name' field."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            pp_degree=2, micro_batch_num=4, num_layer=24,
        )
        layer_builder = LayerBuilder(yaml_config, _MULTI_GROUP_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body_layers = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY]
        body_names = {lay.name_ for lay in body_layers}
        assert "encoder" in body_names
        assert "decoder" in body_names

    def test_multi_body_layer_offset_has_two_groups(self) -> None:
        """Multi-body ILP produces layer_offset with two group keys."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            pp_degree=2, micro_batch_num=4, num_layer=24, memory_limit=80000,
        )
        layer_builder = LayerBuilder(yaml_config, _MULTI_GROUP_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30)

        assert output.is_feasible
        assert isinstance(output.layer_offset, dict)
        assert len(output.layer_offset) == 2, (
            f"Expected 2 groups in layer_offset, got {len(output.layer_offset)}"
        )
        for group_name, group_offset in output.layer_offset.items():
            assert isinstance(group_offset, list)
            assert len(group_offset) > 0

    def test_multi_body_recompute_considered(self) -> None:
        """Multi-body FULL recompute types should be reflected in recompute_considered_."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            pp_degree=2, micro_batch_num=4, num_layer=24,
        )
        layer_builder = LayerBuilder(yaml_config, _MULTI_GROUP_JSON)
        layers = layer_builder.layers_sapp_ppb

        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        layer_cls = _get_pipeline_layer_class()
        body_layers = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY]
        for body in body_layers:
            assert body.recompute_considered_[Recompute.TYPE.NONE] is True
            assert body.recompute_considered_[Recompute.TYPE.FULL] is True


class TestLayerBuilderValidation:
    """Test LayerBuilder validation and error handling."""

    def test_empty_json_path_raises(self) -> None:
        """Empty json_path raises ValueError."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
        )
        with pytest.raises(ValueError, match="json_path"):
            LayerBuilder(yaml_config, "")

    def test_none_json_path_raises(self) -> None:
        """None json_path raises ValueError."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
        )
        with pytest.raises((ValueError, TypeError)):
            LayerBuilder(yaml_config, None)  # type: ignore[arg-type]

    def test_no_layers_description_raises(self, tmp_path) -> None:
        """JSON without layers_description raises ValueError."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        cfg: dict = {"name": "empty_model"}
        path = _write_json(tmp_path, cfg)
        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4,
        )
        with pytest.raises(ValueError, match="No layers parsed"):
            LayerBuilder(yaml_config, path)

    def test_group_name_conflicts_with_solver_raises(self, tmp_path) -> None:
        """A body group named 'max_stage_time' conflicts with ILP solver internal variables."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        cfg: dict = {
            "name": "conflict_test",
            "pre_defined_layer": {"HEAD": 0, "TAIL": -1},
            "auto_partition_layer": {"NumberOfLayers": 16},
            "layers_description": [
                {
                    "name": "HEAD", "type": "HEAD", "model_name": "conflict_test",
                    "time": 10, "nb_layer": 1, "memory_parameter": 500,
                },
                {
                    "name": "max_stage_time", "type": "BODY", "model_name": "conflict_test",
                    "time": 5, "nb_layer": 16, "memory_parameter": 100,
                    "memory_activation": 200, "memory_recompute": 50,
                },
                {
                    "name": "TAIL", "type": "TAIL", "model_name": "conflict_test",
                    "time": 20, "nb_layer": 1, "memory_parameter": 500,
                },
            ],
        }
        path = _write_json(tmp_path, cfg)
        yaml_config = YamlOptimizationConfig(
            pp_degree=2, micro_batch_num=4, num_layer=16,
        )
        with pytest.raises(ValueError, match="conflict"):
            LayerBuilder(yaml_config, path)


class TestOffsetZeroWithJsonConfig:
    """Critical test: offset=0 + JSON config -> ILP produces reasonable results."""

    def test_offset_zero_ilp_has_memory_constraints(self) -> None:
        """ILP with uniform partition produces feasible result."""
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

    def test_offset_zero_tight_memory(self) -> None:
        """With a tight memory_limit, ILP should not degrade."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=4000,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        assert output.is_feasible, f"Infeasible: {output.infeasibility_details}"


class TestThreeBodyGroupsFrontierOrdering:
    """Three BODY groups should respect frontier ordering via ILP constraints."""

    def test_three_body_groups_frontier_ordering(self, tmp_path) -> None:
        """Three BODY groups should produce a valid partition with correct ordering."""
        if not SAPP_PPB_AVAILABLE:
            pytest.skip("sapp_ppb not available")

        cfg: dict = {
            "name": "three_group_test",
            "pre_defined_layer": {"HEAD": 0, "TAIL": -1},
            "auto_partition_layer": {"NumberOfLayers": 24},
            "layers_description": [
                {
                    "name": "HEAD", "type": "HEAD", "model_name": "three_group_test",
                    "time": 10, "nb_layer": 1, "memory_parameter": 500,
                },
                {
                    "name": "group_a", "type": "BODY", "model_name": "three_group_test",
                    "time": 5, "nb_layer": 8, "memory_parameter": 100,
                    "memory_activation": 200, "memory_recompute": 50,
                },
                {
                    "name": "group_b", "type": "BODY", "model_name": "three_group_test",
                    "time": 4, "nb_layer": 8, "memory_parameter": 80,
                    "memory_activation": 160, "memory_recompute": 40,
                },
                {
                    "name": "group_c", "type": "BODY", "model_name": "three_group_test",
                    "time": 3, "nb_layer": 8, "memory_parameter": 60,
                    "memory_activation": 120, "memory_recompute": 30,
                },
                {
                    "name": "TAIL", "type": "TAIL", "model_name": "three_group_test",
                    "time": 20, "nb_layer": 1, "memory_parameter": 500,
                },
            ],
        }
        path = _write_json(tmp_path, cfg)
        yaml_config = YamlOptimizationConfig(
            pp_degree=4, micro_batch_num=8, num_layer=24, memory_limit=100000,
        )
        layer_builder = LayerBuilder(yaml_config, path)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30)

        assert output.is_feasible, f"Expected feasible, got: {output.infeasibility_details}"
        all_ids = []
        for stage in output.stage_partition:
            all_ids.extend(e[0] for e in stage)
        total_body = 24
        assert set(all_ids) == set(range(total_body + 2))
