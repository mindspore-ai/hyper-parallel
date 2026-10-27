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
"""Unit tests for the reusable AutoModel infrastructure build boundary."""

import unittest
from unittest.mock import Mock, patch

from torch import nn

from hyper_parallel.models._transformers import model_builder


_TARGET = "hyper_parallel.models._transformers.model_builder"


class TestPrepareParallelModel(unittest.TestCase):
    """Verify the parallel-model-ready boundary without distributed hardware."""

    def test_prepare_parallel_model_stops_before_execution_infrastructure(self):
        """Stop after replacement and native sharding without later build stages."""
        source_model = nn.Linear(2, 2)
        replaced_model = nn.Sequential(source_model)
        parallel_model = nn.Sequential(replaced_model)
        weights_mapping = object()
        updated_mapping = object()
        source_shard_info = {"weight": object()}
        sharding_plan = object()
        call_order = []

        with patch(_TARGET + "._apply_pre_sharding_features",
                   side_effect=lambda *args: call_order.append("features")), \
                patch(_TARGET + ".get_model_conversion_mapping",
                      side_effect=lambda model: call_order.append("mapping") or weights_mapping), \
                patch(_TARGET + "._maybe_apply_module_replacements",
                      side_effect=lambda *args: (
                          call_order.append("replacement") or replaced_model,
                          updated_mapping,
                      )), \
                patch(_TARGET + "._apply_materialization_adapter",
                      side_effect=lambda model: call_order.append("materialization_adapter")), \
                patch(_TARGET + "._apply_parameter_freezing",
                      side_effect=lambda *args: call_order.append("freezing")), \
                patch(_TARGET + "._parallelize_from_generated",
                      side_effect=lambda *args, **kwargs: (
                          call_order.append("generated") or False,
                          None,
                      )), \
                patch(_TARGET + "._plan_and_apply_sharding",
                      side_effect=lambda *args: (
                          call_order.append("native_plan") or parallel_model,
                          source_shard_info,
                          sharding_plan,
                      )), \
                patch(_TARGET + "._apply_activation_features") as activation, \
                patch(_TARGET + "._apply_fsdp2_with_external") as fsdp, \
                patch(_TARGET + "._materialize_and_load_model") as materialize, \
                patch(_TARGET + ".apply_compile") as compile_model:
            result = model_builder.prepare_parallel_model(
                source_model,
                mesh=object(),
                sharding_planner=object(),
                peft_config=object(),
                qat_config=object(),
                fp8_config=object(),
                freeze_config=object(),
                is_meta_device=True,
            )

        self.assertEqual(
            call_order,
            [
                "features",
                "mapping",
                "replacement",
                "materialization_adapter",
                "freezing",
                "generated",
                "native_plan",
            ],
        )
        self.assertIs(result.model, parallel_model)
        self.assertIs(result.sharding_plan, sharding_plan)
        self.assertIs(result.source_shard_info, source_shard_info)
        self.assertIs(result.weights_mapping, updated_mapping)
        activation.assert_not_called()
        fsdp.assert_not_called()
        materialize.assert_not_called()
        compile_model.assert_not_called()

    def test_apply_model_infrastructure_continues_after_parallel_boundary(self):
        """Keep the complete native build path after the reusable boundary."""
        source_model = nn.Linear(2, 2)
        parallel_model = nn.Sequential(source_model)
        activated_model = nn.Sequential(parallel_model)
        fsdp_model = nn.Sequential(activated_model)
        loaded_model = nn.Sequential(fsdp_model)
        weights_mapping = object()
        source_shard_info = {"weight": object()}
        parallel_result = model_builder.ParallelModelBuildResult(
            model=parallel_model,
            sharding_plan=object(),
            source_shard_info=source_shard_info,
            weights_mapping=weights_mapping,
        )
        call_order = []

        with patch(_TARGET + "._resolve_compile_config", return_value=(None, False)), \
                patch(_TARGET + ".prepare_parallel_model",
                      side_effect=lambda *args, **kwargs: (
                          call_order.append("parallel") or parallel_result
                      )), \
                patch(_TARGET + "._apply_activation_features",
                      side_effect=lambda *args, **kwargs: (
                          call_order.append("activation") or activated_model
                      )), \
                patch(_TARGET + "._create_external_state",
                      side_effect=lambda *args: call_order.append("external")), \
                patch(_TARGET + "._apply_fsdp2_with_external",
                      side_effect=lambda *args: call_order.append("fsdp") or fsdp_model), \
                patch(_TARGET + "._materialize_and_load_model",
                      side_effect=lambda *args, **kwargs: (
                          call_order.append("materialize") or loaded_model
                      )), \
                patch(_TARGET + ".apply_model_init_dtype",
                      side_effect=lambda *args: call_order.append("dtype")):
            result = model_builder.apply_model_infrastructure(source_model)

        self.assertIs(result, loaded_model)
        self.assertEqual(
            call_order,
            ["parallel", "activation", "external", "fsdp", "materialize", "dtype"],
        )

    def test_plan_only_boundary_returns_native_plan_without_applying_it(self):
        """Offline tooling may share planner decisions without a process group."""
        model = nn.Linear(2, 2)
        plan = object()
        mesh = type("Mesh", (), {
            "device_mesh": object(),
            "tp_size": 2,
            "cp_size": 1,
            "ep_size": 1,
            "sequence_parallel": False,
            "loss_parallel": False,
        })()
        planner = Mock()
        planner.plan.return_value = plan

        with patch(_TARGET + ".apply_sharding_plan") as apply_plan:
            result = model_builder._plan_and_apply_sharding(
                model,
                mesh,
                planner,
                is_hf_model=True,
                validate_placement=False,
                parallel_plan_mode="plan_only",
            )

        self.assertEqual(result, (model, None, plan))
        planner.plan.assert_called_once()
        apply_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
