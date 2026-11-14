# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Keep source-visible MLA dispatch aligned with native overlap semantics."""

from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.modules.attention.multi_latent_attention import (
    cp_multi_latent_attention as context_parallel,
)
from hyper_parallel.codegen.model_build_overlay import use_codegen_model_build_overlay
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.components.modules import MultiLatentAttention
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.deepseek_v3.generated.modeling_Deepseek_V3_gen_npu import (
    DeepseekV3ForCausalLM as BundledDeepseekV3ForCausalLM,
)
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.mla_cp_utils import make_deepseek_model


class TestMLAContextParallelCompatibility(unittest.TestCase):
    """Exercise the shared forward without devices or collectives."""

    def test_output_projection_and_overlap_follow_runtime_plan(self):
        """Feature: Native projection boundaries remain visible.

        Description: Exercise supported overlap and TP/CP combinations.
        Expectation: Dispatch retains native hooks and the correct output projection state.
        """
        hidden = torch.ones(1, 2, 4)
        for overlap, tp_degree, ulysses_degree, gather_degree, grad_enabled in (
            (False, 1, 2, 1, True),
            (True, 1, 2, 1, True),
            (True, 1, 2, 2, True),
            (True, 2, 2, 1, True),
            (True, 1, 1, 2, True),
            (True, 1, 2, 1, False),
        ):
            with self.subTest(overlap=overlap, tp=tp_degree, ulysses=ulysses_degree,
                              gather=gather_degree, grad=grad_enabled):
                runtime = Mock(return_value=(hidden, None))
                runtime.plan = SimpleNamespace(
                    overlap=overlap, tp_degree=tp_degree,
                    ulysses_degree=ulysses_degree, all_gather_degree=gather_degree,
                )
                module = SimpleNamespace(
                    cp_runtime=runtime, cp_plan=runtime.plan, v_head_dim=4,
                    prepare_input=lambda value: value,
                    project_latent_inputs=Mock(return_value=(hidden,)),
                    project_output=Mock(return_value=hidden),
                    o_proj=Mock(return_value=hidden),
                )
                output_overlap = overlap and tp_degree == 1 and ulysses_degree > 1 and grad_enabled
                with torch.set_grad_enabled(grad_enabled), patch.object(
                    context_parallel, "MLAOutputProjectionState",
                ) as projection_state:
                    output, weights = context_parallel.CPMultiLatentAttentionModule.forward(module, hidden)
                self.assertIs(output, hidden)
                self.assertIsNone(weights)
                self.assertEqual(runtime.call_args.kwargs["restore_output"], not output_overlap)
                self.assertEqual(runtime.call_args.kwargs["overlap_input_backward"],
                                 output_overlap and gather_degree == 1)
                if output_overlap:
                    projection_state.assert_called_once_with(runtime.ulysses_mesh, 4)
                    module.o_proj.assert_called_once_with(hidden, mla_output_state=projection_state.return_value)
                    module.project_output.assert_not_called()
                else:
                    module.project_output.assert_called_once()
                    projected, residual = module.project_output.call_args.args
                    torch.testing.assert_close(projected, hidden)
                    self.assertIs(residual, hidden)
                    module.o_proj.assert_not_called()
                    projection_state.assert_not_called()

    def test_kv_cache_is_rejected_before_runtime_execution(self):
        """Feature: CP rejects unsupported KV-cache inputs.

        Description: Pass a cache before projections or runtime state are available.
        Expectation: Validation fails before accessing those resources.
        """
        with self.assertRaisesRegex(ValueError, "KV cache"):
            context_parallel.CPMultiLatentAttentionModule.forward(
                SimpleNamespace(), torch.ones(1, 2, 4), past_key_values=object(),
            )

    def test_generated_family_preserves_layer_requests_and_layout(self) -> None:
        """Feature: Generated MLA keeps the factory request and owned parallel layout.

        Description: Generate two layers with different CP schedules, then construct and plan the artifact.
        Expectation: CP/TP degrees, strategy, overlap, parameter keys and family ownership match native.
        """
        config = make_deepseek_model().config
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (1, 2, 2)),
                           tp_size=2, cp_size=2, dp_size=1, dp_shard_size=2)
        requests = tuple(ModuleReplacementRequest(
            (f"model.layers.{index}.self_attn",), MultiLatentAttention,
            kwargs={"strategy": strategy, "overlap": bool(index)},
        ) for index, strategy in enumerate(("expanded_ulysses", "latent_kv_head")))
        setup = DistributedSetup(mesh_context=mesh, module_replacement_requests=requests)
        with TemporaryDirectory() as directory, \
                patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                use_codegen_model_build_overlay(CodegenBuildSession(), directory) as observed:
            native = HyperAutoModelForCausalLM.from_config(config, distributed_setup=setup, torch_dtype=torch.float32)
            self.assertEqual(observed.snapshot.to_dict()["status"], "generated")
            modeling = next(Path(directory).rglob("modeling_*_gen_npu.py"))
            spec = util.spec_from_file_location("mla_family_codegen_test", modeling)
            generated_module = util.module_from_spec(spec)
            with patch.dict(sys.modules, {"mla_family_codegen_test": generated_module}):
                spec.loader.exec_module(generated_module)
                with torch.device("meta"):
                    generated = getattr(generated_module, "DeepseekV3ForCausalLM")(config)
        # The tracked bundle must also stop importing the removed compatibility
        # module; model preparation must select CP before the layout planner runs.
        with torch.device("meta"):
            bundled = BundledDeepseekV3ForCausalLM(config)
        self.assertEqual(set(bundled.state_dict()), set(native.state_dict()))
        bundled, _ = prepare_model_components(bundled, build_module_context(DistributedSetup(mesh_context=mesh)))
        bundled_plan = ShardingPlanner().plan(bundled, mesh.device_mesh, tp_size=2, cp_size=2)
        self.assertEqual(set(bundled_plan.parallel_modules), {"model.layers.0.self_attn", "model.layers.1.self_attn"})
        for layer in bundled.model.layers:
            self.assertEqual(get_family_build_info(layer.self_attn).resolved.variant, "cp")
        self.assertEqual(set(generated.state_dict()), set(native.state_dict()))
        for index, (expected, actual) in enumerate(zip(native.model.layers, generated.model.layers)):
            expected, actual = expected.self_attn, actual.self_attn
            self.assertEqual(get_family_build_info(actual).resolved.variant, "cp")
            self.assertEqual(actual.cp_plan, expected.cp_plan)
            self.assertEqual(actual.cp_plan.overlap, bool(index))
        planner = ShardingPlanner()
        expected = planner.plan(native, mesh.device_mesh, tp_size=2, cp_size=2)
        actual = planner.plan(generated, mesh.device_mesh, tp_size=2, cp_size=2)
        self.assertEqual(actual.parallel_modules, expected.parallel_modules)
        for name in expected.parallel_modules:
            self.assertEqual(actual.modules[name].params, expected.modules[name].params)
