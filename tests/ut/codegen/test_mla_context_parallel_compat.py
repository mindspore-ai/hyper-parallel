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
from hyper_parallel.codegen.project import project_model_build, publish_codegen_ir
from hyper_parallel.components.modules import MultiLatentAttention
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.deepseek_v3.generated.modeling_Deepseek_V3_gen_npu import (
    DeepseekV3ForCausalLM as BundledDeepseekV3ForCausalLM,
)
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.build_capture import capture_model_build_plan
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.mla_cp_utils import make_deepseek_model, make_mla


class TestMLAContextParallelCompatibility(unittest.TestCase):
    """Exercise the shared forward without devices or collectives."""

    def test_output_projection_and_overlap_follow_runtime_plan(self):
        """Feature: Native projection boundaries remain visible.

        Description: Exercise supported overlap and TP/CP combinations through class stages.
        Expectation: Shared forward retains native hooks and the correct output projection state.
        """
        hidden = torch.ones(1, 2, 4)
        implementation = context_parallel.CPMultiLatentAttentionModule
        self.assertIs(implementation.forward, context_parallel.MultiLatentAttentionModule.forward)
        for overlap, tp_degree, ulysses_degree, gather_degree, grad_enabled in (
            (False, 1, 2, 1, True), (True, 1, 2, 1, True), (True, 1, 2, 2, True),
            (True, 2, 2, 1, True), (True, 1, 1, 2, True), (True, 1, 2, 1, False),
        ):
            with self.subTest(overlap=overlap, tp=tp_degree, ulysses=ulysses_degree,
                              gather=gather_degree, grad=grad_enabled):
                module = implementation(module=make_mla(), context={"cp_size": 2})
                module.cp_plan = SimpleNamespace(
                    overlap=overlap, tp_degree=tp_degree,
                    ulysses_degree=ulysses_degree, all_gather_degree=gather_degree,
                    local_heads=ulysses_degree, dimensions=SimpleNamespace(value_dim=4))
                module.cp_context = object()
                module.ulysses_mesh, module.v_head_dim = object(), 4
                restored = torch.ones(1, 2, ulysses_degree, 4)
                attention = torch.ones(1, 2 * ulysses_degree, 1, 4)
                module.ulysses_context = SimpleNamespace(head_to_sequence=Mock(return_value=restored))
                module.rotary_fn, module.rotary_interleaved = None, False
                module.apply_position = lambda value, *args, **kwargs: value
                module._attention_layout = Mock(return_value=(None, None, None, {}))
                module._prepare_distributed_inputs = Mock(return_value=(hidden, hidden, hidden))
                module.o_proj.forward = Mock(return_value=hidden)
                output_overlap = overlap and tp_degree == 1 and ulysses_degree > 1 and grad_enabled
                with torch.set_grad_enabled(grad_enabled), patch.object(
                    context_parallel.MultiLatentAttentionModule, "project_output", return_value=hidden,
                ) as project_output:
                    latents = context_parallel.MLALatentInputs(hidden, hidden, hidden)
                    inputs = module.exchange_attention(latents)
                    output, state = module.restore_attention(attention, hidden, inputs.output_state)
                    output = module.project_output(output, hidden, state)
                self.assertIs(output, hidden)
                self.assertEqual(module._prepare_distributed_inputs.call_args.args[4],
                                 output_overlap and gather_degree == 1)
                if output_overlap:
                    self.assertIsInstance(state, context_parallel.MLAOutputProjectionState)
                    self.assertIs(state.mesh, module.ulysses_mesh)
                    self.assertEqual(state.head_dim, 4)
                    module.o_proj.forward.assert_called_once_with(attention, mla_output_state=state)
                    module.ulysses_context.head_to_sequence.assert_not_called()
                    project_output.assert_not_called()
                else:
                    self.assertIsNone(state)
                    if ulysses_degree > 1:
                        module.ulysses_context.head_to_sequence.assert_called_once_with(attention)
                    else:
                        module.ulysses_context.head_to_sequence.assert_not_called()
                    project_output.assert_called_once()
                    projected, residual, projection_state = project_output.call_args.args
                    torch.testing.assert_close(projected, restored)
                    self.assertIs(residual, hidden)
                    self.assertIsNone(projection_state)
                    module.o_proj.forward.assert_not_called()

    def test_kv_cache_is_rejected_before_runtime_execution(self):
        """Feature: CP rejects unsupported KV-cache inputs.

        Description: Pass a cache to the shared forward before resources are bound.
        Expectation: Validation fails before any projection or communication.
        """
        module = context_parallel.CPMultiLatentAttentionModule(module=make_mla(), context={"cp_size": 2})
        with patch.object(module, "project_latent_inputs") as project:
            with self.assertRaisesRegex(ValueError, "KV cache"):
                module(torch.ones(1, 2, 12), past_key_values=object())
            project.assert_not_called()

    def test_generated_family_preserves_layer_requests_and_layout(self) -> None:
        """Feature: Generated MLA directly constructs the selected concrete members.

        Description: Generate and directly construct two layers with different CP schedules.
        Expectation: CP/TP degrees, strategy, overlap and parameter keys match native without installing patches.
        """
        config = make_deepseek_model().config
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (1, 2, 2)),
                           tp_size=2, cp_size=2, dp_size=1, dp_shard_size=2)
        requests = tuple(ModuleReplacementRequest(
            (f"model.layers.{index}.self_attn",), MultiLatentAttention,
            kwargs={"strategy": strategy, "overlap": bool(index)},
        ) for index, strategy in enumerate(("expanded_ulysses", "latent_kv_head")))
        setup = DistributedSetup(mesh_context=mesh, module_replacement_requests=requests)
        with TemporaryDirectory() as directory:
            with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                    capture_model_build_plan() as capture:
                native = HyperAutoModelForCausalLM.from_config(
                    config, distributed_setup=setup, torch_dtype=torch.float32,
                )
            observed = publish_codegen_ir(project_model_build(capture), directory)
            self.assertEqual(observed.snapshot.to_dict()["status"], "generated")
            modeling = next(Path(directory).rglob("modeling_*_gen_npu.py"))
            spec = util.spec_from_file_location("mla_family_codegen_test", modeling)
            generated_module = util.module_from_spec(spec)
            with patch.dict(sys.modules, {"mla_family_codegen_test": generated_module}):
                spec.loader.exec_module(generated_module)
                with torch.device("meta"):
                    generated = getattr(generated_module, "DeepseekV3ForCausalLM")(config)
        self.assertEqual(set(generated.state_dict()), set(native.state_dict()))
        for index, (expected, actual) in enumerate(zip(native.model.layers, generated.model.layers)):
            expected, actual = expected.self_attn, actual.self_attn
            self.assertIsInstance(actual, context_parallel.CPMultiLatentAttentionModule)
            self.assertEqual(actual.cp_plan, expected.cp_plan)
            self.assertEqual(actual.cp_plan.overlap, bool(index))

    def test_tracked_deepseek_bundle_uses_recipe_topology(self) -> None:
        """Direct construction materializes the tracked CP4 member and its native plan."""
        config = make_deepseek_model().config
        config.num_hidden_layers = 4
        config.first_k_dense_replace = 4
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (1, 4, 1)),
                           tp_size=1, cp_size=4, dp_size=1, dp_shard_size=4)
        request = ModuleReplacementRequest(
            ("model.layers.*.self_attn",), MultiLatentAttention,
            kwargs={"strategy": "expanded_ulysses", "overlap": False, "config": None,
                    "ulysses_degree": None, "kv_exchange": "all_gather"},
            declaration_id="plan_override:0:replace",
        )
        setup = DistributedSetup(mesh_context=mesh, module_replacement_requests=(request,))
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                capture_model_build_plan():
            native = HyperAutoModelForCausalLM.from_config(
                config, distributed_setup=setup, torch_dtype=torch.float32,
            )
        with torch.device("meta"):
            bundled = BundledDeepseekV3ForCausalLM(config)
        self.assertEqual(set(bundled.state_dict()), set(native.state_dict()))
        for expected, actual in zip(native.model.layers, bundled.model.layers):
            self.assertIsInstance(actual.self_attn, context_parallel.CPMultiLatentAttentionModule)
            self.assertEqual(actual.self_attn.cp_plan, expected.self_attn.cp_plan)
            self.assertEqual(actual.self_attn.cp_plan.tp_degree, 1)
            self.assertEqual(actual.self_attn.cp_plan.ulysses_degree, 4)
        for name, parameter in bundled.named_parameters():
            expected = native.get_parameter(name)
            self.assertEqual(parameter.shape, expected.shape)
            self.assertEqual(parameter.requires_grad, expected.requires_grad)
