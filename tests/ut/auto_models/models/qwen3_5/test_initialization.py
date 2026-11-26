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
"""Native Qwen3.5 initialization survives component replacement and materialization."""

import tempfile
import unittest
from itertools import product
from copy import deepcopy
from unittest.mock import patch

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet, Qwen3_5PreTrainedModel, Qwen3_5RMSNorm, Qwen3_5TextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel.components.modules import GatedDeltaNet, GatedGroupedQueryAttention, OffsetRMSNorm
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.module_build import attach_build_context, resolve_effective_model_spec
from hyper_parallel.models.qwen3_5.adapter.initialization import register_native_initialization
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.mark_utils import arg_mark


_MODEL_TYPES = ((Qwen3_5ForCausalLM, Qwen3_5TextConfig), (Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig))
_REQUESTS = (
    ModuleReplacementRequest(match=("model.layers.*.linear_attn",), target=GatedDeltaNet,
                             kwargs={"ulysses_degree": None}),
    ModuleReplacementRequest(match=("model.layers.*.self_attn",), target=GatedGroupedQueryAttention),
    ModuleReplacementRequest(match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm",
                                    "model.norm"), target=OffsetRMSNorm),
)
_MOVE_MODEL = model_builder._move_model_to_device


def _config(config_type):
    config = config_type.from_dict({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 128, "initializer_range": 0.017, "use_cache": False,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
        "num_experts": 4, "num_experts_per_tok": 2,
    })
    config._attn_implementation = "eager"
    return config


def _move_and_poison(model, is_meta_device, device):
    """NaN storage makes missing initialization deterministic instead of allocator-dependent."""
    model = _MOVE_MODEL(model, is_meta_device, device)
    if is_meta_device:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
    return model


def _build(model, *, meta=True, cp_size=1, requests=_REQUESTS, **kwargs):
    attach_build_context(model, {"module_replacement_requests": requests,
                                 "config": model.config, "cp_size": cp_size,
                                 "model_spec": resolve_effective_model_spec(model.config)})
    with patch.object(model_builder, "_move_model_to_device", _move_and_poison):
        return model_builder.apply_model_infrastructure(
            model, is_meta_device=meta, is_hf_model=True, device=torch.device("cpu"), **kwargs,
        )


class TestQwen35Initialization(unittest.TestCase):
    """Use real dense/MoE HF models and independent native initializer oracles."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_source_special_parameters_are_initialized(self):
        """Feature: Native initialization through module/config and local/CP family construction.
        Description: Materialize dense/MoE models with poisoned parameter storage.
        Expectation: All parameters are finite; norms, GDN values and buffers match native references.
        """
        for (model_type, config_type), cp_size, direct in product(_MODEL_TYPES, (1, 2), (False, True)):
            with self.subTest(model=model_type.__name__, cp_size=cp_size, direct=direct):
                config = _config(config_type)
                reference = model_type(config)
                with torch.device("meta"):
                    candidate = model_type(config)
                    if direct:
                        candidate.model.layers[0].linear_attn = GatedDeltaNet(config=config, ulysses_degree=None)
                candidate = _build(candidate, cp_size=cp_size, requests=_REQUESTS[1:] if direct else _REQUESTS)
                for name, parameter in candidate.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all(), name)
                for name, buffer in candidate.named_buffers():
                    torch.testing.assert_close(buffer, reference.get_buffer(name), atol=0, rtol=0, msg=name)
                for name, module in candidate.named_modules():
                    if "RMSNorm" in type(module).__name__:
                        torch.testing.assert_close(module.weight, reference.get_submodule(name).weight,
                                                   atol=0, rtol=0, msg=name)
                actual = candidate.model.layers[0].linear_attn
                self.assertEqual(type(actual).__name__.startswith("CP"), cp_size > 1)
                expected = reference.model.layers[0].linear_attn
                # Compare the native distribution exactly without depending on traversal RNG consumption.
                torch.manual_seed(35)
                reference.model._init_weights(expected)
                torch.manual_seed(35)
                candidate.model._init_weights(actual)
                torch.testing.assert_close(actual.A_log, expected.A_log, atol=0, rtol=0)
                torch.testing.assert_close(actual.dt_bias, expected.dt_bias, atol=0, rtol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_loaded_module_conversion_preserves_special_weights(self):
        """Feature: Adoption of initialized source parameters.
        Description: Replace modules in real models with known GDN and norm values.
        Expectation: Parameter identity and values are preserved without random initialization.
        """
        for model_type, config_type in _MODEL_TYPES:
            with self.subTest(model=model_type.__name__):
                source = model_type(_config(config_type))
                with torch.no_grad():
                    source.model.norm.weight.fill_(0.37)
                    source.model.layers[0].linear_attn.A_log.fill_(0.52)
                    source.model.layers[0].linear_attn.dt_bias.fill_(0.73)
                retained = {name: parameter for name, parameter in source.named_parameters()
                            if name.endswith(("A_log", "dt_bias")) or "layernorm" in name
                            or name == "model.norm.weight"}
                snapshots = {name: value.detach().clone() for name, value in retained.items()}
                with patch.object(model_builder, "_initialize_model_weights",
                                  side_effect=AssertionError("reinitialized")):
                    candidate = _build(source, meta=False)
                for name, parameter in retained.items():
                    self.assertIs(candidate.get_parameter(name), parameter)
                    torch.testing.assert_close(parameter, snapshots[name], atol=0, rtol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_checkpoint_loading_preserves_all_weights(self):
        """Feature: Checkpoint loading after family replacement.
        Description: Load a real HF safetensors checkpoint into a meta model with layout conversion.
        Expectation: Every loaded value matches the converted source without random initialization.
        """
        source = Qwen3_5ForCausalLM(_config(Qwen3_5TextConfig))
        with torch.no_grad():
            source.model.norm.weight.fill_(0.37)
            source.model.layers[0].linear_attn.A_log.fill_(0.52)
            source.model.layers[0].linear_attn.dt_bias.fill_(0.73)
        with tempfile.TemporaryDirectory() as path:
            source.save_pretrained(path)
            expected = _build(deepcopy(source), meta=False)
            # Source GQA adoption declares its fused layout; checkpoint loading applies the conversion.
            for name, module in expected.named_modules():
                if not callable(getattr(module, "make_transforms", None)):
                    continue
                source_state = source.get_submodule(name).state_dict()
                converted = dict(source_state)
                for transform in module.make_transforms():
                    converted.update(transform.operations[0].convert(
                        source_state, transform.source_patterns, transform.target_patterns))
                module.load_state_dict({key: converted[key] for key in module.state_dict()})
            with torch.device("meta"):
                candidate = Qwen3_5ForCausalLM(source.config)
            with patch.object(model_builder, "_initialize_model_weights",
                                  side_effect=AssertionError("reinitialized")):
                candidate = _build(candidate, load_base_model=True, pretrained_path=path)
        self.assertEqual(set(candidate.state_dict()), set(expected.state_dict()))
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], atol=0, rtol=0, msg=name)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_policy_and_copied_owner_are_retained(self):
        """Feature: Model-owned initialization policy and copied initializer binding.
        Description: Override native initialization and deepcopy an adapted meta model.
        Expectation: The copied owner's policy determines the special parameter values.
        """
        native_init = Qwen3_5PreTrainedModel._init_weights

        def custom_init(owner: Qwen3_5PreTrainedModel, module: torch.nn.Module) -> None:
            """Apply owner-specific values after the native initialization policy."""
            native_init(owner, module)
            with torch.no_grad():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    module.dt_bias.fill_(owner.config.special_init_value)
                elif isinstance(module, Qwen3_5RMSNorm):
                    module.weight.fill_(-owner.config.special_init_value)

        config = _config(Qwen3_5TextConfig)
        config.special_init_value = 2.75
        with patch.object(Qwen3_5PreTrainedModel, "_init_weights", custom_init):
            with torch.device("meta"):
                source = Qwen3_5ForCausalLM(config)
            register_native_initialization(source)
            candidate = deepcopy(source)
            candidate.config.special_init_value = 3.25
            candidate = _build(candidate)
        self.assertIs(candidate.model._init_weights.__self__, candidate.model)
        self.assertTrue(source.model.norm.weight.is_meta)
        torch.testing.assert_close(candidate.model.norm.weight, torch.full((32,), -3.25), atol=0, rtol=0)
        torch.testing.assert_close(candidate.model.layers[0].linear_attn.dt_bias,
                                   torch.full((4,), 3.25), atol=0, rtol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_parameter_guards_are_retained(self):
        """Feature: HF guards for externally initialized parameters.
        Description: Invoke adapted initialization on parameters marked as already initialized.
        Expectation: Guarded GDN and norm values remain unchanged.
        """
        with torch.device("meta"):
            candidate = Qwen3_5ForCausalLM(_config(Qwen3_5TextConfig))
        candidate = _build(candidate)
        attention = candidate.model.layers[0].linear_attn
        guarded = (attention.A_log, attention.dt_bias, candidate.model.norm.weight)
        with torch.no_grad():
            for parameter in guarded:
                parameter.fill_(0.43)
                parameter._is_hf_initialized = True
        candidate.model._init_weights(attention)
        candidate.model._init_weights(candidate.model.norm)
        for parameter in guarded:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.43), atol=0, rtol=0)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_parallel_wrapper_preserves_native_parent_initialization(self):
        """Feature: Native initialization identity across parallel wrapping.
        Description: Apply a dynamic wrapper class before materializing a Qwen MoE model.
        Expectation: The model retains native classification and initializes parent-owned router weights.
        """
        with torch.device("meta"):
            candidate = Qwen3_5MoeForCausalLM(_config(Qwen3_5MoeTextConfig))
        self.assertFalse(candidate.is_custom_code())

        def wrap(model: torch.nn.Module, *_args: object) -> torch.nn.Module:
            """Emulate the dynamic class introduced by parallel wrapping."""
            model.__class__ = type("ParallelQwen", (type(model),), {"__module__": __name__})
            return model

        with patch.object(model_builder, "_apply_fsdp2_with_external", wrap):
            candidate = _build(candidate)
        self.assertFalse(candidate.is_custom_code())
        for name, parameter in candidate.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all(), name)
        router = candidate.model.layers[0].mlp.gate.weight
        self.assertGreater(float(router.std()), 0)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_custom_model_identity_is_not_forced_to_native(self):
        """Feature: Preservation of genuine custom-model initialization identity.
        Description: Bind the source classification of a user-defined Qwen subclass.
        Expectation: The model retains its custom-code guard and source class binding.
        """
        custom_type = type("CustomQwen", (Qwen3_5ForCausalLM,), {"__module__": __name__})
        with torch.device("meta"):
            candidate = custom_type(_config(Qwen3_5TextConfig))
        model_builder._preserve_initialization_identity(candidate)
        self.assertTrue(candidate.is_custom_code())
        self.assertIs(candidate.is_custom_code.__self__, custom_type)


if __name__ == "__main__":
    unittest.main()
