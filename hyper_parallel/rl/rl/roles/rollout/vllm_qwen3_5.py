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
"""Transformers Qwen3.5 adapter for the Hyper-vLLM runtime."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM as TransformersQwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    apply_rotary_pos_emb,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMRoPE,
)


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _validate_adapter_config(vllm_config: VllmConfig) -> None:
    model_config = vllm_config.model_config
    text_config = model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    if getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise ValueError("HyperQwen3_5ForCausalLM requires model_type='qwen3_5_text'")
    if model_config.dtype != torch.bfloat16:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports only bfloat16")
    if parallel_config.tensor_parallel_size != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports tensor_parallel_size=1")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports pipeline_parallel_size=1")
    if _config_value(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports prefill_context_parallel_size=1")
    if _config_value(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports decode_context_parallel_size=1")
    if vllm_config.quant_config is not None:
        raise ValueError("HyperQwen3_5ForCausalLM requires an unquantized checkpoint")
    if _config_value(text_config, "hidden_act", "silu") != "silu":
        raise ValueError("HyperQwen3_5ForCausalLM supports only hidden_act='silu'")
    if float(_config_value(text_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("HyperQwen3_5ForCausalLM does not support attention dropout")
    if any(layer_type not in ("linear_attention", "full_attention") for layer_type in text_config.layer_types):
        raise ValueError("HyperQwen3_5ForCausalLM requires linear_attention or full_attention layers")

    mamba_cache_mode = _config_value(cache_config, "mamba_cache_mode", "none")
    if cache_config.enable_prefix_caching and mamba_cache_mode != "align":
        raise ValueError("Qwen3.5 prefix caching requires mamba_cache_mode='align'")
    if mamba_cache_mode not in ("none", "align"):
        raise ValueError("HyperQwen3_5ForCausalLM supports mamba_cache_mode='none' or 'align'")
    model_ssm_dtype = _config_value(text_config, "mamba_ssm_dtype", None)
    if cache_config.mamba_ssm_cache_dtype == "auto" and model_ssm_dtype is not None:
        cache_config.mamba_ssm_cache_dtype = model_ssm_dtype
    elif model_ssm_dtype is not None and cache_config.mamba_ssm_cache_dtype != model_ssm_dtype:
        raise ValueError(
            "mamba_ssm_cache_dtype must match the Qwen3.5 checkpoint mamba_ssm_dtype"
        )


class _VLLMQwen3_5Attention(nn.Module):
    """Keep Transformers Qwen3.5 projections and RoPE around paged attention."""

    def __init__(
        self,
        attention: Qwen3_5Attention,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.head_dim = attention.head_dim
        self.num_heads = attention.config.num_attention_heads
        self.num_key_value_heads = attention.config.num_key_value_heads
        self.scaling = attention.scaling
        self.layer_idx = attention.layer_idx
        self.attention = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_key_value_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Run HF attention math with vLLM-owned causal masking and KV state."""
        del kwargs
        if attention_mask is not None:
            raise ValueError("vLLM manages causal masks; explicit attention_mask is unsupported")
        if past_key_values is not None:
            raise ValueError("vLLM owns KV cache state; Transformers past_key_values is unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("HyperQwen3_5ForCausalLM expects packed hidden states with shape [1,T,H]")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)

        num_tokens = hidden_states.shape[1]
        query = query.transpose(1, 2).reshape(num_tokens, -1)
        key = key.transpose(1, 2).reshape(num_tokens, -1)
        value = value.transpose(1, 2).reshape(num_tokens, -1)
        output = self.attention(query, key, value).view(*input_shape, -1)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output), None


class _VLLMQwen3_5GatedDeltaNet(nn.Module):
    """Keep Transformers GDN parameters around vLLM request-local state."""

    _TRANSFORMERS_MODULE_NAMES = (
        "conv1d",
        "norm",
        "out_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    )

    def __init__(
        self,
        gated_delta_net: Qwen3_5GatedDeltaNet,
        *,
        hf_config: object,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        gated_delta_net.A_log = nn.Parameter(
            gated_delta_net.A_log.float(),
            requires_grad=gated_delta_net.A_log.requires_grad,
        )
        gated_delta_net.norm.float()
        for name in self._TRANSFORMERS_MODULE_NAMES:
            self.add_module(name, getattr(gated_delta_net, name))
        self.dt_bias = gated_delta_net.dt_bias
        self.A_log = gated_delta_net.A_log
        for name in (
            "hidden_size",
            "num_v_heads",
            "num_k_heads",
            "head_k_dim",
            "head_v_dim",
            "key_dim",
            "value_dim",
            "conv_dim",
            "conv_kernel_size",
        ):
            setattr(self, name, getattr(gated_delta_net, name))

        self.state_runtime = QwenGatedDeltaNetAttention(
            config=hf_config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=False,
        )
        for name in ("conv1d", "in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
            self.state_runtime._modules.pop(name, None)  # pylint: disable=protected-access
        self.state_runtime._parameters.pop("dt_bias", None)  # pylint: disable=protected-access
        self.state_runtime._parameters.pop("A_log", None)  # pylint: disable=protected-access
        self._bind_state_runtime_parameters()

    def _bind_state_runtime_parameters(self) -> None:
        object.__setattr__(self.state_runtime, "conv1d", self.conv1d)
        object.__setattr__(self.state_runtime, "dt_bias", self.dt_bias)
        object.__setattr__(self.state_runtime, "A_log", self.A_log)

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        result = super()._apply(fn, recurse=recurse)
        self._bind_state_runtime_parameters()
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[object] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run HF projections and gated norm around vLLM's GDN state core."""
        del kwargs
        if cache_params is not None:
            raise ValueError("vLLM owns GDN cache state; Transformers cache_params is unsupported")
        if attention_mask is not None:
            raise ValueError("vLLM manages GDN request boundaries; attention_mask is unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("HyperQwen3_5ForCausalLM expects packed hidden states with shape [1,T,H]")

        batch_size, num_tokens, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).reshape(num_tokens, -1)
        z = self.in_proj_z(hidden_states).reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states).reshape(num_tokens, self.num_v_heads).contiguous()
        a = self.in_proj_a(hidden_states).reshape(num_tokens, self.num_v_heads).contiguous()
        core_output = torch.zeros(
            (num_tokens, self.num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_output,
            False,
            self.state_runtime.prefix,
        )
        core_output = self.norm(
            core_output.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        return self.out_proj(core_output.reshape(batch_size, num_tokens, -1))


def _normalize_positions(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if positions.ndim == 1:
        if positions.shape[0] != num_tokens:
            raise ValueError("positions length must match the packed token count")
        return positions.view(1, -1)
    if positions.ndim == 2:
        if positions.shape[-1] != num_tokens or positions.shape[0] not in (1, 3):
            raise ValueError("positions must have shape [T], [1,T], or [3,T]")
        return positions.unsqueeze(1) if positions.shape[0] == 3 else positions
    if positions.ndim == 3 and positions.shape == (3, 1, num_tokens):
        return positions
    raise ValueError("positions must have shape [T], [1,T], [3,T], or [3,1,T]")


def _map_weight_name(name: str) -> Optional[str]:
    if name.startswith("model.visual.") or name.startswith("model.mtp.") or name.startswith("mtp."):
        return None
    if name.endswith("rotary_emb.inv_freq"):
        return None
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    return name


def _load_parameter(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    default_weight_loader(parameter, loaded_weight)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)
class HyperQwen3_5ForCausalLM(
    TransformersQwen3_5ForCausalLM,
    HasInnerState,
    IsHybrid,
    SupportsMRoPE,
):
    """Run the Transformers Qwen3.5 model with Hyper-vLLM state leaves."""

    is_text_generation_model = True
    supports_pp = False
    supports_multimodal = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _validate_adapter_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        super().__init__(text_config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer_prefix = _join_prefix(prefix, f"model.layers.{layer_idx}")
            if text_config.layer_types[layer_idx] == "full_attention":
                layer.self_attn = _VLLMQwen3_5Attention(
                    layer.self_attn,
                    vllm_config=vllm_config,
                    prefix=f"{layer_prefix}.self_attn",
                )
            else:
                layer.linear_attn = _VLLMQwen3_5GatedDeltaNet(
                    layer.linear_attn,
                    hf_config=text_config,
                    vllm_config=vllm_config,
                    prefix=f"{layer_prefix}.linear_attn",
                )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the Transformers Qwen3.5 token embedding."""
        return self.model.embed_tokens(input_ids)

    def get_input_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> nn.Module | torch.Tensor:
        """Preserve the HF accessor and support vLLM's legacy embedding call."""
        if input_ids is None:
            return self.model.embed_tokens
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run packed tokens through Transformers layers and vLLM state leaves."""
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if hidden_states.ndim != 2:
            raise ValueError("packed input embeddings must have shape [T,H]")

        position_ids = _normalize_positions(positions, hidden_states.shape[0])
        hidden_states = hidden_states.unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
        return self.model.norm(hidden_states).squeeze(0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute full-vocabulary logits with the Transformers LM head."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load streamed composite-checkpoint or online Actor weights."""
        parameters = dict(self.named_parameters())
        loaded_parameters = set()
        tied_embedding_weight = None

        for source_name, loaded_weight in weights:
            target_name = _map_weight_name(source_name)
            if target_name is None:
                continue
            if (
                target_name == "lm_head.weight"
                and target_name not in parameters
                and self.config.tie_word_embeddings
            ):
                continue
            if target_name not in parameters:
                raise ValueError(
                    f"Unexpected Qwen3.5 checkpoint parameter '{source_name}' mapped to '{target_name}'"
                )
            if target_name in loaded_parameters:
                raise ValueError(f"Duplicate Qwen3.5 checkpoint parameter '{source_name}'")
            _load_parameter(parameters[target_name], loaded_weight)
            loaded_parameters.add(target_name)
            if target_name == "model.embed_tokens.weight":
                tied_embedding_weight = loaded_weight

        if (
            self.config.tie_word_embeddings
            and "lm_head.weight" in parameters
            and "lm_head.weight" not in loaded_parameters
            and tied_embedding_weight is not None
        ):
            _load_parameter(parameters["lm_head.weight"], tied_embedding_weight)
            loaded_parameters.add("lm_head.weight")

        missing_parameters = set(parameters).difference(loaded_parameters)
        if missing_parameters:
            raise ValueError(
                "Incomplete Qwen3.5 checkpoint; missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )
        return loaded_parameters

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Return convolution and recurrent-state dtypes for Qwen3.5 GDN."""
        ssm_cache_dtype = vllm_config.cache_config.mamba_ssm_cache_dtype
        if ssm_cache_dtype == "auto":
            ssm_cache_dtype = _config_value(
                vllm_config.model_config.hf_text_config,
                "mamba_ssm_dtype",
                "auto",
            )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return request-local convolution and recurrent-state shapes."""
        hf_config = vllm_config.model_config.hf_text_config
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return vLLM cache copy callbacks for Qwen3.5 GDN states."""
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        """Return text-only MRoPE positions and reject multimodal inputs."""
        if mm_features:
            raise ValueError("HyperQwen3_5ForCausalLM currently supports text-only input")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


__all__ = ["HyperQwen3_5ForCausalLM"]
