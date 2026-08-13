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
"""vLLM model adapter for the native HyperParallel Qwen3.5 dense model."""

import os
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
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
from vllm.platforms import current_platform

from hyper_parallel import DTensor, DeviceMesh, distribute_tensor, mark_created_groups
from hyper_parallel.models.qwen3_5.model import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM as HyperQwen3_5ForCausalLM,
)
from hyper_parallel.models.qwen3_5.parallelize import (
    parallelize_qwen3_5_inference_tp,
    qwen3_5_inference_tp_load_transforms,
)


_GDN_ACTIVATION_COMPATIBILITY_MODE = "separate_bf16"
_GDN_GATING_COMPATIBILITY_MODE = "torch"
_GDN_QK_L2NORM_COMPATIBILITY_MODE = "torch_bf16"
_GDN_RECURRENCE_COMPATIBILITY_MODE = "torch"
_GDN_NORM_COMPATIBILITY_MODE = "torch_bf16"
_NUMERICAL_PROFILE_ENV = "HYPER_VLLM_NUMERICAL_PROFILE"
_FUNCTIONAL_PROFILE = "functional"
_PARITY_PROFILE = "parity"


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _rope_value(text_config: object, name: str, default: Any) -> Any:
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    if isinstance(rope_parameters, dict):
        return rope_parameters.get(name, _config_value(text_config, name, default))
    return _config_value(rope_parameters, name, _config_value(text_config, name, default))


def _build_hyper_config(vllm_config: VllmConfig) -> Qwen3_5Config:
    model_config = vllm_config.model_config
    text_config = model_config.hf_text_config
    hf_config = model_config.hf_config

    if getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise ValueError(
            "HyperQwen3_5ForCausalLM requires hf_text_config.model_type='qwen3_5_text'"
        )
    if _config_value(text_config, "hidden_act", "silu") != "silu":
        raise ValueError("HyperQwen3_5ForCausalLM supports only hidden_act='silu'")
    if float(_config_value(text_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("HyperQwen3_5ForCausalLM does not support attention dropout")

    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    if not isinstance(rope_parameters, dict):
        raise ValueError("HyperQwen3_5ForCausalLM requires dictionary rope_parameters")
    if rope_parameters.get("rope_type", "default") != "default":
        raise ValueError("HyperQwen3_5ForCausalLM currently supports only default RoPE")
    if not rope_parameters.get("mrope_interleaved", True):
        raise ValueError("HyperQwen3_5ForCausalLM requires interleaved MRoPE")
    supported_rope_fields = {
        "mrope_interleaved",
        "mrope_section",
        "partial_rotary_factor",
        "rope_theta",
        "rope_type",
    }
    unsupported_rope_fields = set(rope_parameters).difference(supported_rope_fields)
    if unsupported_rope_fields:
        unsupported = ", ".join(sorted(unsupported_rope_fields))
        raise ValueError(f"Unsupported Qwen3.5 rope_parameters: {unsupported}")

    layer_types = list(text_config.layer_types)
    return Qwen3_5Config(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_hidden_layers=text_config.num_hidden_layers,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        max_position_embeddings=text_config.max_position_embeddings,
        rms_norm_eps=text_config.rms_norm_eps,
        attention_bias=bool(_config_value(text_config, "attention_bias", False)),
        tie_word_embeddings=bool(_config_value(hf_config, "tie_word_embeddings", False)),
        attn_output_gate=bool(_config_value(text_config, "attn_output_gate", True)),
        rope_theta=float(_rope_value(text_config, "rope_theta", 10_000_000.0)),
        partial_rotary_factor=float(_rope_value(text_config, "partial_rotary_factor", 0.25)),
        mrope_section=list(_rope_value(text_config, "mrope_section", [11, 11, 10])),
        full_attention_interval=int(_config_value(text_config, "full_attention_interval", 4)),
        linear_num_value_heads=text_config.linear_num_value_heads,
        linear_num_key_heads=text_config.linear_num_key_heads,
        linear_value_head_dim=text_config.linear_value_head_dim,
        linear_key_head_dim=text_config.linear_key_head_dim,
        linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
        layer_types=layer_types,
        image_token_id=int(_config_value(hf_config, "image_token_id", 248056)),
        video_token_id=int(_config_value(hf_config, "video_token_id", 248057)),
        vision_start_token_id=int(_config_value(hf_config, "vision_start_token_id", 248053)),
        vision_end_token_id=int(_config_value(hf_config, "vision_end_token_id", 248054)),
    )


def _validate_adapter_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    hf_text_config = vllm_config.model_config.hf_text_config

    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports only bfloat16")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports pipeline_parallel_size=1")
    if _config_value(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports prefill_context_parallel_size=1")
    if _config_value(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3_5ForCausalLM currently supports decode_context_parallel_size=1")
    if vllm_config.quant_config is not None:
        raise ValueError("HyperQwen3_5ForCausalLM requires an unquantized checkpoint")
    mamba_cache_mode = _config_value(cache_config, "mamba_cache_mode", "none")
    if cache_config.enable_prefix_caching and mamba_cache_mode != "align":
        raise ValueError("Qwen3.5 prefix caching requires mamba_cache_mode='align'")
    if mamba_cache_mode not in ("none", "align"):
        raise ValueError("HyperQwen3_5ForCausalLM supports mamba_cache_mode='none' or 'align'")
    model_ssm_dtype = _config_value(hf_text_config, "mamba_ssm_dtype", None)
    if cache_config.mamba_ssm_cache_dtype == "auto" and model_ssm_dtype is not None:
        cache_config.mamba_ssm_cache_dtype = model_ssm_dtype
    elif model_ssm_dtype is not None and cache_config.mamba_ssm_cache_dtype != model_ssm_dtype:
        raise ValueError(
            "mamba_ssm_cache_dtype must match the Qwen3.5 checkpoint mamba_ssm_dtype"
        )
    tp_size = parallel_config.tensor_parallel_size
    divisibility_fields = {
        "vocab_size": hf_text_config.vocab_size,
        "num_attention_heads": hf_text_config.num_attention_heads,
        "num_key_value_heads": hf_text_config.num_key_value_heads,
        "intermediate_size": hf_text_config.intermediate_size,
        "linear_num_key_heads": hf_text_config.linear_num_key_heads,
        "linear_num_value_heads": hf_text_config.linear_num_value_heads,
    }
    for field_name, field_value in divisibility_fields.items():
        if field_value % tp_size != 0:
            raise ValueError(f"Qwen3.5 {field_name}={field_value} must be divisible by TP size {tp_size}")


def _enable_gdn_activation_compatibility(gdn: object) -> None:
    supported_modes = getattr(gdn, "supported_causal_conv_activation_modes", frozenset())
    supported_gating_modes = getattr(gdn, "supported_gdn_gating_modes", frozenset())
    supported_l2norm_modes = getattr(gdn, "supported_gdn_qk_l2norm_modes", frozenset())
    supported_recurrence_modes = getattr(gdn, "supported_gdn_recurrence_modes", frozenset())
    supported_norm_modes = getattr(gdn, "supported_gdn_norm_modes", frozenset())
    if (
        _GDN_ACTIVATION_COMPATIBILITY_MODE not in supported_modes
        or _GDN_GATING_COMPATIBILITY_MODE not in supported_gating_modes
        or _GDN_QK_L2NORM_COMPATIBILITY_MODE not in supported_l2norm_modes
        or _GDN_RECURRENCE_COMPATIBILITY_MODE not in supported_recurrence_modes
        or _GDN_NORM_COMPATIBILITY_MODE not in supported_norm_modes
    ):
        raise ValueError(
            "HyperQwen3_5ForCausalLM requires vLLM-Ascend GDN numerical compatibility support"
        )
    setattr(gdn, "causal_conv_activation_mode", _GDN_ACTIVATION_COMPATIBILITY_MODE)
    setattr(gdn, "gdn_gating_mode", _GDN_GATING_COMPATIBILITY_MODE)
    setattr(gdn, "gdn_qk_l2norm_mode", _GDN_QK_L2NORM_COMPATIBILITY_MODE)
    setattr(gdn, "gdn_recurrence_mode", _GDN_RECURRENCE_COMPATIBILITY_MODE)
    setattr(gdn, "gdn_norm_mode", _GDN_NORM_COMPATIBILITY_MODE)


def _configure_gdn_numerical_profile(gdn: object) -> None:
    profile = os.environ.get(_NUMERICAL_PROFILE_ENV, _FUNCTIONAL_PROFILE).strip().lower()
    if profile == _FUNCTIONAL_PROFILE:
        return
    if profile == _PARITY_PROFILE:
        _enable_gdn_activation_compatibility(gdn)
        return
    raise ValueError(
        f"{_NUMERICAL_PROFILE_ENV} must be '{_FUNCTIONAL_PROFILE}' or '{_PARITY_PROFILE}', got '{profile}'"
    )


class _VllmAttentionCore(nn.Module):
    """Adapt Hyper's local BHSD attention core to vLLM paged attention."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Run vLLM paged attention and restore Hyper's local BHSD layout."""
        # pylint: disable=W0613  # Signature matches Hyper's attention core.
        if attention_mask is not None:
            raise ValueError("vLLM manages causal masks; explicit attention_mask is unsupported")
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError(
                "HyperQwen3_5ForCausalLM expects packed attention tensors with batch size 1"
            )

        batch_size, _, seq_len, head_dim = query.shape
        query = query.transpose(1, 2).reshape(-1, self.num_heads, head_dim)
        key = key.transpose(1, 2).reshape(-1, key.shape[1], head_dim)
        value = value.transpose(1, 2).reshape(-1, value.shape[1], head_dim)
        output = self.attention(query, key, value).narrow(0, 0, batch_size * seq_len)
        return output.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)


class _VllmGatedDeltaNet(nn.Module):
    """Keep native Hyper GDN math around vLLM's request-state core."""

    _NATIVE_MODULE_NAMES = (
        "conv1d",
        "norm",
        "out_proj_input",
        "out_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    )

    def __init__(
        self,
        native_gdn: nn.Module,
        *,
        hf_config: object,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        for name in self._NATIVE_MODULE_NAMES:
            self.add_module(name, getattr(native_gdn, name))
        self.dt_bias = native_gdn.dt_bias
        self.A_log = native_gdn.A_log
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
            "kv_groups",
        ):
            setattr(self, name, getattr(native_gdn, name))

        self.state_runtime = QwenGatedDeltaNetAttention(
            config=hf_config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=False,
        )
        _configure_gdn_numerical_profile(self.state_runtime)
        for name in ("conv1d", "in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
            self.state_runtime._modules.pop(name, None)  # pylint: disable=protected-access
        self.state_runtime._parameters.pop("dt_bias", None)  # pylint: disable=protected-access
        self.state_runtime._parameters.pop("A_log", None)  # pylint: disable=protected-access
        self.bind_state_runtime_parameters()

    def bind_state_runtime_parameters(self) -> None:
        """Bind non-owning runtime aliases after native TP replaces parameters."""
        object.__setattr__(self.state_runtime, "conv1d", self.conv1d)
        object.__setattr__(self.state_runtime, "dt_bias", self.dt_bias)
        object.__setattr__(self.state_runtime, "A_log", self.A_log)

    def _apply(self, fn, recurse: bool = True):
        """Refresh runtime aliases after module device or storage transforms."""
        result = super()._apply(fn, recurse=recurse)
        self.bind_state_runtime_parameters()
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run native projections and norm around vLLM's stateful GDN core."""
        # pylint: disable=W0613  # Signature matches Hyper's GDN leaf.
        if attention_mask is not None:
            raise ValueError("vLLM manages GDN request boundaries; attention_mask is unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "HyperQwen3_5ForCausalLM expects packed GDN tensors with batch size 1"
            )

        batch_size, num_tokens, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).reshape(num_tokens, -1)
        z = self.in_proj_z(hidden_states).reshape(
            num_tokens,
            self.num_v_heads // self.state_runtime.tp_size,
            self.head_v_dim,
        )
        b = self.in_proj_b(hidden_states).reshape(num_tokens, -1).contiguous()
        a = self.in_proj_a(hidden_states).reshape(num_tokens, -1).contiguous()
        core_output = torch.zeros(
            (num_tokens, self.num_v_heads // self.state_runtime.tp_size, self.head_v_dim),
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
        core_output = core_output.reshape(batch_size, num_tokens, -1)
        return self.out_proj(self.out_proj_input(core_output))


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


def _reset_rope_inv_freq_from_cpu(rope: Any, model_dtype: torch.dtype) -> None:
    frequency_indices = torch.arange(0, rope.dim, 2, dtype=torch.float32, device="cpu")
    inv_freq = 1.0 / (rope.theta ** (frequency_indices / rope.dim))
    rope.inv_freq.copy_(inv_freq.to(model_dtype).to(rope.inv_freq.device))


def _map_weight_name(name: str) -> tuple[Optional[str], object]:
    if name.startswith("model.visual.") or name.startswith("model.mtp.") or name.startswith("mtp."):
        return None, None
    if name.endswith("rotary_emb.inv_freq"):
        return None, None
    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    return name, None


def _required_weight_shards(target_name: str) -> set[object]:
    del target_name
    return {None}


def _validate_loaded_weight_shards(
    parameter_names: set[str],
    loaded_shards: dict[str, set[object]],
) -> None:
    errors = []
    for target_name in sorted(parameter_names):
        expected = _required_weight_shards(target_name)
        actual = loaded_shards.get(target_name, set())
        if actual != expected:
            errors.append(f"{target_name}: expected shards {expected}, loaded {actual}")
    if errors:
        raise ValueError("Incomplete Qwen3.5 checkpoint: " + "; ".join(errors))


def _device_mesh_from_vllm_tp() -> DeviceMesh:
    """Build a Hyper mesh view over vLLM's existing TP process group."""
    tp_group = get_tp_group()
    process_group = tp_group.device_group
    mark_created_groups(process_group)
    mesh = DeviceMesh.from_group(
        process_group,
        device_type=current_platform.device_type,
        mesh_dim_names=("tp",),
    )
    if mesh.get_group() is not process_group:
        raise RuntimeError("Hyper TP mesh did not retain vLLM's device process group")
    if tuple(mesh.rank_list) != tuple(tp_group.ranks):
        raise RuntimeError(
            f"Hyper TP mesh ranks {mesh.rank_list} differ from vLLM TP ranks {tuple(tp_group.ranks)}"
        )
    return mesh


def _load_native_parameter(
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Copy one full checkpoint tensor into a native parameter or local shard."""
    if transform is not None:
        loaded_weight = transform(loaded_weight)
    if not isinstance(parameter, DTensor):
        default_weight_loader(parameter, loaded_weight)
        return

    local_weight = distribute_tensor(
        loaded_weight,
        parameter.device_mesh,
        parameter.placements,
        src_data_rank=None,
    ).to_local()
    local_parameter = parameter.to_local()
    if tuple(local_parameter.shape) != tuple(local_weight.shape):
        raise ValueError(
            "Qwen3.5 TP checkpoint shard shape mismatch: "
            f"parameter={tuple(local_parameter.shape)}, weight={tuple(local_weight.shape)}"
        )
    with torch.no_grad():
        local_parameter.copy_(
            local_weight.to(device=local_parameter.device, dtype=local_parameter.dtype)
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)
class HyperQwen3_5ForCausalLM(
    HyperQwen3_5ForCausalLM,
    HasInnerState,
    IsHybrid,
    SupportsMRoPE,
):
    """Run the native Hyper Qwen3.5 model inside the vLLM execution engine."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _validate_adapter_config(vllm_config)
        super().__init__(_build_hyper_config(vllm_config))
        _reset_rope_inv_freq_from_cpu(
            self.model.rotary_emb,
            self.model.embed_tokens.weight.dtype,
        )
        hf_text_config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        for layer_idx, layer in enumerate(self.model.layers):
            layer_prefix = _join_prefix(prefix, f"model.layers.{layer_idx}")
            if layer.layer_type == "full_attention":
                layer.self_attn.sdpa_core = _VllmAttentionCore(
                    num_heads=self.config.num_attention_heads // tp_size,
                    num_kv_heads=self.config.num_key_value_heads // tp_size,
                    head_dim=self.config.head_dim,
                    vllm_config=vllm_config,
                    prefix=f"{layer_prefix}.self_attn.attn",
                )
            else:
                layer.linear_attn = _VllmGatedDeltaNet(
                    layer.linear_attn,
                    hf_config=hf_text_config,
                    vllm_config=vllm_config,
                    prefix=f"{layer_prefix}.linear_attn",
                )

        self._tp_mesh = None
        self._tp_load_transforms = {}
        if tp_size > 1:
            self._tp_mesh = _device_mesh_from_vllm_tp()
            parallelize_qwen3_5_inference_tp(self, self._tp_mesh)
            self._tp_load_transforms = qwen3_5_inference_tp_load_transforms(
                self,
                self._tp_mesh,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the native Hyper token embedding."""
        return self.model.embed_tokens(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings through vLLM's legacy model interface."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run packed tokens through Hyper decoder modules and vLLM state leaves."""
        # pylint: disable=W0613  # vLLM may pass model-specific keyword arguments.
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if hidden_states.ndim != 2:
            raise ValueError("packed input embeddings must have shape [T,H]")

        num_tokens = hidden_states.shape[0]
        position_ids = _normalize_positions(positions, num_tokens)
        hidden_states = hidden_states.unsqueeze(0)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, position_ids=position_ids)
        hidden_states = self.model.norm(hidden_states)
        return hidden_states.squeeze(0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute gathered full-vocabulary logits with the native Hyper LM head."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load streamed HF weights into Hyper and vLLM leaf parameters."""
        params = dict(self.named_parameters())
        loaded_shards: dict[str, set[object]] = {}
        tied_embedding_weight = None

        for source_name, loaded_weight in weights:
            target_name, shard_id = _map_weight_name(source_name)
            if target_name is None:
                continue
            if target_name == "lm_head.weight" and target_name not in params and self.config.tie_word_embeddings:
                continue
            if target_name not in params:
                raise ValueError(
                    f"Unexpected Qwen3.5 checkpoint parameter '{source_name}' mapped to '{target_name}'"
                )

            target_shards = loaded_shards.setdefault(target_name, set())
            if shard_id in target_shards:
                raise ValueError(
                    f"Duplicate Qwen3.5 checkpoint shard '{source_name}' for '{target_name}'"
                )

            parameter = params[target_name]
            transform = self._tp_load_transforms.get(target_name)
            _load_native_parameter(parameter, loaded_weight, transform)
            target_shards.add(shard_id)
            if target_name == "model.embed_tokens.weight":
                tied_embedding_weight = loaded_weight

        if (
            self.config.tie_word_embeddings
            and "lm_head.weight" in params
            and "lm_head.weight" not in loaded_shards
            and tied_embedding_weight is not None
        ):
            transform = self._tp_load_transforms.get("lm_head.weight")
            _load_native_parameter(params["lm_head.weight"], tied_embedding_weight, transform)
            loaded_shards["lm_head.weight"] = {None}

        _validate_loaded_weight_shards(set(params), loaded_shards)
        return set(loaded_shards)

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
