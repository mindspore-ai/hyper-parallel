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
"""Local HF configurations for the Kimi K3 training architecture.

Field names describe the released MoonshotAI/Kimi-K3 checkpoint schema. Training
initialization and Quantile Balancing options are explicit additions; they are
not a claim to reproduce Moonshot's unpublished complete pretraining recipe.
"""
from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    """K3 text configuration, retaining the checkpoint's historical class name."""

    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs: Any) -> None:
        """Accept the released text schema plus explicit training options."""
        full_layers = list(range(4, 93, 4)) + [93]
        defaults = {
            "vocab_size": 163840, "hidden_size": 7168, "intermediate_size": 33792,
            "num_hidden_layers": 93, "num_attention_heads": 96, "num_key_value_heads": 96,
            "hidden_act": "situ", "initializer_range": 0.02, "rms_norm_eps": 1e-5,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64, "v_head_dim": 128, "mla_use_nope": True,
            "mla_use_output_gate": True, "attention_dropout": 0.0,
            "num_experts": 896, "num_experts_per_token": 16, "num_shared_experts": 2,
            "moe_intermediate_size": 3072, "routed_expert_hidden_size": 3584,
            "routed_scaling_factor": 1.0, "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid", "first_k_dense_replace": 1,
            "moe_layer_freq": 1, "use_grouped_topk": False, "num_expert_group": 1,
            "topk_group": 1, "topk_method": "noaux_tc", "latent_moe_use_norm": True,
            "moe_backend": "eager", "mla_backend": "eager", "situ_backend": "eager",
            "kda_backend": "eager", "causal_conv_backend": "eager",
            "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 12, "max_position_embeddings": 1048576,
            "num_nextn_predict_layers": 0, "use_cache": False,
            "quantile_token_chunk_size": 0, "quantile_balance": True,
            "quantile_num_bins": 1000, "quantile_ema_decay": 0.0,
            "kda_a_log_init": 0.0, "kda_dt_bias_init": 0.0,
            "attn_res_zero_init": True, "recompute_scope": "layer",
            "attn_res_backend": "eager", "attn_res_history_layout": "stacked",
            "linear_attn_config": {
                "full_attn_layers": full_layers,
                "kda_layers": [idx for idx in range(1, 94) if idx not in full_layers],
                "num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0, "use_full_rank_gate": True,
            },
        }
        settings = {key: kwargs.pop(key, value) for key, value in defaults.items()}
        kwargs.pop("model_type", None)
        pad_token_id = kwargs.pop("pad_token_id", 163839)
        bos_token_id = kwargs.pop("bos_token_id", 163584)
        eos_token_id = kwargs.pop("eos_token_id", 163586)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
        for name, value in settings.items():
            setattr(self, name, value)
        self.validate_training_architecture()

    @property
    def is_mla(self) -> bool:
        """Return whether MLA projections are configured."""
        return self.kv_lora_rank is not None

    @property
    def is_moe(self) -> bool:
        """Return whether routed experts are configured."""
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        """Return whether the layer schedule includes KDA."""
        return bool(self.linear_attn_config and self.linear_attn_config["kda_layers"])

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Resolve a zero-based decoder index against the one-based schedule.

        Args:
            layer_idx: Zero-based decoder layer index.

        Returns:
            True when layer_idx + 1 occurs in linear_attn_config's kda_layers.
        """
        return bool(self.linear_attn_config and layer_idx + 1 in self.linear_attn_config["kda_layers"])

    def validate_training_architecture(self) -> None:
        """Reject incomplete schedules and unsupported architectural variants."""
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_act != "situ" or not self.mla_use_nope:
            raise ValueError("K3 training requires SiTU and NoPE MLA")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("K3 MLA requires equal attention and key/value head counts")
        if self.num_nextn_predict_layers:
            raise ValueError("K3 next-token predictor layers are not implemented")
        self._validate_execution_options()
        self._validate_ffn_configuration()
        self._validate_attention_schedule()

    def _validate_execution_options(self) -> None:
        """Check supported execution backends and activation-memory choices."""
        if self.recompute_scope not in ("layer", "ffn", "attention", "attn_res"):
            raise ValueError("recompute_scope must be layer, ffn, attention, or attn_res")
        if self.attn_res_history_layout not in ("stacked", "blocks"):
            raise ValueError("attn_res_history_layout must be stacked or blocks")
        if self.situ_backend not in ("eager", "ascendc"):
            raise ValueError("situ_backend must be eager or ascendc")
        for name in ("kda_backend", "causal_conv_backend"):
            if getattr(self, name) not in ("eager", "triton"):
                raise ValueError(f"{name} must be eager or triton")
        if self.moe_backend not in ("eager", "grouped"):
            raise ValueError("moe_backend must be eager or grouped")
        if not isinstance(self.quantile_token_chunk_size, int) or self.quantile_token_chunk_size < 0:
            raise ValueError("quantile_token_chunk_size must be a nonnegative integer")
        if self.mla_backend not in ("eager", "npu_fusion"):
            raise ValueError("mla_backend must be eager or npu_fusion")
        if self.attn_res_backend not in ("eager", "ascendc"):
            raise ValueError("attn_res_backend must be eager or ascendc")

    def _validate_ffn_configuration(self) -> None:
        """Check SiTU, block residual and learned expert-routing dimensions."""
        if self.activation_situ_beta <= 0 or self.activation_situ_linear_beta <= 0:
            raise ValueError("SiTU clipping scales must be positive")
        if self.attn_res_block_size is not None and self.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive or None")
        if self.moe_layer_freq <= 0:
            raise ValueError("moe_layer_freq must be positive")
        if self.is_moe:
            if not 0 < self.num_experts_per_token <= self.num_experts:
                raise ValueError("num_experts_per_token must be in [1, num_experts]")
            if self.quantile_balance and self.num_experts_per_token == self.num_experts:
                raise ValueError("Quantile Balancing requires top_k < num_experts")
            if self.moe_router_activation_func != "sigmoid" or self.num_expert_group != 1:
                raise ValueError("K3 training supports sigmoid routing without expert groups")
            if not self.moe_intermediate_size or self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive")

    def _validate_attention_schedule(self) -> None:
        """Require complete KDA/MLA layer ownership and valid KDA dimensions."""
        if self.linear_attn_config is not None:
            kda = self.linear_attn_config.get("kda_layers", [])
            mla = self.linear_attn_config.get("full_attn_layers", [])
            schedule = kda + mla
            if sorted(schedule) != list(range(1, self.num_hidden_layers + 1)):
                raise ValueError("KDA and MLA schedules must partition all decoder layers using one-based indices")
            for name in ("num_heads", "head_dim", "short_conv_kernel_size"):
                if self.linear_attn_config.get(name, 0) <= 0:
                    raise ValueError(f"linear_attn_config.{name} must be positive")
            if self.linear_attn_config.get("gate_lower_bound", 0) >= 0:
                raise ValueError("K3 KDA requires a negative gate_lower_bound")


class KimiK3VisionConfig(PretrainedConfig):
    """Configuration for MoonViT and its spatial patch merger."""

    model_type = "kimi_k3_vision"

    def __init__(self, **kwargs: Any) -> None:
        """Load vision dimensions without requiring a device attention backend."""
        defaults = {
            "attention_backend": "eager", "norm_backend": "eager", "rope_backend": "eager",
            "recompute": False,
            "patch_size": 14, "init_pos_emb_height": 64, "init_pos_emb_width": 64,
            "init_pos_emb_time": 4, "pos_emb_type": "divided_fixed",
            "vt_num_attention_heads": 12, "vt_num_hidden_layers": 27,
            "vt_hidden_size": 1024, "vt_intermediate_size": 4096,
            "merge_kernel_size": (2, 2), "merge_type": "sd2_tpool",
            "mm_projector_type": "patchmergerv2", "mm_hidden_size": None,
            "projector_hidden_act": "gelu", "projector_ln_eps": 1e-5,
            "qkv_hidden_size": 1536, "norm_type": "rmsnorm", "attn_bias": False,
            "patch_embed_proj_bias": False, "mlp_type": "mlp2", "linear_bias": False,
            "activation_func": "gelu_pytorch_tanh", "pos_emb_interpolation_mode": "bilinear",
            "text_hidden_size": 7168,
        }
        settings = {key: kwargs.pop(key, value) for key, value in defaults.items()}
        kwargs.pop("model_type", None)
        kwargs.pop("_attn_implementation", None)
        super().__init__(**kwargs)
        if not isinstance(settings["recompute"], bool):
            raise ValueError("vision recompute must be a boolean")
        if settings["attention_backend"] not in ("eager", "npu_fusion"):
            raise ValueError("vision attention_backend must be eager or npu_fusion")
        for name in ("norm_backend", "rope_backend"):
            if settings[name] not in ("eager", "npu"):
                raise ValueError(f"vision {name} must be eager or npu")
        if settings["mm_hidden_size"] is None:
            settings["mm_hidden_size"] = settings["vt_hidden_size"]
        for name, value in settings.items():
            setattr(self, name, value)


class KimiK3Config(PretrainedConfig):
    """Composite local configuration for K3 image/text pretraining."""

    model_type = "kimi_k3"
    is_composition = True
    sub_configs = {"text_config": KimiLinearConfig, "vision_config": KimiK3VisionConfig}

    def __init__(self, text_config: dict | KimiLinearConfig | None = None,
                 vision_config: dict | KimiK3VisionConfig | None = None,
                 ignore_index: int = -100, media_placeholder_token_id: int = 163605,
                 **kwargs: Any) -> None:
        """Build nested configurations while preserving their checkpoint schema."""
        kwargs.pop("model_type", None)
        kwargs.setdefault("architectures", ["KimiK3ForConditionalGeneration"])
        kwargs.setdefault("pad_token_id", 163839)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.text_config = text_config if isinstance(text_config, KimiLinearConfig) else KimiLinearConfig(
            **(text_config or {}))
        self.vision_config = vision_config if isinstance(vision_config, KimiK3VisionConfig) else KimiK3VisionConfig(
            **(vision_config or {}))
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
