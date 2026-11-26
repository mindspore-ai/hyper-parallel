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
"""Multi-head latent attention module using the Hyper projection layout."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

# This package provides PyTorch-specific high-performance modules.
import torch
from torch import nn
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter

from hyper_parallel.components import functional
from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.components.modules.common.rms_norm import RMSNorm



def prepare_mla_attention_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    """Disambiguate MLA Q/K masks before calling a shared attention interface.

    Args:
        attention_mask: Optional mask. A two-dimensional Q/K matrix takes
            precedence over padding in MLA, including when batch equals query length.
        query_length: Number of query tokens visible to the attention interface.
        key_length: Number of key tokens visible to the attention interface.

    Returns:
        A broadcastable four-dimensional view for a Q/K matrix. Other inputs
        pass through unchanged, preserving padding and compressed causal paths.
    """
    if (attention_mask is not None and attention_mask.ndim == 2
            and attention_mask.shape == (query_length, key_length)):
        # Shared FA interprets 2D masks as batch padding; MLA also accepts Q/K matrices.
        return attention_mask[None, None, :, :]
    return attention_mask


def apply_mla_rope(
    tensor: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotate one BSND tensor so Q and shared K can be scheduled independently.

    Args:
        tensor: Query or unexpanded shared key, with rotary features in the last dimension.
        position_embeddings: Cosine/sine tables in SD or BSD, or None for an identity operation.
        interleaved: Whether the input stores real/imaginary features in adjacent pairs.

    Returns:
        Rotated BSND features. Interleaved inputs become the half-band order consumed by MLA.
        No extra contiguous conversion is imposed on the caller's layout.
    """
    if position_embeddings is None:
        return tensor
    cos, sin = (part.to(tensor.dtype) for part in position_embeddings)
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if interleaved:
        cos, sin = (part[..., :tensor.shape[-1] // 2].repeat_interleave(2, dim=-1)
                    for part in (cos, sin))
    cos, sin = (part.expand(tensor.shape[0], -1, -1).transpose(0, 1) for part in (cos, sin))
    rotated = functional.apply_rotary_pos_emb_single(
        tensor.transpose(0, 1), cos, sin,
        rotary_interleaved=interleaved, use_fused_rotary_pos_emb=True,
    )
    if interleaved:
        # V3 consumes real/imaginary half-bands; the kernel returns adjacent pairs.
        rotated = rotated.permute(1, 2, 0, 3)
        return rotated.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).transpose(1, 2)
    return rotated.transpose(0, 1)


@dataclass(frozen=True)
class MLALatentInputs:
    """Normalized token-local features before query/KV head expansion."""

    query_latent: torch.Tensor
    kv_latent: torch.Tensor
    key_rope: torch.Tensor


@dataclass(frozen=True)
class MLAInputs:
    """Ready BNSD operands and invocation-local attention/output metadata."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    options: dict[str, Any]
    output_state: Any = None


class MultiLatentAttentionModule(nn.Module):
    """Transformers-compatible MLA using a fused latent projection.

    ``q_a_proj`` and ``kv_a_proj_with_mqa`` are fused into ``linear_qkv``.
    Unchanged child modules retain their source names and checkpoint layout.
    Construction only creates the target structure; ``make_transforms``
    declares the checkpoint conversion.
    """

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        config: Any = None,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Construct MLA projections and attention math from a source module or config.

        Args:
            module: Optional source with Q-LoRA, KV-LoRA and managed projection children.
                An already-fused MLA reuses all child modules and their Parameters.
            config: Direct construction dimensions and attention options; exclusive with module.
                Uses the DeepSeek parameter layout: optional down/output bias, bias-free up projections.
            attention_interface: Local FA implementation returning BSND output and optional weights.

        Note:
            Source conversion keeps the existing checkpoint transforms. Direct construction
            initializes native Linear weights and unit RMSNorm scales, without creating an HF model.
        """
        super().__init__()
        self.attention_interface = attention_interface
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        if module is None:
            if config is None:
                raise ValueError("MLA construction requires config or a source module")
            self._initialize_config(config)
            return
        if config is not None:
            raise ValueError("MLA construction accepts config or module, not both")
        self._initialize_source(module)

    def _initialize_source(self, module: nn.Module) -> None:
        """Read source math options and reuse its managed projection children."""
        self._source_is_fused = hasattr(module, "linear_qkv")
        required = (
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
        )
        required += ("linear_qkv",) if self._source_is_fused else ("q_a_proj", "kv_a_proj_with_mqa")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"MultiLatentAttention source module is missing: {missing}")

        config = module.config
        self.config = config
        self.layer_idx = getattr(module, "layer_idx", getattr(module, "layer_number", None))
        self.num_heads = getattr(module, "num_heads", config.num_attention_heads)
        self.num_key_value_heads = getattr(
            module,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", self.num_heads),
        )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_lora_rank = getattr(module, "q_lora_rank", config.q_lora_rank)
        self.kv_lora_rank = getattr(module, "kv_lora_rank", config.kv_lora_rank)
        if self.q_lora_rank is None:
            raise ValueError("MultiLatentAttention requires the Q-LoRA projection path")
        self.qk_rope_head_dim = getattr(module, "qk_rope_head_dim", config.qk_rope_head_dim)
        self.qk_nope_head_dim = getattr(module, "qk_nope_head_dim", config.qk_nope_head_dim)
        self.v_head_dim = getattr(module, "v_head_dim", config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(
            module, "sliding_window", getattr(config, "sliding_window", None)
        )
        self.rotary_interleaved = bool(
            getattr(module, "rotary_interleaved", getattr(config, "rope_interleave", False))
        )
        # These source variants change attention math; ignoring the flags would
        # silently replace them with ordinary MLA. Keep only construction guards.
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "MultiLatentAttention only implements standard MLA; parameter-sink attention "
                "must use the sink high-performance interfaces."
            )
        if getattr(module, "use_mome", False):
            raise NotImplementedError(
                "MultiLatentAttention only implements standard MLA; MOME must use the "
                "aggregate_hidden high-performance path."
            )

        self._initialize_latent_projection(module)

        self.q_a_layernorm = module.q_a_layernorm
        self.kv_a_layernorm = module.kv_a_layernorm
        self.q_b_proj = module.q_b_proj
        self.kv_b_proj = module.kv_b_proj
        self.o_proj = module.o_proj

        self.train(module.training)

    def _initialize_latent_projection(self, module: nn.Module) -> None:
        """Fuse compatible down projections or retain an already-fused source."""
        self._q_latent_output_size = self.q_lora_rank
        self._kv_latent_output_size = self.kv_lora_rank + self.qk_rope_head_dim
        if self._source_is_fused:
            self.linear_qkv = module.linear_qkv
            self.attention_interface = module.attention_interface
        else:
            q_a = module.q_a_proj
            kv_a = module.kv_a_proj_with_mqa
            self._q_latent_output_size = q_a.out_features
            self._kv_latent_output_size = kv_a.out_features
            can_fuse = (
                q_a.in_features == kv_a.in_features
                and q_a.weight.requires_grad == kv_a.weight.requires_grad
                and (q_a.bias is None) == (kv_a.bias is None)
                and (q_a.bias is None or q_a.bias.requires_grad == kv_a.bias.requires_grad)
            )
            if not can_fuse:
                raise ValueError(
                    "Q and KV latent projections cannot be represented by one fused projection"
                )
            self.linear_qkv = nn.Linear(
                q_a.in_features,
                q_a.out_features + kv_a.out_features,
                bias=q_a.bias is not None,
                device=q_a.weight.device,
                dtype=q_a.weight.dtype,
            )
            self.linear_qkv.weight.requires_grad_(q_a.weight.requires_grad)
            if q_a.bias is not None:
                self.linear_qkv.bias.requires_grad_(q_a.bias.requires_grad)

    def _initialize_config(self, config: Any) -> None:
        """Construct the same parameter tree directly, without an intermediate HF model."""
        self.config = config
        self.layer_idx = getattr(config, "layer_idx", None)
        dimensions = {
            "num_heads": config.num_attention_heads,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions.values()):
            raise ValueError("MLA dimensions must be positive integers, including q_lora_rank")
        if dimensions["qk_rope_head_dim"] % 2:
            raise ValueError("MLA rotary dimension must be even")
        for name, value in dimensions.items():
            setattr(self, name, value)
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.scaling = self._config_scaling(config, self.scaling)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = getattr(config, "is_causal", True)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.rotary_interleaved = bool(getattr(config, "rotary_interleaved", getattr(config, "rope_interleave", False)))
        if getattr(config, "use_mome", False) or getattr(config, "param_sink_number", 0) > 0:
            raise ValueError("MLA config construction does not support MOME or parameter-sink attention")
        self._source_is_fused = True
        self._q_latent_output_size = self.q_lora_rank
        self._kv_latent_output_size = self.kv_lora_rank + self.qk_rope_head_dim
        hidden_size = config.hidden_size
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("MLA hidden_size must be a positive integer")
        dtype = getattr(config, "dtype", None)
        bias = getattr(config, "attention_bias", False)
        self.linear_qkv = nn.Linear(
            hidden_size, self._q_latent_output_size + self._kv_latent_output_size, bias=bias, dtype=dtype,
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_a_layernorm = RMSNorm(
            config=SimpleNamespace(hidden_size=self.q_lora_rank, eps=eps), use_fused_ops=True,
        ).to(dtype=dtype)
        self.kv_a_layernorm = RMSNorm(
            config=SimpleNamespace(hidden_size=self.kv_lora_rank, eps=eps), use_fused_ops=True,
        ).to(dtype=dtype)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=dtype,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=bias, dtype=dtype)

    @staticmethod
    def _config_scaling(config: Any, default: float) -> float:
        """Keep the source-compatible YaRN factor and explicit scaling override."""
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" and rope.get("mscale_all_dim", 0):
            factor = rope["factor"]
            mscale = 1.0 if factor <= 1 else 0.1 * rope["mscale_all_dim"] * math.log(factor) + 1.0
            default *= mscale * mscale
        return getattr(config, "scaling", default)

    def reset_parameters(self) -> None:
        """Initialize direct-config parameters without overriding a source model's initializer."""
        # Converted sources retain their model's initialization; config-owned members
        # provide complete defaults before the enclosing model applies its own policy.
        if not self._config_initialization:
            return
        for projection in (self.linear_qkv, self.q_b_proj, self.kv_b_proj, self.o_proj):
            projection.reset_parameters()
        nn.init.ones_(self.q_a_layernorm.weight)
        nn.init.ones_(self.kv_a_layernorm.weight)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if self._source_is_fused:
            return []
        suffixes = ("weight", "bias") if self.linear_qkv.bias is not None else ("weight",)
        sections = (self._q_latent_output_size, self._kv_latent_output_size)
        return [
            WeightConverter(
                source_patterns=[f"q_a_proj.{suffix}", f"kv_a_proj_with_mqa.{suffix}"],
                target_patterns=f"linear_qkv.{suffix}",
                operations=[ConcatenateWithSections(sections=sections, dim=0)],
            )
            for suffix in suffixes
        ]

    def project_latent_inputs(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute shared latent inputs before head expansion.

        Args:
            hidden_states: Input tokens with shape [batch, sequence, hidden].

        Returns:
            Normalized query and KV latents, and the unrotated shared RoPE key.
        """
        latent_states = self.linear_qkv(hidden_states)
        query_latent, kv_nope, key_rope = torch.split(
            latent_states,
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1,
        )
        return self.q_a_layernorm(query_latent), self.kv_a_layernorm(kv_nope), key_rope

    def apply_position(
        self, tensor: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *, interleaved: bool | None = None,
    ) -> torch.Tensor:
        """Rotate query/shared-key features before head replication.

        Args:
            tensor: Features in BSND; shared K has a singleton head dimension.
            position_embeddings: Local token positions in SD or BSD.
            interleaved: Optional input-order override; defaults to the module's source convention.

        Returns:
            BSND features in the MLA key/query feature order.
        """
        return apply_mla_rope(
            tensor, position_embeddings,
            interleaved=self.rotary_interleaved if interleaved is None else interleaved,
        )

    def project_query(
        self, latent: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *, heads: int | None = None, rotary_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Expand query heads without changing token ownership.

        Args:
            latent: Normalized query latent in BSR.
            position_embeddings: Local token cosine/sine tables, or None.
            heads: Number of heads produced by the query projection.
            rotary_fn: Optional legacy local callback; defaults to this algorithm's position method.

        Returns:
            BNSD queries. The NoPE identity path preserves the projection view.
        """
        heads = self.num_heads if heads is None else heads
        projected = self.q_b_proj(latent).view(*latent.shape[:2], heads, self.qk_head_dim)
        content, rotary = projected.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        rotation = self.apply_position if rotary_fn is None else rotary_fn
        rotated = rotation(rotary, position_embeddings, interleaved=self.rotary_interleaved)
        if rotated is rotary:
            return projected.transpose(1, 2)
        return torch.cat((content, rotated), dim=-1).transpose(1, 2)

    def project_key_value_parts(
        self, latent: torch.Tensor, *, heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project K content and V before assembling the shared key features.

        Args:
            latent: Normalized KV latent in BSR.
            heads: Projection head count, or the module's local head count.

        Returns:
            K content and V views in BSND. A scheduler can consume V before K is assembled.
        """
        heads = self.num_heads if heads is None else heads
        projected = self.kv_b_proj(latent).view(
            *latent.shape[:2], heads, self.qk_nope_head_dim + self.v_head_dim,
        )
        return projected.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)

    @staticmethod
    def assemble_key(content: torch.Tensor, shared_key: torch.Tensor) -> torch.Tensor:
        """Join BSND content with one already-transformed shared key per token.

        Args:
            content: Head-specific key features in BSND.
            shared_key: BSR features after RoPE/NoPE; expansion itself is a view.

        Returns:
            Full BSND keys with content followed by shared features.
        """
        return torch.cat((content, shared_key.unsqueeze(2).expand(-1, -1, content.shape[2], -1)), dim=-1)

    def assemble_key_value(
        self, projected: torch.Tensor, shared_key: torch.Tensor, *, heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack the result of a managed KV projection.

        Args:
            projected: Projection output with K/V features interleaved per head.
            shared_key: Already-positioned BSR shared key features.
            heads: Number of heads represented by projected.

        Returns:
            BNSD keys and values, preserving the projection's token order.
        """
        heads = self.num_heads if heads is None else heads
        projected = projected.view(*projected.shape[:2], heads, self.qk_nope_head_dim + self.v_head_dim)
        content, value = projected.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        return self.assemble_key(content, shared_key).transpose(1, 2), value.transpose(1, 2)

    def project_key_value(
        self, latent: torch.Tensor, shared_key: torch.Tensor, *, heads: int | None = None,
        head_range: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project owned heads through the child Module, preserving its parameter hooks.

        Args:
            latent: Normalized KV features in BSR.
            shared_key: Positioned shared features for the same tokens.
            heads: Number of heads in the returned tensors.
            head_range: Optional half-open head slice within the managed projection.
                Requires kv_b_proj to accept the mla_head_range keyword.

        Returns:
            Expanded keys and values in BNSD.
        """
        projected = (self.kv_b_proj(latent) if head_range is None
                     else self.kv_b_proj(latent, mla_head_range=head_range))
        return self.assemble_key_value(projected, shared_key, heads=heads)

    def validate_inputs(self, past_key_values: Any) -> None:
        """Validate per-call constraints before executing projections.

        Args:
            past_key_values: Optional compressed cache accepted by the local algorithm.
        """
        del past_key_values

    @staticmethod
    def prepare_positions(position_embeddings: Any) -> Any:
        """Preserve the algorithm's positional inputs; NoPE overrides this stage.

        Args:
            position_embeddings: Model-provided cosine/sine tables, or None.
        """
        return position_embeddings

    def prepare_attention_inputs(self, hidden_states: torch.Tensor) -> tuple[MLALatentInputs, Any]:
        """Compute normalized Q/KV latents and shared-key features before head expansion.

        Args:
            hidden_states: Original local input tokens in BSH layout.

        Returns:
            Local Q/KV latents, shared-key features and optional output-stage state.
        """
        return MLALatentInputs(*self.project_latent_inputs(hidden_states)), None

    def exchange_attention(self, state: MLALatentInputs,
                           position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                           attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                           actual_seq_len: torch.Tensor | Sequence[int] | None = None, *,
                           output_state: Any = None, **kwargs: Any) -> MLAInputs:
        """Expand local operands without changing sequence or head ownership.

        Args:
            state: Already-computed normalized latents and shared-key features.
            position_embeddings: Local cosine/sine tables.
            attention_mask: Optional padding or visibility mask.
            past_key_values: Optional compressed inference cache.
            actual_seq_len: Cumulative document endpoints.
            output_state: Optional invocation-local state passed to the output stages.
            **kwargs: Additional attention options.
        """
        query = self.project_query(state.query_latent, position_embeddings)
        shared_key = self.apply_position(state.key_rope.unsqueeze(2), position_embeddings).squeeze(2)
        kv_latent = state.kv_latent
        if past_key_values is not None:
            kv_latent, shared_key = past_key_values.update(
                kv_latent.unsqueeze(1), shared_key.unsqueeze(1), self.layer_idx,
            )
            kv_latent, shared_key = kv_latent.squeeze(1), shared_key.squeeze(1)
        key, value = self.project_key_value(kv_latent, shared_key)
        return MLAInputs(query, key, value, attention_mask,
                         dict(kwargs, actual_seq_len=actual_seq_len), output_state)

    def run_attention(self, inputs: MLAInputs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the shared FA contract to ready BNSD inputs and return BSND output.

        Args:
            inputs: Expanded Q/K/V, mask and attention options for this invocation.
                Its optional attention_interface selects the FA callback for this invocation.

        Returns:
            BSND attention output and optional attention weights.
        """
        attention_mask = prepare_mla_attention_mask(
            inputs.attention_mask, inputs.query.shape[2], inputs.key.shape[2])
        options = {"dropout": self.attention_dropout if self.training else 0.0,
                   "scaling": self.scaling, "sliding_window": self.sliding_window}
        options.update(inputs.options)
        interface = options.pop("attention_interface", None)
        if interface is None:
            interface = self.attention_interface
        return interface(self, inputs.query, inputs.key, inputs.value, attention_mask, **options)

    def restore_attention(self, output: torch.Tensor, hidden_states: torch.Tensor,
                          output_state: Any) -> tuple[torch.Tensor, Any]:
        """Preserve local query ownership before the output projection.

        Args:
            output: Attention output in BSND layout.
            hidden_states: Original local input for output-stage algorithms; unused here.
            output_state: Optional invocation-local output state.

        Returns:
            Unchanged local-token output and state consumed by project_output.
        """
        del hidden_states
        return output, output_state

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor,
                       output_state: Any) -> torch.Tensor:
        """Flatten local heads and project to the original hidden dimension.

        Args:
            attention_output: Token-local attention output in BSND layout.
            hidden_states: Original local input defining batch and sequence dimensions.
            output_state: Unused by ordinary local MLA; specialized members own their output state.

        Returns:
            Projected local hidden states in BSH layout.
        """
        del output_state
        output = attention_output.reshape(*hidden_states.shape[:2], -1).contiguous()
        return self.o_proj(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the shared MLA algorithm through operand and output stages.

        Args:
            hidden_states: Input tokens in batch, sequence, hidden order.
            position_embeddings: Optional precomputed RoPE cosine and sine.
            attention_mask: Optional mask in the attention interface convention.
            past_key_values: Optional cache for incremental decoding.
            actual_seq_len: Optional cumulative packed document endpoints.
            **kwargs: Additional arguments for the attention interface.

        Returns:
            Projected attention output and optional attention weights.
        """
        self.validate_inputs(past_key_values)
        positions = self.prepare_positions(position_embeddings)
        state, output_state = self.prepare_attention_inputs(hidden_states)
        inputs = self.exchange_attention(
            state, positions, attention_mask, past_key_values, actual_seq_len,
            output_state=output_state, **kwargs)
        del state
        output, weights = self.run_attention(inputs)
        output_state = inputs.output_state
        # Output stages need only output_state, not an extra owner of QKV.
        # Tensors saved for backward remain owned by autograd.
        del inputs
        output, output_state = self.restore_attention(output, hidden_states, output_state)
        return self.project_output(output, hidden_states, output_state), weights
