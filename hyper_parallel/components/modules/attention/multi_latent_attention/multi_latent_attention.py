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

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

# This package provides PyTorch-specific high-performance modules.
# pylint: disable=forbidden-backend-import
import torch  # pylint: disable=forbidden-backend-import
from torch import nn
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter

from hyper_parallel.components import functional
from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.tensor_parallel.execution import TensorParallelExecution
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    validate_mla_layout,
)


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


class MLAOutputLinear(nn.Linear):
    """Rowwise output projection retaining its Parameter names and identities."""

    def __init__(self, source: nn.Linear) -> None:
        """Reuse a plain projection before FSDP/optimizer construction."""
        if type(source) is not nn.Linear and not isinstance(source, MLAOutputLinear):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA parallel output requires a plain Linear projection")
        nn.Module.__init__(self)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.tensor_parallel = TensorParallelExecution()
        self.train(source.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Reduce the partial result before adding the replicated bias exactly once.

        Args:
            input: Tokens with TP-local value heads; output reduction precedes bias addition.
        """
        output = torch.nn.functional.linear(input, self.weight, None)
        output = self.tensor_parallel.finalize_output(output)
        return output if self.bias is None else output + self.bias


@module_replacement
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
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        strategy: str = "expanded_ulysses", overlap: bool = False,
        ulysses_degree: int | None = None, kv_exchange: str = "all_gather",
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Construct an MLA algorithm without binding CP or TP communication.

        Args:
            module: Optional source with Q-LoRA, KV-LoRA and managed projection children.
                An already-fused MLA reuses all child modules and their Parameters.
            config: Direct construction dimensions and attention options; exclusive with module.
                Uses the DeepSeek parameter layout: optional down/output bias, bias-free up projections.
            module_fqn: Source path supplied by the replacement framework.
            context: Construction context; parallel resources are bound in later stages.
            attention_interface: Local FA implementation returning BSND output and optional weights.

        Note:
            Source conversion keeps the existing checkpoint transforms. Direct construction
            initializes native Linear weights and unit RMSNorm scales, without creating an HF model.
        """
        super().__init__()
        del module_fqn
        self.tensor_parallel = TensorParallelExecution()
        # CP options remain in the family's construction request for later promotion.
        # The local algorithm has no CP execution state.
        del strategy, overlap, ulysses_degree, kv_exchange
        self.tp_degree = (context or {}).get("tp_size", 1)
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
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "MultiLatentAttention only implements standard MLA; parameter-sink attention "
                "must use the sink high-performance interfaces."
            )
        self.use_mome = bool(getattr(module, "use_mome", False))
        if self.use_mome:
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
        self.use_mome = bool(getattr(config, "use_mome", False))
        if self.use_mome or getattr(config, "param_sink_number", 0) > 0:
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
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps).to(dtype=dtype)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps).to(dtype=dtype)
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
        # HF initialization already visits converted Linear/Norm children. The replacement
        # framework also invokes this method after it, so source conversions must not reset them again.
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
        transforms: list[WeightConverter] = [
            WeightConverter(
                source_patterns=[
                    "q_a_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                ],
                target_patterns="linear_qkv.weight",
                operations=[
                    ConcatenateWithSections(
                        sections=(
                            self._q_latent_output_size,
                            self._kv_latent_output_size,
                        ),
                        dim=0,
                    )
                ],
            ),
        ]
        if self.linear_qkv.bias is not None:
            transforms.insert(
                1,
                WeightConverter(
                    source_patterns=[
                        "q_a_proj.bias",
                        "kv_a_proj_with_mqa.bias",
                    ],
                    target_patterns="linear_qkv.bias",
                    operations=[
                        ConcatenateWithSections(
                            sections=(
                                self._q_latent_output_size,
                                self._kv_latent_output_size,
                            ),
                            dim=0,
                        )
                    ],
                ),
            )
        return transforms

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare class-owned TP preparation and output restoration for local MLA."""
        return ParallelModuleSpec(
            "attention.mla.local", "mesh",
            {"prepare_input": ("hidden_states",), "project_output": ("attention_output", "hidden_states")},
            validate=validate_mla_layout,
        )

    def bind_parallel(self, binding: Any) -> None:
        """Bind final layouts to shared TP stages, without implementing collectives here.

        Args:
            binding: Final validated layout and job-owned mesh resources.
        """
        spec = binding.sharding
        compute_input = dict(spec.in_src["hidden_states"])
        compute_output = dict(spec.out_dst["output"])
        # Parameter ownership is validated before binding. A row-sharded Wo produces
        # TP Partial; replicated Wo produces a complete result on each TP rank.
        sharded = spec.params["o_proj.weight"].get("tp", Replicate()).is_shard()
        if sharded:
            compute_input["tp"] = Replicate()
            compute_output["tp"] = Partial()
        self.tensor_parallel = TensorParallelExecution.from_binding(
            binding, input_layout=compute_input, output_layout=compute_output,
        )
        if sharded:
            if not isinstance(self.o_proj, MLAOutputLinear):
                self.o_proj = MLAOutputLinear(self.o_proj)
            self.o_proj.tensor_parallel = self.tensor_parallel
        self.num_heads = self.config.num_attention_heads // self.tp_degree
        self.num_key_value_heads = self.num_heads

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore TP sequence shards before local or CP projection math.

        Args:
            hidden_states: Tokens in the declared external TP/SP layout.
        """
        return self.tensor_parallel.prepare_input(hidden_states)

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
            heads: Projection head count, including any existing TP localization.
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
                The existing CP projection wrapper supplies this capability during migration.

        Returns:
            Expanded keys and values in BNSD.
        """
        projected = (self.kv_b_proj(latent) if head_range is None
                     else self.kv_b_proj(latent, mla_head_range=head_range))
        return self.assemble_key_value(projected, shared_key, heads=heads)

    def _project_attention_inputs(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None, past_key_values: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the same projection/position stages as CP, retaining the compressed cache contract."""
        query_latent, kv_latent, shared_key = self.project_latent_inputs(hidden_states)
        query = self.project_query(query_latent, position_embeddings)
        shared_key = self.apply_position(shared_key.unsqueeze(2), position_embeddings).squeeze(2)
        if past_key_values is not None:
            kv_latent, shared_key = past_key_values.update(
                kv_latent.unsqueeze(1), shared_key.unsqueeze(1), self.layer_idx,
            )
            kv_latent, shared_key = kv_latent.squeeze(1), shared_key.squeeze(1)
        key, value = self.project_key_value(kv_latent, shared_key)
        return query, key, value

    def local_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attention_mask: torch.Tensor | None = None, *, attention_interface: Callable | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the shared FA contract to BNSD inputs and return BSND output.

        Args:
            query: Expanded queries, in computation head and sequence order.
            key: Expanded keys in the same computation head order.
            value: Values with the same key token and head order.
            attention_mask: Model-convention boolean mask, or None for causal metadata.
            attention_interface: Optional local implementation, preserving the existing CP callback contract.
            **kwargs: Attention metadata; CP supplies global document ends and head coordinates.

        Returns:
            BSND attention output and optional attention weights.
        """
        attention_mask = prepare_mla_attention_mask(attention_mask, query.shape[2], key.shape[2])
        options = {"dropout": self.attention_dropout if self.training else 0.0,
                   "scaling": self.scaling, "sliding_window": self.sliding_window}
        options.update(kwargs)
        interface = self.attention_interface if attention_interface is None else attention_interface
        return interface(self, query, key, value, attention_mask, **options)

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project [B, local_tokens, heads * value_dim] to [B, local_tokens, hidden].

        This model-math extension point lets gated MLA multiply its gate before
        Wo; the default MLA keeps the original output projection unchanged.
        This method owns model math only; CP communication belongs to its adapter.

        Args:
            attention_output: Flattened local-token attention output.
            hidden_states: Original local-token input, available to gated variants.

        Returns:
            Projected local hidden states.
        """
        del hidden_states
        return self.o_proj(attention_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run MLA with the same external contract as Transformers attention.

        Args:
            hidden_states: Input tokens in batch, sequence, hidden order.
            position_embeddings: Optional precomputed RoPE cosine and sine.
            attention_mask: Optional mask in the attention interface convention.
            past_key_values: Optional cache for incremental decoding.
            actual_seq_len: Optional cumulative packed document ends.
            **kwargs: Additional arguments for the attention interface.

        Returns:
            Projected attention output and optional attention weights.
        """
        hidden_states = self.prepare_input(hidden_states)
        batch_size, seq_length = hidden_states.shape[:-1]
        query_states, key_states, value_states = self._project_attention_inputs(
            hidden_states,
            position_embeddings,
            past_key_values,
        )

        attn_output, attn_weights = self.local_attention(
            query_states, key_states, value_states, attention_mask,
            actual_seq_len=actual_seq_len, **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        # Gated MLA derives its gate from the original input before Wo; ordinary
        # MLA only applies Wo. Keep this math extension shared by CP1 and CP.
        attn_output = self.project_output(attn_output, hidden_states)
        return attn_output, attn_weights
