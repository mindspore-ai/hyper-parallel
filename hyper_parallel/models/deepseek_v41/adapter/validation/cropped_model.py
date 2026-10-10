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
"""Build a configurable DeepSeek-V4.1 validation crop from local assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import CompileConfig


def _scaled_dimension(value: int, divisor: int, field_name: str) -> int:
    """Scale one architecture dimension without silently changing its ratio."""
    if divisor < 1:
        raise ValueError(f"{field_name} divisor must be positive, got {divisor}")
    if value % divisor:
        raise ValueError(
            f"{field_name}={value} must be divisible by its parameter divisor {divisor}"
        )
    return value // divisor


def _load_validation_sources(
        config_path: str,
        engram_assets_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, int]:
    """Read the released config and Engram assets and check their consistency."""
    model_dir = Path(config_path).expanduser().resolve()
    assets_path = Path(engram_assets_path).expanduser().resolve()
    with (model_dir / "config.json").open("r", encoding="utf-8") as config_file:
        source = json.load(config_file)
    with assets_path.open("r", encoding="utf-8") as assets_file:
        assets = json.load(assets_file)
    if source.get("model_type") != "deepseek_v41":
        raise ValueError(
            "config_path must contain DeepSeek-V4.1; "
            f"got model_type={source.get('model_type')!r}"
        )
    if assets.get("source_model_type") != "deepseek_v41":
        raise ValueError("engram_assets_path is not a DeepSeek-V4.1 validation asset")
    text = source["text_config"]
    released_hidden_layers = int(text["num_hidden_layers"])
    if assets.get("num_hidden_layers") != released_hidden_layers:
        raise ValueError(
            "Engram assets must retain the released decoder depth: expected "
            f"{released_hidden_layers}, got {assets.get('num_hidden_layers')}"
        )
    return source, text, assets, assets_path, released_hidden_layers


def _resolve_crop_depth(
        requested_depth: int | None,
        released_depth: int,
        field_name: str,
) -> int:
    """Validate an optional layer crop while retaining released depth by default."""
    resolved_depth = released_depth if requested_depth is None else int(requested_depth)
    if not 0 < resolved_depth <= released_depth:
        raise ValueError(
            f"{field_name} must be in [1, {released_depth}], got {resolved_depth}"
        )
    return resolved_depth


def _resolve_routed_experts(text: dict[str, Any], num_routed_experts: int) -> int:
    """Validate the requested routed-expert crop against the released config."""
    released_routed_experts = int(text["n_routed_experts"])
    resolved_routed_experts = int(num_routed_experts)
    if not 0 < resolved_routed_experts <= released_routed_experts:
        raise ValueError(
            "num_routed_experts must be in [1, "
            f"{released_routed_experts}], got {resolved_routed_experts}"
        )
    if resolved_routed_experts < int(text["num_experts_per_tok"]):
        raise ValueError("num_routed_experts must be at least num_experts_per_tok")
    return resolved_routed_experts


def _scale_text_dimensions(
        text: dict[str, Any],
        text_parameter_divisor: int,
        assets: dict[str, Any],
) -> dict[str, int]:
    """Compute the uniformly scaled text dimensions and check their ratios."""
    dimensions = {
        "hidden_size": _scaled_dimension(
            int(text["hidden_size"]), text_parameter_divisor, "hidden_size"
        ),
        "moe_intermediate_size": _scaled_dimension(
            int(text["moe_intermediate_size"]), text_parameter_divisor, "moe_intermediate_size"
        ),
        "num_attention_heads": _scaled_dimension(
            int(text["num_attention_heads"]), text_parameter_divisor, "num_attention_heads"
        ),
        "head_dim": int(text["head_dim"]),
        "q_lora_rank": _scaled_dimension(
            int(text["q_lora_rank"]), text_parameter_divisor, "q_lora_rank"
        ),
        "o_lora_rank": _scaled_dimension(
            int(text["o_lora_rank"]), text_parameter_divisor, "o_lora_rank"
        ),
        "index_n_heads": _scaled_dimension(
            int(text["index_n_heads"]), text_parameter_divisor, "index_n_heads"
        ),
        "index_head_dim": int(text["index_head_dim"]),
    }
    # ``o_groups`` partitions attention heads; it is not an independent
    # parameter-width dimension. Dividing it changes the grouped projection
    # semantics and can make a memory-equivalent crop incompatible with TP.
    dimensions["o_groups"] = int(text["o_groups"])
    if dimensions["num_attention_heads"] % dimensions["o_groups"]:
        raise ValueError(
            "scaled num_attention_heads must remain divisible by released o_groups: "
            f"{dimensions['num_attention_heads']} versus {dimensions['o_groups']}"
        )
    expected_engram_head_dim = _scaled_dimension(
        int(text["engram_head_dim"]), text_parameter_divisor, "engram_head_dim"
    )
    if int(assets["head_dim"]) != expected_engram_head_dim:
        raise ValueError(
            "Engram assets and text parameter crop use different head dimensions: "
            f"expected {expected_engram_head_dim}, got {assets['head_dim']}"
        )
    return dimensions


def _build_text_config(
        source: dict[str, Any],
        text: dict[str, Any],
        dimensions: dict[str, int],
        resolved_hidden_layers: int,
        resolved_routed_experts: int,
) -> DeepseekV4Config:
    """Assemble the Transformers config for the parameter-cropped text stack."""
    return DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=text["vocab_size"],
        hidden_size=dimensions["hidden_size"],
        moe_intermediate_size=dimensions["moe_intermediate_size"],
        num_hidden_layers=resolved_hidden_layers,
        num_attention_heads=dimensions["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=dimensions["head_dim"],
        q_lora_rank=dimensions["q_lora_rank"],
        num_experts_per_tok=text["num_experts_per_tok"],
        n_routed_experts=resolved_routed_experts,
        n_shared_experts=text["n_shared_experts"],
        scoring_func=text["scoring_func"],
        norm_topk_prob=text["norm_topk_prob"],
        routed_scaling_factor=text["routed_scaling_factor"],
        max_position_embeddings=text["max_position_embeddings"],
        rope_theta=text["rope_theta"],
        rope_parameters=text["rope_scaling"],
        layer_types=["sliding_attention"] * resolved_hidden_layers,
        mlp_layer_types=["moe"] * resolved_hidden_layers,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        compress_rope_theta=text["compress_rope_theta"],
        hc_mult=text["hc_mult"],
        hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
        hc_eps=text["hc_eps"],
        swiglu_limit=text["swiglu_limit"],
        sliding_window=text["sliding_window"],
        o_groups=dimensions["o_groups"],
        o_lora_rank=dimensions["o_lora_rank"],
        index_n_heads=dimensions["index_n_heads"],
        index_head_dim=dimensions["index_head_dim"],
        index_topk=text["index_topk"],
        hidden_act=text["hidden_act"],
        initializer_range=text["initializer_range"],
        rms_norm_eps=text["rms_norm_eps"],
        use_cache=False,
        pad_token_id=source["pad_token_id"],
        bos_token_id=source["bos_token_id"],
        eos_token_id=source["eos_token_id"],
        tie_word_embeddings=text["tie_word_embeddings"],
        partial_rotary_factor=text["qk_rope_head_dim"] / text["head_dim"],
        attention_bias=text["attention_bias"],
        attention_dropout=text["attention_dropout"],
    )


def _apply_v41_extension_fields(
        config: DeepseekV4Config,
        source: dict[str, Any],
        text: dict[str, Any],
        assets: dict[str, Any],
        assets_path: Path,
        resolved_hidden_layers: int,
        exercise_post_training_indexer: bool,
        indexer_loss_coeff: float,
) -> None:
    """Fill the V4.1 shared-attention and Engram fields for retained layers."""
    config.architectures = ["DeepseekV41ForCausalLM"]
    config.v41_compress_ratios = list(text["compress_ratios"][:resolved_hidden_layers])
    config.v41_kv_source_layer_ids = [
        layer_id for layer_id in text["kv_source_layer_ids"]
        if layer_id < resolved_hidden_layers
    ]
    config.v41_index_source_layer_ids = [
        layer_id for layer_id in text["index_source_layer_ids"]
        if layer_id < resolved_hidden_layers
    ]
    _apply_validation_indexer(config, text, exercise_post_training_indexer, indexer_loss_coeff)
    retained_engram_layers = [
        (layer_id, num_embeddings)
        for layer_id, num_embeddings in zip(
            assets["layer_ids"], assets["num_embeddings"], strict=True
        )
        if layer_id < resolved_hidden_layers
    ]
    config.v41_engram_layer_ids = [layer_id for layer_id, _ in retained_engram_layers]
    config.v41_engram_num_embeddings = [
        num_embeddings for _, num_embeddings in retained_engram_layers
    ]
    config.v41_engram_bucket_base = int(assets["bucket_base"])
    config.v41_engram_table_pad_multiple = int(assets.get("table_pad_multiple", 16))
    config.v41_engram_assets_path = str(assets_path)
    config.v41_source_model_type = source["model_type"]
    config.v41_model_mode = "validation_crop"


def _apply_validation_indexer(
        config: DeepseekV4Config,
        text: dict[str, Any],
        exercise_post_training_indexer: bool,
        indexer_loss_coeff: float,
) -> None:
    """Keep the Full/Reindex hierarchy valid when the decoder depth is cropped."""
    source_candidate_layer = int(text.get("candidate_source_layer_id", -1))
    config.v41_candidate_source_layer_id = (
        source_candidate_layer if source_candidate_layer < config.num_hidden_layers else -1
    )
    config.v41_candidate_topk_blocks = int(text.get("candidate_topk_blocks", 0))
    config.v41_candidate_block_size = int(text.get("candidate_block_size", 1))
    config.v41_indexer_loss_coeff = float(indexer_loss_coeff)
    if not exercise_post_training_indexer:
        return
    if config.v41_candidate_source_layer_id < 0:
        if not config.v41_kv_source_layer_ids:
            raise ValueError("the validation crop has no shared-attention source for Reindex")
        source_layer = config.v41_kv_source_layer_ids[-1]
        reindex_layer = config.num_hidden_layers - 1
        if reindex_layer <= source_layer:
            raise ValueError("the validation crop has no layer available for Reindex")
        config.v41_index_source_layer_ids = sorted(
            set(config.v41_index_source_layer_ids + [reindex_layer])
        )
        config.v41_candidate_source_layer_id = source_layer
        config.v41_validation_reindex_remap = {
            "released_full_layer": source_candidate_layer,
            "released_reindex_layer": next(
                layer_id for layer_id in text["index_source_layer_ids"]
                if layer_id > source_candidate_layer
            ),
            "crop_full_layer": source_layer,
            "crop_reindex_layer": reindex_layer,
        }
    # At 4K with eight-token blocks this retains 1024 candidates for
    # Top-512. The released 2048-block value would retain every key.
    config.v41_candidate_topk_blocks = min(config.v41_candidate_topk_blocks, 128)


def _apply_v41_vision_fields(
        config: DeepseekV4Config,
        source: dict[str, Any],
        enable_vision: bool,
        vision_num_hidden_layers: int | None,
        vision_parameter_divisor: int,
) -> None:
    """Fill the V4.1 vision-tower extension fields."""
    vision = source["vision_config"]
    if vision_num_hidden_layers is not None and not enable_vision:
        raise ValueError("vision_num_hidden_layers requires enable_vision=true")
    resolved_vision_layers = _resolve_crop_depth(
        vision_num_hidden_layers, int(vision["num_hidden_layers"]), "vision_num_hidden_layers"
    )
    config.v41_vision_enabled = bool(enable_vision)
    config.v41_vision_num_hidden_layers = resolved_vision_layers
    config.v41_vision_hidden_size = _scaled_dimension(
        int(vision["hidden_size"]), vision_parameter_divisor, "vision_hidden_size"
    )
    config.v41_vision_num_attention_heads = _scaled_dimension(
        int(vision["num_attention_heads"]),
        vision_parameter_divisor,
        "vision_num_attention_heads",
    )
    config.v41_vision_intermediate_size = _scaled_dimension(
        int(vision["intermediate_size"]),
        vision_parameter_divisor,
        "vision_intermediate_size",
    )
    config.v41_vision_patch_size = int(vision["patch_size"])
    config.v41_vision_rope_theta = float(vision["rope_theta"])
    config.v41_vision_downsample_ratio = int(vision["downsample_ratio"])
    config.v41_vision_max_image_tokens = int(vision["max_image_tokens"])
    config.v41_vision_min_pixels = int(vision["min_pixels"])
    config.v41_vision_max_wh_ratio = vision["max_wh_ratio"]
    config.v41_image_token_id = int(source["image_token_id"])


def build_deepseek_v41_validation_config(
        config_path: str,
        engram_assets_path: str,
        *,
        num_hidden_layers: int | None = None,
        text_parameter_divisor: int = 4,
        enable_vision: bool = False,
        vision_num_hidden_layers: int | None = None,
        vision_parameter_divisor: int = 4,
        num_routed_experts: int = 16,
        exercise_post_training_indexer: bool = True,
        indexer_loss_coeff: float = 1.0e-3,
) -> DeepseekV4Config:
    """Translate the released config into a configurable validation crop.

    Args:
        config_path: Local DeepSeek-V4.1 repository.
        engram_assets_path: Scaled Engram assets prepared from its tokenizer.
        num_hidden_layers: Optional decoder crop depth. ``None`` retains the
            released decoder depth.
        text_parameter_divisor: Uniform divisor for text hidden, MLP, attention
            head count, low-rank dimensions, Indexer head count, and Engram
            width. Per-head dimensions and the grouped-output partition count
            stay unchanged so attention semantics and supported TP degrees are
            preserved.
        enable_vision: Enable the native multimodal branch. Its released depth
            is retained unless ``vision_num_hidden_layers`` is set.
        vision_num_hidden_layers: Optional visual-tower crop depth. This is
            valid only when ``enable_vision`` is true.
        vision_parameter_divisor: Uniform divisor for vision hidden, MLP, and
            attention-head dimensions.
        num_routed_experts: Routed-expert count for the parameter crop. Routing
            top-k and every released MoE layer are retained.
        exercise_post_training_indexer: Bound the candidate pool so a 4K smoke
            exercises candidate selection before the released Reindex layers.
        indexer_loss_coeff: Sparse-stage Indexer KL coefficient. The released
            report does not disclose its production value.

    Returns:
        A Transformers DeepSeek-V4 config carrying V4.1 extension fields.

    Raises:
        ValueError: If the source, parameter crop, or assets are inconsistent.
    """
    source, text, assets, assets_path, released_hidden_layers = _load_validation_sources(
        config_path, engram_assets_path
    )
    resolved_hidden_layers = _resolve_crop_depth(
        num_hidden_layers, released_hidden_layers, "num_hidden_layers"
    )
    resolved_routed_experts = _resolve_routed_experts(text, num_routed_experts)
    dimensions = _scale_text_dimensions(text, text_parameter_divisor, assets)
    config = _build_text_config(
        source, text, dimensions, resolved_hidden_layers, resolved_routed_experts
    )
    _apply_v41_extension_fields(
        config,
        source,
        text,
        assets,
        assets_path,
        resolved_hidden_layers,
        exercise_post_training_indexer,
        indexer_loss_coeff,
    )
    _apply_v41_vision_fields(
        config, source, enable_vision, vision_num_hidden_layers, vision_parameter_divisor
    )
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    return config


def build_cropped_deepseek_v41(
        config_path: str,
        engram_assets_path: str,
        num_hidden_layers: int | None = None,
        text_parameter_divisor: int = 4,
        enable_vision: bool = False,
        vision_num_hidden_layers: int | None = None,
        vision_parameter_divisor: int = 4,
        num_routed_experts: int = 16,
        exercise_post_training_indexer: bool = True,
        indexer_loss_coeff: float = 1.0e-3,
        torch_dtype: str = "bfloat16",
        validate_placement: bool = False,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: str | None = None,
        activation_checkpoint_selection: Any | None = None,
        activation_swap: str = "none",
        model_init_dtype: str = "float32",
) -> PreTrainedModel:
    """Build and parallelize a configurable V4.1 validation crop.

    Args:
        config_path: Local DeepSeek-V4.1 repository.
        engram_assets_path: Prepared scaled-Engram JSON file.
        num_hidden_layers: Optional decoder crop depth.
        text_parameter_divisor: Uniform text-dimension divisor.
        enable_vision: Enable the parameter-scaled visual tower.
        vision_num_hidden_layers: Optional visual-tower crop depth.
        vision_parameter_divisor: Uniform visual-dimension divisor.
        num_routed_experts: Routed-expert count for the parameter crop.
        exercise_post_training_indexer: Exercise the released Full/Reindex
            hierarchy at its native layer indices.
        indexer_loss_coeff: Sparse-stage Indexer KL coefficient.
        torch_dtype: Forward dtype accepted by the model builder.
        validate_placement: Enable DTensor placement validation.
        distributed_setup: Trainer-provided parallel topology.
        peft_config: Optional PEFT configuration.
        compile_config: Optional compilation configuration.
        activation_checkpoint: Activation-checkpoint mode.
        activation_checkpoint_selection: Optional adapter-safe region selection.
        activation_swap: Activation-swap mode.
        model_init_dtype: Final parameter initialization dtype.

    Returns:
        Parallelized, randomly initialized V4.1 validation model.
    """
    config = build_deepseek_v41_validation_config(
        config_path,
        engram_assets_path,
        num_hidden_layers=num_hidden_layers,
        text_parameter_divisor=text_parameter_divisor,
        enable_vision=enable_vision,
        vision_num_hidden_layers=vision_num_hidden_layers,
        vision_parameter_divisor=vision_parameter_divisor,
        num_routed_experts=num_routed_experts,
        exercise_post_training_indexer=exercise_post_training_indexer,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        validate_placement=validate_placement,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        activation_checkpoint_selection=activation_checkpoint_selection,
        activation_swap=activation_swap,
        model_init_dtype=model_init_dtype,
    )


__all__ = ["build_cropped_deepseek_v41", "build_deepseek_v41_validation_config"]
