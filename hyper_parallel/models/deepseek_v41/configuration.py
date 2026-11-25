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
"""Released DeepSeek-V4.1 configuration translation and explicit validation-crop helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def _is_prime(value: int) -> bool:
    """Return whether ``value`` is prime."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = math.isqrt(value)
    return all(value % divisor for divisor in range(3, limit + 1, 2))


def _next_unused_prime(start: int, seen: set[int]) -> int:
    """Return the first unused prime strictly greater than ``start``."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_scaled_engram_buckets(
        layer_ids: Iterable[int],
        *,
        bucket_base: int,
        max_ngram_size: int,
        num_heads: int,
) -> tuple[list[list[list[int]]], list[int]]:
    """Build synchronized prime buckets and table sizes for scaled Engram.

    Args:
        layer_ids: Decoder layers that contain an Engram module.
        bucket_base: Approximate rows assigned to each n-gram/head bucket.
        max_ngram_size: Largest n-gram included in the hash.
        num_heads: Independent Engram hash heads.

    Returns:
        Nested prime moduli ``[layer][ngram][head]`` and the exact number of
        embedding rows required by every layer.

    Raises:
        ValueError: If a shape argument cannot describe a useful table.
    """
    layer_ids = list(layer_ids)
    if not layer_ids:
        raise ValueError("scaled Engram requires at least one layer")
    if bucket_base < 2:
        raise ValueError("engram bucket_base must be at least 2")
    if max_ngram_size < 2:
        raise ValueError("engram max_ngram_size must be at least 2")
    if num_heads <= 0:
        raise ValueError("engram num_heads must be positive")

    seen: set[int] = set()
    all_primes: list[list[list[int]]] = []
    table_sizes: list[int] = []
    for _ in layer_ids:
        layer_primes: list[list[int]] = []
        for _ in range(max_ngram_size - 1):
            head_primes: list[int] = []
            current = bucket_base - 1
            for _ in range(num_heads):
                current = _next_unused_prime(current, seen)
                seen.add(current)
                head_primes.append(current)
            layer_primes.append(head_primes)
        all_primes.append(layer_primes)
        table_sizes.append(sum(prime for row in layer_primes for prime in row))
    return all_primes, table_sizes


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
    # Asset preparation also uses this module without optional native V4 model support.
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # pylint: disable=C0415

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


def _apply_v41_dspark_fields(
        config: DeepseekV4Config,
        source: dict[str, Any],
        text: dict[str, Any],
        num_hidden_layers: int,
        resolved_routed_experts: int,
        dspark_depth: int,
        dspark_target_layer_ids: list[int] | None,
        dspark_loss_coeff: float,
        dspark_confidence_coeff: float,
) -> None:
    """Fill the DSpark drafter extension fields."""
    config.v41_dspark_depth = int(dspark_depth)
    if config.v41_dspark_depth <= 0:
        config.v41_dspark_target_layer_ids = []
        return
    # Depth crops must keep the drafter targets inside the retained backbone.
    default_targets = list(range(max(0, num_hidden_layers - 3), num_hidden_layers))
    config.v41_dspark_target_layer_ids = list(dspark_target_layer_ids or default_targets)
    config.v41_dspark_block_size = int(text.get("dspark_block_size", 5))
    config.v41_dspark_markov_rank = int(text.get("dspark_markov_rank", 256))
    config.v41_dspark_noise_token_id = int(
        text.get("dspark_noise_token_id", source["pad_token_id"]))
    config.v41_dspark_n_routed_experts = min(
        int(text.get("dspark_n_routed_experts", resolved_routed_experts)),
        resolved_routed_experts,
    )
    config.v41_dspark_top_k = min(
        int(text.get("dspark_num_experts_per_tok", 3)),
        config.v41_dspark_n_routed_experts,
    )
    # Released drafter sliding window (technical report 2.4.3).
    config.v41_dspark_window = 128
    config.v41_dspark_loss_coeff = float(dspark_loss_coeff)
    config.v41_dspark_confidence_coeff = float(dspark_confidence_coeff)


def _apply_engram_host_fields(
        config: DeepseekV4Config,
        engram_storage_backend: str,
        engram_max_pending_entries: int,
        engram_max_sparse_rows_per_step: int,
        engram_sparse_adam_config: dict[str, Any] | None,
) -> None:
    """Validate and fill the host-side Engram storage extension fields."""
    if engram_storage_backend not in ("device", "host"):
        raise ValueError("engram_storage_backend must be 'device' or 'host'")
    config.engram_storage_backend = engram_storage_backend
    config.engram_max_pending_entries = engram_max_pending_entries
    config.engram_max_sparse_rows_per_step = engram_max_sparse_rows_per_step
    config.engram_sparse_adam_config = engram_sparse_adam_config or {"lr": 1.0e-3}


def _released_engram_assets(source: dict[str, Any], assets_path: str, *,
                            allow_padding: bool = False) -> dict[str, Any]:
    """Reject scaled hashing assets when loading released Engram table weights."""
    assets = json.loads(Path(assets_path).expanduser().read_text(encoding="utf-8"))
    text = source["text_config"]
    expected = {
        "source_model_type": source["model_type"],
        "num_hidden_layers": text["num_hidden_layers"],
        "layer_ids": text["engram_layer_ids"],
        "num_embeddings": text["engram_num_embeddings"],
        "head_dim": text["engram_head_dim"],
        "bucket_base": text["engram_vocab_size"],
    }
    for name, value in expected.items():
        if assets.get(name) != value:
            raise ValueError(f"Released Engram assets mismatch for {name}: expected {value}, got {assets.get(name)}")
    padded_sizes = assets.get("padded_num_embeddings", assets["num_embeddings"])
    if (len(padded_sizes) != len(assets["num_embeddings"])
            or any(not isinstance(padded, int) or padded < logical
                   for padded, logical in zip(padded_sizes, assets["num_embeddings"]))):
        raise ValueError("Released Engram assets padding must cover every checkpoint table row")
    if not allow_padding and padded_sizes != assets["num_embeddings"]:
        raise ValueError("Released Engram assets must preserve checkpoint table row counts")
    return assets


def build_deepseek_v41_config(
    config_path: str, engram_assets_path: str, *, enable_vision: bool = False,
    engram_storage_backend: str = "device",
) -> PretrainedConfig:
    """Translate released dimensions and layer roles without a validation crop.

    Prepare tokenizer-derived Engram assets with ``prepare_deepseek_v41_assets.py --released``.
    Construction reads configuration and hash metadata only; it does not download weights.

    Args:
        config_path: Local directory containing the released composite config.json.
        engram_assets_path: Released-size tokenizer and hash metadata for every Engram layer.
        enable_vision: Construct the complete released visual tower when true.
        engram_storage_backend: Device-resident tables or the public host-table family.

    Returns:
        The complete translated configuration accepted by DeepseekV41ForCausalLM.
    """
    source, text, assets, assets_path, depth = _load_validation_sources(config_path, engram_assets_path)
    _released_engram_assets(source, str(assets_path), allow_padding=engram_storage_backend == "host")
    dimensions = _scale_text_dimensions(text, 1, assets)
    config = _build_text_config(source, text, dimensions, depth, text["n_routed_experts"])
    _apply_v41_extension_fields(config, source, text, assets, assets_path, depth, False, 0.001)
    config.v41_model_mode = "full"
    _apply_v41_vision_fields(config, source, enable_vision, None, 1)
    _apply_v41_dspark_fields(config, source, text, depth, text["n_routed_experts"], 0, None, 1.0, 0.1)
    _apply_engram_host_fields(config, engram_storage_backend, 1000000, 1000000, None)
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    config._name_or_path = str(Path(config_path).expanduser().resolve())  # pylint: disable=protected-access
    return config


__all__ = ["build_deepseek_v41_config", "build_scaled_engram_buckets"]
