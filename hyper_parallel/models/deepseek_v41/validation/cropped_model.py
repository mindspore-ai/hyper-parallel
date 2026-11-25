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
"""Build configurable DeepSeek-V4.1 validation crops, with optional HF weight loading."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import PreTrainedModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)

from hyper_parallel.components.checkpoint.quantized_hf_storage import validate_quantization_config
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.models.deepseek_v41.configuration import (
    _apply_engram_host_fields,
    _apply_v41_dspark_fields,
    _apply_v41_extension_fields,
    _apply_v41_vision_fields,
    _build_text_config,
    _load_validation_sources,
    _released_engram_assets,
    _resolve_crop_depth,
    _resolve_routed_experts,
    _scale_text_dimensions,
)


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
        enable_candidate_pool: bool = False,
        indexer_loss_coeff: float = 1.0e-3,
        moe_aux_loss_coeff: float = 0.0,
        moe_router_bias_update_rate: float = 0.0,
        moe_router_force_load_balancing: bool = False,
        dspark_depth: int = 0,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_loss_coeff: float = 1.0,
        dspark_confidence_coeff: float = 0.1,
        engram_storage_backend: str = "device",
        engram_max_pending_entries: int = 1000000,
        engram_max_sparse_rows_per_step: int = 1000000,
        engram_sparse_adam_config: dict[str, Any] | None = None,
) -> DeepseekV4Config:
    """Translate the released config into a configurable validation crop.

    Args:
        config_path: Local DeepSeek-V4.1 repository.
        engram_assets_path: Scaled Engram assets prepared from its tokenizer.
        num_hidden_layers: Optional decoder crop depth; ``None`` retains the released decoder depth.
        text_parameter_divisor: Uniform divisor for text hidden, MLP, attention
            head count, low-rank dimensions, Indexer head count, and Engram
            width; per-head dimensions stay unchanged.
        enable_vision: Enable the native multimodal branch; retain its released depth unless explicitly cropped.
        vision_num_hidden_layers: Optional visual-tower crop depth, valid only when ``enable_vision`` is true.
        vision_parameter_divisor: Uniform divisor for vision hidden, MLP, and attention-head dimensions.
        num_routed_experts: Routed-expert count for the crop; routing top-k and every MoE layer are retained.
        exercise_post_training_indexer: Bound the candidate pool for 4K smoke tests when it is enabled.
        enable_candidate_pool: Retain hierarchical candidates. False preserves the released layer roles
            with full-range causal indexing.
        indexer_loss_coeff: Sparse-stage Indexer KL coefficient.
        moe_aux_loss_coeff: Sequence balancing coefficient; zero disables it.
        moe_router_bias_update_rate: Per-step text/image correction-bias update speed; zero disables it.
        moe_router_force_load_balancing: Enable deterministic destinations when using the new Gate family.
        dspark_depth: Trainable DSpark stage count; 0 disables the drafter.
        dspark_target_layer_ids: Backbone layers feeding the drafter; defaults to the released last-three layout.
        dspark_loss_coeff: Drafter objective weight.
        dspark_confidence_coeff: Confidence-head BCE weight.

    Returns:
        A Transformers DeepSeek-V4 config carrying V4.1 extension fields.

    Raises:
        ValueError: If the source, parameter crop, or assets are inconsistent.
    """
    if not isinstance(enable_candidate_pool, bool):
        raise ValueError(f"enable_candidate_pool must be a bool, got {enable_candidate_pool!r}")
    source, text, assets, assets_path, released_hidden_layers = _load_validation_sources(
        config_path, engram_assets_path
    )
    resolved_hidden_layers = _resolve_crop_depth(num_hidden_layers, released_hidden_layers, "num_hidden_layers")
    resolved_routed_experts = _resolve_routed_experts(text, num_routed_experts)
    dimensions = _scale_text_dimensions(text, text_parameter_divisor, assets)
    config = _build_text_config(source, text, dimensions, resolved_hidden_layers, resolved_routed_experts)
    _apply_v41_extension_fields(
        config, source, text, assets, assets_path, resolved_hidden_layers,
        exercise_post_training_indexer and enable_candidate_pool, indexer_loss_coeff,
    )
    if not enable_candidate_pool:
        config.v41_candidate_source_layer_id = -1
        config.v41_candidate_topk_blocks = 0
    config.moe_aux_loss_coeff = float(moe_aux_loss_coeff)
    config.moe_router_bias_update_rate = float(moe_router_bias_update_rate)
    config.moe_router_force_load_balancing = moe_router_force_load_balancing
    # The native HF model still consumes these fields before module replacement.
    config.router_aux_loss_coef = config.moe_aux_loss_coeff
    config.router_bias_update_rate = config.moe_router_bias_update_rate
    _apply_v41_vision_fields(
        config, source, enable_vision, vision_num_hidden_layers, vision_parameter_divisor
    )
    _apply_v41_dspark_fields(
        config,
        source,
        text,
        resolved_hidden_layers,
        resolved_routed_experts,
        dspark_depth,
        dspark_target_layer_ids,
        dspark_loss_coeff,
        dspark_confidence_coeff,
    )
    _apply_engram_host_fields(
        config, engram_storage_backend, engram_max_pending_entries,
        engram_max_sparse_rows_per_step, engram_sparse_adam_config,
    )
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    # Record the source identity the way AutoConfig.from_pretrained does; the
    # codegen from_config path resolves the modeling source from it.
    config._name_or_path = str(Path(config_path).expanduser().resolve())  # pylint: disable=protected-access
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
        moe_aux_loss_coeff: float = 0.0,
        moe_router_bias_update_rate: float = 0.0,
        moe_router_force_load_balancing: bool = False,
        dspark_depth: int = 0,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_loss_coeff: float = 1.0,
        dspark_confidence_coeff: float = 0.1,
        engram_storage_backend: str = "device",
        engram_max_pending_entries: int = 1000000,
        engram_max_sparse_rows_per_step: int = 1000000,
        engram_sparse_adam_config: dict[str, Any] | None = None,
        torch_dtype: str = "bfloat16",
        validate_placement: bool = False,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: Any | None = None,
        model_init_dtype: str = "float32",
        codegen: bool = False,
        codegen_config: Any | None = None,
        load_base_model: bool = False,
        enable_candidate_pool: bool = True,
        verify_hf_weights: str | None = None,
) -> PreTrainedModel:
    """Build and parallelize a configurable V4.1 validation crop.
    Args:
        config_path: Local DeepSeek-V4.1 repository.
        engram_assets_path: Prepared scaled-Engram JSON file.
        num_routed_experts: Routed-expert count for the parameter crop.
        exercise_post_training_indexer: Bound and remap candidates in shallow crops when the pool is enabled.
        enable_candidate_pool: Retain the released Full/Reindex candidate pool.
        indexer_loss_coeff: Sparse-stage Indexer KL coefficient.
        moe_aux_loss_coeff: Sequence balancing coefficient; zero disables it.
        moe_router_bias_update_rate: Per-step text/image correction-bias update speed; zero disables it.
        moe_router_force_load_balancing: Enable deterministic destinations when using the new Gate family.
        dspark_depth: Trainable DSpark stage count; 0 disables the drafter.
        dspark_target_layer_ids: Backbone layers feeding the drafter; defaults
            to the last three.
        dspark_loss_coeff: Drafter objective weight.
        dspark_confidence_coeff: Confidence-head BCE weight.
        torch_dtype: Forward dtype accepted by the model builder.
        validate_placement: Enable DTensor placement validation.
        distributed_setup: Trainer-provided parallel topology.
        peft_config: Optional PEFT configuration.
        compile_config: Optional compilation configuration.
        activation_checkpoint: Normalized activation-checkpoint configuration.
        model_init_dtype: Final parameter initialization dtype.
        codegen: Select the codegen ``gen`` backend for this build.
        codegen_config: Optional codegen backend configuration.
        load_base_model: Load local HF weights from config_path.
        verify_hf_weights: Optional report directory for checking loaded weights
            against the HF checkpoint before the first forward or optimizer update.
    """
    config = build_deepseek_v41_validation_config(
        config_path, engram_assets_path,
        num_hidden_layers=num_hidden_layers, text_parameter_divisor=text_parameter_divisor,
        enable_vision=enable_vision, vision_num_hidden_layers=vision_num_hidden_layers,
        vision_parameter_divisor=vision_parameter_divisor, num_routed_experts=num_routed_experts,
        exercise_post_training_indexer=exercise_post_training_indexer,
        enable_candidate_pool=enable_candidate_pool, indexer_loss_coeff=indexer_loss_coeff,
        moe_aux_loss_coeff=moe_aux_loss_coeff, moe_router_bias_update_rate=moe_router_bias_update_rate,
        moe_router_force_load_balancing=moe_router_force_load_balancing,
        dspark_depth=dspark_depth, dspark_target_layer_ids=dspark_target_layer_ids,
        dspark_loss_coeff=dspark_loss_coeff, dspark_confidence_coeff=dspark_confidence_coeff,
        engram_storage_backend=engram_storage_backend, engram_max_pending_entries=engram_max_pending_entries,
        engram_max_sparse_rows_per_step=engram_max_sparse_rows_per_step,
        engram_sparse_adam_config=engram_sparse_adam_config,
    )
    if not enable_candidate_pool:
        config.v41_candidate_source_layer_id = -1
        config.v41_candidate_topk_blocks = 0
    return _run_validation_model_builder(
        config,
        torch_dtype=torch_dtype, validate_placement=validate_placement,
        distributed_setup=distributed_setup, peft_config=peft_config,
        compile_config=compile_config, activation_checkpoint=activation_checkpoint,
        model_init_dtype=model_init_dtype, codegen=codegen, codegen_config=codegen_config,
        pretrained_path=config_path if load_base_model else None,
        verify_hf_weights=verify_hf_weights,
    )


def _run_validation_model_builder(
        config: DeepseekV4Config,
        *,
        torch_dtype: str,
        validate_placement: bool,
        distributed_setup: DistributedSetup | None,
        peft_config: Any | None,
        compile_config: CompileConfig | dict[str, Any] | None,
        activation_checkpoint: Any | None,
        model_init_dtype: str,
        codegen: bool,
        codegen_config: Any | None,
        pretrained_path: str | None = None,
        verify_hf_weights: str | None = None,
) -> PreTrainedModel:
    """Invoke the model builder with the trainer wiring."""
    kwargs = {
        "distributed_setup": distributed_setup,
        "peft_config": peft_config,
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",
        "validate_placement": validate_placement,
        "compile_config": compile_config,
        "activation_checkpoint": activation_checkpoint,
        "model_init_dtype": model_init_dtype,
        "codegen": codegen,
        "codegen_config": codegen_config,
    }
    if pretrained_path is None:
        if verify_hf_weights is not None:
            raise ValueError("verify_hf_weights requires load_base_model=true")
        return HyperAutoModelForCausalLM.from_config(config, **kwargs)
    return _load_validation_hf_weights(config, pretrained_path, verify_hf_weights=verify_hf_weights, **kwargs)


def _load_validation_hf_weights(config: DeepseekV4Config, pretrained_path: str, *,
                                verify_hf_weights: str | None = None, **kwargs: Any) -> PreTrainedModel:
    """Load the selected text/vision prefix through the shared HF checkpointer."""
    source = Path(pretrained_path).expanduser().resolve()
    official = json.loads((source / "config.json").read_text(encoding="utf-8"))
    _validate_pretrained_dimensions(config, official)
    quantization = official.get("quantization_config")
    if quantization:
        validate_quantization_config(quantization)
        config.quantization_config = quantization
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    vision_layers = config.v41_vision_num_hidden_layers if config.v41_vision_enabled else 0
    names = select_layer_crop_parameters(list(index["weight_map"]),
                                         text_layers=config.num_hidden_layers, vision_layers=vision_layers)
    names.extend(name for name in ("hc_head_fn", "hc_head_base", "hc_head_scale") if name in index["weight_map"])
    if not config.v41_vision_enabled:
        names = [name for name in names if not name.endswith(".bias_vl")]
    if not names:
        raise ValueError("No checkpoint parameters match the requested validation model")
    selected = {name: index["weight_map"][name] for name in names}
    for name in names:
        if name.endswith("weight"):
            for suffix in ("scale", "weight_scale_inv"):
                scale_name = name[:-6] + suffix
                if scale_name in index["weight_map"]:
                    selected[scale_name] = index["weight_map"][scale_name]
    with tempfile.TemporaryDirectory(prefix="deepseek_v41_validation_hf_") as temporary:
        view = Path(temporary)
        config.save_pretrained(view)
        _write_hf_crop_index(source, view, selected, quantization)
        return HyperAutoModelForCausalLM.from_pretrained(
            str(view), verify_hf_weights=verify_hf_weights, **kwargs)


def _validate_pretrained_dimensions(config: DeepseekV4Config, source: dict[str, Any]) -> None:
    """Reject incompatible parameter crops before reading any weight payload."""
    text = source["text_config"]
    for name in ("hidden_size", "moe_intermediate_size", "num_attention_heads", "q_lora_rank", "o_lora_rank",
                 "index_n_heads"):
        actual = getattr(config, name)
        if actual != text[name]:
            raise ValueError(f"HF weight loading requires released {name}: expected {text[name]}, got {actual}")
    if config.num_local_experts != text["n_routed_experts"]:
        raise ValueError("HF weight loading requires the released num_routed_experts")
    if config.v41_dspark_depth or hasattr(config, "v41_validation_reindex_remap"):
        raise ValueError("HF weight loading does not support DSpark or remapped validation indexers")
    if config.v41_engram_layer_ids:
        _released_engram_assets(
            source, config.v41_engram_assets_path, allow_padding=True,
        )
    if config.v41_vision_enabled:
        for name in ("hidden_size", "intermediate_size", "num_attention_heads"):
            if getattr(config, f"v41_vision_{name}") != source["vision_config"][name]:
                raise ValueError(f"HF weight loading requires released vision {name}")


def build_layer_cropped_config(
        source: dict[str, Any], destination: Path, *, text_layers: int = 1, vision_layers: int = 2,
        engram_assets_path: str | None = None, engram_storage_backend: str = "device",
) -> Any:
    """Translate a released config, retaining a dependency-closed layer prefix.

    All parameter dimensions, expert counts, routing top-k, and modality widths
    retain their released values. Prefixes containing Engram require matching
    released hashing assets, rather than the scaled assets of a random-init crop.

    Args:
        source: Official composite DeepSeek-V4.1 configuration.
        destination: Directory for the translated config and its Engram assets.
        text_layers: Number of original decoder prefix layers to retain.
        vision_layers: Number of original vision prefix layers; zero disables vision.
        engram_assets_path: Released-size hashing assets required by retained Engram layers.
        engram_storage_backend: Keep Engram tables on the device or their EP owners' hosts.

    Returns:
        Translated configuration accepted by the production model class.

    Raises:
        ValueError: A crop is out of range or requires unavailable Engram assets.
    """
    text = source["text_config"]
    vision = source["vision_config"]
    if not 1 <= text_layers <= text["num_hidden_layers"]:
        raise ValueError("text_layers must select a nonempty prefix of the released decoder")
    if not 0 <= vision_layers <= vision["num_hidden_layers"]:
        raise ValueError("vision_layers must select a prefix of the released vision tower")
    retained_engram = [layer for layer in text["engram_layer_ids"] if layer < text_layers]
    if retained_engram and engram_assets_path is None:
        raise ValueError(f"Retained Engram layers require released hashing assets: {retained_engram}")
    destination.mkdir(parents=True, exist_ok=True)
    assets = {"source_model_type": source["model_type"], "num_hidden_layers": text_layers,
              "layer_ids": [], "num_embeddings": [], "head_dim": text["engram_head_dim"],
              "bucket_base": text["engram_vocab_size"], "table_pad_multiple": 1}
    if retained_engram:
        assets = _released_engram_assets(source, engram_assets_path, allow_padding=engram_storage_backend == "host")
    assets_path = (destination / "engram_assets.json").resolve()
    assets_path.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    # Share the model-owned translation while selecting divisor one; independent
    # crop math here could silently change checkpoint projection shapes.
    dimensions = _scale_text_dimensions(text, 1, assets)
    config = _build_text_config(source, text, dimensions, text_layers, text["n_routed_experts"])
    _apply_v41_extension_fields(config, source, text, assets, assets_path, text_layers, False, 0.001)
    config.v41_kv_source_layer_ids = [layer for layer in text["kv_source_layer_ids"] if layer < text_layers]
    config.v41_index_source_layer_ids = [layer for layer in text["index_source_layer_ids"] if layer < text_layers]
    if config.v41_candidate_source_layer_id >= text_layers:
        config.v41_candidate_source_layer_id = -1
        config.v41_candidate_topk_blocks = 0
    config.v41_dspark_depth = 0
    config.v41_dspark_target_layer_ids = []
    _apply_engram_host_fields(config, engram_storage_backend, 1000000, 1000000, None)
    _apply_v41_vision_fields(
        config, source, enable_vision=vision_layers > 0,
        vision_num_hidden_layers=vision_layers if vision_layers > 0 else None, vision_parameter_divisor=1,
    )
    config.v41_vision_num_hidden_layers = vision_layers
    config.save_pretrained(destination)
    return config


def select_layer_crop_parameters(
        names: list[str], *, text_layers: int = 1, vision_layers: int = 2,
) -> list[str]:
    """Select native HF parameter names for the same prefix and global modules.

    Args:
        names: Parameter names from the source HF checkpoint index.
        text_layers: Number of decoder prefix layers to retain.
        vision_layers: Number of visual prefix layers to retain.

    Returns:
        Source names belonging to retained layers and required global modules.
    """
    selected = []
    global_names = {"embed.weight", "head.weight", "norm.weight"}
    for name in names:
        if name.endswith((".scale", ".weight_scale_inv")):
            continue
        parts = name.split(".")
        if name in global_names:
            selected.append(name)
        elif len(parts) > 2 and parts[0] == "layers" and int(parts[1]) < text_layers:
            selected.append(name)
        elif vision_layers > 0:
            if name.startswith(("aligner.", "image_")):
                selected.append(name)
            elif name.startswith("vision.blocks."):
                if int(parts[2]) < vision_layers:
                    selected.append(name)
            elif name.startswith("vision."):
                selected.append(name)
    return selected


def _write_hf_crop_index(source: Path, view: Path, selected: dict[str, str], quantization: Any) -> None:
    """Link selected source shards and record their physical tensor byte sizes."""
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)
    total_size = 0
    for number, (shard, shard_names) in enumerate(by_shard.items()):
        path = (source / shard).resolve()
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in shard_names:
                tensor_slice = reader.get_slice(name)
                dtype = tensor_slice.get_dtype()
                if not quantization and dtype not in ("BF16", "F32"):
                    raise ValueError(f"Expected BF16/FP32 source parameter: {name}")
                bytes_per_element = {"BF16": 2, "F32": 4}.get(dtype, 1)
                total_size += math.prod(tensor_slice.get_shape()) * bytes_per_element
        link = view / f"source-{number:05d}.safetensors"
        link.symlink_to(path)
        for name in shard_names:
            selected[name] = link.name
    (view / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": selected}), encoding="utf-8")


__all__ = [
    "build_cropped_deepseek_v41",
    "build_deepseek_v41_validation_config",
    "build_layer_cropped_config",
    "select_layer_crop_parameters",
]
