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
"""Shared Transformers-config resolution for the Auto Parallel cost model.

Both the SAPP-ND parser (``CostModelParserHyperV2``) and the config-adapter
loader need the same model dimensions out of an AutoModels ``model`` section.
This module owns that translation exactly once, so the two callers cannot
drift apart on field names, aliases, or fallback behaviour.

It is also the single place that imports ``transformers``.  The import is
function-local because ``transformers`` is not a hard dependency of
``hyper_parallel`` (``requirements.txt`` only pins numpy) and the non-Hyper
cost-model backends must keep working without it.
"""
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# AutoModels root sections that identify the current Trainer schema.
_AUTO_MODELS_SECTIONS = ("training", "accelerator", "fsdp_config")

# ``model`` keys forwarded verbatim to ``AutoConfig.from_pretrained``.
_CONFIG_KWARG_NAMES = (
    "cache_dir", "local_files_only", "revision", "subfolder",
    "token", "trust_remote_code",
)

# Canonical text-tower fields, with the Transformers aliases that carry them.
# First populated alias wins, so Hyper-internal names keep priority over the
# Hugging Face spelling when a config happens to define both.
_TEXT_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("num_hidden_layers",),
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "head_dim": ("head_dim",),
    "intermediate_size": ("intermediate_size",),
    "vocab_size": ("vocab_size",),
    "max_position_embeddings": ("max_position_embeddings",),
    "num_experts": ("num_experts", "n_routed_experts"),
    "num_experts_per_tok": ("num_experts_per_tok",),
    "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
    "moe_intermediate_size": ("moe_intermediate_size",),
    "shared_expert_intermediate_size": ("shared_expert_intermediate_size",),
    "first_k_dense_replace": ("first_k_dense_replace",),
    "mtp_depth": ("mtp_depth", "num_nextn_predict_layers"),
    "multiple_of": ("multiple_of",),
    "ffn_dim_multiplier": ("ffn_dim_multiplier",),
    "kv_lora_rank": ("kv_lora_rank",),
    "q_lora_rank": ("q_lora_rank",),
    "qk_rope_head_dim": ("qk_rope_head_dim",),
}

# Vision towers use their own spelling for the shared concepts.
_VISION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("depth", "num_hidden_layers"),
    "num_attention_heads": ("num_heads", "num_attention_heads"),
    "intermediate_size": ("intermediate_size",),
    "out_hidden_size": ("out_hidden_size",),
    "patch_size": ("patch_size",),
    "spatial_merge_size": ("spatial_merge_size",),
    "num_position_embeddings": ("num_position_embeddings",),
}

# Fallback when a vision tower declares no positional-embedding grid.
_DEFAULT_VISUAL_SEQ_LEN = 1024


def is_auto_models_schema(mapping: Any) -> bool:
    """Return whether *mapping* looks like an AutoModels Trainer config.

    Accepts a plain mapping or any object exposing the root sections as
    attributes, so the SAPP-ND ``Config`` tree and a parsed YAML dict can be
    tested with the same call.
    """
    if isinstance(mapping, Mapping):
        return any(name in mapping for name in _AUTO_MODELS_SECTIONS)
    holder = getattr(mapping, "__dict__", None)
    if isinstance(holder, dict):
        return any(name in holder for name in _AUTO_MODELS_SECTIONS)
    return False


def _first_attr(config: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first populated attribute of *config* among *names*."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _text_tower(model_config: Any) -> Any:
    """Return the language-model sub-config of a composite config.

    Multimodal Transformers configs (``Qwen3VLMoeConfig`` and friends) keep
    the language model under ``text_config`` and expose none of its fields at
    the top level, so reading ``hidden_size`` directly raises AttributeError.
    """
    return getattr(model_config, "text_config", None) or model_config


def _spec_from_aliases(config: Any, aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """Collect the canonical fields declared by *config*."""
    spec = {}
    for canonical, names in aliases.items():
        value = _first_attr(config, names)
        if value is not None:
            spec[canonical] = value
    return spec


def _derive_shared_experts(spec: Dict[str, Any]) -> None:
    """Infer the shared-expert count from its total feed-forward width.

    Qwen2-MoE style configs declare one wide shared expert instead of a
    count.  The cost model only ever uses ``n_shared_exp * hff_exp``, so
    encoding that width as several narrow experts is width-equivalent.
    """
    if spec.get("num_shared_experts"):
        return
    moe_inter = int(spec.get("moe_intermediate_size", 0) or 0)
    shared_inter = int(spec.get("shared_expert_intermediate_size", 0) or 0)
    if moe_inter and shared_inter:
        spec["num_shared_experts"] = max(1, shared_inter // moe_inter)


def _visual_seq_len(vision_spec: Dict[str, Any], override: Optional[int]) -> int:
    """Resolve the encoder sequence length in merged visual tokens.

    The true count depends on the image resolution served by the dataset,
    which no configuration we read carries.  The positional-embedding grid
    divided by the spatial merge is the tightest bound available from the
    model alone, so it is the default and *override* exists to correct it.
    """
    if override:
        logger.info("visual sequence length taken from context.visual_seq_len: %d", override)
        return int(override)
    grid = int(vision_spec.get("num_position_embeddings", 0) or 0)
    merge = max(1, int(vision_spec.get("spatial_merge_size", 1) or 1))
    if not grid:
        logger.warning(
            "vision tower declares no num_position_embeddings; assuming %d visual tokens. "
            "Set context.visual_seq_len to the value your dataset produces.",
            _DEFAULT_VISUAL_SEQ_LEN,
        )
        return _DEFAULT_VISUAL_SEQ_LEN
    derived = max(1, grid // (merge * merge))
    logger.info(
        "visual sequence length derived from the vision config: %d "
        "(num_position_embeddings=%d, spatial_merge_size=%d). "
        "Set context.visual_seq_len to override.",
        derived, grid, merge,
    )
    return derived


def _get_hf_config(model_raw: Mapping[str, Any]) -> Any:
    """Call the AutoModels Transformers entry point for *model_raw*."""
    # Transformers is an optional dependency; import only when a config
    # actually needs it so the other cost-model backends stay importable.
    from hyper_parallel.auto_models._transformers.registry import get_hf_config  # pylint: disable=C0415

    config_kwargs = {
        name: model_raw[name]
        for name in _CONFIG_KWARG_NAMES
        if model_raw.get(name) is not None
    }
    return get_hf_config(
        str(model_raw.get("pretrained_model_name_or_path")),
        str(model_raw.get("attn_implementation", "sdpa")),
        model_raw.get("torch_dtype", "auto"),
        **config_kwargs,
    )


def _explicit_overrides(model_raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``config_overrides`` fallback as a plain dict."""
    overrides = model_raw.get("config_overrides")
    return dict(overrides) if isinstance(overrides, Mapping) else {}


def resolve_hf_model_spec(
    model_raw: Mapping[str, Any],
    visual_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Return canonical cost-model fields for a Trainer ``model`` section.

    Resolves ``model.pretrained_model_name_or_path`` through the same
    Transformers path AutoModels uses.  A composite (vision-language) config
    contributes its language tower under the canonical keys and its vision
    tower under ``"vision"``; a plain config leaves ``"vision"`` absent.

    ``model.config_overrides`` stays supported for standalone cost-model
    search files, and doubles as the fallback when the Transformers config
    cannot be reached (offline node, unreachable repository).  Explicit
    overrides always win over resolved values.

    Args:
        model_raw: The ``model`` section, as a plain mapping.
        visual_seq_len: Optional override for the encoder sequence length.

    Returns:
        A dict of canonical model fields, always carrying ``"name"``.

    Raises:
        ValueError: If neither a pretrained path nor overrides can supply
            the model dimensions.
    """
    explicit = _explicit_overrides(model_raw)
    model_path = model_raw.get("pretrained_model_name_or_path")

    if not model_path:
        if explicit:
            explicit.setdefault("name", model_raw.get("name", "custom"))
            return explicit
        raise ValueError(
            "AutoModels train.yaml requires model.pretrained_model_name_or_path "
            "or model.config_overrides for Auto Parallel search"
        )

    try:
        model_config = _get_hf_config(model_raw)
    except (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        if explicit:
            logger.warning(
                "Transformers config resolution failed (%s); "
                "falling back to model.config_overrides", exc,
            )
            explicit.setdefault("name", model_raw.get("name", "custom"))
            return explicit
        raise ValueError(
            f"cannot resolve model.pretrained_model_name_or_path '{model_path}'; "
            "install transformers, set model.config_overrides, or make the config "
            "available offline (warm the HF_HOME cache, or pass local_files_only)"
        ) from exc

    spec = _spec_from_aliases(_text_tower(model_config), _TEXT_FIELD_ALIASES)
    _derive_shared_experts(spec)
    spec["name"] = str(getattr(model_config, "model_type", None) or model_path)

    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        vision_spec = _spec_from_aliases(vision_config, _VISION_FIELD_ALIASES)
        vision_spec["name"] = f"{spec['name']}_vision"
        vision_spec["max_position_embeddings"] = _visual_seq_len(vision_spec, visual_seq_len)
        spec["vision"] = vision_spec

    spec.update(explicit)
    return spec
