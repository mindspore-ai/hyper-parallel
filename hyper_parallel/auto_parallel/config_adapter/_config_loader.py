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
"""Configuration loader for auto parallel strategy search.

Reads Search Config (``search.yaml``) and HyperParallel training config
(``train.yaml``) files, producing :class:`NormalizedConfig` instances.
"""

import logging
import os
from typing import Any, Dict, List, Tuple, Optional

try:
    import yaml  # type: ignore[import-untyped]  # pylint: disable=C0415
except ImportError:
    yaml = None  # pragma: no cover

from hyper_parallel.auto_parallel.config_adapter._normalized_config import NormalizedConfig

logger = logging.getLogger(__name__)

# Mapping from HuggingFace-style config_overrides keys to internal names.
# After field-name alignment, most HP YAML keys already match the internal
# model_spec names.  Only ``seq_length`` (MF variant) needs remapping.
_HP_TO_INTERNAL: Dict[str, str] = {
    "seq_length": "max_position_embeddings",
}

_HF_MODEL_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "intermediate_size",
    "num_key_value_heads",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "num_shared_experts",
    "moe_intermediate_size",
    "first_k_dense_replace",
    "mtp_depth",
    "multiple_of",
    "ffn_dim_multiplier",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
)


def _normalize_model_spec(model_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Rename non-standard config overrides keys to canonical names.

    Only maps a key if the target name is not already present, so
    explicit canonical names in the YAML take precedence.
    """
    for hf_key, internal_key in _HP_TO_INTERNAL.items():
        if hf_key in model_spec and internal_key not in model_spec:
            model_spec[internal_key] = model_spec.pop(hf_key)
    return model_spec


# Mapping from short dimension names (used in search config YAML parallelism section)
# to canonical NormalizedConfig search_space keys.
_UNIFIED_DIM_MAP: Dict[str, str] = {
    "dp": "data_parallel_replicate_degree",
    "fsdp": "data_parallel_shard_degree",
    "tp": "tensor_parallel_degree",
    "pp": "pipeline_parallel_degree",
    "cp": "context_parallel_degree",
    "ep": "expert_parallel_degree",
    "etp": "expert_tensor_parallel_degree",
    "micro_batch_num": "micro_batch_num",
}

# Mapping from short dimension names to constraint fixed_*_degree keys.
_FIXED_DIM_MAP: Dict[str, str] = {
    "dp": "fixed_dp_degree",
    "fsdp": "fixed_fsdp_degree",
    "tp": "fixed_tp_degree",
    "pp": "fixed_pp_degree",
    "cp": "fixed_cp_degree",
    "ep": "fixed_ep_degree",
    "etp": "fixed_etp_degree",
    "micro_batch_num": "fixed_micro_batch_num",
}


def _get_dict(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return the value of a key if it is a dict, otherwise an empty dict."""
    val = raw.get(key, {})
    return val if isinstance(val, dict) else {}


def _first_attr(config: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first populated attribute from a Transformers config."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _load_auto_models_model_spec(model_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model dimensions through the AutoModels Transformers path."""
    model_path = model_raw.get("pretrained_model_name_or_path")
    explicit = model_raw.get("config_overrides", {})
    explicit = dict(explicit) if isinstance(explicit, dict) else {}
    if not model_path:
        if explicit:
            explicit.setdefault("name", model_raw.get("name", "custom"))
            return _normalize_model_spec(explicit)
        raise ValueError(
            "AutoModels train.yaml requires model.pretrained_model_name_or_path "
            "or model.config_overrides for Auto Parallel search"
        )

    # Transformers is optional for non-Hyper backends, so import it only when
    # an AutoModels train.yaml needs its model metadata.
    from hyper_parallel.auto_models._transformers.registry import get_hf_config  # pylint: disable=C0415

    config_kwargs = {
        name: model_raw[name]
        for name in (
            "cache_dir", "local_files_only", "revision", "subfolder",
            "token", "trust_remote_code",
        )
        if model_raw.get(name) is not None
    }
    model_config = get_hf_config(
        str(model_path),
        str(model_raw.get("attn_implementation", "sdpa")),
        model_raw.get("torch_dtype", "auto"),
        **config_kwargs,
    )
    model_spec = {
        name: getattr(model_config, name)
        for name in _HF_MODEL_FIELDS
        if getattr(model_config, name, None) is not None
    }
    model_spec["name"] = str(
        getattr(model_config, "model_type", None) or model_path
    )
    model_spec["num_experts"] = int(_first_attr(
        model_config, ("num_experts", "n_routed_experts"), 1,
    ))
    model_spec["num_shared_experts"] = int(_first_attr(
        model_config, ("num_shared_experts", "n_shared_experts"), 0,
    ))
    moe_intermediate_size = int(model_spec.get("moe_intermediate_size", 0) or 0)
    shared_intermediate_size = int(
        getattr(model_config, "shared_expert_intermediate_size", 0) or 0
    )
    if (
        not model_spec["num_shared_experts"]
        and moe_intermediate_size
        and shared_intermediate_size
    ):
        model_spec["num_shared_experts"] = max(
            1, shared_intermediate_size // moe_intermediate_size,
        )
    model_spec.update(explicit)
    return _normalize_model_spec(model_spec)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Read and parse a YAML file, returning the raw dict."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read HyperParallel YAML configs. "
            "Install it with: pip install pyyaml"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".yaml", ".yml"):
        raise ValueError(
            f"Unsupported config file format: {ext!r}. "
            "Supported formats: .yaml, .yml"
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML file {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file {path} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )
    return raw


# ── Search Config YAML reader (primary) ──────────────────────────────


def _parse_unified_parallelism(
    para_raw: Dict[str, Any],
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Convert the unified parallelism declaration into search_space + constraint.

    Rules:
    *   Scalar integer value → fixed dimension (both ``constraint.fixed_*``
        and ``search_space`` with a single-element list).
    *   List value → search candidates (placed into ``search_space`` only).
    *   String ``"auto"`` → dimension is left to the searcher's ``bound_space``
        (neither fixed nor explicitly enumerated).

    Returns:
        A ``(search_space, constraint)`` tuple.
    """
    search_space: Dict[str, List[int]] = {}
    constraint: Dict[str, Any] = {}

    for short_key, canonical_key in _UNIFIED_DIM_MAP.items():
        if short_key not in para_raw:
            continue
        value = para_raw[short_key]

        if isinstance(value, int):
            constraint[_FIXED_DIM_MAP[short_key]] = value
            search_space[canonical_key] = [value]
        elif isinstance(value, list):
            search_space[canonical_key] = [int(v) for v in value]
        elif isinstance(value, str) and value.strip().lower() == "auto":
            continue

    return search_space, constraint


def _build_config_from_search_yaml(raw: Dict[str, Any]) -> NormalizedConfig:
    """Construct a NormalizedConfig from a parsed Search Config YAML dict.

    Supports two modes:

    * **Standalone** (no ``train_yaml``) — ``model`` section and all other
      info must be present in the search config file.
    * **With train_yaml** — loads the specified ``train.yaml`` for model
      parameters and current parallelism values, then overlays the search
      config's ``cluster``, ``parallelism``, and ``constraint`` sections.

    Undeclared parallelism dimensions are inherited from ``train.yaml``
    as fixed (single-element) entries.
    """
    train_yaml_path = raw.get("train_yaml")
    base_config: Optional[NormalizedConfig] = None

    if train_yaml_path:
        if not isinstance(train_yaml_path, str):
            raise ValueError("'train_yaml' must be a file path string")
        base_raw = _load_yaml(train_yaml_path)
        base_config = _build_config_from_hp_yaml(base_raw)

    model_spec: Dict[str, Any]
    if base_config:
        model_spec = dict(base_config.model_spec)
    else:
        model_spec = {}

    # Override or supply model section from search.yaml
    search_model = _get_dict(raw, "model")
    if search_model:
        model_spec.update(search_model)

    model_spec.setdefault("max_position_embeddings", 4096)
    model_spec.setdefault("local_batch_size", 1)

    model_spec = _normalize_model_spec(model_spec)

    cluster_spec = _get_dict(raw, "cluster")

    pp_raw = _get_dict(raw, "pp_config")
    parallelism_raw = _get_dict(raw, "parallelism")
    constraint_raw = _get_dict(raw, "constraint")

    search_space, parallelism_constraint = _parse_unified_parallelism(parallelism_raw)

    # Inherit undeclared dimensions from train.yaml as fixed values.
    if base_config:
        for space_key, candidates in base_config.search_space.items():
            if space_key not in search_space:
                search_space[space_key] = candidates

        if constraint_raw.get("global_batch_size", 0) is None or constraint_raw.get("global_batch_size", 0) == 0:
            constraint_raw.setdefault(
                "global_batch_size", base_config.constraint.get("global_batch_size", 0)
            )

    pp_config: Dict[str, Any] = {
        "pp_degree": pp_raw.get("pp_degree",
                                 parallelism_raw.get("pp", 1)),
        "stage_partition_mode": pp_raw.get("stage_partition_mode", "uniform"),
        "stage_partition": pp_raw.get("stage_partition", []),
        "layer_offset_range": tuple(pp_raw.get("layer_offset_range", [0, 0])),
        "layer_recompute_layers": pp_raw.get("layer_recompute_layers", []),
        "micro_batch_num": pp_raw.get("micro_batch_num", 1),
        "pp_interleave_num": pp_raw.get("pp_interleave_num", 1),
        "pipeline_parallel_schedule": pp_raw.get("pipeline_schedule", "1F1B"),
    }

    estimator: Dict[str, Any] = {
        "type": "symbolic",
        "recompute_strategy": str(raw.get("recompute", "none")),
        "enable_profiling_calibration": False,
    }

    constraint: Dict[str, Any] = {
        "global_batch_size": constraint_raw.get("global_batch_size", 0),
        "memory_limit_gb": constraint_raw.get("memory_limit_gb", 0.0),
        **parallelism_constraint,
    }

    return NormalizedConfig(
        model_spec=model_spec,
        cluster_spec=cluster_spec,
        search_space=search_space,
        constraint=constraint,
        estimator=estimator,
        pp_config=pp_config,
    )


def read_search_config(path: str) -> NormalizedConfig:
    """Read a Search Config YAML file and return a :class:`NormalizedConfig`.

    The Search Config YAML format uses a unified ``parallelism`` section
    where each dimension is declared as::

        parallelism:
          tp: 4              # scalar → fixed input
          dp: [1, 2, 4]      # list → search candidate
          pp: auto            # string → let the searcher decide

    To reuse model parameters from an existing ``train.yaml`` without
    duplicating them, set the ``train_yaml`` key::

        train_yaml: "./train.yaml"   # load model params from here
        cluster:
          num_nodes: 4
          cards_per_node: 8
        parallelism:
          dp: [1, 2, 4]
          tp: [1, 2, 4, 8]

    Dimensions absent from ``parallelism`` are inherited from
    ``train.yaml`` as fixed values.  A ``model`` section in the search
    config overrides values read from ``train_yaml``.

    See ``auto_parallel/examples/dense_llm_search.yaml`` for a complete
    standalone example.

    Args:
        path: Path to the YAML config file (``.yaml`` or ``.yml``).

    Returns:
        A :class:`NormalizedConfig` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed, or if ``cluster``
            is missing when ``train_yaml`` is not used.
        ImportError: If PyYAML is not installed.
    """
    raw = _load_yaml(path)
    return _build_config_from_search_yaml(raw)


# ── HyperParallel training YAML reader (secondary) ──────────────────

_ACCEL_TO_SEARCH = {
    "dp_shard": "data_parallel_shard_degree",
    "dp_replicate": "data_parallel_replicate_degree",
    "tp_degree": "tensor_parallel_degree",
    "pipeline_parallel_degree": "pipeline_parallel_degree",
    "context_parallel_degree": "context_parallel_degree",
    "expert_parallel_degree": "expert_parallel_degree",
    "expert_tensor_parallel_degree": "expert_tensor_parallel_degree",
}

_AUTO_MODELS_ACCEL_TO_SEARCH = {
    "tp_size": "tensor_parallel_degree",
    "pp_size": "pipeline_parallel_degree",
    "cp_size": "context_parallel_degree",
    "ep_size": "expert_parallel_degree",
}


def _build_config_from_auto_models_yaml(raw: Dict[str, Any]) -> NormalizedConfig:
    """Construct a normalized config from the current AutoModels schema."""
    model_raw = _get_dict(raw, "model")
    training_raw = _get_dict(raw, "training")
    accelerator_raw = _get_dict(raw, "accelerator")
    fsdp_raw = _get_dict(raw, "fsdp_config")
    activation_raw = _get_dict(raw, "activation_checkpoint")
    dataset_raw = _get_dict(raw, "dataset")
    data_transform_raw = _get_dict(dataset_raw, "data_transform")

    model_spec = _load_auto_models_model_spec(model_raw)
    model_spec["max_position_embeddings"] = data_transform_raw.get(
        "max_seq_len", model_spec.get("max_position_embeddings", 4096),
    )
    model_spec["local_batch_size"] = training_raw.get("micro_batch_size", 1)
    model_spec["compute_dtype"] = model_raw.get("torch_dtype", "bfloat16")

    search_space: Dict[str, List[int]] = {}
    dp_shard_size = fsdp_raw.get("dp_shard_size")
    if dp_shard_size is not None:
        search_space["data_parallel_shard_degree"] = [int(dp_shard_size)]
    for field_name, search_name in _AUTO_MODELS_ACCEL_TO_SEARCH.items():
        value = accelerator_raw.get(field_name)
        if value is not None:
            search_space[search_name] = [int(value)]

    global_batch_size = int(training_raw.get("global_batch_size", 0) or 0)
    local_batch_size = int(model_spec["local_batch_size"] or 1)
    data_parallel_size = int(dp_shard_size or 1)
    micro_batch_num = (
        global_batch_size // (local_batch_size * data_parallel_size)
        if global_batch_size
        and global_batch_size % (local_batch_size * data_parallel_size) == 0
        else 1
    )
    pp_degree = max(1, int(accelerator_raw.get("pp_size", 1) or 1))

    mode = str(activation_raw.get("mode", "off"))
    recompute_map = {
        "off": "none",
        "none": "none",
        "full": "full",
        "selective": "selective",
    }
    return NormalizedConfig(
        model_spec=model_spec,
        cluster_spec={},
        search_space=search_space,
        constraint={
            "global_batch_size": global_batch_size,
            "memory_limit_gb": 0.0,
        },
        estimator={
            "type": "symbolic",
            "recompute_strategy": recompute_map.get(mode, "none"),
        },
        pp_config={
            "pp_degree": pp_degree,
            "stage_partition_mode": "uniform",
            "micro_batch_num": max(1, micro_batch_num),
        },
    )


def _build_config_from_hp_yaml(raw: Dict[str, Any]) -> NormalizedConfig:
    """Construct a NormalizedConfig from a parsed HyperParallel YAML dict.

    Extracts current AutoModels fields when root-level ``training`` and
    ``accelerator`` sections are present. The legacy ``model.name`` /
    ``model.config_overrides`` schema remains supported for compatibility.

    Legacy parsing extracts model identifiers from ``model.config_overrides``,
    parallelism from ``train.accelerator.*``, batch settings from ``train.*``,
    sequence length from ``data.max_seq_len``, and recompute mode from
    ``train.gradient_checkpointing.activation_checkpoint``.

    Model hyperparameters are extracted from ``model.config_overrides``.
    """
    if "training" in raw or "accelerator" in raw or "fsdp_config" in raw:
        return _build_config_from_auto_models_yaml(raw)

    model_raw = _get_dict(raw, "model")
    train_raw = _get_dict(raw, "train")
    data_raw = _get_dict(raw, "data")
    accel_raw = _get_dict(train_raw, "accelerator")
    gc_raw = _get_dict(train_raw, "gradient_checkpointing")

    # --- model_spec ---
    model_spec: Dict[str, Any] = {}
    model_spec["name"] = model_raw.get("name", "unknown")
    overrides = model_raw.get("config_overrides", {})
    if isinstance(overrides, dict):
        model_spec.update(overrides)
    model_spec["max_position_embeddings"] = data_raw.get("max_seq_len", 4096)
    model_spec["local_batch_size"] = train_raw.get("micro_batch_size", 1)

    # dtype from train.mixed_precision
    mp_raw = _get_dict(train_raw, "mixed_precision")
    if mp_raw.get("enabled", True):
        model_spec["compute_dtype"] = mp_raw.get("param_dtype", "bfloat16")

    # --- cluster_spec (users should set via search config or directly) ---
    cluster_spec: Dict[str, Any] = {}

    # --- search_space from train.accelerator ---
    search_space: Dict[str, List[int]] = {}
    for hkey, skey in _ACCEL_TO_SEARCH.items():
        val = accel_raw.get(hkey)
        if val is not None:
            search_space[skey] = [int(val)]

    # --- constraint from train ---
    gbs = train_raw.get("global_batch_size", 0)
    constraint: Dict[str, Any] = {
        "global_batch_size": gbs or 0,
        "memory_limit_gb": 0.0,
    }
    mb_num = int(gbs) // int(model_spec["local_batch_size"]) if gbs and model_spec.get("local_batch_size") else 1

    # --- pp_config ---
    pp_degree = accel_raw.get("pipeline_parallel_degree", 1)
    pp_degree = max(1, int(pp_degree) if pp_degree else 1)
    pp_config: Dict[str, Any] = {
        "pp_degree": pp_degree,
        "stage_partition_mode": "uniform",
        "micro_batch_num": max(1, mb_num // pp_degree),
    }

    # --- estimator from gradient_checkpointing ---
    ac_mode = str(gc_raw.get("activation_checkpoint", "none"))
    recompute_map = {"none": "none", "full": "full", "selective": "selective"}
    estimator: Dict[str, Any] = {
        "type": "symbolic",
        "recompute_strategy": recompute_map.get(ac_mode, "none"),
    }

    model_spec = _normalize_model_spec(model_spec)

    return NormalizedConfig(
        model_spec=model_spec,
        cluster_spec=cluster_spec,
        search_space=search_space,
        constraint=constraint,
        estimator=estimator,
        pp_config=pp_config,
    )


def read_hp_yaml_config(path: str) -> NormalizedConfig:
    """Read a HyperParallel YAML configuration file.

    For the current AutoModels Trainer schema, model dimensions are resolved
    from ``model.pretrained_model_name_or_path`` through the shared
    Transformers config path. Parallelism is read from root-level
    ``accelerator`` / ``fsdp_config`` sections. The legacy
    ``model.config_overrides`` / ``train.accelerator`` schema remains
    supported for standalone cost-model configurations.

    .. note::
        Cluster configuration is **not** present in ``train.yaml``.
        To perform a full strategy search, use :func:`read_search_config`
        which accepts cluster and search-space parameters.

    See :func:`_build_config_from_hp_yaml` for the full list of recognised
    YAML sections.

    Args:
        path: Path to the YAML config file (``.yaml`` or ``.yml``).

    Returns:
        A :class:`NormalizedConfig` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
        ImportError: If PyYAML is not installed.
    """
    raw = _load_yaml(path)
    return _build_config_from_hp_yaml(raw)
