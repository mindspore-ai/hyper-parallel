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
"""Strategy output writer for auto parallel strategy search.

Serializes :class:`NormalizedConfig` instances to JSON or YAML files,
generates human-readable summaries, and provides a PPB (PR524)
configuration stub.
"""

import json
import logging
import os
from typing import Any, Dict

import yaml  # type: ignore[import-untyped]

from hyper_parallel.auto_parallel.config_adapter._normalized_config import NormalizedConfig


logger = logging.getLogger(__name__)


def _resolve_pp_degree(pp_cfg: Dict) -> int:
    """Resolve pp_degree to a scalar from pp_config."""
    pp_degree_val = pp_cfg.get("pp_degree", 1)
    if isinstance(pp_degree_val, list):
        return pp_degree_val[0] if pp_degree_val else 1
    return pp_degree_val


def _write_json(data: Dict[str, Any], output_path: str) -> None:
    """Write a dict as a pretty-printed JSON file."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.write("\n")


# Mapping from resolved strategy keys to HP YAML accelerator field names.
_YAML_KEY_MAP: Dict[str, str] = {
    "dp_shard": "dp_shard",
    "dp_replicate": "dp_replicate",
    "data_parallel_shard_degree": "dp_shard",
    "data_parallel_replicate_degree": "dp_replicate",
    "data_parallel_degree": "dp_replicate",
    "dp": "dp_replicate",
    "tensor_parallel_degree": "tp_degree",
    "tp_degree": "tp_degree",
    "tp": "tp_degree",
    "pipeline_parallel_degree": "pipeline_parallel_degree",
    "context_parallel_degree": "context_parallel_degree",
    "expert_parallel_degree": "expert_parallel_degree",
    "expert_tensor_parallel_degree": "expert_tensor_parallel_degree",
    "pp_degree": "pipeline_parallel_degree",
    "pp": "pipeline_parallel_degree",
    "cp_degree": "context_parallel_degree",
    "cp": "context_parallel_degree",
    "ep_degree": "expert_parallel_degree",
    "ep": "expert_parallel_degree",
    "etp_degree": "expert_tensor_parallel_degree",
    "etp": "expert_tensor_parallel_degree",
}

_AUTO_MODELS_YAML_KEY_MAP: Dict[str, str] = {
    "tensor_parallel_degree": "tp_size",
    "tp_degree": "tp_size",
    "tp": "tp_size",
    "pipeline_parallel_degree": "pp_size",
    "pp_degree": "pp_size",
    "pp": "pp_size",
    "context_parallel_degree": "cp_size",
    "cp_degree": "cp_size",
    "cp": "cp_size",
    "expert_parallel_degree": "ep_size",
    "ep_degree": "ep_size",
    "ep": "ep_size",
}


def _validate_strategy_and_yaml(
    config: NormalizedConfig,
    original_yaml_path: str,
) -> None:
    """Validate that resolved_strategy is set and the original YAML exists."""
    if config.resolved_strategy is None:
        raise ValueError(
            "config.resolved_strategy is None — no strategy to write. "
            "Set config.resolved_strategy first."
        )
    if not os.path.isfile(original_yaml_path):
        raise FileNotFoundError(
            f"Original YAML not found: {original_yaml_path}"
        )


def _load_yaml_to_inject(original_yaml_path: str) -> Dict[str, Any]:
    """Load the original YAML and initialize its strategy sections."""
    with open(original_yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None or not isinstance(data, dict):
        raise ValueError(
            f"Original YAML {original_yaml_path} must contain a top-level mapping."
        )
    if "training" in data or "accelerator" in data or "fsdp_config" in data:
        if not isinstance(data.get("training"), dict):
            data["training"] = {}
        if not isinstance(data.get("accelerator"), dict):
            data["accelerator"] = {}
        if not isinstance(data.get("fsdp_config"), dict):
            data["fsdp_config"] = {}
    else:
        if "train" not in data or not isinstance(data["train"], dict):
            data["train"] = {}
        if "accelerator" not in data["train"] or not isinstance(data["train"]["accelerator"], dict):
            data["train"]["accelerator"] = {}
    return data


def _inject_resolved_strategy(data: Dict[str, Any], resolved: Dict[str, Any]) -> None:
    """Inject resolved strategy values into the YAML data dict."""
    if "training" in data or "fsdp_config" in data:
        accelerator = data["accelerator"]
        for src_key, dst_key in _AUTO_MODELS_YAML_KEY_MAP.items():
            if src_key in resolved:
                accelerator[dst_key] = int(resolved[src_key])

        dp_shard_size = resolved.get("dp_shard")
        if dp_shard_size is None:
            dp_shard_size = resolved.get(
                "data_parallel_shard_degree",
                resolved.get("dp"),
            )
        if dp_shard_size is not None:
            data["fsdp_config"]["dp_shard_size"] = int(dp_shard_size)
        if "global_batch_size" in resolved:
            data["training"]["global_batch_size"] = int(
                resolved["global_batch_size"]
            )
        return

    train = data["train"]
    accel = train["accelerator"]

    for src_key, dst_key in _YAML_KEY_MAP.items():
        if src_key in resolved:
            accel[dst_key] = int(resolved[src_key])

    if "global_batch_size" in resolved:
        train["global_batch_size"] = int(resolved["global_batch_size"])

    if "micro_batch_num" in resolved:
        train["micro_batch_num"] = int(resolved["micro_batch_num"])


def _write_output_yaml(
    data: Dict[str, Any],
    output_path: str,
    overwrite: bool,
    original_yaml_path: str,
) -> None:
    """Write the data dict to the output YAML file."""
    write_path = original_yaml_path if overwrite else output_path

    parent_dir = os.path.dirname(os.path.abspath(write_path))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(write_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info(
        "Resolved YAML written to %s (overwrite=%s)",
        write_path, overwrite,
    )


def write_strategy_config(
    config: NormalizedConfig,
    output_path: str,
    fmt: str = "json",
) -> None:
    """Write a normalized configuration to a file.

    Args:
        config: The normalized configuration to serialize.
        output_path: Destination file path.
        fmt: Output format, ``"json"`` (default) only.

    Raises:
        ValueError: If the output format is unsupported.
        OSError: If the file cannot be written.
    """
    data = config.to_dict()

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    try:
        if fmt == "json":
            _write_json(data, output_path)
        else:
            raise ValueError(
                f"Unsupported output format: {fmt!r}. "
                "Supported format: 'json'"
            )
    except OSError as exc:
        raise OSError(f"Failed to write config to {output_path}: {exc}") from exc

    logger.info("Strategy config written to %s", output_path)


def write_resolved_strategy(
    config: NormalizedConfig,
    output_path: str,
    fmt: str = "json",
) -> None:
    """Write only the resolved strategy to a file.

    This produces a compact config snippet suitable for consumption
    by training scripts.

    Args:
        config: The normalized configuration (must have ``resolved_strategy`` set).
        output_path: Destination file path.
        fmt: Output format, ``"json"`` (default) only.

    Raises:
        ValueError: If ``resolved_strategy`` is ``None``.
    """
    if config.resolved_strategy is None:
        raise ValueError("config.resolved_strategy is None — no strategy to write")

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    data = {"resolved_strategy": dict(config.resolved_strategy)}

    try:
        if fmt == "json":
            _write_json(data, output_path)
        else:
            raise ValueError(
                f"Unsupported output format: {fmt!r}. "
                "Supported format: 'json'"
            )
    except OSError as exc:
        raise OSError(f"Failed to write resolved strategy to {output_path}: {exc}") from exc

    logger.info("Resolved strategy written to %s", output_path)


def write_ppb_config(
    config: NormalizedConfig,
    output_path: str,
) -> None:
    """Write a PPB (PR524 pipeline balancer) input JSON file.

    Writes the PP-relevant fields from ``NormalizedConfig``
    in a structure compatible with ``args_for_pipeline_parallel.json``.

    Args:
        config: The normalized configuration.
        output_path: Destination JSON file path.

    Raises:
        OSError: If the file cannot be written.
    """
    model = config.model_spec
    pp_cfg = config.pp_config
    constraint = config.constraint

    ppb_data = {
        "llm_class": "0",
        "train_yaml": "",
        "mindformers_dir": "",
        "layer_ratio": 0.33,
        "backward_ratio": 2.0,
        "head_loss": 1.5,
        "recompute_ratio": 1,
        "time_limit": 2147483647,
        "dryrun": True,
        "check": True,
        "fit_level": 0,
        "extract": False,
        "env_json": "./config/env_config.json",
        "dryrun_lim": 16,
        "_hyper_model": {
            "num_hidden_layers": model.get("num_hidden_layers", 0),
            "hidden_size": model.get("hidden_size", 0),
            "num_attention_heads": model.get("num_attention_heads", 0),
            "vocab_size": model.get("vocab_size", 0),
            "max_position_embeddings": model.get("max_position_embeddings", 4096),
            "moe_enabled": model.get("moe_enabled", False),
            "num_experts": model.get("num_experts", 1),
        },
        "_hyper_pp": {
            "pp_degree": _resolve_pp_degree(pp_cfg),
            "stage_partition_mode": pp_cfg.get("stage_partition_mode", "uniform"),
            "micro_batch_num": pp_cfg.get("micro_batch_num", 1),
            "layer_offset_range": list(pp_cfg.get("layer_offset_range", (0, 0))),
            "layer_recompute_layers": pp_cfg.get("layer_recompute_layers", []),
        },
        "_hyper_constraint": {
            "global_batch_size": constraint.get("global_batch_size", 0),
            "memory_limit_gb": constraint.get("memory_limit_gb", 0.0),
        },
    }

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(ppb_data, fh, indent=4)
            fh.write("\n")
    except OSError as exc:
        raise OSError(f"Failed to write PPB config to {output_path}: {exc}") from exc

    logger.info("PPB config stub written to %s", output_path)


def write_resolved_yaml(
    config: NormalizedConfig,
    original_yaml_path: str,
    output_path: str,
    overwrite: bool = False,
) -> None:
    """Write a resolved strategy into a HyperParallel YAML file.

    Copies the original ``train.yaml`` and replaces the parallel
    dimension fields with the resolved strategy values.  This produces
    a complete, immediately launchable training configuration.

    The resolved strategy is read from ``config.resolved_strategy``.
    Supported keys: ``dp_shard``, ``dp_replicate``, ``tp_degree``,
    ``pipeline_parallel_degree``, ``context_parallel_degree``,
    ``expert_parallel_degree``, ``global_batch_size``.

    Args:
        config: NormalizedConfig with ``resolved_strategy`` set.
        original_yaml_path: Path to the original ``train.yaml``.
        output_path: Destination file path for the resolved YAML.
        overwrite: If ``True``, write to ``original_yaml_path`` instead
            of ``output_path`` (default ``False``).

    Raises:
        ValueError: If ``resolved_strategy`` is ``None``.
        FileNotFoundError: If ``original_yaml_path`` does not exist.
    """
    _validate_strategy_and_yaml(config, original_yaml_path)
    data = _load_yaml_to_inject(original_yaml_path)
    _inject_resolved_strategy(data, config.resolved_strategy)
    _write_output_yaml(data, output_path, overwrite, original_yaml_path)


def normalized_to_summary(config: NormalizedConfig) -> Dict[str, Any]:
    """Generate a human-readable summary of the configuration.

    Args:
        config: The normalized configuration to summarize.

    Returns:
        A dictionary with summary fields suitable for logging or display.
    """
    model = config.model_spec
    cluster = config.cluster_spec
    search = config.search_space
    constraint = config.constraint
    estimator = config.estimator
    pp_cfg = config.pp_config

    return {
        "model": {
            "name": model.get("name", "unknown"),
            "num_hidden_layers": model.get("num_hidden_layers", 0),
            "hidden_size": model.get("hidden_size", 0),
            "intermediate_size": model.get("intermediate_size", 0),
            "num_attention_heads": model.get("num_attention_heads", 0),
            "num_key_value_heads": model.get("num_key_value_heads", 0),
            "vocab_size": model.get("vocab_size", 0),
            "max_position_embeddings": model.get("max_position_embeddings", 0),
            "moe_enabled": model.get("moe_enabled", False),
            "num_experts": model.get("num_experts", 0),
        },
        "cluster": {
            "num_nodes": cluster.get("num_nodes", 0),
            "cards_per_node": cluster.get("cards_per_node", 0),
            "total_cards": (
                cluster.get("num_nodes", 0) * cluster.get("cards_per_node", 0)
            ),
            "device_memory_gb": cluster.get("device_memory_gb", 0),
            "device_type": cluster.get("device_type", "unknown"),
        },
        "search_space": dict(sorted(search.items())),
        "constraints": {
            "global_batch_size": constraint.get("global_batch_size", 0),
            "memory_limit_gb": constraint.get("memory_limit_gb", 0.0),
            "fixed_dimensions": {
                k: constraint.get(f"fixed_{k}_degree")
                for k in ("dp", "tp", "pp", "cp", "ep")
                if constraint.get(f"fixed_{k}_degree") is not None
            },
        },
        "estimator": {
            "type": estimator.get("type", "symbolic"),
            "recompute_strategy": estimator.get("recompute_strategy", "none"),
        },
        "pipeline": {
            "pp_degree": _resolve_pp_degree(pp_cfg),
            "stage_partition_mode": pp_cfg.get("stage_partition_mode", "uniform"),
            "micro_batch_num": pp_cfg.get("micro_batch_num", 1),
        },
        "resolved_strategy": config.resolved_strategy,
    }
