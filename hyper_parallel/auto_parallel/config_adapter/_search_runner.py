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
"""Search runner — bridges NormalizedConfig to the ND search engine.

Converts a :class:`NormalizedConfig` into a temporary HyperParallel
``train.yaml``, runs the ND search via :class:`Parallelize`,
post-filters by user candidate lists, and returns the optimal strategy.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, TYPE_CHECKING

import yaml  # type: ignore[import-untyped]

from hyper_parallel.auto_parallel.config_adapter._normalized_config import NormalizedConfig

if TYPE_CHECKING:
    import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
    import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
    import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard

logger = logging.getLogger(__name__)

def _get_dim_module():
    """Lazy-import the sapp_nd dimensions module."""
    import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as dim_mod  # pylint: disable=C0415
    return dim_mod


def _get_machine_mod():
    """Lazy-import the sapp_nd hardware module."""
    import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as hw_mod  # pylint: disable=C0415
    return hw_mod


def _search_dim_map():
    """Return the mapping of NormalizedConfig keys to sapp_nd Dimension objects.

    Lazily loaded to avoid importing sapp_nd at module-import time.
    """
    dim_mod = _get_dim_module()
    return {
        "data_parallel_replicate_degree": dim_mod.DP,
        "tensor_parallel_degree": dim_mod.TP,
        "pipeline_parallel_degree": dim_mod.PP,
        "context_parallel_degree": dim_mod.CP,
        "expert_parallel_degree": dim_mod.EP,
        "micro_batch_num": dim_mod.MBN,
    }

# Accelerator field names for fixed dimensions (written into the temp YAML).
_ACCEL_FIELD_MAP: Dict[str, str] = {
    "data_parallel_shard_degree": "dp_shard",
    "data_parallel_replicate_degree": "dp_replicate",
    "tensor_parallel_degree": "tp_degree",
    "pipeline_parallel_degree": "pipeline_parallel_degree",
    "context_parallel_degree": "context_parallel_degree",
    "expert_parallel_degree": "expert_parallel_degree",
    "micro_batch_num": "micro_batch_num",
}


def _validate_before_search(config: NormalizedConfig) -> None:
    """Check required model fields are populated (>0) before search.

    Raises:
        ValueError: If any required field is missing or zero.
    """
    model = config.model_spec
    required = {
        "model_spec.n_layers": model.get("n_layers", 0),
        "model_spec.dim": model.get("dim", 0),
        "model_spec.n_heads": model.get("n_heads", 0),
        "model_spec.vocab_size": model.get("vocab_size", 0),
        "cluster_spec": config.cluster_spec,
    }
    missing = []
    for name, value in required.items():
        if name == "cluster_spec":
            if not isinstance(value, dict) or not value:
                missing.append(name)
        elif value <= 0:
            missing.append(name)
    if missing:
        raise ValueError(
            "Required fields missing or zero before ND search: "
            f"{', '.join(missing)}"
        )


def _build_hp_yaml_dict(config: NormalizedConfig) -> dict:
    """Build a HyperParallel ``train.yaml`` dict from *config*.

    Fixed dimensions (``constraint.fixed_*_degree``) are written directly
    into ``train.accelerator``.  Dimensions with search-space candidates
    use the first candidate as a placeholder — the actual search is driven
    by the ``dimensions`` parameter passed to :class:`Parallelize`.
    """
    model = config.model_spec
    constraint = config.constraint
    space = config.search_space

    accel: Dict[str, Any] = {}

    # Fixed dimensions → write actual value.
    fixed_map = {
        "fixed_dp_degree": ("dp_replicate", "data_parallel_replicate_degree", [1]),
        "fixed_fsdp_degree": ("dp_shard", "data_parallel_shard_degree", [1]),
        "fixed_tp_degree": ("tp_degree", "tensor_parallel_degree", [1]),
        "fixed_pp_degree": ("pipeline_parallel_degree", "pipeline_parallel_degree", [1]),
        "fixed_cp_degree": ("context_parallel_degree", "context_parallel_degree", [1]),
        "fixed_ep_degree": ("expert_parallel_degree", "expert_parallel_degree", [1]),
    }
    for constraint_key, (accel_key, space_key, default) in fixed_map.items():
        fixed_val = constraint.get(constraint_key)
        if fixed_val is not None and fixed_val > 0:
            accel[accel_key] = fixed_val
        else:
            candidates = space.get(space_key, default)
            accel[accel_key] = candidates[0]

    # Enable parallel optimizer by default.
    accel.setdefault("enable_parallel_optimizer", True)

    recompute = config.estimator.get("recompute_strategy", "none")

    hp_yaml: dict = {
        "model": {
            "name": model.get("name", "custom"),
            "config_overrides": {
                "hidden_size": model.get("dim", 4096),
                "num_hidden_layers": model.get("n_layers", 32),
                "num_attention_heads": model.get("n_heads", 32),
                "vocab_size": model.get("vocab_size", 128256),
            },
        },
        "train": {
            "global_batch_size": constraint.get("global_batch_size", 0) or 1,
            "micro_batch_size": model.get("local_batch_size", 1),
            "micro_batch_num": accel.pop("micro_batch_num", 1),
            "accelerator": accel,
            "gradient_checkpointing": {
                "activation_checkpoint": recompute,
            },
            "mixed_precision": {
                "enabled": True,
                "param_dtype": model.get("compute_dtype", "bfloat16"),
            },
        },
        "data": {
            "max_seq_len": model.get("seq_len", 4096),
        },
    }

    # Optional model fields.
    overrides = hp_yaml["model"]["config_overrides"]
    if model.get("inter_dim"):
        overrides["intermediate_size"] = model["inter_dim"]
    if model.get("n_kv_heads"):
        overrides["num_key_value_heads"] = model["n_kv_heads"]

    return hp_yaml


def _write_temp_hp_yaml(config: NormalizedConfig) -> str:
    """Write a temp ``train.yaml`` and return its absolute path."""
    data = _build_hp_yaml_dict(config)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="hp_search_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    logger.debug("Temp HP YAML written to %s", path)
    return path


def _build_machine(config: NormalizedConfig) -> Any:
    """Build a ``Hard.Machine`` from cluster_spec."""
    hw_mod = _get_machine_mod()
    cluster = config.cluster_spec
    nodes = max(1, cluster.get("num_nodes", 1))
    cards_per_node = max(1, cluster.get("cards_per_node", 8))
    total_devices = nodes * cards_per_node
    device_type = cluster.get("device_type", "A2")
    # Map generic names to sapp_nd device codes.
    device_code_map = {"ascend": "A2", "ascend910": "A2", "ascend910b": "A3"}
    device_type = device_code_map.get(str(device_type).lower(), device_type)
    return hw_mod.Machine(total_devices, device_type)


def _resolve_search_dimensions(config: NormalizedConfig) -> List[Any]:
    """Return a list of ``Dim`` objects whose candidates contain >1 value.

    List-valued entries in ``config.search_space`` are treated as
    **output** (search) dimensions.  Entries absent from
    ``search_space`` (``"auto"`` in YAML) are also included — they
    will be determined by ND's ``bound_space()``.
    """
    dims: List[Any] = []
    space = config.search_space
    for space_key, dim_obj in _search_dim_map().items():
        candidates = space.get(space_key)
        if candidates is not None and len(candidates) > 1:
            dims.append(dim_obj)
        elif space_key not in space:
            dims.append(dim_obj)
    return dims


def _post_filter(
    scored_space: list,
    config: NormalizedConfig,
) -> list:
    """Keep only entries whose dimension values are in the user's candidate lists."""
    space = config.search_space
    candidate_map: Dict[Any, List[int]] = {}
    for space_key, dim_obj in _search_dim_map().items():
        candidates = space.get(space_key)
        if candidates is not None and len(candidates) > 1:
            candidate_map[dim_obj] = candidates

    filtered = []
    for entry in scored_space:
        dims_val = entry[0].dims_val  # type: ignore[index]
        keep = True
        for dim_obj, allowed in candidate_map.items():
            actual = dims_val.get(dim_obj)
            if actual is not None and actual not in allowed:
                keep = False
                break
        if keep:
            filtered.append(entry)

    if not filtered and scored_space:
        logger.warning(
            "Post-filter removed ALL %d candidates. "
            "Returning unfiltered best entry.",
            len(scored_space),
        )
        return scored_space[:1]
    return filtered


def _format_result(best_entry: tuple) -> Dict[str, Any]:
    """Convert the best ND result entry into a flat result dict."""
    dim_mod = _get_dim_module()
    dims_val = best_entry[0].dims_val  # type: ignore[index]
    dim_to_key = {
        dim_mod.DP: "dp",
        dim_mod.TP: "tp",
        dim_mod.PP: "pp",
        dim_mod.CP: "cp",
        dim_mod.EP: "ep",
        dim_mod.MBN: "micro_batch_num",
    }
    result: Dict[str, Any] = {
        "memory_estimate_mb": float(best_entry[1]),
        "score": float(best_entry[2]),
    }
    for dim_obj, key in dim_to_key.items():
        if dim_obj in dims_val:
            result[key] = int(dims_val[dim_obj])
    return result


def search_strategies(config: NormalizedConfig) -> Dict[str, Any]:
    """Run the ND strategy search and return the optimal strategy.

    This is the main entry point for end-to-end strategy search:

    1. Validates required model fields.
    2. Converts the ``NormalizedConfig`` to a temporary HyperParallel
       ``train.yaml`` and writes it to disk.
    3. Launches the ND search engine (:class:`Parallelize`).
    4. Post-filters results against the user's candidate lists.
    5. Returns the best strategy as a flat dictionary.

    Args:
        config: A fully populated ``NormalizedConfig`` from
            :func:`read_search_config` or :func:`read_hp_yaml_config`.

    Returns:
        A dict with keys ``dp``, ``tp``, ``pp``, ``cp``, ``ep``,
        ``micro_batch_num``, ``memory_estimate_mb``, and ``score``.

    Raises:
        ValueError: If required fields are missing or no strategy is found.
        ImportError: If PyYAML is not installed.
    """
    _validate_before_search(config)

    yaml_path = _write_temp_hp_yaml(config)
    machine = _build_machine(config)
    dims = _resolve_search_dimensions(config)

    import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as _Par  # pylint: disable=C0415
    try:
        nd_runner = _Par.Parallelize(
            "hyper_v2",
            yaml_path,
            machine,
            global_batch_size=config.constraint.get("global_batch_size", 0),
            dimensions=dims,
        )
        scored_space = nd_runner.run_generation_to_ordering(
            yaml_folder=None,
            threads_num=None,
            top_num=None,
        )
    finally:
        try:
            os.remove(yaml_path)
        except OSError:
            pass

    if not scored_space:
        raise ValueError("ND search returned no valid strategies.")

    filtered = _post_filter(scored_space, config)
    best = filtered[0]
    result = _format_result(best)

    logger.info(
        "Optimal strategy found: dp=%(dp)s tp=%(tp)s pp=%(pp)s "
        "cp=%(cp)s ep=%(ep)s mb_num=%(micro_batch_num)s "
        "mem=%(memory_estimate_mb).0f MB score=%(score).2e",
        result,
    )
    return result
