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
"""Search runner -- bridges NormalizedConfig to the ND search engine.

Converts a :class:`NormalizedConfig` into a temporary HyperParallel
``train.yaml``, runs the ND search via :class:`Parallelize`,
post-filters by user candidate lists, and returns the optimal strategy.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import yaml  # type: ignore[import-untyped]

from hyper_parallel.auto_parallel.config_adapter._normalized_config import NormalizedConfig


CONFIG_OVERRIDE_FIELDS = [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
    "intermediate_size", "num_key_value_heads", "max_position_embeddings",
    "num_experts", "num_experts_per_tok", "num_shared_experts",
    "moe_intermediate_size", "first_k_dense_replace", "mtp_depth",
    "multiple_of", "ffn_dim_multiplier", "kv_lora_rank", "q_lora_rank",
    "qk_rope_head_dim", "v_head_dim", "capacity_factor", "offset",
    "head_dim", "vision",
    "param_init_type", "compute_dtype", "softmax_compute_type",
]

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
        # OP carries the FSDP/optimizer shard degree (ccfg.os_max_shard),
        # so mapping it here is what lets ND search that dimension.
        "data_parallel_shard_degree": dim_mod.OP,
    }

def _validate_before_search(config: NormalizedConfig) -> None:
    """Check required model fields are populated (>0) before search.

    Raises:
        ValueError: If any required field is missing or zero.
    """
    model = config.model_spec
    required = {
        "model_spec.num_hidden_layers": model.get("num_hidden_layers", 0),
        "model_spec.hidden_size": model.get("hidden_size", 0),
        "model_spec.num_attention_heads": model.get("num_attention_heads", 0),
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


def _build_model_dict(model: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``model`` section of the HP YAML from *model* spec.

    All ``config_overrides`` field names in *model* already match the
    HP YAML convention, so they are passed through directly without
    any name mapping.

    Args:
        model: The ``model_spec`` dict from :class:`NormalizedConfig`.

    Returns:
        A dict suitable for the ``model`` key of a HP ``train.yaml``.
    """
    model_dict: Dict[str, Any] = {
        "name": model.get("name", "custom"),
        "config_overrides": {},
    }
    overrides = model_dict["config_overrides"]
    for key in CONFIG_OVERRIDE_FIELDS:
        val = model.get(key)
        if val is not None:
            overrides[key] = val

    return model_dict


def _build_hp_yaml_dict(config: NormalizedConfig) -> dict:
    """Build an AutoModels-shaped cost-model YAML dict from *config*.

    Fixed dimensions (``constraint.fixed_*_degree``) are written directly
    into the strategy sections. Dimensions with search-space candidates
    use the first candidate as a placeholder -- the actual search is driven
    by the ``dimensions`` parameter passed to :class:`Parallelize`.
    """
    model = config.model_spec
    constraint = config.constraint
    space = config.search_space

    accel: Dict[str, Any] = {}
    fsdp: Dict[str, Any] = {}

    # Fixed dimensions -- write actual value.
    fixed_map = {
        "fixed_dp_degree": ("dp_replicate", "data_parallel_replicate_degree", [1]),
        "fixed_tp_degree": ("tp_size", "tensor_parallel_degree", [1]),
        "fixed_pp_degree": ("pp_size", "pipeline_parallel_degree", [1]),
        "fixed_cp_degree": ("cp_size", "context_parallel_degree", [1]),
        "fixed_ep_degree": ("ep_size", "expert_parallel_degree", [1]),
        "fixed_etp_degree": ("expert_tensor_parallel_degree", "expert_tensor_parallel_degree", [0]),
    }
    for constraint_key, (accel_key, space_key, default) in fixed_map.items():
        fixed_val = constraint.get(constraint_key)
        if fixed_val is not None and fixed_val > 0:
            accel[accel_key] = fixed_val
        else:
            candidates = space.get(space_key, default)
            accel[accel_key] = candidates[0]

    fixed_fsdp = constraint.get("fixed_fsdp_degree")
    fsdp_candidates = space.get("data_parallel_shard_degree", [1])
    fsdp["dp_shard_size"] = int(
        fixed_fsdp if fixed_fsdp is not None and fixed_fsdp > 0
        else fsdp_candidates[0]
    )

    # CP algorithm: propagate to yaml so CostModelParserHyperV2 can read it.
    cp_algo = config.estimator.get("cp_algo")
    if cp_algo:
        accel["context_parallel_algo"] = cp_algo

    # Optional accelerator fields that affect memory estimation.
    owss = model.get("optimizer_weight_shard_size")
    if owss and owss > 0:
        accel["optimizer_weight_shard_size"] = owss

    use_sp = model.get("use_seq_parallel", True)
    accel.setdefault("sequence_parallel", bool(use_sp))

    recompute = config.estimator.get("recompute_strategy", "none")

    cluster = config.cluster_spec
    device_mem_gb = cluster.get("device_memory_gb", 0)
    context: Dict[str, Any] = {}
    if device_mem_gb > 0:
        context["max_device_memory"] = f"{device_mem_gb}GB"
    device_num = cluster.get("num_nodes", 0) * cluster.get("cards_per_node", 0)
    if device_num > 0:
        context["device_num"] = int(device_num)
    visual_seq_len = model.get("visual_seq_len")
    if visual_seq_len:
        context["visual_seq_len"] = int(visual_seq_len)

    gc_dict: Dict[str, Any] = {"mode": "off" if recompute == "none" else recompute}
    recompute_slice = model.get("recompute_slice_activation")
    if recompute_slice is not None:
        gc_dict["recompute_slice_activation"] = bool(recompute_slice)

    model_dict = _build_model_dict(model)

    hp_yaml: dict = {
        "model": model_dict,
        "training": {
            "global_batch_size": constraint.get("global_batch_size", 0),
            "micro_batch_size": model.get("local_batch_size", 1),
            "micro_batch_num": accel.pop("micro_batch_num", 1),
        },
        "accelerator": accel,
        "fsdp_config": fsdp,
        "activation_checkpoint": gc_dict,
        "dataset": {
            "data_transform": {
                "max_seq_len": model.get("max_position_embeddings", 4096),
            },
        },
    }

    if context:
        hp_yaml["context"] = context

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


def _resolve_search_dimensions(config: NormalizedConfig) -> Tuple[List[Any], Set[Any]]:
    """Return search dimensions and the set of dimensions with user candidates.

    List-valued entries in ``config.search_space`` are treated as
    **output** (search) dimensions.  Entries absent from
    ``config.search_space`` (``"auto"`` in YAML) are also included -- they
    will be determined by ND's ``bound_space()``.

    Returns:
        A tuple ``(dims, candidate_dims)`` where *dims* is the list of
        ``Dim`` objects to pass to ND and *candidate_dims* is the set of
        Dim objects for which the user supplied an explicit candidate list
        (used by :func:`_post_filter`).
    """
    dims: List[Any] = []
    candidate_dims: Set[Any] = set()
    space = config.search_space
    for space_key, dim_obj in _search_dim_map().items():
        candidates = space.get(space_key)
        if candidates is not None and len(candidates) > 1:
            dims.append(dim_obj)
            candidate_dims.add(dim_obj)
        elif space_key not in space:
            dims.append(dim_obj)
    return dims, candidate_dims


def _post_filter(
    scored_space: list,
    config: NormalizedConfig,
    candidate_dims: Optional[Set[Any]] = None,
) -> list:
    """Keep only entries whose dimension values are in the user's candidate lists.

    Args:
        scored_space: The scored strategy list from ND engine.
        config: The normalized config containing ``search_space``.
        candidate_dims: The set of Dim objects that have user-supplied
            candidate lists with more than one value.  If *None*, the
            set is derived from *config* (backward-compatible).

    Returns:
        A filtered list.  May be empty if no entry satisfies all
        candidate constraints -- the caller decides how to handle this.
    """
    space = config.search_space
    if candidate_dims is None:
        candidate_dims = set()
        for space_key, dim_obj in _search_dim_map().items():
            candidates = space.get(space_key)
            if candidates is not None and len(candidates) > 1:
                candidate_dims.add(dim_obj)

    candidate_map: Dict[Any, List[int]] = {}
    for space_key, dim_obj in _search_dim_map().items():
        if dim_obj not in candidate_dims:
            continue
        candidates = space.get(space_key)
        if candidates is not None:
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
            "Post-filter removed ALL %d candidates; "
            "no strategy matches the user's candidate constraints.",
            len(scored_space),
        )
        return scored_space[:1]
    return filtered


def _format_result(best_entry: tuple, config: NormalizedConfig) -> Dict[str, Any]:
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
        dim_mod.OP: "dp_shard",
    }
    result: Dict[str, Any] = {
        "memory_estimate_mb": float(best_entry[1]),
        "score": float(best_entry[2]),
    }
    for dim_obj, key in dim_to_key.items():
        if dim_obj in dims_val:
            result[key] = int(dims_val[dim_obj])
    result.setdefault("cp", 1)
    result.setdefault("ep", 1)
    total_dp = result.get("dp", 1)
    if "dp_shard" not in result:
        # OP absent from the searched dimensions: fall back to the declared
        # degree, which is what the parser used for the whole run.
        fsdp_candidates = config.search_space.get("data_parallel_shard_degree", [1])
        configured_fsdp = config.constraint.get("fixed_fsdp_degree")
        result["dp_shard"] = int(configured_fsdp or fsdp_candidates[0])
    dp_shard = max(1, min(int(result["dp_shard"]), total_dp))
    result["dp_shard"] = dp_shard
    result["dp_replicate"] = max(1, total_dp // dp_shard)
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
    dims, candidate_dims = _resolve_search_dimensions(config)

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

    filtered = _post_filter(scored_space, config, candidate_dims)
    best = filtered[0]
    result = _format_result(best, config)

    logger.info(
        "Optimal strategy found: dp=%(dp)s tp=%(tp)s pp=%(pp)s "
        "cp=%(cp)s ep=%(ep)s mb_num=%(micro_batch_num)s "
        "mem=%(memory_estimate_mb).0f MB score=%(score).2e",
        result,
    )
    return result
