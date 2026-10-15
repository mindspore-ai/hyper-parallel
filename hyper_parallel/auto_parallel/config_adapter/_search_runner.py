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

Hands a :class:`NormalizedConfig` to the ND search as the mapping
``Parallelize`` parses directly, post-filters by user candidate lists and
memory budget, and returns the optimal strategy.
"""

import copy
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

from hyper_parallel.auto_parallel._hf_model_spec import EXEC_OVERRIDE_KEYS
from hyper_parallel.auto_parallel._model_spec import ModelSpec, model_fields
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

    The model's fields go through the model spec, whose names are the HP
    YAML's, and the run keys the parser reads from ``config_overrides``
    pass through as they are.  The adapter's own keys, such as the
    micro-batch size, reach the HP YAML through their own sections.

    Args:
        model: The ``model_spec`` dict from :class:`NormalizedConfig`.

    Returns:
        A dict suitable for the ``model`` key of a HP ``train.yaml``.
    """
    spec = ModelSpec.from_dict(model_fields(model))
    overrides = spec.to_dict()
    overrides.pop("name", None)
    overrides.update({key: model[key] for key in EXEC_OVERRIDE_KEYS if key in model})
    return {"name": spec.name, "config_overrides": overrides}


# The strategy, under every name the cost model reads it by, in the sections
# a train.yaml states its run in: the search sets it, from its candidates or
# its own config, so the train.yaml's values give way.  The rest of each
# section is the run's.
_STRATEGY_KEYS: Dict[str, FrozenSet[str]] = {
    "accelerator": frozenset({
        "dp_replicate", "dp_shard", "tp_size", "tp_degree", "pp_size", "pipeline_parallel_degree",
        "cp_size", "context_parallel_degree", "ep_size", "expert_parallel_degree",
        "expert_tensor_parallel_degree", "micro_batch_num", "optimizer_weight_shard_size",
        "pipeline_scheduler", "pp_interleave_num",
    }),
    "fsdp_config": frozenset({"dp_shard_size"}),
    "training": frozenset({"global_batch_size", "micro_batch_size", "micro_batch_num"}),
}


def _stated_run(config: NormalizedConfig) -> Dict[str, Any]:
    """Return the run the train.yaml states, less the strategy the search sets.

    Args:
        config: The normalized config whose ``run`` the train.yaml filled.

    Returns:
        A copy of ``config.run``, its sections without the strategy's keys.
    """
    run = copy.deepcopy(config.run)
    for section, strategy in _STRATEGY_KEYS.items():
        stated = run.get(section)
        if isinstance(stated, dict):
            run[section] = {key: value for key, value in stated.items() if key not in strategy}
    return run


def _memory_budget_gb(config: NormalizedConfig) -> float:
    """Return the per-device memory budget in GB, or ``0.0`` if unconstrained.

    ``constraint.memory_limit_gb`` is the budget the user asked for and
    ``cluster_spec.device_memory_gb`` the hardware ceiling, so the search
    must respect whichever of the two is tighter.

    Args:
        config: The normalized config carrying constraint and cluster spec.

    Returns:
        The tighter positive budget in GB, or ``0.0`` when neither is set.
    """
    limit = float(config.constraint.get("memory_limit_gb", 0.0) or 0.0)
    device = float(config.cluster_spec.get("device_memory_gb", 0.0) or 0.0)
    budgets = [value for value in (limit, device) if value > 0]
    return min(budgets) if budgets else 0.0


def _pinned_or_first(config: NormalizedConfig, constraint_key: str, space_key: str,
                     default: List[Any]) -> Any:
    """Return the degree *constraint_key* pins, else the first candidate.

    Args:
        config: The normalized config carrying constraint and search space.
        constraint_key: The ``constraint`` entry that pins the degree.
        space_key: The ``search_space`` entry listing its candidates.
        default: The candidates when the search space lists none.

    Returns:
        The pinned degree when it is positive, else the first candidate,
        a placeholder the search replaces.
    """
    pinned = config.constraint.get(constraint_key)
    if pinned is not None and pinned > 0:
        return pinned
    return config.search_space.get(space_key, default)[0]


def _build_strategy_dicts(
    config: NormalizedConfig, run: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the ``accelerator`` and ``fsdp_config`` sections of the HP YAML.

    The strategy goes over the sections the train.yaml states, which this
    takes out of *run*.
    """
    accel: Dict[str, Any] = dict(run.pop("accelerator", None) or {})
    fsdp: Dict[str, Any] = dict(run.pop("fsdp_config", None) or {})

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
        accel[accel_key] = _pinned_or_first(config, constraint_key, space_key, default)

    # micro_batch_num has no accelerator entry of its own, so a fixed value
    # would be dropped and the global-batch-size check would then reject the
    # only config the user asked for.
    accel["micro_batch_num"] = int(
        _pinned_or_first(config, "fixed_micro_batch_num", "micro_batch_num", [1])
    )
    fsdp["dp_shard_size"] = int(
        _pinned_or_first(config, "fixed_fsdp_degree", "data_parallel_shard_degree", [1])
    )

    # CP algorithm: propagate to yaml so CostModelParserHyperV2 can read it.
    cp_algo = config.estimator.get("cp_algo")
    if cp_algo:
        accel["context_parallel_algo"] = cp_algo

    # Optional accelerator fields that affect memory estimation.
    owss = config.model_spec.get("optimizer_weight_shard_size")
    if owss and owss > 0:
        accel["optimizer_weight_shard_size"] = owss

    use_sp = config.model_spec.get("use_seq_parallel", True)
    accel.setdefault("sequence_parallel", bool(use_sp))
    return accel, fsdp


def _build_hp_yaml_dict(config: NormalizedConfig) -> dict:
    """Build an AutoModels-shaped cost-model YAML dict from *config*.

    Fixed dimensions (``constraint.fixed_*_degree``) are written directly
    into the strategy sections. Dimensions with search-space candidates
    use the first candidate as a placeholder -- the actual search is driven
    by the ``dimensions`` parameter passed to :class:`Parallelize`.  The
    strategy goes over the run the train.yaml states, which stays as stated.
    """
    model = config.model_spec
    constraint = config.constraint
    run = _stated_run(config)
    accel, fsdp = _build_strategy_dicts(config, run)

    recompute = config.estimator.get("recompute_strategy", "none")

    cluster = config.cluster_spec
    # The train.yaml's pricing options, beneath the search's own device count
    # and memory budget.
    context: Dict[str, Any] = dict(run.pop("context", None) or {})
    # ND prunes the space against ccfg.device_capacity, which the parser reads
    # from this field, so the user's memory_limit_gb has to reach it here.
    budget_gb = _memory_budget_gb(config)
    if budget_gb > 0:
        context["max_device_memory"] = f"{budget_gb}GB"
    device_num = cluster.get("num_nodes", 0) * cluster.get("cards_per_node", 0)
    if device_num > 0:
        context["device_num"] = int(device_num)
    visual_seq_len = model.get("visual_seq_len")
    if visual_seq_len:
        context["visual_seq_len"] = int(visual_seq_len)

    gc_dict: Dict[str, Any] = {"mode": {"none": "off"}.get(recompute, recompute)}
    recompute_slice = model.get("recompute_slice_activation")
    if recompute_slice is not None:
        gc_dict["recompute_slice_activation"] = bool(recompute_slice)

    model_dict = _build_model_dict(model)

    hp_yaml: dict = {
        **run,
        "model": {**run.get("model", {}), **model_dict},
        "training": {
            **run.get("training", {}),
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



# The names a search config may give its devices, and the sapp_nd device
# codes they mean: the Ascend 910B is the Atlas A2 series' chip, and the
# 910C, CANN's ascend910_93, the A3's.  A generic "ascend" stays A2.
_DEVICE_CODES: Dict[str, str] = {
    "a2": "A2", "a3": "A3", "v100": "V100",
    "ascend": "A2", "ascend910": "A2", "ascend910b": "A2",
    "ascend910c": "A3", "ascend910_93": "A3",
}


def _build_machine(config: NormalizedConfig) -> Any:
    """Build a ``Hard.Machine`` from cluster_spec."""
    hw_mod = _get_machine_mod()
    cluster = config.cluster_spec
    nodes = max(1, cluster.get("num_nodes", 1))
    cards_per_node = max(1, cluster.get("cards_per_node", 8))
    total_devices = nodes * cards_per_node
    device_type = str(cluster.get("device_type", "A2"))
    return hw_mod.Machine(total_devices, _DEVICE_CODES.get(device_type.lower(), device_type))


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


def _filter_by_memory(scored_space: list, budget_gb: float) -> list:
    """Drop entries whose peak-memory estimate exceeds *budget_gb*.

    ND already prunes against the device capacity while it generates the
    space; this is the gate that holds when a strategy is scored under a
    capacity looser than the budget the caller asked for.

    Args:
        scored_space: Scored entries ``(strategy, memory_mb, score, ...)``.
        budget_gb: Per-device budget in GB; ``0`` disables the gate.

    Returns:
        The entries that fit.  May be empty, which means the budget rules
        out every strategy the search found.
    """
    if budget_gb <= 0:
        return list(scored_space)
    budget_mb = budget_gb * 1024.0
    kept = [entry for entry in scored_space if float(entry[1]) <= budget_mb]
    if len(kept) < len(scored_space):
        logger.info(
            "Memory filter dropped %d of %d strategies above %.1f GB.",
            len(scored_space) - len(kept),
            len(scored_space),
            budget_gb,
        )
    return kept


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
        The entries that fit the memory budget and sit in every list.  Empty
        when none does: a strategy outside the lists is never returned in
        their place, and the caller refuses the search.
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

    feasible = _filter_by_memory(scored_space, _memory_budget_gb(config))

    filtered = []
    for entry in feasible:
        dims_val = entry[0].dims_val  # type: ignore[index]
        keep = True
        for dim_obj, allowed in candidate_map.items():
            actual = dims_val.get(dim_obj)
            if actual is not None and actual not in allowed:
                keep = False
                break
        if keep:
            filtered.append(entry)
    return filtered


def _no_candidate_match(feasible: list, config: NormalizedConfig, candidate_dims: Set[Any]) -> str:
    """Say why no strategy that fits the budget is in the candidate lists.

    ND searches each dimension within its own bounds and the lists only
    filter what it found, so a list can ask for degrees the search never
    reached.  Naming the degrees it did reach shows which.

    Args:
        feasible: The scored entries that fit the memory budget.
        config: The normalized config whose ``search_space`` holds the lists.
        candidate_dims: The dimensions that have a list of their own.

    Returns:
        The refusal, naming each list no fitting strategy meets.
    """
    unmet = []
    for space_key, dim_obj in _search_dim_map().items():
        if dim_obj not in candidate_dims:
            continue
        asked = list(config.search_space.get(space_key, []))
        found = sorted({entry[0].dims_val.get(dim_obj) for entry in feasible} - {None})
        if not set(found) & set(asked):
            unmet.append(f"{space_key} asks {asked}, the strategies that fit take {found}")
    why = "; ".join(unmet) if unmet else "each list is met by some of them, but never all at once"
    return (
        f"None of the {len(feasible)} strategies that fit the memory budget is in the candidate "
        f"lists: {why}. ND searches each dimension within its own bounds and the lists only "
        "filter what it finds: widen the lists, or the budget."
    )


def _pinned_degree(config: NormalizedConfig, constraint_key: str, space_key: str) -> int:
    """Return the degree of a dimension the search did not vary.

    Args:
        config: The normalized config carrying constraint and search space.
        constraint_key: The ``constraint`` entry that pins the degree.
        space_key: The ``search_space`` entry listing its candidates.

    Returns:
        The pinned degree, else the only candidate, else 1.
    """
    pinned = config.constraint.get(constraint_key)
    if not pinned:
        candidates = config.search_space.get(space_key) or []
        pinned = candidates[0] if len(candidates) == 1 else 1
    return int(pinned)


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
    # A pinned dimension is absent from dims_val, so fill it from what pinned
    # it: the searcher never varied it, but every consumer still reads it.
    fixed_from = {
        "dp": ("fixed_dp_degree", "data_parallel_replicate_degree"),
        "tp": ("fixed_tp_degree", "tensor_parallel_degree"),
        "pp": ("fixed_pp_degree", "pipeline_parallel_degree"),
        "cp": ("fixed_cp_degree", "context_parallel_degree"),
        "ep": ("fixed_ep_degree", "expert_parallel_degree"),
        "micro_batch_num": ("fixed_micro_batch_num", "micro_batch_num"),
    }
    for key, (constraint_key, space_key) in fixed_from.items():
        if key not in result:
            result[key] = _pinned_degree(config, constraint_key, space_key)
    total_dp = result.get("dp", 1)
    if "dp_shard" not in result:
        # OP absent from the searched dimensions: fall back to the declared
        # degree, which is what the parser used for the whole run.
        fsdp_candidates = config.search_space.get("data_parallel_shard_degree", [1])
        configured_fsdp = config.constraint.get("fixed_fsdp_degree")
        result["dp_shard"] = int(configured_fsdp or fsdp_candidates[0])
    # The yaml the search prices is AutoModels-shaped, whose FSDP domain is
    # DP times CP: the shard may span it, and the trainer replicates the rest
    # (MeshContext.build_meshs).  Bounded by DP, a shard the search priced
    # under CP was cut back and its replicas undercounted.
    domain = total_dp * max(1, int(result.get("cp") or 1))
    dp_shard = max(1, min(int(result["dp_shard"]), domain))
    result["dp_shard"] = dp_shard
    result["dp_replicate"] = max(1, domain // dp_shard)
    return result


def search_strategies(config: NormalizedConfig) -> Dict[str, Any]:
    """Run the ND strategy search and return the optimal strategy.

    This is the main entry point for end-to-end strategy search:

    1. Validates required model fields.
    2. Shapes the ``NormalizedConfig`` into the cost-model config ND
       parses, as a mapping rather than a file.
    3. Launches the ND search engine (:class:`Parallelize`).
    4. Post-filters results against the user's candidate lists.
    5. Returns the best strategy as a flat dictionary.

    Args:
        config: A fully populated ``NormalizedConfig`` from
            :func:`read_search_config` or :func:`read_hp_yaml_config`.
            ``config.estimator["ratios_file"]``, when set, names the ratios
            file ND scores with (``run_nd -s ... -c`` sets it); without it
            ND scores raw.

    Returns:
        A dict with keys ``dp``, ``tp``, ``pp``, ``cp``, ``ep``,
        ``micro_batch_num``, ``memory_estimate_mb``, ``score`` and
        ``activation_checkpoint``.

    Raises:
        ValueError: If required fields are missing or no strategy is found.
    """
    _validate_before_search(config)

    hp_config = _build_hp_yaml_dict(config)
    machine = _build_machine(config)
    dims, candidate_dims = _resolve_search_dimensions(config)

    import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as _Par  # pylint: disable=C0415
    nd_runner = _Par.Parallelize(
        "hyper_v2",
        hp_config,
        machine,
        global_batch_size=config.constraint.get("global_batch_size", 0),
        dimensions=dims,
    )
    scored_space = nd_runner.run_generation_to_ordering(
        yaml_folder=None,
        threads_num=None,
        top_num=None,
        cache_file=config.estimator.get("ratios_file"),
    )

    if not scored_space:
        raise ValueError("ND search returned no valid strategies.")

    filtered = _post_filter(scored_space, config, candidate_dims)
    if not filtered:
        feasible = _filter_by_memory(scored_space, _memory_budget_gb(config))
        if feasible:
            raise ValueError(_no_candidate_match(feasible, config, candidate_dims))
        lightest_gb = min(float(entry[1]) for entry in scored_space) / 1024.0
        raise ValueError(
            "No strategy fits the memory budget of "
            f"{_memory_budget_gb(config):.1f} GB; the lightest strategy "
            f"found needs {lightest_gb:.1f} GB. Raise "
            "constraint.memory_limit_gb or widen the search space."
        )
    best = filtered[0]
    result = _format_result(best, config)
    # The search prices every candidate fully recomputed, whatever the search
    # yaml says, so the trainer runs what was priced.
    result["activation_checkpoint"] = "full"

    logger.info(
        "Optimal strategy found: dp=%(dp)s tp=%(tp)s pp=%(pp)s "
        "cp=%(cp)s ep=%(ep)s mb_num=%(micro_batch_num)s "
        "recompute=%(activation_checkpoint)s "
        "mem=%(memory_estimate_mb).0f MB score=%(score).2e",
        result,
    )
    return result
