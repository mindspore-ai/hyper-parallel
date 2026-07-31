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
"""Constraint checker for auto parallel strategy search.

Validates cross-field constraints on a :class:`NormalizedConfig` instance:
divisibility checks, device count limits, pipeline stage consistency,
and required-field presence.
"""

from typing import Dict, List, Optional

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
    NormalizedConfig,
    ValidationError,
)


def _err(field_path: str, message: str) -> ValidationError:
    """Create an error-severity validation error."""
    return ValidationError(field_path=field_path, message=message, severity="error")


def _warn(field_path: str, message: str) -> ValidationError:
    """Create a warning-severity validation error."""
    return ValidationError(field_path=field_path, message=message, severity="warning")


def _check_divisibility(
    numerator_name: str,
    numerator: int,
    denominator_name: str,
    denominator: int,
) -> Optional[ValidationError]:
    """Check that numerator is divisible by denominator. Returns None if OK."""
    if denominator <= 1:
        return None
    if numerator <= 0:
        return None
    if numerator % denominator != 0:
        return _err(
            numerator_name,
            f"{numerator_name} ({numerator}) must be divisible by "
            f"{denominator_name} ({denominator}), "
            f"remainder is {numerator % denominator}",
        )
    return None


def _get_fixed_value(constraint: Dict, dim_key: str) -> Optional[int]:
    """Look up a fixed dimension value from constraint dict."""
    fixed_map = {
        "dp": constraint.get("fixed_dp_degree"),
        "tp": constraint.get("fixed_tp_degree"),
        "pp": constraint.get("fixed_pp_degree"),
        "cp": constraint.get("fixed_cp_degree"),
        "ep": constraint.get("fixed_ep_degree"),
    }
    return fixed_map.get(dim_key)


def _resolve_candidates(search_space: Dict, dim_key: str,
                        constraint: Dict, dim_label: str) -> List[int]:
    """Return the effective candidate list for a dimension,
    respecting fixed values from constraints."""
    fixed = _get_fixed_value(constraint, dim_label)
    if fixed is not None and fixed > 0:
        return [fixed]
    return search_space.get(dim_key, [1])


def _candidates_or_default(search_space: Dict, dim_key: str,
                           constraint: Dict, dim_label: str) -> List[int]:
    """Return candidates; if empty, default to [1]."""
    candidates = _resolve_candidates(search_space, dim_key, constraint, dim_label)
    if not candidates:
        candidates = [1]
    return candidates


def _total_cards(cluster_spec: Dict) -> int:
    """Compute total devices (num_nodes * cards_per_node)."""
    nodes = cluster_spec.get("num_nodes", 1)
    cards_per_node = cluster_spec.get("cards_per_node", 8)
    if nodes <= 0 or cards_per_node <= 0:
        return 0
    return nodes * cards_per_node


def validate(config: NormalizedConfig) -> List[ValidationError]:
    """Validate cross-field constraints on a normalized configuration.

    Performs all checks listed in Issue 126 Section 5:
    divisibility, device product limit, pipeline constraints,
    batch size relationships, and required-field presence.

    Args:
        config: The normalized configuration to validate.

    Returns:
        List of :class:`ValidationError` objects. An empty list
        means the configuration is valid.

    Example:
        >>> errors = validate(config)
        >>> for err in errors:
        ...     print(f"[{err.severity}] {err.field_path}: {err.message}")
    """
    errors: List[ValidationError] = []

    model = config.model_spec
    cluster = config.cluster_spec
    search = config.search_space
    constraint = config.constraint
    pp_cfg = config.pp_config

    _check_required_fields(errors, model)
    _check_batch_size_relationships(errors, search, constraint)
    _check_tp_divisibility(errors, model, search, constraint)
    _check_cp_divisibility(errors, model, search, constraint)
    _check_ep_divisibility(errors, model, search, constraint)
    _check_fixed_dims_vs_search_space(errors, search, constraint)
    _check_pipeline_constraints(errors, model, pp_cfg)
    _check_layer_offset(errors, model, pp_cfg)
    _check_layer_recompute(errors, model, pp_cfg)
    _check_device_product_limit(errors, search, constraint, cluster)
    _check_memory_limit(errors, cluster, constraint)
    _check_dense_model_ep_cp_warning(errors, model, search)
    _check_fsdp_hsdp_device_product(errors, search, constraint, cluster)

    return errors


def validate_strict(config: NormalizedConfig) -> None:
    """Validate and raise ``ValueError`` on any ``"error"`` severity issues.

    Args:
        config: The normalized configuration to validate.

    Raises:
        ValueError: If one or more ``"error"`` severity issues are found,
            with all messages concatenated.
    """
    errors = validate(config)
    fatal = [e for e in errors if e.severity == "error"]
    if fatal:
        lines = "\n".join(f"  [{e.severity}] {e.field_path}: {e.message}" for e in fatal)
        raise ValueError(f"Configuration validation failed with {len(fatal)} error(s):\n{lines}")


def _check_required_fields(errors: List[ValidationError], model: Dict) -> None:
    """Check required model fields (num_hidden_layers, hidden_size,
    num_attention_heads, vocab_size) are present and > 0."""
    required = [
        ("model.num_hidden_layers", model.get("num_hidden_layers", 0), "num_hidden_layers must be > 0"),
        ("model.hidden_size", model.get("hidden_size", 0), "hidden_size must be > 0"),
        ("model.num_attention_heads", model.get("num_attention_heads", 0), "num_attention_heads must be > 0"),
        ("model.vocab_size", model.get("vocab_size", 0), "vocab_size must be > 0"),
    ]
    for field_path, value, message in required:
        if value <= 0:
            errors.append(_err(field_path, message))


def _check_batch_size_relationships(
    errors: List[ValidationError],
    search: Dict,
    constraint: Dict,
) -> None:
    """Check that global_batch_size is divisible by micro_batch_num and dp.

    In FSDP/HSDP scenarios the effective data-parallel degree is
    ``dp_shard * dp_replicate``, so both components are validated.
    """
    gbs = constraint.get("global_batch_size", 0)
    if gbs <= 0:
        return

    mbn_list = search.get("micro_batch_num", [1])
    for mbn in mbn_list:
        if mbn > 0 and gbs % mbn != 0:
            errors.append(_err(
                "constraint.global_batch_size",
                f"global_batch_size ({gbs}) must be divisible by "
                f"micro_batch_num ({mbn}), remainder is {gbs % mbn}",
            ))

    dp_repl = _candidates_or_default(search, "data_parallel_replicate_degree", constraint, "dp")
    dp_shard = _candidates_or_default(search, "data_parallel_shard_degree", constraint, "fsdp")
    for repl in dp_repl:
        for shard in dp_shard:
            effective_dp = max(1, repl) * max(1, shard)
            if effective_dp > 1 and gbs % effective_dp != 0:
                errors.append(_err(
                    "constraint.global_batch_size",
                    f"global_batch_size ({gbs}) must be divisible by "
                    f"effective DP ({repl}*{shard}={effective_dp}), "
                    f"remainder is {gbs % effective_dp}",
                ))


def _check_tp_divisibility(
    errors: List[ValidationError],
    model: Dict,
    search: Dict,
    constraint: Dict,
) -> None:
    """Check that hidden_size, num_attention_heads, intermediate_size are divisible by tp."""
    tp_list = _candidates_or_default(search, "tensor_parallel_degree", constraint, "tp")
    dim = model.get("hidden_size", 0)
    n_heads = model.get("num_attention_heads", 0)
    inter_dim = model.get("intermediate_size", 0)

    for tp in tp_list:
        if tp <= 1:
            continue
        if dim > 0:
            err = _check_divisibility("model.hidden_size", dim, "tp", tp)
            if err:
                errors.append(err)
        if n_heads > 0:
            err = _check_divisibility("model.num_attention_heads", n_heads, "tp", tp)
            if err:
                errors.append(err)
        if inter_dim > 0:
            err = _check_divisibility("model.intermediate_size", inter_dim, "tp", tp)
            if err:
                errors.append(err)


def _check_cp_divisibility(
    errors: List[ValidationError],
    model: Dict,
    search: Dict,
    constraint: Dict,
) -> None:
    """Check that max_position_embeddings is divisible by cp."""
    cp_list = _candidates_or_default(search, "context_parallel_degree", constraint, "cp")
    seq_len = model.get("max_position_embeddings", 0)

    for cp in cp_list:
        if cp <= 1:
            continue
        if seq_len > 0:
            err = _check_divisibility("model.max_position_embeddings", seq_len, "cp", cp)
            if err:
                errors.append(err)


def _check_ep_divisibility(
    errors: List[ValidationError],
    model: Dict,
    search: Dict,
    constraint: Dict,
) -> None:
    """Check that num_experts is divisible by ep."""
    ep_list = _candidates_or_default(search, "expert_parallel_degree", constraint, "ep")
    num_experts = model.get("num_experts", 0)

    for ep in ep_list:
        if ep <= 1:
            continue
        if num_experts > 0:
            err = _check_divisibility("model.num_experts", num_experts, "ep", ep)
            if err:
                errors.append(err)


def _check_fixed_dims_vs_search_space(
    errors: List[ValidationError],
    search: Dict,
    constraint: Dict,
) -> None:
    """Check that fixed dimension values are within the candidate search space."""
    dim_map = {
        "dp": "data_parallel_replicate_degree",
        "tp": "tensor_parallel_degree",
        "pp": "pipeline_parallel_degree",
        "cp": "context_parallel_degree",
        "ep": "expert_parallel_degree",
    }

    for dim_label, search_key in dim_map.items():
        fixed = _get_fixed_value(constraint, dim_label)
        if fixed is None:
            continue
        candidates = search.get(search_key, [])
        if candidates and fixed not in candidates:
            errors.append(_err(
                f"constraint.fixed_{dim_label}_degree",
                f"Fixed {dim_label}_degree ({fixed}) is not in the "
                f"search space {candidates}",
            ))


def _check_pipeline_constraints(
    errors: List[ValidationError],
    model: Dict,
    pp_cfg: Dict,
) -> None:
    """Check pipeline stage count, num_hidden_layers, and stage_partition consistency."""
    pp_degree_raw = pp_cfg.get("pp_degree", 1)
    pp_values = pp_degree_raw if isinstance(pp_degree_raw, list) else [pp_degree_raw]
    pp_values = [v for v in pp_values if v > 1]
    if not pp_values:
        return

    n_layers = model.get("num_hidden_layers", 0)
    if n_layers <= 0:
        return

    for pp_degree_val in pp_values:
        if pp_degree_val > n_layers:
            errors.append(_err(
                "pp_config.pp_degree",
                f"pp_degree ({pp_degree_val}) exceeds the number of splittable "
                f"layers ({n_layers})",
            ))

    stage_partition = pp_cfg.get("stage_partition", [])
    stage_mode = pp_cfg.get("stage_partition_mode", "uniform")
    if stage_mode == "manual" and stage_partition:
        # Validate against every pp_degree candidate.
        for pp_for_stages in pp_values:
            if len(stage_partition) != pp_for_stages:
                errors.append(_err(
                    "pp_config.stage_partition",
                    f"stage_partition has {len(stage_partition)} stages, "
                    f"but pp_degree is {pp_for_stages}",
                ))
            all_layers: set = set()
            for stage_layers in stage_partition:
                all_layers.update(stage_layers)
            expected = set(range(n_layers))
            missing = expected - all_layers
            extra = all_layers - expected
            if missing:
                errors.append(_err(
                    "pp_config.stage_partition",
                    f"stage_partition does not cover layers: {sorted(missing)}",
                ))
            if extra:
                errors.append(_err(
                    "pp_config.stage_partition",
                    f"stage_partition references non-existent layers: {sorted(extra)}",
                ))


def _check_layer_offset(
    errors: List[ValidationError],
    model: Dict,
    pp_cfg: Dict,
) -> None:
    """Check that layer_offset_range is valid and within num_hidden_layers bounds."""
    offset_range = pp_cfg.get("layer_offset_range", (0, 0))
    if not isinstance(offset_range, (tuple, list)):
        return
    lo_min, lo_max = offset_range[0], offset_range[1]
    if lo_min == 0 and lo_max == 0:
        return

    n_layers = model.get("num_hidden_layers", 0)
    if lo_min > lo_max:
        errors.append(_err(
            "pp_config.layer_offset_range",
            f"layer_offset_range min ({lo_min}) must be <= max ({lo_max})",
        ))
    if n_layers > 0:
        if abs(lo_min) >= n_layers or abs(lo_max) >= n_layers:
            errors.append(_err(
                "pp_config.layer_offset_range",
                f"layer_offset_range ({lo_min}, {lo_max}) exceeds "
                f"num_layers ({n_layers})",
            ))


def _check_layer_recompute(
    errors: List[ValidationError],
    model: Dict,
    pp_cfg: Dict,
) -> None:
    """Check that layer_recompute_layers indices are within [0, num_hidden_layers)."""
    recompute_layers = pp_cfg.get("layer_recompute_layers", [])
    if not recompute_layers:
        return

    n_layers = model.get("num_hidden_layers", 0)
    if n_layers <= 0:
        return

    invalid = [idx for idx in recompute_layers
               if idx < 0 or idx >= n_layers]
    if invalid:
        errors.append(_err(
            "pp_config.layer_recompute_layers",
            f"layer_recompute_layers references non-existent layers: {invalid}. "
            f"Valid range: [0, {n_layers - 1}]",
        ))


def _check_device_product_limit(
    errors: List[ValidationError],
    search: Dict,
    constraint: Dict,
    cluster: Dict,
) -> None:
    """Check that parallel dimension product does not exceed available devices."""
    total_devices = _total_cards(cluster)
    if total_devices <= 0:
        return

    # DP is decomposed into replicate * shard when FSDP/HSDP is used.
    # Compute the minimum product across all combinations to correctly
    # check whether any valid DP decomposition fits the device budget.
    dp_repl_vals = search.get("data_parallel_replicate_degree", [1]) or [1]
    dp_shard_vals = search.get("data_parallel_shard_degree", [1]) or [1]
    fixed_dp = constraint.get("fixed_dp_degree")
    if fixed_dp is not None and fixed_dp > 0:
        dp_min = fixed_dp
    else:
        dp_min = min(dp_repl_vals) * min(dp_shard_vals)

    dim_keys = {
        "tensor_parallel_degree": "tp",
        "pipeline_parallel_degree": "pp",
        "context_parallel_degree": "cp",
        "expert_parallel_degree": "ep",
    }
    fixed_overrides = {
        "tp": constraint.get("fixed_tp_degree"),
        "pp": constraint.get("fixed_pp_degree"),
        "cp": constraint.get("fixed_cp_degree"),
        "ep": constraint.get("fixed_ep_degree"),
    }

    # Use the *minimum* product to check that at least one valid
    # combination fits within the device budget.  The enumerator
    # (the strategy enumerator) will filter invalid combos.
    min_product = dp_min
    for search_key, dim_label in dim_keys.items():
        fixed_val = fixed_overrides.get(dim_label)
        if fixed_val is not None and fixed_val > 0:
            min_product *= fixed_val
        else:
            candidates = search.get(search_key, [1])
            if not candidates:
                candidates = [1]
            min_product *= min(candidates)

    if min_product > total_devices:
        errors.append(_err(
            "search_space",
            f"Minimum product of parallel dimensions ({min_product}) exceeds "
            f"total available devices ({total_devices})",
        ))


def _check_memory_limit(
    errors: List[ValidationError],
    cluster: Dict,
    constraint: Dict,
) -> None:
    """Check that memory_limit_gb is non-negative and does not exceed device memory."""
    memory_limit = constraint.get("memory_limit_gb", 0.0)
    if memory_limit < 0:
        errors.append(_err(
            "constraint.memory_limit_gb",
            f"memory_limit_gb must be >= 0, got {memory_limit}",
        ))

    device_memory = cluster.get("device_memory_gb", 0.0)
    if memory_limit > 0 and device_memory > 0:
        if memory_limit > device_memory:
            errors.append(_warn(
                "constraint.memory_limit_gb",
                f"memory_limit_gb ({memory_limit}) exceeds "
                f"device_memory_gb ({device_memory})",
            ))


def _check_dense_model_ep_cp_warning(
    errors: List[ValidationError],
    model: Dict,
    search: Dict,
) -> None:
    """Issue 126 Section 5 rule 9: warn when EP/CP are enabled on Dense models.

    If ``moe_enabled`` is ``False`` and ``ep_degree`` or ``cp_degree``
    candidates contain values > 1, emit a warning since EP/CP are
    typically used for MoE and long-sequence scenarios respectively.
    """
    moe_enabled = model.get("moe_enabled", False)
    if moe_enabled:
        return

    ep_vals = search.get("expert_parallel_degree", [1])
    if ep_vals and any(v > 1 for v in ep_vals):
        errors.append(_warn(
            "search_space.expert_parallel_degree",
            "Expert Parallelism (ep > 1) is configured but moe_enabled is "
            "False. EP has no effect on Dense LLMs.",
        ))

    cp_vals = search.get("context_parallel_degree", [1])
    if cp_vals and any(v > 1 for v in cp_vals):
        errors.append(_warn(
            "search_space.context_parallel_degree",
            "Context Parallelism (cp > 1) is configured on a Dense LLM. "
            "Ensure this is intentional for long-sequence scenarios.",
        ))


def _check_fsdp_hsdp_device_product(
    errors: List[ValidationError],
    search: Dict,
    constraint: Dict,
    cluster: Dict,
) -> None:
    """Issue 127 constraint: FSDP/HSDP device product validation.

    Verifies that the sum of shard-degree and replicate-degree
    (which together form a complete DP decomposition) does not
    exceed available devices when combined with TP/PP/CP/EP.
    """
    total_devices = _total_cards(cluster)
    if total_devices <= 0:
        return

    fixed_dp = constraint.get("fixed_dp_degree")
    dp_shard_vals = search.get("data_parallel_shard_degree", [1])
    dp_repl_vals = search.get("data_parallel_replicate_degree", [1])

    if fixed_dp is not None and fixed_dp > 0:
        dp_shard_vals = [fixed_dp]
        dp_repl_vals = [1]

    fixed_overrides = {
        "tp": constraint.get("fixed_tp_degree"),
        "pp": constraint.get("fixed_pp_degree"),
        "cp": constraint.get("fixed_cp_degree"),
        "ep": constraint.get("fixed_ep_degree"),
    }

    dim_keys = {
        "tensor_parallel_degree": "tp",
        "pipeline_parallel_degree": "pp",
        "context_parallel_degree": "cp",
        "expert_parallel_degree": "ep",
    }

    has_over_product = True
    for dp_shard in (dp_shard_vals or [1]):
        for dp_repl in (dp_repl_vals or [1]):
            product = dp_shard * dp_repl
            for search_key, dim_label in dim_keys.items():
                fixed_val = fixed_overrides.get(dim_label)
                if fixed_val is not None and fixed_val > 0:
                    product *= fixed_val
                else:
                    candidates = search.get(search_key, [1]) or [1]
                    product *= min(candidates)
            if product <= total_devices:
                has_over_product = False
                break
        if not has_over_product:
            break

    if has_over_product:
        errors.append(_err(
            "search_space",
            f"No FSDP/HSDP decomposition (dp_shard * dp_replicate) "
            f"fits within total available devices ({total_devices}) "
            f"when combined with TP/PP/CP/EP dimensions.",
        ))
