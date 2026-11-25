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
"""Versioned, JSON-only transport of the Actor's explicit computation plan."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from rl.unified_execution.batch_invariant import normalize_policy
from rl.unified_execution.execution_plan import (
    resolve_pik_leaves,
    has_unified_ep,
    validate_cp_plan,
    validate_pcp_plan,
    validate_pik_contract,
)
from rl.unified_execution.models import get_model_family

_PLAN_FIELDS = frozenset((
    "match", "replace", "when", "low_precision_dtype_scheme", "region_dispatch",
    "params", "in_src", "in_dst", "out_src", "out_dst", "tp_divide_attrs",
))
_PAYLOAD_FIELDS = frozenset((
    "schema_version", "mode", "plan_overrides", "compute_plan_overrides", "excluded_plan_overrides",
    "attn_implementation", "plan_fingerprint", "training_structure",
    "parallelism", "training_parallelism", "attention_backend", "batch_invariant", "model_family", "expert_parallelism",
))


def fingerprint(value: Any) -> str:
    """Hash canonical JSON without allowing unserializable or nonfinite values."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_plan_overrides(entries: Any) -> list[dict]:
    """Validate the complete shared plan without dropping layout or method rules."""
    if not isinstance(entries, list):
        raise ValueError("train.plan_overrides must be an explicit list")
    plan = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) - _PLAN_FIELDS:
            raise ValueError(f"Unknown plan_overrides fields at entry {index}")
        match = entry.get("match")
        if not (isinstance(match, str) and match or isinstance(match, list)
                and match and all(isinstance(item, str) and item for item in match)):
            raise ValueError(f"plan_overrides[{index}].match must be a nonempty string or list")
        condition = entry.get("when")
        if condition not in (None, "cp", "ep", "sequence_parallel", "low_precision"):
            raise ValueError(f"Unsupported plan_overrides condition: {condition!r}")
        action = entry.get("replace")
        if action is not None:
            _validate_action(action, index)
        # HP accepts grouped pure module replacements. Its layout and method
        # builders require individual strings, in the same declaration order.
        if isinstance(match, list) and (action is None or action["kind"] == "method"
                                        or set(entry) - {"match", "replace", "when", "low_precision_dtype_scheme"}):
            plan.extend({**entry, "match": pattern} for pattern in match)
        else:
            plan.append(dict(entry))
    return plan


def _validate_action(action: Any, index: int) -> None:
    """Validate the replacement kind and its import target."""
    if not isinstance(action, Mapping) or set(action) - {"kind", "implementation", "source_type", "exact_type"}:
        raise ValueError(f"Unknown replacement action at plan_overrides[{index}]")
    if action.get("kind") not in ("module", "method"):
        raise ValueError("Replacement kind must be module or method")
    implementation = action.get("implementation")
    target = implementation.get("_target_") if isinstance(implementation, Mapping) else implementation
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Replacement implementation must declare an importable target")


def _mapping(config: Mapping, name: str) -> Mapping:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_numerical_selection(consistency: Mapping) -> dict[str, Any] | None:
    """Keep numerical policy opt-in and tied to the shared V4 attention path."""
    mode = consistency.get("mode")
    if mode not in ("architecture", "batch_invariant"):
        raise ValueError("consistency.mode must be architecture or batch_invariant")
    if mode == "architecture":
        if "batch_invariant" in consistency:
            raise ValueError("consistency.batch_invariant requires mode=batch_invariant")
        return None
    if consistency.get("attention_backend") != "flash_attention_npu_v4":
        raise ValueError("consistency.mode=batch_invariant requires attention_backend=flash_attention_npu_v4")
    policy = consistency.get("batch_invariant")
    if not isinstance(policy, Mapping):
        raise ValueError("consistency.batch_invariant must be an explicit mapping")
    return normalize_policy(policy)


def _validate_consistency_selection(consistency: Mapping) -> str:
    """Validate schema and numerical selection before reading model options."""
    if not {"schema_version", "mode"} <= set(consistency) or set(consistency) - {
        "schema_version", "mode", "attention_backend", "batch_invariant",
    }:
        raise ValueError("consistency requires schema_version=1 and mode=architecture or batch_invariant")
    attention_backend = consistency.get("attention_backend", "native")
    if attention_backend not in ("native", "flash_attention_npu_v4"):
        raise ValueError("consistency.attention_backend must be native or flash_attention_npu_v4")
    version = consistency["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError("consistency.schema_version must be 1")
    _validate_numerical_selection(consistency)
    if "batch_invariant" in consistency and set(consistency["batch_invariant"]) != {
        "provider", "fill_uninitialized_memory",
    }:
        raise ValueError("consistency.batch_invariant requires only provider and fill_uninitialized_memory")
    return attention_backend


def _validate_model_parallelism(config, family, train, accelerator, vllm, training, inference) -> None:
    """Check model-specific expert topology and graph execution requirements."""
    if family.moe:
        _validate_moe_parallelism(training, inference)
        if vllm.get("enable_expert_parallel", False) or vllm.get("enable_eplb", False):
            raise ValueError(
                "Unified MoE does not use vLLM engine flags for expert ownership. Configure expert_parallel_size."
            )
        if train.get("router_replay", {}).get("mode", "off") not in ("off", "disabled", None):
            raise ValueError("Unified MoE router replay is not integrated")
        expert = _expert_options(config)
        if expert["inference"] > 1 and not has_unified_ep(train.get("plan_overrides", [])):
            raise ValueError("Unified HP EP requires the shared EP recipe in both eager and graph mode")
    elif accelerator.get("ep", 1) != 1 or vllm.get("expert_parallel_size", 1) != 1:
        raise ValueError("Dense unified execution does not use expert parallelism")


def _validate_precision(train: Mapping, vllm: Mapping) -> None:
    """Require the shared BF16 path without independent rollout quantization."""
    if vllm.get("batch_invariant", False) is not False:
        raise ValueError("Unified execution requires rollout.vllm.batch_invariant=false")
    if vllm.get("quantization") is not None or _mapping(train, "low_precision").get("enabled", False):
        raise ValueError("Unified execution does not support quantization or low_precision")
    precision = _mapping(train, "mixed_precision")
    bf16_names = ("bfloat16", "bf16")
    if (not precision.get("enabled", True) or precision.get("param_dtype", "bfloat16") not in bf16_names
            or vllm.get("dtype", "bfloat16") not in bf16_names):
        raise ValueError("Unified execution requires BF16 training and rollout computation")


def _validate_native_config(config: Mapping[str, Any]) -> None:
    """Require explicit consistency configuration before selecting unified CP or worker paths."""
    if _mapping(_mapping(config, "train"), "accelerator").get("cp", 1) != 1:
        raise ValueError(
            "Hyper-RL invalid topology: cp=1 is required without explicit unified batch-invariant execution"
        )
    vllm = _mapping(_mapping(config, "rollout"), "vllm")
    if vllm.get("context_parallel_size", 1) != 1:
        raise ValueError("Inference PCP requires explicit unified batch-invariant execution")
    if vllm.get("model_implementation") == "unified" or "unified_execution" in vllm:
        raise ValueError("Unified model selection requires an explicit consistency configuration")


def validate_unified_config(config: Mapping[str, Any]) -> None:
    """Validate model execution independently of numerical acceptance topologies."""
    if "consistency" not in config:
        _validate_native_config(config)
        return
    consistency = _mapping(config, "consistency")
    attention_backend = _validate_consistency_selection(consistency)
    model, train, rollout = (_mapping(config, name) for name in ("model", "train", "rollout"))
    family = get_model_family(model.get("name"))
    accelerator, vllm = _mapping(train, "accelerator"), _mapping(rollout, "vllm")
    if rollout.get("engine", "vllm") != "vllm":
        raise ValueError("Unified execution requires vLLM rollout")
    training, inference = _parallel_options(config)
    train = {**train, "plan_overrides": _normalize_plan_overrides(train.get("plan_overrides"))}
    _validate_training_cp(config, train, training)
    _validate_model_parallelism(config, family, train, accelerator, vllm, training, inference)
    if vllm.get("decode_context_parallel_size", 1) != 1:
        raise ValueError("Unified inference requires DCP=1")
    if inference["cp"] > 1:
        validate_pcp_plan(train.get("plan_overrides", []), inference, attention_backend,
                          _validate_numerical_selection(consistency))
        if family.moe and (_expert_options(config)["inference"] <= 1
                           or not has_unified_ep(train.get("plan_overrides", []))):
            raise ValueError("Inference MoE PCP requires graph-capable HP EP > 1")
    if attention_backend == "native" and not vllm.get("enforce_eager", False):
        raise ValueError("Unified execution requires rollout.vllm.enforce_eager=true")
    if attention_backend == "flash_attention_npu_v4":
        if vllm.get("attention_backend") not in (None, "CUSTOM"):
            raise ValueError("A unified V4 attention selection cannot be overridden on rollout only")
    _validate_precision(train, vllm)
    if ("async_training" in config or "colocate_async" in config or train.get("async_training")
            or train.get("colocate_async") or train.get("mode", "sync") != "sync"):
        raise ValueError("Unified execution currently supports synchronous training only")


def _validate_training_cp(config: Mapping, train: Mapping, topology: Mapping) -> None:
    """Keep unvalidated RL combinations out of the new sequence-sharded path."""
    if topology["cp"] == 1:
        return
    accelerator = _mapping(train, "accelerator")
    if (topology["cp"] not in (2, 4) or topology["pp"] != 1
            or _mapping(config, "algorithm").get("name") != "grpo"
            or _mapping(config, "algorithm").get("loss_aggregation", "token-mean") != "token-mean"
            or train.get("token_log_probs_mode", "dense") != "dense"):
        raise ValueError("Training CP supports CP2/CP4, PP1 and dense token-mean GRPO")
    unsupported = [key for key in ("max_tokens_per_micro_batch", "old_log_prob_max_tokens_per_micro_batch",
                                   "reference_log_prob_max_tokens_per_micro_batch", "balance_dp_tokens")
                   if train.get(key)]
    checkpoint_mode = accelerator.get("activation_checkpoint", "off")
    if isinstance(checkpoint_mode, Mapping):
        checkpoint_mode = checkpoint_mode.get("mode")
    if (unsupported or checkpoint_mode not in ("off", False)
            or accelerator.get("loss_parallel", False) or "distillation" in config
            or _mapping(train, "router_replay").get("mode", "off") not in ("off", "disabled", None)):
        raise ValueError("Training CP does not support dynamic batching, AC, loss parallel, distillation or replay")
    validate_cp_plan(train.get("plan_overrides", []), topology,
                     _mapping(config, "consistency").get("attention_backend"),
                     _validate_numerical_selection(_mapping(config, "consistency")))
    if get_model_family(_mapping(config, "model").get("name")).moe:
        experts = _expert_options(config)
        if (min(experts.values()) <= 1
                or not has_unified_ep(train.get("plan_overrides", []))):
            raise ValueError("Training MoE CP requires the graph-capable EP recipe on both sides with EP > 1")


def _validate_parallelism(value: Mapping) -> dict[str, Any]:
    """Normalize the independent compute axes carried to inference workers."""
    if set(value) != {"tp", "cp", "pp", "sequence_parallel"}:
        raise ValueError("Parallelism requires tp, cp, pp and sequence_parallel")
    result = dict(value)
    for axis in ("tp", "cp", "pp"):
        size = result[axis]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"parallelism.{axis} must be a positive integer")
    if not isinstance(result["sequence_parallel"], bool):
        raise ValueError("parallelism.sequence_parallel must be a boolean")
    if result["sequence_parallel"] and result["tp"] == 1:
        raise ValueError("sequence_parallel requires TP > 1")
    return result


def _validate_inference_parallelism(value: Mapping) -> dict[str, Any]:
    """Reject axes without a shared HP paged-inference execution path."""
    result = _validate_parallelism(value)
    if result["cp"] not in (1, 2, 4) or result["pp"] != 1:
        raise ValueError(
            "Unified inference supports PCP1/PCP2/PCP4 with PP=1 and DCP=1."
        )
    return result


def _parallel_options(config: Mapping) -> tuple[dict, dict]:
    """Normalize independent training and inference compute axes."""
    train = _mapping(_mapping(config, "train"), "accelerator")
    vllm = _mapping(_mapping(config, "rollout"), "vllm")
    training = _validate_parallelism({
        **{axis: train.get(axis, 1) for axis in ("tp", "cp", "pp")},
        "sequence_parallel": train.get("sequence_parallel", False),
    })
    inference = _validate_inference_parallelism({
        "tp": vllm.get("tensor_parallel_size", training["tp"]),
        "cp": vllm.get("context_parallel_size", 1),
        "pp": vllm.get("pipeline_parallel_size", training["pp"]),
        "sequence_parallel": vllm.get("sequence_parallel", training["sequence_parallel"]),
    })
    return training, inference


def _validate_moe_parallelism(training: Mapping, inference: Mapping) -> None:
    """Keep unsupported context and pipeline execution out of the MoE contract."""
    for parallelism in (training, inference):
        if parallelism["pp"] != 1:
            raise ValueError("Unified Qwen3-MoE requires PP=1")


def _expert_options(config: Mapping) -> dict[str, int]:
    """Validate expert degrees against the full training and inference rank domains."""
    train = _mapping(_mapping(config, "train"), "accelerator")
    rollout = _mapping(_mapping(config, "rollout"), "vllm")
    result = _validate_expert_options({
        "training": train.get("ep", 1), "inference": rollout.get("expert_parallel_size", train.get("ep", 1)),
    })
    domains = {
        "training": (train.get("dp_replicate", 1) * train.get("dp_shard", 1)
                     * train.get("cp", 1) * train.get("tp", 1)),
        "inference": (rollout.get("data_parallel_size", 1) * rollout.get("context_parallel_size", 1)
                      * rollout.get("tensor_parallel_size", train.get("tp", 1))),
    }
    for side, size in result.items():
        if domains[side] % size:
            axes = "DP x CP x TP" if side == "training" else "DP x PCP x TP"
            raise ValueError(f"Unified {side} EP={size} must divide {axes}={domains[side]}")
    return result


def _validate_expert_options(value: Mapping) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"training", "inference"}:
        raise ValueError("expert_parallelism requires training and inference sizes")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in value.values()):
        raise ValueError("expert_parallelism sizes must be positive integers")
    return dict(value)


def _make_payload(entries: list, attn_implementation: str, training: Mapping, inference: Mapping,
                  attention_backend: str = "native", mode: str = "architecture",
                  batch_invariant: Mapping | None = None, model_family: str = "qwen3",
                  expert_parallelism: Mapping | None = None) -> dict[str, Any]:
    """Serialize the shared execution plan and its fingerprint."""
    entries = _normalize_plan_overrides(json.loads(json.dumps(entries, allow_nan=False)))
    training = _validate_parallelism(training)
    inference = _validate_inference_parallelism(inference)
    resolve_pik_leaves(entries, training, inference)
    compute = [entry for entry in entries if (entry.get("replace") or {}).get("kind") == "module"]
    result = {
        "schema_version": 1, "mode": mode, "plan_overrides": entries,
        "compute_plan_overrides": compute, "excluded_plan_overrides": [],
        "attn_implementation": attn_implementation,
        "parallelism": inference,
        "training_parallelism": training,
        "attention_backend": attention_backend,
    }
    family = get_model_family(model_family)
    validate_pik_contract(entries, batch_invariant, model_family, training, inference)
    if family.moe:
        _validate_moe_parallelism(result["training_parallelism"], result["parallelism"])
        result["model_family"] = family.model_type
        if expert_parallelism is not None:
            result["expert_parallelism"] = _validate_expert_options(expert_parallelism)
    elif expert_parallelism is not None:
        raise ValueError("Dense unified execution cannot declare expert_parallelism")
    if mode == "batch_invariant":
        result["batch_invariant"] = batch_invariant
    policy = _validate_numerical_selection(result)
    if policy is not None:
        result["batch_invariant"] = policy
    validate_cp_plan(entries, result["training_parallelism"], attention_backend, policy)
    validate_pcp_plan(entries, result["parallelism"], attention_backend, policy)
    result["plan_fingerprint"] = fingerprint(result)
    return result


def build_unified_payload(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Snapshot the explicit Actor YAML plan for each inference worker."""
    validate_unified_config(config)
    if "consistency" not in config:
        return None
    model = _mapping(config, "model")
    selection = model.get("attention_implementation")
    if selection is not None and (selection != "transformers_builtin" or "attn_implementation" in model):
        raise ValueError("Unified execution requires one supported model attention selection")
    attention = model.get("attn_implementation", "sdpa")
    if attention not in ("sdpa", "eager", "flash_attention_2"):
        raise ValueError("Unsupported unified execution attention implementation")
    training, inference = _parallel_options(config)
    consistency = config["consistency"]
    return _make_payload(config["train"]["plan_overrides"], attention, training, inference,
                         consistency.get("attention_backend", "native"), consistency["mode"],
                         consistency.get("batch_invariant"), model.get("name"),
                         _expert_options(config) if get_model_family(model.get("name")).moe else None)


def parse_unified_payload(payload: Any) -> dict[str, Any]:
    """Recompute derived rules and the digest at a worker trust boundary."""
    if not isinstance(payload, Mapping) or set(payload) - _PAYLOAD_FIELDS:
        raise ValueError("Invalid unified execution payload fields")
    version = payload.get("schema_version")
    if (not isinstance(version, int) or isinstance(version, bool) or version != 1
            or payload.get("mode") not in ("architecture", "batch_invariant")):
        raise ValueError("Unsupported unified execution payload version or mode")
    attention = payload.get("attn_implementation")
    if attention not in ("sdpa", "eager", "flash_attention_2"):
        raise ValueError("Unsupported unified execution attention implementation")
    backend = payload.get("attention_backend", "native")
    if backend not in ("native", "flash_attention_npu_v4"):
        raise ValueError("Unsupported unified numerical attention backend")
    _validate_numerical_selection(payload)
    expected = _make_payload(payload.get("plan_overrides"), attention,
                             _mapping(payload, "training_parallelism"), _mapping(payload, "parallelism"), backend,
                             payload["mode"], payload.get("batch_invariant"), payload.get("model_family", "qwen3"),
                             payload.get("expert_parallelism"))
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Unified execution payload plan or fingerprint mismatch")
    if "training_structure" in payload:
        structure = payload["training_structure"]
        if (not isinstance(structure, Mapping)
                or structure.get("fingerprint") != fingerprint(structure.get("structure"))):
            raise ValueError("Invalid training structure fingerprint")
        expected["training_structure"] = json.loads(json.dumps(structure, allow_nan=False))
    return expected
