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
"""RL configuration limits."""
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from rl.config.schema import _positive_integer, _trainer_data_parallel_size, required_mapping
from rl.config.validate import _reject_removed_vllm_topology


def _load_automatic_limit_text_config(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and validate the model metadata used for automatic vLLM limits."""
    model_path = Path(str(model["weights_path"])) / "config.json"
    try:
        with model_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Automatic max_num_seqs requires a readable model config: {model_path}"
        ) from error
    text_config = model_config.get("text_config", model_config)
    layer_types = text_config.get("layer_types") or []
    if text_config.get("model_type") not in {"qwen3", "qwen3_moe"} or any(
        layer_type != "full_attention" for layer_type in layer_types
    ):
        raise ValueError(
            "Automatic max_num_seqs currently supports Qwen3 "
            "full-attention models only"
        )
    return text_config


def _validate_automatic_limit_inputs(
    data: Mapping[str, Any],
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    train: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    vllm: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Validate automatic-limit inputs and return DP, TP, and observation limits."""
    required_vllm_fields = (
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_batched_tokens",
        "block_size",
    )
    missing = [field for field in required_vllm_fields if vllm.get(field) is None]
    if missing:
        raise ValueError(
            f"Automatic max_num_seqs requires explicit rollout.vllm fields: {missing}"
        )
    positive_fields = (
        (data, "max_prompt_length", "data"),
        (rollout, "max_new_tokens", "rollout"),
        (rollout, "num_return_sequences", "rollout"),
        (agentic, "max_turns", "agentic"),
        (train, "prompt_batch_size", "train"),
        (accelerator, "dp_shard", "train.accelerator"),
        (vllm, "kv_cache_memory_bytes", "rollout.vllm"),
        (vllm, "max_model_len", "rollout.vllm"),
        (vllm, "max_num_batched_tokens", "rollout.vllm"),
        (vllm, "block_size", "rollout.vllm"),
    )
    for section, field, prefix in positive_fields:
        _ = _positive_integer(section, field, prefix)
    data_parallel_size = _positive_integer(
        vllm, "data_parallel_size", "rollout.vllm"
    )
    tensor_parallel_size = _positive_integer(
        vllm, "tensor_parallel_size", "rollout.vllm"
    )
    max_observation_tokens = agentic.get("max_observation_tokens", 0)
    if (
        not isinstance(max_observation_tokens, int)
        or isinstance(max_observation_tokens, bool)
        or max_observation_tokens < 0
    ):
        raise ValueError("agentic.max_observation_tokens must be a non-negative integer")
    return data_parallel_size, tensor_parallel_size, max_observation_tokens


def _automatic_context_tokens(
    data: Mapping[str, Any],
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    vllm: Mapping[str, Any],
    max_observation_tokens: int,
) -> int:
    """Calculate and validate the maximum per-sequence context length."""
    max_turns = int(agentic["max_turns"])
    context_tokens = (
        int(data["max_prompt_length"])
        + max_turns * int(rollout["max_new_tokens"])
        + (max_turns - 1) * max_observation_tokens
    )
    max_model_len = int(vllm["max_model_len"])
    if context_tokens > max_model_len:
        raise ValueError(
            "Automatic max_num_seqs requires workload context within max_model_len: "
            f"context_tokens={context_tokens}, max_model_len={max_model_len}"
        )
    return context_tokens


def _automatic_kv_capacity(
    text_config: Mapping[str, Any],
    vllm: Mapping[str, Any],
    tensor_parallel_size: int,
    context_tokens: int,
    dtype_bytes: int,
) -> int:
    """Calculate how many maximum-length sequences fit in the KV cache."""
    num_kv_heads = int(text_config["num_key_value_heads"])
    if num_kv_heads % tensor_parallel_size != 0:
        raise ValueError(
            "Automatic max_num_seqs requires num_key_value_heads divisible by "
            f"tensor_parallel_size: {num_kv_heads} % {tensor_parallel_size} != 0"
        )
    kv_heads_per_rank = num_kv_heads // tensor_parallel_size
    head_dim = int(
        text_config.get(
            "head_dim",
            int(text_config["hidden_size"])
            // int(text_config["num_attention_heads"]),
        )
    )
    block_size = int(vllm["block_size"])
    block_bytes = (
        int(text_config["num_hidden_layers"])
        * block_size
        * 2
        * kv_heads_per_rank
        * head_dim
        * dtype_bytes
    )
    pool_blocks = int(vllm["kv_cache_memory_bytes"]) // block_bytes
    blocks_per_sequence = math.ceil(context_tokens / block_size)
    kv_capacity = pool_blocks // blocks_per_sequence
    if kv_capacity <= 0:
        raise ValueError(
            "Automatic max_num_seqs found insufficient KV cache for one "
            "maximum-length sequence"
        )
    return kv_capacity


def resolve_vllm_automatic_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve workload- and KV-bounded vLLM limits before validation.

    Args:
        config: Fully merged Hyper-RL configuration.

    Returns:
        A detached configuration with ``max_num_seqs`` resolved when it is null.

    Raises:
        ValueError: If automatic capacity lacks an explicit resource bound or the
            model uses an unsupported hybrid KV layout.
    """
    resolved = deepcopy(dict(config))
    rollout = resolved.get("rollout")
    if not isinstance(rollout, dict) or rollout.get("engine") != "vllm":
        return resolved
    vllm = rollout.get("vllm")
    if isinstance(vllm, Mapping):
        _reject_removed_vllm_topology(vllm)
    if (
        not isinstance(vllm, dict)
        or "max_num_seqs" not in vllm
        or vllm["max_num_seqs"] is not None
    ):
        return resolved
    model = required_mapping(resolved, "model")
    data = required_mapping(resolved, "data")
    agentic = required_mapping(resolved, "agentic")
    train = required_mapping(resolved, "train")
    accelerator = required_mapping(train, "accelerator")
    text_config = _load_automatic_limit_text_config(model)
    data_parallel_size, tensor_parallel_size, max_observation_tokens = (
        _validate_automatic_limit_inputs(
            data, rollout, agentic, train, accelerator, vllm
        )
    )
    kv_capacity = _automatic_kv_limit(
        data, rollout, agentic, vllm, text_config, tensor_parallel_size, max_observation_tokens
    )
    workload_capacity = _automatic_workload_capacity(train, accelerator, rollout, data_parallel_size)
    vllm["max_num_seqs"] = min(
        workload_capacity, kv_capacity, int(vllm["max_num_batched_tokens"])
    )
    return resolved


def _automatic_kv_limit(
    data: Mapping[str, Any], rollout: Mapping[str, Any], agentic: Mapping[str, Any],
    vllm: Mapping[str, Any], text_config: Any, tensor_parallel_size: int, max_observation_tokens: int,
) -> int:
    """Resolve the existing dtype and context bounded KV capacity."""
    dtype = str(vllm.get("dtype", "bfloat16"))
    dtype_bytes = {"bfloat16": 2, "bf16": 2}.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"Automatic max_num_seqs does not support dtype {dtype!r}")
    context_tokens = _automatic_context_tokens(
        data, rollout, agentic, vllm, max_observation_tokens
    )
    kv_capacity = _automatic_kv_capacity(
        text_config,
        vllm,
        tensor_parallel_size,
        context_tokens,
        dtype_bytes,
    )
    return kv_capacity


def _automatic_workload_capacity(
    train: Mapping[str, Any], accelerator: Mapping[str, Any],
    rollout: Mapping[str, Any], data_parallel_size: int,
) -> int:
    """Compute the number of generated children per rollout replica."""
    trainer_dp_size = _trainer_data_parallel_size(accelerator)
    global_children = (
        trainer_dp_size
        * int(train.get("prompt_batch_size", 1))
        * int(rollout["num_return_sequences"])
    )
    return math.ceil(global_children / data_parallel_size)
