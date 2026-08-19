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
"""Validate a real Qwen CPU weight refit through native or Hyper vLLM."""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from safetensors import safe_open
from vllm import LLM

from rl.config import build_model_registration
from rl.roles.model import VLLMModelRegistration, resolve_vllm_model
from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.vllm import (
    VLLMGenerationEngine,
    _policy_weight_fingerprint,
    _verify_policy_fingerprints,
)
from rl.roles.rollout.vllm_plugin import register_hyper_models
from rl.roles.weight_sync import CPUWeightTransfer, PolicySnapshot


_LOGGER = logging.getLogger(__name__)
_MUTATED_WEIGHT = "model.norm.weight"
_PROMPT = "Explain why the sky appears blue in two sentences."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--implementation", choices=("native", "hyper"), required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _actor_weight_name(
    source_name: str,
    rollout_model: VLLMModelRegistration,
) -> Optional[str]:
    if rollout_model.family == "qwen3":
        return source_name
    if source_name.startswith("model.language_model."):
        return "model." + source_name.removeprefix("model.language_model.")
    if rollout_model.model.native_uses_language_model_prefix:
        return source_name if source_name == "lm_head.weight" else None
    if source_name.startswith("model.") or source_name == "lm_head.weight":
        return source_name
    return None


def _load_actor_state_dict(
    model_path: Path,
    rollout_model: VLLMModelRegistration,
) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    state_dict = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        names = sorted(
            name
            for name, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        )
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for source_name in names:
                target_name = _actor_weight_name(source_name, rollout_model)
                if target_name is None:
                    continue
                if target_name in state_dict:
                    raise ValueError(f"Duplicate Actor weight mapping for {target_name!r}")
                state_dict[target_name] = shard.get_tensor(source_name)
    if _MUTATED_WEIGHT not in state_dict:
        raise ValueError(f"Checkpoint does not contain {_MUTATED_WEIGHT}")
    return state_dict


def _response_record(result: Any) -> dict[str, Any]:
    response_tokens = result.sequences[0, -result.response_mask.shape[1]:]
    selected_tokens = response_tokens[result.response_mask[0]].tolist()
    selected_logprobs = result.rollout_log_probs[0, result.response_mask[0]].tolist()
    if not selected_tokens or not all(math.isfinite(value) for value in selected_logprobs):
        raise ValueError("Rollout must return tokens with finite selected-token logprobs")
    return {
        "response_token_ids": selected_tokens,
        "selected_logprobs": selected_logprobs,
    }


def main() -> None:
    """Load, mutate, refit, and validate one real Qwen policy update."""
    args = _parse_args()
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2 to execute a decode step")

    registration = build_model_registration(
        {
            "model": {
                "registry_name": "qwen_refit_gate",
                "name": "qwen",
                "weights_path": str(args.model),
                "tokenizer_path": str(args.model),
            }
        }
    )
    rollout_model = resolve_vllm_model(registration, args.implementation)
    options = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max(1024, args.max_model_len),
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if rollout_model.is_hyper:
        register_hyper_models()
        options["hf_overrides"] = {"architectures": [rollout_model.architecture]}
    client = LLM(**options)
    tokenizer = client.get_tokenizer()
    tokenized = tokenizer(_PROMPT, return_tensors="pt")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id and a usable padding token")

    engine = VLLMGenerationEngine(
        registration,
        {"vllm": {"model_implementation": args.implementation}},
        client=client,
        refitter=CPUWeightTransfer(rollout_model),
        rollout_model=rollout_model,
    )
    request = GenerationRequest(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        settings=GenerationSettings(
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            do_sample=False,
            pad_token_id=int(pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            collect_log_probs=True,
        ),
    )

    before = _response_record(engine.generate(request))
    state_dict = _load_actor_state_dict(args.model, rollout_model)
    state_dict[_MUTATED_WEIGHT].zero_()
    expected_fingerprint = _policy_weight_fingerprint(state_dict)
    engine.update_weights(
        PolicySnapshot(
            version=1,
            model_name=registration.name,
            payload=state_dict,
            metadata={"reason": "real_refit_gate"},
        )
    )
    after = _response_record(engine.generate(request))
    logprob_differences = [
        abs(before_value - after_value)
        for before_value, after_value in zip(
            before["selected_logprobs"],
            after["selected_logprobs"],
        )
    ]
    if (
        before["response_token_ids"] == after["response_token_ids"]
        and max(logprob_differences, default=0.0) <= 1.0e-6
    ):
        raise ValueError("The mutated policy did not change tokens or selected log-probabilities")
    if engine.policy_version != 1:
        raise ValueError(f"Expected policy version 1 after refit, got {engine.policy_version}")
    worker_fingerprints = client.collective_rpc(
        "get_policy_weight_fingerprint",
        kwargs={},
    )
    _verify_policy_fingerprints(
        expected_fingerprint,
        worker_fingerprints,
        expected_version=1,
    )

    report = {
        "implementation": args.implementation,
        "model_family": rollout_model.family,
        "architecture": client.llm_engine.model_config.architecture,
        "before": before,
        "after": after,
        "max_abs_selected_logprob_change": max(logprob_differences, default=0.0),
        "mutated_weight": _MUTATED_WEIGHT,
        "policy_fingerprint": expected_fingerprint,
        "worker_fingerprints": worker_fingerprints,
        "policy_version": engine.policy_version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    _LOGGER.info("Validated CPU refit and policy version %d", engine.policy_version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
