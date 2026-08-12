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
"""Validate a real CPU weight refit for native or Hyper vLLM."""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
from vllm import LLM

from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.vllm import (
    CPUStateDictRefitter,
    VLLMGenerationEngine,
    _policy_weight_fingerprint,
    _verify_policy_fingerprints,
)
from rl.roles.rollout.vllm_plugin import register_hyper_models
from rl.roles.rollout.vllm_policy import architecture_for_implementation
from rl.roles.weight_sync import PolicySnapshot


_LOGGER = logging.getLogger(__name__)
_MUTATED_WEIGHT = "model.language_model.norm.weight"
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


def _load_text_state_dict(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    state_dict = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        names = sorted(
            name
            for name, mapped_shard in weight_map.items()
            if mapped_shard == shard_name and name.startswith("model.language_model.")
        )
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                state_dict[name] = shard.get_tensor(name)
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
    """Load, mutate, refit, and validate one real Qwen3.5 policy update."""
    args = _parse_args()
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2 to execute a decode step")

    register_hyper_models()
    client = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(1024, args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_overrides={
            "architectures": [architecture_for_implementation(args.implementation)]
        },
    )
    tokenizer = client.get_tokenizer()
    tokenized = tokenizer(_PROMPT, return_tensors="pt")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id and a usable padding token")

    registration = ModelRegistration(
        "qwen3_5_refit_gate",
        "qwen3_5",
        str(args.model),
        str(args.model),
    )
    engine = VLLMGenerationEngine(
        registration,
        {"vllm": {"model_implementation": args.implementation}},
        client=client,
        refitter=CPUStateDictRefitter(args.implementation),
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
    state_dict = _load_text_state_dict(args.model)
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
        kwargs={"version": "1"},
    )
    _verify_policy_fingerprints(expected_fingerprint, worker_fingerprints)

    report = {
        "implementation": args.implementation,
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
