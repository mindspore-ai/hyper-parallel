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
"""Compare serial and concurrent production HTTP rollout numerics and throughput."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Optional

from transformers import AutoTokenizer

from rl.dataset import PromptDataset
from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationSettings
from rl.roles.rollout.vllm import VLLMGenerationEngine


TokenRecord = tuple[list[int], Optional[list[float]]]


def _record_digest(records: list[TokenRecord]) -> str:
    """Return a stable digest over token IDs and sampled-token log probabilities."""
    payload = [
        {"token_ids": token_ids, "log_probs": log_probs}
        for token_ids, log_probs in records
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _compare_records(reference: list[TokenRecord], candidate: list[TokenRecord]) -> dict[str, Any]:
    """Compare tokens globally and log probabilities only while contexts match."""
    if len(reference) != len(candidate):
        raise RuntimeError(
            "Serial/concurrent response count mismatch: "
            f"serial={len(reference)}, concurrent={len(candidate)}"
        )
    token_mismatches = 0
    identical_records = 0
    common_prefix_tokens = 0
    log_prob_differences = []
    for (reference_tokens, reference_log_probs), (candidate_tokens, candidate_log_probs) in zip(
        reference,
        candidate,
    ):
        token_mismatches += abs(len(reference_tokens) - len(candidate_tokens))
        token_mismatches += sum(
            int(reference_token != candidate_token)
            for reference_token, candidate_token in zip(reference_tokens, candidate_tokens)
        )
        if reference_tokens == candidate_tokens:
            identical_records += 1
        if reference_log_probs is None or candidate_log_probs is None:
            raise RuntimeError("Production benchmark requires sampled-token log probabilities")
        if len(reference_log_probs) != len(reference_tokens):
            raise RuntimeError("Serial sampled-token log probabilities do not align with token IDs")
        if len(candidate_log_probs) != len(candidate_tokens):
            raise RuntimeError("Concurrent sampled-token log probabilities do not align with token IDs")
        common_prefix_length = 0
        for reference_token, candidate_token in zip(reference_tokens, candidate_tokens):
            if reference_token != candidate_token:
                break
            common_prefix_length += 1
        common_prefix_tokens += common_prefix_length
        log_prob_differences.extend(
            abs(reference_value - candidate_value)
            for reference_value, candidate_value in zip(
                reference_log_probs[:common_prefix_length],
                candidate_log_probs[:common_prefix_length],
            )
        )
    return {
        "token_mismatches": token_mismatches,
        "identical_records": identical_records,
        "common_prefix_tokens": common_prefix_tokens,
        "log_prob_compared": len(log_prob_differences),
        "log_prob_max_abs_diff": max(log_prob_differences, default=0.0),
        "log_prob_mean_abs_diff": (
            sum(log_prob_differences) / max(len(log_prob_differences), 1)
        ),
    }


def _run_mode(
    client: Any,
    prompt_token_ids: list[list[int]],
    settings: GenerationSettings,
    request_concurrency: int,
    rounds: int,
) -> tuple[list[list[TokenRecord]], dict[str, Any]]:
    """Run one concurrency mode repeatedly and record replay differences."""
    first_records = None
    round_records = []
    elapsed_seconds = 0.0
    generated_tokens = 0
    round_metrics = []
    for _ in range(rounds):
        started = time.perf_counter()
        records = client.generate_tokens(
            prompt_token_ids,
            settings,
            request_concurrency=request_concurrency,
        )
        elapsed = time.perf_counter() - started
        token_count = sum(len(token_ids) for token_ids, _ in records)
        digest = _record_digest(records)
        if first_records is None:
            first_records = records
        replay_comparison = _compare_records(first_records, records)
        round_records.append(records)
        round_metrics.append(
            {
                "seconds": elapsed,
                "generated_tokens": token_count,
                "tokens_per_second": token_count / max(elapsed, 1.0e-9),
                "digest": digest,
                "comparison_to_first": replay_comparison,
            }
        )
        elapsed_seconds += elapsed
        generated_tokens += token_count
    if first_records is None:
        raise RuntimeError("Production benchmark completed no measured rounds")
    return round_records, {
        "request_concurrency": request_concurrency,
        "rounds": round_metrics,
        "generated_tokens": generated_tokens,
        "seconds": elapsed_seconds,
        "tokens_per_second": generated_tokens / max(elapsed_seconds, 1.0e-9),
        "digest": _record_digest(first_records),
    }


def main() -> None:
    """Run one same-server serial/concurrent numerical and throughput comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("hyper", "native"), required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--response-count", type=int, default=6)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-model-len", type=int, default=640)
    parser.add_argument("--request-concurrency", type=int, default=12)
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-token-mismatches", type=int, default=0)
    parser.add_argument("--max-log-prob-diff", type=float, default=0.05)
    parser.add_argument("--min-speedup", type=float, default=1.2)
    args = parser.parse_args()
    for name in ("prompt_count", "response_count", "max_prompt_length", "max_tokens", "rounds"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.request_concurrency <= 1:
        raise ValueError("request-concurrency must be greater than one")
    if args.max_prompt_length + args.max_tokens > args.max_model_len:
        raise ValueError("max-prompt-length plus max-tokens must not exceed max-model-len")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = PromptDataset(
        str(args.source),
        tokenizer,
        args.max_prompt_length,
        prompt_column="prompt",
        answer_column="extra_info",
        max_samples=args.prompt_count,
    )
    prompts = [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
    prompt_token_ids = [
        prompt
        for prompt in prompts
        for _ in range(args.response_count)
    ]
    settings = GenerationSettings(
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=True,
        pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        collect_log_probs=True,
        seed=args.seed,
    )
    warmup_settings = GenerationSettings(
        **{**settings.__dict__, "max_new_tokens": min(16, args.max_tokens)}
    )
    model = ModelRegistration(
        "qwen3_5_0_8b",
        "qwen3_5",
        str(args.model),
        str(args.model),
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5",
        "qwen3_5_text",
        True,
    )
    engine = VLLMGenerationEngine(
        model,
        {
            "vllm": {
                "deployment": "disjoint",
                "visible_devices": args.visible_devices,
                "model_implementation": args.implementation,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "trust_remote_code": False,
                "enforce_eager": True,
                "batch_invariant": args.batch_invariant,
                "enable_prefix_caching": False,
                "skip_mm_profiling": True,
                "kv_cache_memory_bytes": 2147483648,
                "max_model_len": args.max_model_len,
                "max_num_seqs": max(args.request_concurrency, len(prompt_token_ids)),
                "max_num_batched_tokens": 4096,
                "startup_timeout": 600,
                "request_timeout": 600,
            }
        },
    )
    try:
        client = engine._ensure_client()  # pylint: disable=protected-access
        client.generate_tokens(prompt_token_ids, warmup_settings, request_concurrency=1)
        client.generate_tokens(
            prompt_token_ids,
            warmup_settings,
            request_concurrency=args.request_concurrency,
        )
        serial_round_records, serial_metrics = _run_mode(
            client,
            prompt_token_ids,
            settings,
            request_concurrency=1,
            rounds=args.rounds,
        )
        concurrent_round_records, concurrent_metrics = _run_mode(
            client,
            prompt_token_ids,
            settings,
            request_concurrency=args.request_concurrency,
            rounds=args.rounds,
        )
    finally:
        engine.close()

    serial_records = serial_round_records[0]
    concurrent_records = concurrent_round_records[0]
    cross_mode_comparisons = [
        _compare_records(serial_records, records)
        for records in concurrent_round_records
    ]
    replay_comparisons = [
        metrics["comparison_to_first"]
        for mode_metrics in (serial_metrics, concurrent_metrics)
        for metrics in mode_metrics["rounds"]
    ]
    acceptance_comparisons = cross_mode_comparisons + replay_comparisons
    speedup = concurrent_metrics["tokens_per_second"] / max(
        serial_metrics["tokens_per_second"],
        1.0e-9,
    )
    comparison = {
        "serial_to_concurrent_rounds": cross_mode_comparisons,
        "token_mismatches": max(item["token_mismatches"] for item in acceptance_comparisons),
        "log_prob_max_abs_diff": max(item["log_prob_max_abs_diff"] for item in acceptance_comparisons),
        "throughput_speedup": speedup,
    }
    comparison["passed"] = (
        comparison["token_mismatches"] <= args.max_token_mismatches
        and comparison["log_prob_max_abs_diff"] <= args.max_log_prob_diff
        and speedup >= args.min_speedup
    )
    report = {
        "implementation": args.implementation,
        "prompt_count": args.prompt_count,
        "response_count": args.response_count,
        "max_tokens": args.max_tokens,
        "batch_invariant": args.batch_invariant,
        "seed": args.seed,
        "serial": serial_metrics,
        "concurrent": concurrent_metrics,
        "comparison": comparison,
        "serial_records": [
            {"token_ids": token_ids, "log_probs": log_probs}
            for token_ids, log_probs in serial_records
        ],
        "concurrent_records": [
            {"token_ids": token_ids, "log_probs": log_probs}
            for token_ids, log_probs in concurrent_records
        ],
        "serial_round_records": [
            [
                {"token_ids": token_ids, "log_probs": log_probs}
                for token_ids, log_probs in records
            ]
            for records in serial_round_records
        ],
        "concurrent_round_records": [
            [
                {"token_ids": token_ids, "log_probs": log_probs}
                for token_ids, log_probs in records
            ]
            for records in concurrent_round_records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_output.replace(args.output)
    if not comparison["passed"]:
        raise RuntimeError(f"Production vLLM benchmark failed acceptance: {comparison}")


if __name__ == "__main__":
    main()
