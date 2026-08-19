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
"""Benchmark Qwen3 native, Hyper, and batch-invariant Hyper-vLLM rollout."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from operator import methodcaller
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from rl.config import build_model_registration
from rl.consistency import (
    QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1,
    validate_consistency_model_identity,
)
from rl.dataset import PromptDataset
from rl.roles.model import resolve_vllm_model
from rl.roles.rollout.base import GenerationRequest, GenerationResult, GenerationSettings
from rl.roles.rollout.vllm import VLLMGenerationEngine


TokenRecord = tuple[list[int], Optional[list[float]]]
_ARMS = ("native", "hyper", "hyper-fa3", "hyper-bi")


def _parse_args() -> argparse.Namespace:
    """Parse one isolated benchmark arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=_ARMS, required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--response-count", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--request-concurrency", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--runtime-source-sha256", required=True)
    parser.add_argument("--runtime-git-state", required=True)
    parser.add_argument("--profile-dir")
    parser.add_argument("--no-logprobs", action="store_false", dest="collect_log_probs")
    parser.set_defaults(collect_log_probs=True)
    args = parser.parse_args()
    for field in (
        "prompt_count",
        "response_count",
        "max_prompt_length",
        "max_tokens",
        "max_model_len",
        "request_concurrency",
        "rounds",
    ):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    if args.max_prompt_length + args.max_tokens > args.max_model_len:
        raise ValueError("max-prompt-length plus max-tokens must not exceed max-model-len")
    if args.rounds > 10:
        raise ValueError("rounds must not exceed 10 because the report uses a small-sample t table")
    if args.profile_dir is not None and args.rounds != 1:
        raise ValueError("Profiler runs must use exactly one measured round")
    return args


def _file_sha256(path: Path) -> str:
    """Return one artifact digest without loading the whole file."""
    digest = sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path: Path) -> dict[str, str]:
    """Hash checkpoint weights, config, and tokenizer artifacts used by one arm."""
    required = (model_path / "config.json", model_path / "model.safetensors.index.json")
    if any(not path.is_file() for path in required):
        raise ValueError(f"Qwen3 checkpoint metadata is incomplete under {model_path}")
    artifacts = set(required)
    for pattern in ("*.safetensors", "tokenizer*", "vocab*", "merges.txt", "generation_config.json"):
        artifacts.update(path for path in model_path.glob(pattern) if path.is_file())
    identity = {}
    for path in sorted(artifacts):
        identity[path.name] = _file_sha256(path)
    return identity


def _cann_identity() -> dict[str, Optional[str]]:
    """Record available CANN installation metadata from the fixed image."""
    candidates = (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
        Path("/etc/ascend_install.info"),
    )
    return {
        str(path): path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        for path in candidates
    }


def _package_versions(names: tuple[str, ...]) -> dict[str, Optional[str]]:
    """Record package versions while leaving profile-specific dependencies optional."""
    versions = {}
    for name in names:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _record_digest(records: list[TokenRecord]) -> str:
    """Return a stable digest over generated IDs and raw selected-token logprobs."""
    payload = [
        {"token_ids": token_ids, "log_probs": log_probs}
        for token_ids, log_probs in records
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _call_client_method(client: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call one required dynamic vLLM client method or fail clearly."""
    method = getattr(client, name, None)
    if not callable(method):
        raise RuntimeError(f"The benchmark vLLM client does not expose {name}")
    return methodcaller(name, *args, **kwargs)(client)


def _validate_records(records: list[TokenRecord], expected_count: int, collect_log_probs: bool) -> int:
    """Validate ordered rollout records and return their generated-token count."""
    if len(records) != expected_count:
        raise RuntimeError(
            f"Response count mismatch: expected={expected_count}, received={len(records)}"
        )
    generated_tokens = 0
    for token_ids, log_probs in records:
        if not token_ids:
            raise RuntimeError("Every measured request must generate at least one token")
        generated_tokens += len(token_ids)
        if collect_log_probs:
            if log_probs is None or len(log_probs) != len(token_ids):
                raise RuntimeError("Raw selected-token logprobs must align with generated tokens")
            if not all(math.isfinite(value) for value in log_probs):
                raise RuntimeError("Raw selected-token logprobs must be finite")
        elif log_probs is not None:
            raise RuntimeError("The no-logprobs arm unexpectedly returned logprobs")
    return generated_tokens


def _student_t_critical_95(sample_count: int) -> float:
    """Return the two-sided 95% Student-t critical value for a small sample."""
    critical_values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    return critical_values.get(sample_count, 1.96)


def _summary(values: list[float]) -> dict[str, Optional[float]]:
    """Return descriptive statistics and a small-sample Student-t interval."""
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    confidence_half_width = (
        _student_t_critical_95(len(values)) * standard_deviation / math.sqrt(len(values))
        if len(values) > 1
        else None
    )
    return {
        "mean": mean,
        "std": standard_deviation,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "ci95_low": None if confidence_half_width is None else mean - confidence_half_width,
        "ci95_high": None if confidence_half_width is None else mean + confidence_half_width,
    }


def _build_vllm_config(args: argparse.Namespace, request_count: int) -> dict[str, Any]:
    """Build one explicit server config without relying on ambient profile state."""
    is_batch_invariant = args.arm == "hyper-bi"
    uses_fa3 = args.arm in ("hyper-fa3", "hyper-bi")
    implementation = "native" if args.arm == "native" else "hyper"
    vllm_config: dict[str, Any] = {
        "deployment": "disjoint",
        "visible_devices": args.visible_devices,
        "model_implementation": implementation,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "enforce_eager": True,
        "batch_invariant": is_batch_invariant,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "skip_mm_profiling": True,
        "gpu_memory_utilization": 0.45,
        "kv_cache_memory_bytes": 2147483648,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.request_concurrency, request_count),
        "max_num_batched_tokens": 4096,
        "logprobs_mode": "raw_logprobs",
        "request_concurrency": args.request_concurrency,
        "startup_timeout": 1200,
        "request_timeout": 1200,
    }
    if uses_fa3:
        vllm_config.update(
            {
                "attention_backend": "FLASH_ATTN",
                "block_size": 128,
            }
        )
    if args.arm == "hyper-bi":
        vllm_config["consistency_profile"] = QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1
    if args.profile_dir is not None:
        vllm_config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": args.profile_dir,
            "torch_profiler_with_memory": True,
        }
    return {"vllm": vllm_config}


def _reset_prefix_cache(client: Any) -> None:
    """Hard-reset request and prefix caches while admission is paused."""
    client.pause()
    if not client.is_paused():
        raise RuntimeError("vLLM did not remain paused after hard cache reset")
    client.resume()
    if client.is_paused():
        raise RuntimeError("vLLM remained paused after hard cache reset resume")


def _build_request(
    prompt_token_ids: list[list[int]],
    settings: GenerationSettings,
) -> GenerationRequest:
    """Right-pad tokenized prompts for the production generation interface."""
    max_prompt_length = max(len(prompt) for prompt in prompt_token_ids)
    input_ids = torch.full(
        (len(prompt_token_ids), max_prompt_length),
        settings.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompt_token_ids):
        length = len(prompt)
        input_ids[row, :length] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, :length] = 1
    return GenerationRequest(
        input_ids=input_ids,
        attention_mask=attention_mask,
        settings=settings,
    )


def _result_records(result: GenerationResult) -> list[TokenRecord]:
    """Extract generated IDs and selected-token logprobs from one engine result."""
    if result.response_mask is None:
        raise RuntimeError("The production generation interface did not return a response mask")
    response_width = result.response_mask.shape[1]
    response_ids = result.sequences[:, -response_width:]
    records = []
    for row in range(response_ids.shape[0]):
        mask = result.response_mask[row].bool()
        token_ids = response_ids[row, mask].tolist()
        log_probs = None
        if result.rollout_log_probs is not None:
            log_probs = result.rollout_log_probs[row, mask].tolist()
        records.append((token_ids, log_probs))
    return records


def _run_round(engine: VLLMGenerationEngine, request: GenerationRequest) -> tuple[list[TokenRecord], dict[str, Any]]:
    """Run one measured round and return auditable records and throughput."""
    started = time.perf_counter()
    result = engine.generate(request)
    elapsed_seconds = time.perf_counter() - started
    records = _result_records(result)
    generated_tokens = _validate_records(
        records,
        request.input_ids.shape[0],
        request.settings.collect_log_probs,
    )
    metrics = {
        "seconds": elapsed_seconds,
        "engine_generation_seconds": result.generation_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed_seconds,
        "requests_per_second": len(records) / elapsed_seconds,
        "response_length_mean": generated_tokens / len(records),
        "response_length_min": min(len(token_ids) for token_ids, _ in records),
        "response_length_max": max(len(token_ids) for token_ids, _ in records),
        "digest": _record_digest(records),
    }
    identity = None
    if result.worker_policy_version is not None:
        identity = {
            "policy_version": result.worker_policy_version,
            "policy_fingerprint": result.worker_policy_fingerprint,
        }
    metrics["worker_identity"] = identity
    return records, metrics


def main() -> None:
    """Execute one benchmark arm in a fresh process and write an atomic JSON report."""
    started_at_utc = datetime.now(timezone.utc).isoformat()
    args = _parse_args()
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
    prompt_token_ids = [prompt for prompt in prompts for _ in range(args.response_count)]
    if not prompt_token_ids:
        raise RuntimeError("The benchmark dataset produced no prompts")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define EOS and a usable padding token")
    settings = GenerationSettings(
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=True,
        pad_token_id=int(pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        collect_log_probs=bool(args.collect_log_probs),
        seed=args.seed,
        ignore_eos=True,
    )
    warmup_settings = GenerationSettings(
        **{**settings.__dict__, "max_new_tokens": min(16, args.max_tokens)}
    )
    model = build_model_registration(
        {
            "model": {
                "registry_name": "qwen3_consistency_performance",
                "name": "qwen3",
                "weights_path": str(args.model),
                "tokenizer_path": str(args.model),
            }
        }
    )
    if model.family != "qwen3":
        raise ValueError(f"The benchmark requires a Qwen3 checkpoint, got family={model.family!r}")
    selected_profile = QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1 if args.arm == "hyper-bi" else "off"
    validate_consistency_model_identity(
        {"consistency": {"profile": selected_profile}},
        model,
    )
    engine_config = _build_vllm_config(args, len(prompt_token_ids))
    engine = VLLMGenerationEngine(model, engine_config)
    request = _build_request(prompt_token_ids, settings)
    warmup_request = _build_request(prompt_token_ids, warmup_settings)
    round_records = []
    round_metrics = []
    worker_fingerprints = None
    try:
        client = engine._ensure_client()  # pylint: disable=protected-access
        worker_fingerprints = _call_client_method(client, "get_policy_weight_fingerprints")
        rollout_model = resolve_vllm_model(
            model,
            engine_config["vllm"]["model_implementation"],
        )
        architectures = {str(item.get("architecture")) for item in worker_fingerprints}
        if architectures != {rollout_model.architecture}:
            raise RuntimeError(
                "vLLM worker architecture mismatch: "
                f"expected={rollout_model.architecture!r}, actual={sorted(architectures)}"
            )
        engine.generate(warmup_request)
        if args.cache_mode == "cold":
            _reset_prefix_cache(client)
        profile_active = False
        try:
            if args.profile_dir is not None:
                _call_client_method(client, "_request", "POST", "start_profile")
                profile_active = True
            for round_index in range(args.rounds):
                if round_index and args.cache_mode == "cold":
                    _reset_prefix_cache(client)
                records, metrics = _run_round(engine, request)
                round_records.append(
                    [
                        {"token_ids": token_ids, "log_probs": log_probs}
                        for token_ids, log_probs in records
                    ]
                )
                round_metrics.append(metrics)
        finally:
            if profile_active:
                _call_client_method(
                    client,
                    "_request",
                    "POST",
                    "stop_profile",
                    timeout=3600,
                )
                time.sleep(10)
    finally:
        engine.close()

    throughput = [float(metrics["tokens_per_second"]) for metrics in round_metrics]
    request_throughput = [float(metrics["requests_per_second"]) for metrics in round_metrics]
    report = {
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arm": args.arm,
        "model": str(args.model),
        "model_identity": _model_identity(args.model),
        "source_sha256": _file_sha256(args.source),
        "prompt_token_ids_sha256": sha256(
            json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "runtime_source_sha256": args.runtime_source_sha256,
        "runtime_git_state": args.runtime_git_state,
        "server_visible_devices": args.visible_devices,
        "parent_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cann": _cann_identity(),
        "packages": _package_versions(
            (
                "torch",
                "torch-npu",
                "transformers",
                "vllm",
                "vllm-ascend",
                "batch-invariant-ops",
                "flash-attn-npu",
            )
        ),
        "workload": {
            "requested_prompt_count": args.prompt_count,
            "actual_prompt_count": len(prompts),
            "response_count": args.response_count,
            "request_count": len(prompt_token_ids),
            "prompt_lengths": [len(prompt) for prompt in prompts],
            "max_prompt_length": args.max_prompt_length,
            "max_tokens": args.max_tokens,
            "request_concurrency": args.request_concurrency,
            "cache_mode": args.cache_mode,
            "collect_log_probs": bool(args.collect_log_probs),
            "seed": args.seed,
            "profile_dir": args.profile_dir,
        },
        "effective_config": engine_config,
        "worker_fingerprints": worker_fingerprints,
        "ci95_method": "two-sided Student-t interval across measured rounds",
        "rounds": round_metrics,
        "tokens_per_second": _summary(throughput),
        "requests_per_second": _summary(request_throughput),
        "records": round_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_output.replace(args.output)


if __name__ == "__main__":
    main()
