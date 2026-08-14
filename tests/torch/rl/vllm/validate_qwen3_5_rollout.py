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
"""Validate forced multi-step rollout through native or Hyper vLLM models."""

import argparse
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

from vllm import LLM, SamplingParams

from rl.roles.rollout.vllm_plugin import register_hyper_models


_LOGGER = logging.getLogger(__name__)
_HYPER_ARCHITECTURE = "HyperQwen3_5ForCausalLM"
_NATIVE_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_ALIGNMENT_ENV = "HYPER_VLLM_ALIGNMENT"
_SAMPLING_PROFILES = ("greedy", "temperature", "top-k", "top-p")
_DEFAULT_PROMPTS = (
    "Explain why the sky appears blue in two sentences.",
    "Calculate 17 times 23 and show the result.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3.5 checkpoint directory")
    parser.add_argument("--implementation", required=True, choices=("native", "hyper"))
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt; repeat for multiple requests")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--termination", choices=("length", "eos"), default="length")
    parser.add_argument(
        "--sampling-profile",
        choices=(*_SAMPLING_PROFILES, "suite"),
        default="greedy",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _selected_logprobs(completion: Any) -> list[float]:
    token_ids = list(completion.token_ids)
    logprob_steps = completion.logprobs
    if logprob_steps is None or len(logprob_steps) != len(token_ids):
        raise ValueError("vLLM must return one logprob record per generated token")

    selected = []
    for token_id, step in zip(token_ids, logprob_steps):
        if step is None or token_id not in step:
            raise ValueError(f"Missing selected-token logprob for token {token_id}")
        value = float(step[token_id].logprob)
        if not math.isfinite(value):
            raise ValueError(f"Selected-token logprob must be finite, got {value}")
        selected.append(value)
    return selected


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: Path) -> str:
    files = [model_path / "config.json"]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        files.append(index_path)
    weight_files = sorted(model_path.glob("*.safetensors"))
    files.extend(weight_files)
    if not weight_files or any(not path.is_file() for path in files):
        raise ValueError(f"Incomplete model fingerprint inputs under {model_path}")
    digest = sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _create_llm(args: argparse.Namespace) -> LLM:
    options: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": len(args.prompts),
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.implementation == "hyper":
        register_hyper_models()
        options["hf_overrides"] = {"architectures": [_HYPER_ARCHITECTURE]}
    else:
        options["hf_overrides"] = {"architectures": [_NATIVE_ARCHITECTURE]}
    return LLM(**options)


def _validate_architecture(implementation: str, architecture: str) -> None:
    expected = _HYPER_ARCHITECTURE if implementation == "hyper" else _NATIVE_ARCHITECTURE
    if architecture != expected:
        raise ValueError(f"{implementation} rollout loaded {architecture}, expected {expected}")


def _restore_output_ownership(path: Path) -> None:
    host_uid = os.environ.get("HYPER_HOST_UID")
    host_gid = os.environ.get("HYPER_HOST_GID")
    if host_uid is None and host_gid is None:
        return
    if host_uid is None or host_gid is None:
        raise ValueError("HYPER_HOST_UID and HYPER_HOST_GID must be set together")
    os.chown(path, int(host_uid), int(host_gid))


def _alignment_enabled() -> bool:
    value = os.environ.get(_ALIGNMENT_ENV, "false").strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"{_ALIGNMENT_ENV} must be true or false, got '{value}'")
    return value == "true"


def _sampling_options(sampling_profile: str) -> dict[str, Any]:
    if sampling_profile == "greedy":
        return {"temperature": 0.0}
    if sampling_profile == "temperature":
        return {"temperature": 0.8, "seed": 2026}
    if sampling_profile == "top-k":
        return {"temperature": 1.0, "top_k": 20, "seed": 2026}
    if sampling_profile == "top-p":
        return {"temperature": 1.0, "top_p": 0.9, "seed": 2026}
    raise ValueError(f"Unsupported sampling profile: {sampling_profile}")


def _build_sampling_params(
    llm: LLM,
    args: argparse.Namespace,
    sampling_profile: str,
) -> tuple[SamplingParams, int | None]:
    options: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "logprobs": 1,
        "detokenize": False,
    }
    options.update(_sampling_options(sampling_profile))
    if args.termination == "length":
        options["ignore_eos"] = True
        sampling_params = SamplingParams(**options)
        _validate_sampling_params(sampling_params, options)
        return sampling_params, None

    eos_token_id = llm.get_tokenizer().eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id for EOS validation")
    options["ignore_eos"] = False
    options["allowed_token_ids"] = [eos_token_id]
    sampling_params = SamplingParams(**options)
    _validate_sampling_params(sampling_params, options)
    return sampling_params, int(eos_token_id)


def _validate_sampling_params(sampling_params: SamplingParams, expected: dict[str, Any]) -> None:
    for name, expected_value in expected.items():
        actual_value = getattr(sampling_params, name)
        if actual_value != expected_value:
            raise ValueError(
                f"SamplingParams changed {name}: expected {expected_value}, got {actual_value}"
            )


def _validate_termination(
    completion: Any,
    token_ids: list[int],
    args: argparse.Namespace,
    eos_token_id: int | None,
) -> None:
    if args.termination == "length":
        if len(token_ids) != args.max_tokens:
            raise ValueError(f"Expected {args.max_tokens} generated tokens, got {len(token_ids)}")
        if completion.finish_reason != "length":
            raise ValueError(f"Expected length termination, got {completion.finish_reason}")
        return

    if token_ids != [eos_token_id]:
        raise ValueError(f"Expected one EOS token {eos_token_id}, got {token_ids}")
    if completion.finish_reason != "stop":
        raise ValueError(f"Expected EOS stop termination, got {completion.finish_reason}")


def main() -> None:
    """Run and validate one rollout implementation."""
    args = _parse_args()
    args.prompts = tuple(args.prompts or _DEFAULT_PROMPTS)
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2 to execute a decode step")
    sampling_profiles = (
        _SAMPLING_PROFILES
        if args.sampling_profile == "suite"
        else (args.sampling_profile,)
    )
    if args.termination == "eos" and sampling_profiles != ("greedy",):
        raise ValueError("EOS validation supports only --sampling-profile=greedy")
    alignment_enabled = _alignment_enabled()
    if alignment_enabled and args.implementation != "hyper":
        raise ValueError("HYPER_VLLM_ALIGNMENT=true requires --implementation=hyper")
    output_parent_existed = args.output.parent.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not output_parent_existed:
        _restore_output_ownership(args.output.parent)

    llm = _create_llm(args)
    architecture = llm.llm_engine.model_config.architecture
    _validate_architecture(args.implementation, architecture)

    sampling_runs = []
    eos_token_id = None
    for sampling_profile in sampling_profiles:
        sampling_params, eos_token_id = _build_sampling_params(llm, args, sampling_profile)
        started = time.perf_counter()
        outputs = llm.generate(list(args.prompts), sampling_params, use_tqdm=False)
        generation_seconds = time.perf_counter() - started
        if len(outputs) != len(args.prompts):
            raise ValueError(f"Expected {len(args.prompts)} request outputs, got {len(outputs)}")

        requests = []
        for request_output in outputs:
            if len(request_output.outputs) != 1:
                raise ValueError("The functional smoke expects one completion per request")
            completion = request_output.outputs[0]
            token_ids = list(completion.token_ids)
            _validate_termination(completion, token_ids, args, eos_token_id)
            requests.append(
                {
                    "request_id": request_output.request_id,
                    "prompt_token_ids": list(request_output.prompt_token_ids),
                    "response_token_ids": token_ids,
                    "selected_logprobs": _selected_logprobs(completion),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
            )
        request_ids = [request["request_id"] for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError(f"vLLM returned duplicate request IDs: {request_ids}")
        sampling_runs.append(
            {
                "sampling_profile": sampling_profile,
                "sampling_options": _sampling_options(sampling_profile),
                "generation_seconds": generation_seconds,
                "generated_tokens": sum(
                    len(request["response_token_ids"]) for request in requests
                ),
                "tokens_per_second": (
                    sum(len(request["response_token_ids"]) for request in requests)
                    / max(generation_seconds, 1.0e-9)
                ),
                "requests": requests,
            }
        )

    report = {
        "implementation": args.implementation,
        "model": args.model,
        "model_fingerprint": _model_fingerprint(Path(args.model)),
        "transformers_version": package_version("transformers"),
        "vllm_version": package_version("vllm"),
        "vllm_ascend_version": package_version("vllm-ascend"),
        "alignment_enabled": alignment_enabled,
        "architecture": architecture,
        "tensor_parallel_size": args.tensor_parallel_size,
        "world_size": llm.llm_engine.vllm_config.parallel_config.world_size,
        "max_tokens": args.max_tokens,
        "termination": args.termination,
        "eos_token_id": eos_token_id,
        "sampling_runs": sampling_runs,
    }
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    _restore_output_ownership(args.output)
    _LOGGER.info(
        "Validated %s rollout for %d sampling profiles",
        args.implementation,
        len(sampling_runs),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
