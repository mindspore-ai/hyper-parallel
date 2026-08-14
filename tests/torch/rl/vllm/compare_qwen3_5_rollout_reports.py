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
"""Compare functional native and Hyper vLLM rollout reports."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_ARCHITECTURES = {
    "native": "Qwen3_5ForConditionalGeneration",
    "hyper": "HyperQwen3_5ForCausalLM",
}


def _load_report(path: Path, implementation: str) -> dict[str, Any]:
    """Load and validate one implementation report."""
    with path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if report.get("implementation") != implementation:
        raise ValueError(f"{path} does not contain a {implementation} report")
    if report.get("architecture") != _ARCHITECTURES[implementation]:
        raise ValueError(
            f"{implementation} report loaded {report.get('architecture')}, "
            f"expected {_ARCHITECTURES[implementation]}"
        )
    return report


def _index_runs(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index sampling runs and reject duplicate profile names."""
    runs = report.get("sampling_runs", [])
    indexed = {run["sampling_profile"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError("Sampling report contains duplicate profile names")
    return indexed


def _flatten(run: Mapping[str, Any], field: str) -> list[Any]:
    """Flatten one request field in stable request order."""
    return [value for request in run["requests"] for value in request[field]]


def _compare_profile(
    native_run: Mapping[str, Any],
    hyper_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare token/log-prob accuracy and report measured throughput."""
    if native_run["sampling_options"] != hyper_run["sampling_options"]:
        raise ValueError("Native and Hyper sampling options differ")
    native_requests = native_run["requests"]
    hyper_requests = hyper_run["requests"]
    if len(native_requests) != len(hyper_requests):
        raise ValueError("Native and Hyper request counts differ")
    for native_request, hyper_request in zip(native_requests, hyper_requests):
        if native_request["prompt_token_ids"] != hyper_request["prompt_token_ids"]:
            raise ValueError("Native and Hyper prompt token IDs differ")
        native_response_length = len(native_request["response_token_ids"])
        hyper_response_length = len(hyper_request["response_token_ids"])
        if native_response_length != hyper_response_length:
            raise ValueError("Native and Hyper per-request response lengths differ")
    native_tokens = _flatten(native_run, "response_token_ids")
    hyper_tokens = _flatten(hyper_run, "response_token_ids")
    native_logprobs = _flatten(native_run, "selected_logprobs")
    hyper_logprobs = _flatten(hyper_run, "selected_logprobs")
    if len(native_tokens) != len(hyper_tokens) or len(native_logprobs) != len(hyper_logprobs):
        raise ValueError("Native and Hyper output lengths differ")
    differences = [
        abs(native_value - hyper_value)
        for native_value, hyper_value in zip(native_logprobs, hyper_logprobs)
    ]
    return {
        "tokens_compared": len(native_tokens),
        "token_mismatches": sum(
            native_token != hyper_token
            for native_token, hyper_token in zip(native_tokens, hyper_tokens)
        ),
        "max_abs_selected_logprob_difference": max(differences, default=0.0),
        "mean_abs_selected_logprob_difference": (
            sum(differences) / len(differences) if differences else 0.0
        ),
        "native_tokens_per_second": native_run["tokens_per_second"],
        "hyper_tokens_per_second": hyper_run["tokens_per_second"],
    }


def main() -> None:
    """Compare reports produced from the same checkpoint and test configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--hyper", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    native_report = _load_report(args.native, "native")
    hyper_report = _load_report(args.hyper, "hyper")
    for field in (
        "model_fingerprint",
        "transformers_version",
        "vllm_version",
        "vllm_ascend_version",
        "alignment_enabled",
        "max_tokens",
        "termination",
        "tensor_parallel_size",
        "world_size",
    ):
        if field not in native_report or field not in hyper_report:
            raise ValueError(f"Native and Hyper rollout reports require provenance field: {field}")
        if native_report[field] != hyper_report[field]:
            raise ValueError(f"Native and Hyper rollout provenance differs: {field}")
    native_runs = _index_runs(native_report)
    hyper_runs = _index_runs(hyper_report)
    if set(native_runs) != set(hyper_runs):
        raise ValueError("Native and Hyper sampling profile sets differ")
    summary = {
        "profiles": {
            profile: _compare_profile(native_runs[profile], hyper_runs[profile])
            for profile in native_runs
        },
        "comparison_is_functional_gate": False,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
