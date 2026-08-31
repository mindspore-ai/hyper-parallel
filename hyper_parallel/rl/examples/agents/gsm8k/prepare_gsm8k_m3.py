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
"""Materialize deterministic GSM8K rows with mixed grouped rewards."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
from vllm import LLM, SamplingParams, TokensPrompt

from examples.agents.gsm8k.agent import (
    PROMPT_INSTRUCTION,
    compute_gsm8k_reward,
    extract_answer,
)
from rl.dataset import PromptDataset
from rl.roles.rollout.vllm_plugin import register_hyper_models
from rl.roles.model import architecture_for_implementation


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one input artifact."""
    digest = sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generate(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    seed: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Generate one deterministic response per prompt and preserve token evidence."""
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids],
        SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            seed=seed,
        ),
        use_tqdm=False,
    )
    records = []
    for output in outputs:
        completion = output.outputs[0]
        records.append(
            {
                "token_ids": list(completion.token_ids),
                "text": completion.text,
            }
        )
    return records


def _parse_args() -> argparse.Namespace:
    """Parse and validate materialization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation", choices=("native", "hyper"), default="native")
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--answer-column", default="extra_info")
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=512)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--output-repeats", type=int, default=1)
    parser.add_argument("--response-count", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    if args.output_repeats <= 0:
        raise ValueError("output-repeats must be positive")
    if args.candidate_offset < 0:
        raise ValueError("candidate-offset must be non-negative")
    if args.candidate_limit - args.candidate_offset < args.sample_count:
        raise ValueError(
            "candidate-limit minus candidate-offset must be at least sample-count"
        )
    if args.response_count < 2:
        raise ValueError("response-count must be at least 2")
    if args.max_prompt_length + args.max_tokens > args.max_model_len:
        raise ValueError("max-prompt-length plus max-tokens must not exceed max-model-len")
    return args


def _architecture(args: argparse.Namespace) -> str:
    """Resolve the vLLM architecture override for the selected model."""
    return args.architecture or architecture_for_implementation(args.implementation)


def _load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    """Load and validate the bounded source prefix."""
    frame = pd.read_parquet(args.source)
    missing = {args.prompt_column, args.answer_column} - set(frame.columns)
    if missing:
        raise ValueError(f"GSM8K source is missing columns: {sorted(missing)}")
    candidate_count = min(len(frame), args.candidate_limit)
    if candidate_count - args.candidate_offset < args.sample_count:
        raise ValueError(
            "GSM8K source provides only "
            f"{max(0, candidate_count - args.candidate_offset)} candidates after offset "
            f"{args.candidate_offset} for {args.sample_count} samples"
        )
    return frame, candidate_count


def _build_llm(args: argparse.Namespace, candidate_count: int) -> LLM:
    """Create the deterministic vLLM instance used for selection."""
    register_hyper_models()
    return LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_mm_profiling=True,
        max_num_seqs=min(64, candidate_count - args.candidate_offset),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.7,
        hf_overrides={
            "architectures": [_architecture(args)]
        },
    )


def _build_candidates(
    args: argparse.Namespace,
    llm: LLM,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Tokenize source rows and sort them in deterministic generation order."""
    tokenizer = llm.get_tokenizer()
    dataset = PromptDataset(
        str(args.source),
        tokenizer,
        args.max_prompt_length,
        prompt_column=args.prompt_column,
        answer_column=args.answer_column,
        max_samples=candidate_count,
        prompt_instruction=PROMPT_INSTRUCTION,
    )
    candidates = []
    for source_index, sample in enumerate(dataset):
        if source_index < args.candidate_offset:
            continue
        candidates.append(
            {
                "source_index": source_index,
                "prompt": sample["prompt"],
                "ground_truth": sample["ground_truth"],
                "prompt_token_ids": sample["input_ids"].tolist(),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            len(candidate["prompt_token_ids"]),
            candidate["source_index"],
        )
    )
    return candidates


def _individual_responses(
    llm: LLM,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate one response at a time for batch-invariance validation."""
    return [
        _generate(
            llm,
            [candidate["prompt_token_ids"]],
            args.seed + offset,
            args.max_tokens,
        )[0]
        for offset in range(args.response_count)
    ]


def _select_candidates(
    llm: LLM,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Select rows with mixed rewards and reproducible individual decoding."""
    prompt_token_ids = [candidate["prompt_token_ids"] for candidate in candidates]
    generated = [
        _generate(llm, prompt_token_ids, args.seed + offset, args.max_tokens)
        for offset in range(args.response_count)
    ]

    accepted = []
    for candidate, *responses in zip(candidates, *generated):
        rewards = [
            compute_gsm8k_reward(record["text"], candidate["ground_truth"])
            for record in responses
        ]
        if set(rewards) != {0.0, 1.0}:
            continue
        individual_responses = _individual_responses(llm, candidate, args)
        individual_rewards = [
            compute_gsm8k_reward(record["text"], candidate["ground_truth"])
            for record in individual_responses
        ]
        if set(individual_rewards) != {0.0, 1.0}:
            continue
        replay_responses = _individual_responses(llm, candidate, args)
        if [record["token_ids"] for record in individual_responses] != [
            record["token_ids"] for record in replay_responses
        ]:
            continue
        accepted.append(
            {
                **candidate,
                "responses": [
                    {
                        "seed": args.seed + offset,
                        "token_ids": record["token_ids"],
                        "extracted_answer": extract_answer(record["text"]),
                        "reward": individual_rewards[offset],
                    }
                    for offset, record in enumerate(individual_responses)
                ],
            }
        )
        if len(accepted) == args.sample_count:
            break
    if len(accepted) != args.sample_count:
        raise RuntimeError(
            f"Found only {len(accepted)} mixed-reward rows in {len(candidates)} candidates"
        )
    return accepted


def _write_outputs(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    accepted: list[dict[str, Any]],
) -> None:
    """Write the selected parquet rows and their reproducibility manifest."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_indices = [record["source_index"] for record in accepted]
    output_frame = frame.iloc[source_indices]
    if args.output_repeats > 1:
        output_frame = pd.concat(
            [output_frame] * args.output_repeats,
            ignore_index=True,
        )
    dataset_path = args.output_dir / "train.parquet"
    output_frame.to_parquet(dataset_path, index=False)
    manifest = {
        "implementation": args.implementation,
        "architecture": _architecture(args),
        "source": str(args.source),
        "source_sha256": _file_sha256(args.source),
        "dataset_sha256": _file_sha256(dataset_path),
        "model": str(args.model),
        "output_repeats": args.output_repeats,
        "seed": args.seed,
        "candidate_offset": args.candidate_offset,
        "candidate_limit": args.candidate_limit,
        "response_count": args.response_count,
        "max_prompt_length": args.max_prompt_length,
        "max_tokens": args.max_tokens,
        "records": accepted,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Search a bounded source prefix and write a reproducible mixed-reward subset."""
    args = _parse_args()
    frame, candidate_count = _load_source(args)
    llm = _build_llm(args, candidate_count)
    candidates = _build_candidates(args, llm, candidate_count)
    accepted = _select_candidates(llm, candidates, args)
    _write_outputs(args, frame, accepted)


if __name__ == "__main__":
    main()
