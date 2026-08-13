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

from rl.algorithm.reward.gsm8k import compute_rule_reward, extract_answer
from rl.dataset import PromptDataset
from rl.roles.rollout.vllm_plugin import register_hyper_models
from rl.roles.rollout.vllm_policy import architecture_for_implementation


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


def main() -> None:
    """Search a bounded source prefix and write a reproducible mixed-reward subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation", choices=("native", "hyper"), default="hyper")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--answer-column", default="extra_info")
    parser.add_argument("--candidate-limit", type=int, default=512)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--response-count", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.sample_count <= 0 or args.candidate_limit < args.sample_count:
        raise ValueError("candidate-limit must be at least the positive sample-count")
    if args.response_count < 2:
        raise ValueError("response-count must be at least 2")
    if args.max_prompt_length + args.max_tokens > args.max_model_len:
        raise ValueError("max-prompt-length plus max-tokens must not exceed max-model-len")

    frame = pd.read_parquet(args.source)
    missing = {args.prompt_column, args.answer_column} - set(frame.columns)
    if missing:
        raise ValueError(f"GSM8K source is missing columns: {sorted(missing)}")
    candidate_count = min(len(frame), args.candidate_limit)
    if candidate_count < args.sample_count:
        raise ValueError(
            f"GSM8K source provides only {candidate_count} candidates for {args.sample_count} samples"
        )

    register_hyper_models()
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=min(64, candidate_count),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.7,
        hf_overrides={
            "architectures": [architecture_for_implementation(args.implementation)]
        },
    )
    tokenizer = llm.get_tokenizer()
    dataset = PromptDataset(
        str(args.source),
        tokenizer,
        args.max_prompt_length,
        prompt_column=args.prompt_column,
        answer_column=args.answer_column,
        max_samples=candidate_count,
    )
    candidates = []
    for source_index in range(len(dataset)):
        sample = dataset[source_index]
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
    prompt_token_ids = [candidate["prompt_token_ids"] for candidate in candidates]
    generated = [
        _generate(llm, prompt_token_ids, args.seed + offset, args.max_tokens)
        for offset in range(args.response_count)
    ]

    accepted = []
    for candidate, *responses in zip(candidates, *generated):
        rewards = [
            compute_rule_reward(record["text"], candidate["ground_truth"])
            for record in responses
        ]
        if set(rewards) != {0.0, 1.0}:
            continue
        individual_responses = [
            _generate(
                llm,
                [candidate["prompt_token_ids"]],
                args.seed + offset,
                args.max_tokens,
            )[0]
            for offset in range(args.response_count)
        ]
        individual_rewards = [
            compute_rule_reward(record["text"], candidate["ground_truth"])
            for record in individual_responses
        ]
        if set(individual_rewards) != {0.0, 1.0}:
            continue
        replay_responses = [
            _generate(
                llm,
                [candidate["prompt_token_ids"]],
                args.seed + offset,
                args.max_tokens,
            )[0]
            for offset in range(args.response_count)
        ]
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_indices = [record["source_index"] for record in accepted]
    output_frame = frame.iloc[source_indices]
    dataset_path = args.output_dir / "train.parquet"
    output_frame.to_parquet(dataset_path, index=False)
    manifest = {
        "implementation": args.implementation,
        "architecture": architecture_for_implementation(args.implementation),
        "source": str(args.source),
        "source_sha256": _file_sha256(args.source),
        "dataset_sha256": _file_sha256(dataset_path),
        "model": str(args.model),
        "seed": args.seed,
        "response_count": args.response_count,
        "max_prompt_length": args.max_prompt_length,
        "max_tokens": args.max_tokens,
        "records": accepted,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
