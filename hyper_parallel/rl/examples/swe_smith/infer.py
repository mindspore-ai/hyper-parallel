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
"""Run SWE-Smith inference through the shared Codex/vLLM path without an Actor."""

import argparse
from datetime import timedelta
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch.distributed as dist
from transformers import AutoTokenizer

from rl.agentic.codex import CodexRuntime
from rl.data.contracts import PromptRecord
from rl.data.prompts import PromptDataset, build_prompt_records, collate_prompt_samples
from rl.workers.engine.model import resolve_model
from rl.rollout.registry import build_rollout_engine
from rl.rollout.manager import CodexRolloutManager
from rl.utils.process_cleanup import cleanup_processes
from rl.config.loader import load_config


def load_prompts(paths: Sequence[Path], tokenizer: Any,
                 instance_ids: Sequence[str] | None = None) -> list[PromptRecord]:
    """Use the training data path, including the token metadata required by trajectories."""
    prompts = []
    for path in paths:
        dataset = PromptDataset(str(path), tokenizer, max_prompt_length=65536,
                                row_adapter="examples.swe_smith.prepare_data:adapt_row")
        for sample in dataset:
            if instance_ids and sample["ground_truth"]["instance_id"] not in instance_ids:
                continue
            batch = collate_prompt_samples([sample], tokenizer.pad_token_id)
            prompts.extend(build_prompt_records(batch, batch["input_ids"], batch["attention_mask"]))
    if not prompts or (instance_ids and {item.ground_truth["instance_id"] for item in prompts} != set(instance_ids)):
        raise ValueError("Requested instances are missing or no prepared records were selected")
    return prompts


def main() -> None:
    """Use torchrun ranks only to coordinate the native tensor-parallel service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", action="append")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config, [])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"], trust_remote_code=False)
    prompts = load_prompts(args.dataset, tokenizer, args.instance_id)
    timeout = float(config["agentic"]["codex"]["timeout_seconds"])
    startup = float(config["rollout"]["vllm"].get("startup_timeout", 600))
    dist.init_process_group("gloo", timeout=timedelta(seconds=timeout + startup + 60))
    engine = build_rollout_engine(config["rollout"], resolve_model(config["model"]))
    if dist.get_world_size() > 1:
        engine.configure_trainer_tensor_parallel(group=dist.group.WORLD, tp_rank=dist.get_rank(),
                                                tp_size=dist.get_world_size(), request_rank=0, request_size=1)
    agentic = config["agentic"]
    runtime = CodexRuntime(engine, {**agentic["codex"], "max_turns": agentic["max_turns"],
                                    "max_episode_tokens": agentic["max_episode_tokens"]})
    rollout = config["rollout"]
    manager = CodexRolloutManager(runtime, num_return_sequences=1, max_new_tokens=rollout["max_new_tokens"],
                                 temperature=rollout["temperature"], top_p=rollout["top_p"], top_k=rollout["top_k"],
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                 collect_old_log_probs=False, seed=rollout.get("seed"))
    try:
        for prompt in prompts:
            batch = manager.generate([prompt], policy_version=0)
            if dist.get_rank() == 0:
                first = batch.trajectories[0]
                result = {"instance_id": prompt.ground_truth["instance_id"], "reward": first.reward,
                          "seconds": batch.generation_seconds, "model_calls": len(batch.trajectories),
                          "output_tokens": sum(int(item.action_mask.sum()) for item in batch.trajectories),
                          "terminal_reason": first.terminal_reason, "metadata": first.metadata}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                logging.info("SWE-Smith result: %s", {key: value for key, value in result.items() if key != "metadata"})
    finally:
        cleanup_processes(None, manager, engine, True)


if __name__ == "__main__":
    main()
