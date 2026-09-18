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
"""RL system-test recipes shared by independently deployed UT and ST suites."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import yaml

EXAMPLES = Path(find_spec("hyper_parallel").origin).parent / "rl" / "examples"


@dataclass(frozen=True)
class Case:
    """Describe a production recipe and its required hardware."""
    name: str
    tp: int = 1
    strategy: str = "full_gather"
    family: str = "qwen3"
    runner: str = "internal"
    disjoint: bool = False
    resume: bool = False
    exact: bool = False
    algorithm: str = "grpo"

    @property
    def world(self) -> int:
        """Trainer uses two FSDP shards and the requested TP degree."""
        return 2 * self.tp

    @property
    def cards(self) -> int:
        """Disjoint rollout consumes a separate device set."""
        return self.world * (2 if self.disjoint else 1)


CASES = (
    Case("dense-tp2-consistency-full", tp=2, exact=True),
    Case("dense-tp2-direct", tp=2, strategy="direct_reshard"),
    Case("checkpoint-resume", resume=True),
    Case("codex-agent", runner="codex"),
    Case("deepseek-agent", runner="deepseek"),
    Case("ppo-tp1-full", algorithm="ppo"),
)

AGENT_INSTRUCTIONS = {
    "codex": """/no_think
Solve the arithmetic word problem below.
You must first use the local shell tool exactly once to run Python for the calculation.
Call exec_command with login=false so shell startup does not delay the calculation.
Do not return a final answer before observing the command output.
After observing the first command output, never call any tool again, even to retry or verify it.
Return only the final answer in the form "#### NUMBER" without an explanation.

Problem: {prompt}""",
    "deepseek": """Solve the arithmetic word problem below.
Immediately call the Bash tool exactly once and use Python for the calculation.
Run Bash in the foreground: omit run_in_background or set it to false.
Do not write analysis before the tool call.
Never use job_output, job_list, or job_kill.
After the Bash result, never call any tool again and return exactly "#### NUMBER".

Problem: {prompt}""",
}


def prepare_config(case: Case, phase: int, ports: tuple[int, int], devices: list[int]) -> dict:
    """Derive a bounded ST workload from the shipped production YAML."""
    if case.runner != "internal":
        recipe = EXAMPLES / f"gsm8k/configs/{case.runner}_multi_turn.yaml"
    elif case.algorithm == "ppo":
        recipe = EXAMPLES / "gsm8k/configs/qwen3_4b_gsm8k_ppo.yaml"
    else:
        recipe = EXAMPLES / "gsm8k/configs/qwen3_4b_gsm8k_vllm_production.yaml"
    config = yaml.safe_load(recipe.read_text())
    config["model"].update(weights_path="/model", tokenizer_path="/model")
    config["consistency"]["enabled"] = case.exact
    config["data"].update(train_path="/data/train.parquet", test_path="/data/test.parquet",
                          max_train_samples=8, shuffle=False, num_workers=0)
    # Dense Qwen3 needs room to finish reasoning before the answer is scored.
    generation_budget = 512 if case.runner == "internal" else 256
    config["rollout"].update(num_return_sequences=4, max_new_tokens=generation_budget, seed=20260908)
    if case.name == "dense-tp2-direct":
        # More diverse responses make a non-zero GRPO update observable.
        config["rollout"].update(num_return_sequences=8, temperature=1.1)
    vllm = config["rollout"]["vllm"]
    vllm.update(deployment="disjoint" if case.disjoint else "colocated",
                data_parallel_size=2, tensor_parallel_size=case.tp, port=ports[0],
                model_implementation="native" if case.runner != "internal" else "hyper",
                enforce_eager=True, batch_invariant=case.exact,
                max_num_seqs=4)
    vllm["weight_sync"].update(strategy=case.strategy, bucket_size_mb=128)
    if case.disjoint:
        vllm["visible_devices"] = ",".join(map(str, devices[case.world:]))
    train = config["train"]
    train.update(max_steps=1 if case.resume and phase == 1 else 2,
                 prompt_batch_size=2, micro_batch_size=1 if case.runner != "internal" else 2,
                 response_mini_batch_size=8)
    if case.algorithm == "ppo" and case.resume and phase == 2:
        train["max_steps"] = 3
    train["accelerator"].update(dp_replicate=1, dp_shard=2, tp=case.tp,
                                 ep=1,
                                 cpu_offload=True, activation_checkpoint="full")
    # Real rewards are preserved. Zero-advantage batches may occur; the validator
    # requires evidence of at least one actual update across the completed run.
    train["learning_gate"]["enabled"] = False
    # Production evaluation is scheduled at checkpoint boundaries.
    train["checkpoint"].update(output_dir="/results/checkpoints", save_steps=0,
                                save_final=case.resume or case.name == "dense-tp1-full", verify_reload=False,
                                load_path="/results/checkpoints/step_1" if phase == 2 else None)
    config["evaluation"].update(enabled=case.name == "dense-tp1-full",
                                 batch_size=1, max_samples=8, max_new_tokens=512)
    config["logging"].update(backends=["console"], log_steps=1)
    config["logging"]["wandb"]["mode"] = "disabled"
    if case.runner != "internal":
        config["agentic"][case.runner].update(
            gateway_port=ports[1], session_root="/results/sessions",
            instruction_template=AGENT_INSTRUCTIONS[case.runner])
    return config
