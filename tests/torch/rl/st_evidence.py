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
"""Validate production metrics and artifacts without importing a training framework."""

import json
import math
from pathlib import Path
import re
from typing import Any


def require(condition: bool, message: str) -> None:
    """Reject missing or inconsistent system-test evidence."""
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict:
    """Load mandatory JSON evidence."""
    require(path.is_file(), f"Missing evidence: {path}")
    return json.loads(path.read_text())


def metrics(path: Path) -> dict[int, dict[str, float]]:
    """Parse console metric records, merging evaluation into its training step."""
    result = {}
    for line in path.read_text().splitlines():
        match = re.search(r"step=(\d+) \| (.+)", line)
        if match is None:
            continue
        row = result.setdefault(int(match[1]), {})
        for field in match[2].split(", "):
            key, value = field.split("=", 1)
            row[key] = float(value)
    require(bool(result), f"No step metrics in {path}")
    return result


def validate_phase(output: Path, case: Any, phase: int) -> None:
    """Require committed versions, real updates, and the selected sync strategy."""
    steps = _phase_steps(case, phase)
    rows = metrics(output / f"phase-{phase}.log")
    require(set(step for step, row in rows.items() if "train/global_step" in row) == set(steps),
            f"Expected training steps {steps}, got {list(rows)}")
    changed = False
    critic_changed = False
    negative_control = False
    for step in steps:
        row = rows[step]
        require(all(math.isfinite(value) for value in row.values()), f"Non-finite metrics at {step}")
        for key in ("train/valid_tokens", "train/optimizer_steps", "rollout/generated_tokens"):
            require(row.get(key, 0) > 0, f"No {key} at step {step}")
        require(row["train/global_step"] == row.get("policy/version") == step, "Policy version mismatch")
        require("train/total_loss" in row and "train/gradient_norm" in row, "Missing optimizer metrics")
        changed |= row["train/gradient_norm"] > 0
        if case.algorithm == "ppo":
            require(row.get("critic/valid_tokens", 0) > 0 and row.get("critic/optimizer_steps", 0) > 0,
                    "Missing Critic optimization")
            critic_changed |= row.get("critic/gradient_norm", 0) > 0
        require(row.get(f"weight_sync/last_{case.strategy}") == 1,
                "Wrong completed transfer strategy")
        if case.exact:
            negative_control |= _validate_exact_metrics(row)
        if case.strategy == "full_gather":
            _validate_streaming_metrics(row)
    require(changed, "No non-zero learning update: inspect real rewards/advantages; do not relabel rewards")
    if case.algorithm == "ppo":
        require(critic_changed, "No non-zero Critic update")
    if case.exact:
        require(negative_control, "No post-update change against the old policy")
    if case.name == "dense-tp1-full":
        _validate_evaluation(rows, steps)
    if case.resume or case.name == "dense-tp1-full":
        _validate_checkpoint(output, case, steps)


def validate_sessions(root: Path, versions: tuple[int, ...]) -> None:
    """Require real external Agent tool interaction, token evidence and released sessions."""
    paths = list(root.rglob("gateway-events.jsonl"))
    require(bool(paths), "Missing external Agent traces")
    seen = set()
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        require(events[0]["type"] == "session.registered" and events[-1]["type"] == "session.released",
                f"Session lifecycle incomplete: {path}")
        session_id = events[0]["session_id"]
        require(all(event["session_id"] == session_id for event in events), "Mixed session identities")
        identity = events[0]["payload"]
        seen.add(identity["policy_version"])
        require(isinstance(identity["policy_version"], int), "Session has no policy version")
        completions = [event["payload"] for event in events if event["type"] == "completion.recorded"]
        require(len(completions) >= 2, "Agent did not complete a tool round trip")
        require(any(record["response"]["choices"][0]["message"].get("tool_calls")
                    for record in completions), "No model-originated tool call")
        require(any(message.get("role") == "tool" for record in completions
                    for message in record["request"].get("messages", [])), "Missing tool feedback in next request")
        for ordinal, record in enumerate(completions):
            require(record["ordinal"] == ordinal, "Completion ordering mismatch")
            response = record["response"]
            require(bool(response.get("prompt_token_ids")), "Missing exact prompt tokens")
            choice = response["choices"][0]
            require(bool(choice.get("token_ids")) and bool(choice.get("logprobs")),
                    "Missing sampled tokens or raw logprobs")
            logprobs = choice["logprobs"]
            if "token_logprobs" in logprobs:
                log_probs = logprobs["token_logprobs"]
            else:
                content = logprobs.get("content")
                require(isinstance(content, list), "Missing sampled chat logprobs")
                require(all(isinstance(item, dict) and "logprob" in item for item in content),
                        "Malformed sampled chat logprobs")
                log_probs = [item["logprob"] for item in content]
            require(isinstance(log_probs, list), "Invalid sampled logprob sequence")
            require(len(log_probs) == len(choice["token_ids"]), "Sampled token/logprob lengths differ")
            require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value <= 0 for value in log_probs),
                    "Invalid sampled raw logprobs")
    require(seen == set(versions), f"Expected Agent policy versions {versions}, got {seen}")


def _validate_exact_metrics(row):
    """Validate exact replay and return whether the negative control changed."""
    require(row.get("training/pre_update_exact_valid") == 1, "Missing successful consistency gate")
    require(row.get("training/pre_update_exact_tokens", 0) > 0, "Empty consistency comparison")
    for key in ("mismatch_count", "max_abs_diff", "mean_abs_diff"):
        require(row.get("training/pre_update_" + key) == 0, f"Consistency failed: {key}")
    tokens = row.get("training/post_update_old_policy_tokens", 0)
    mismatches = row.get("training/post_update_old_policy_mismatch_count", -1)
    require(tokens > 0 and 0 <= mismatches <= tokens, "Invalid post-update comparison")
    require(row.get("training/post_update_negative_control_valid") == int(mismatches > 0),
            "Inconsistent post-update negative control")
    return mismatches > 0


def _validate_streaming_metrics(row):
    """Require bounded, acknowledged and released full-gather buckets."""
    prefix = "weight_sync/streaming_"
    count = row.get(prefix + "bucket_count", 0)
    require(count > 0 and row.get(prefix + "acked_buckets") == count
            and row.get(prefix + "released_buckets") == count, "Unacknowledged streaming buckets")
    require(row.get(prefix + "max_inflight_buckets") == 1, "Unbounded in-flight buckets")
    gathered = row.get(prefix + "max_gathered_bytes", 0)
    packed = row.get(prefix + "max_packed_bytes", 0)
    require(gathered > 0 and gathered == packed, "Invalid packed weight bytes")


def _validate_evaluation(rows, steps):
    """Check evaluation counts and their reported accuracy."""
    evaluation = rows[steps[-1]]
    total = evaluation.get("validation/total", 0)
    correct = evaluation.get("validation/correct", -1)
    accuracy = evaluation.get("validation/accuracy", -1)
    require(total == 8 and 0 <= correct <= total, "Incomplete evaluation sample count")
    require(math.isclose(accuracy, correct / total, rel_tol=0, abs_tol=1e-6),
            "Evaluation accuracy disagrees with counts")
    require(evaluation.get("validation/generated_tokens", 0) > 0, "Evaluation generated no tokens")


def _validate_checkpoint(output, case, steps):
    """Require complete rank state and all exported Hugging Face artifacts."""
    saved_step = steps[-1]
    checkpoint = output / "checkpoints" / f"step_{saved_step}"
    marker = read_json(checkpoint / "checkpoint_complete.json")
    require(marker["step"] == saved_step and marker["world_size"] == case.world, "Incomplete checkpoint")
    if case.algorithm == "ppo":
        require(marker.get("critic") is True, "Checkpoint does not include Critic")
    require(read_json(checkpoint / "extra_state.json")["global_step"] == saved_step, "Checkpoint step mismatch")
    require(any(checkpoint.glob("*.safetensors")), "Missing model checkpoint tensors")
    for rank in range(case.world):
        rank_dir = checkpoint / f"rank_{rank}"
        for suffix in ("bytes", "metadata"):
            require(any(path.stat().st_size > 0 for path in rank_dir.glob(f"*.{suffix}")),
                    f"Missing rank-local checkpoint {suffix}: rank={rank}")
    export = checkpoint / "hf"
    model_config = read_json(export / "config.json")
    require(model_config.get("model_type") == "qwen3", "Invalid HF model config")
    require(model_config.get("architectures") == ["Qwen3ForCausalLM"], "Invalid HF architecture")
    read_json(export / "tokenizer_config.json")
    require((export / "tokenizer.json").is_file(), "Missing HF tokenizer")
    index = export / "model.safetensors.index.json"
    shards = set(read_json(index).get("weight_map", {}).values()) if index.is_file() else {"model.safetensors"}
    require(bool(shards), "Empty HF weight index")
    for name in shards:
        shard = export / name
        require(shard.is_file() and shard.stat().st_size > 0, f"Missing HF weight shard: {name}")


def _phase_steps(case, phase):
    """Return the exact expected step sequence for each resume phase."""
    steps = (phase,) if case.resume else (1, 2)
    if case.algorithm == "ppo" and case.resume and phase == 2:
        steps = (2, 3)
    return steps
