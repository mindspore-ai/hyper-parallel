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

import hashlib
import json
import math
from pathlib import Path
import re
import struct
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


def tensor(record: dict) -> list[list[Any]]:
    """Check serialized tensor shape and exact byte digest."""
    values, shape = record["values"], record["shape"]
    require(len(shape) == 2 and len(values) == shape[0] and shape[0] > 0, "Invalid tensor rows")
    require(all(len(row) == shape[1] for row in values), "Invalid tensor columns")
    formats = {"int64": "q", "int32": "i", "float32": "f"}
    require(record["dtype"] in formats, "Unexpected artifact dtype")
    payload = b"".join(struct.pack("<" + formats[record["dtype"]], value)
                       for row in values for value in row)
    require(hashlib.sha256(payload).hexdigest() == record["sha256"], "Tensor digest mismatch")
    return values


def validate_phase(output: Path, case: Any, phase: int) -> None:
    """Require committed steps, real updates and matching rollout/worker evidence."""
    steps = (phase,) if case.resume else (1, 2)
    rows = metrics(output / f"phase-{phase}.log")
    require(set(step for step, row in rows.items() if "train/global_step" in row) == set(steps),
            f"Expected training steps {steps}, got {list(rows)}")
    changed = False
    for step in steps:
        row = rows[step]
        require(all(math.isfinite(value) for value in row.values()), f"Non-finite metrics at {step}")
        for key in ("train/valid_tokens", "train/optimizer_steps", "rollout/generated_tokens"):
            require(row.get(key, 0) > 0, f"No {key} at step {step}")
        require(row["train/global_step"] == row.get("policy/version") == step, "Policy version mismatch")
        require("train/total_loss" in row and "train/gradient_norm" in row, "Missing optimizer metrics")
        if case.family != "qwen3":
            require(
                row.get("reward/min") == 0 and row.get("reward/max") == 1
                and row.get("reward/zero_std_groups") == 0,
                "MoE control samples did not produce mixed exact-match rewards",
            )
        changed |= row["train/gradient_norm"] > 0 and row.get("policy/fingerprint_changed") == 1
        require(row.get(f"weight_sync/completed_{case.strategy}") == 1, "Wrong completed transfer strategy")
        require(row.get("weight_sync/fallback_count") == 0, "Unexpected fallback")
        if case.exact:
            require(row.get("training/pre_update_exact_tokens", 0) > 0, "Empty consistency comparison")
            for key in ("mismatch_count", "max_abs_diff", "mean_abs_diff"):
                require(row.get("training/pre_update_" + key) == 0, f"Consistency failed: {key}")
        if case.strategy == "full_gather":
            prefix = "weight_sync/streaming_"
            count = row.get(prefix + "bucket_count", 0)
            require(count > 0 and row.get(prefix + "acked_buckets") == count
                    and row.get(prefix + "released_buckets") == count, "Unacknowledged streaming buckets")
            require(row.get(prefix + "max_inflight_buckets") == 1, "Unbounded in-flight buckets")
            for key in ("max_gathered_bytes", "max_packed_bytes"):
                require(0 < row.get(prefix + key, 0) <= 128 * 1024**2, f"Bucket bound exceeded: {key}")
        version = step - 1
        fingerprints = set()
        for rank in range(case.world):
            artifact = read_json(output / "rollouts" / f"{case.strategy}-policy{version}-rank{rank}.json")
            require(artifact["oracle_run_id"] == output.name, "Stale rollout artifact")
            require(artifact["trainer_rank"] == rank, "Wrong Trainer rank")
            require(artifact["policy_version"] == artifact["worker_policy_version"] == version,
                    "Generation used the wrong policy")
            fingerprints.add(artifact["worker_policy_fingerprint"])
            sequences = tensor(artifact["sequences"])
            attention = tensor(artifact["attention_mask"])
            actions = tensor(artifact["action_mask"])
            logprobs = tensor(artifact["raw_logprobs"])
            require(len(sequences) == len(attention) == len(actions) == len(logprobs), "Batch mismatch")
            for seq, mask, action, probs in zip(sequences, attention, actions, logprobs):
                require(len(seq) == len(mask) == len(action) == len(probs) + 1, "Token alignment mismatch")
                require(action[0] == 0 and sum(action) > 0, "Missing action tokens")
                require(all(a in (0, 1) and m in (0, 1) and a <= m for a, m in zip(action, mask)),
                        "Padding contributes to policy loss")
                require(all(math.isfinite(value) for value in probs), "Non-finite raw logprob")
        require(len(fingerprints) == 1 and all(fingerprints), "Workers disagree on generation identity")
        manifests = list((output / "manifests").glob(f"{case.strategy}-version{step}-*.json"))
        require(len(manifests) == case.world, "Missing worker manifests")
        coordinates = set()
        for path in manifests:
            manifest = read_json(path)
            require(manifest["oracle_run_id"] == output.name and manifest["policy_version"] == step,
                    "Stale worker manifest")
            coordinates.add((manifest["dp_rank"], manifest["tp_rank"]))
            require(manifest["dp_size"] == 2 and manifest["tp_size"] == case.tp, "Worker topology mismatch")
            tensors = manifest["tensors"]
            require(bool(tensors) and manifest["parameter_count"] == len(tensors), "Empty weight evidence")
            payload = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
            require(hashlib.sha256(payload).hexdigest() == manifest["manifest_sha256"], "Manifest hash mismatch")
            require(manifest["total_bytes"] == sum(value["num_bytes"] for value in tensors.values()),
                    "Manifest byte count mismatch")
            require(manifest.get("source_match", True) is True, "Source comparison failed")
            require(manifest.get("model_type") == case.family, "Wrong worker model family")
            if case.family != "qwen3":
                require(manifest.get("ep_size") == 4, "EP4 was not active")
                require(any(".experts." in name for name in tensors), "Missing expert weights")
            if case.family == "deepseek_v3":
                require(manifest.get("deepseek_runtime", {}).get("absorbed_mla_layer_count", 0) > 0,
                        "Missing absorbed MLA runtime evidence")
        require(coordinates == {(dp, tp) for dp in range(2) for tp in range(case.tp)}, "Wrong worker ranks")
    if not case.resume or phase == 2:
        versions = []
        for version in (1, 2):
            manifests = [read_json(path) for path in
                         (output / "manifests").glob(f"{case.strategy}-version{version}-*.json")]
            versions.append({(item["dp_rank"], item["tp_rank"]): item["manifest_sha256"] for item in manifests})
        require(versions[0] != versions[1], "Published weight bytes did not change across updates")
    require(changed, "No non-zero learning update: inspect real rewards/advantages; do not relabel rewards")
    if case.name == "dense-tp1-full":
        require(any(row.get("validation/total", 0) > 0 and 0 <= row.get("validation/accuracy", -1) <= 1
                    for row in rows.values()), "Missing evaluation results")
    if case.resume:
        checkpoint = output / "checkpoints" / f"step_{phase}"
        marker = read_json(checkpoint / "checkpoint_complete.json")
        require(marker["step"] == phase and marker["world_size"] == case.world, "Incomplete checkpoint")
        require(read_json(checkpoint / "extra_state.json")["global_step"] == phase, "Checkpoint step mismatch")
        require(any(checkpoint.glob("*.safetensors")), "Missing model checkpoint tensors")
        for rank in range(case.world):
            rank_dir = checkpoint / f"rank_{rank}"
            for suffix in ("bytes", "metadata"):
                require(any(path.stat().st_size > 0 for path in rank_dir.glob(f"*.{suffix}")),
                        f"Missing rank-local checkpoint {suffix}: rank={rank}")


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
        require(bool(identity["policy_fingerprint"]), "Session has no policy identity")
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
    require(seen == set(versions), f"Expected Agent policy versions {versions}, got {seen}")
