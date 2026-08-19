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
"""Distributed numerical consistency gates."""

from typing import Any, Optional

from hyper_parallel import get_platform

platform = get_platform()


def validate_consistency_forward_inputs(
    experience: Any,
    *,
    group: Any,
    group_size: int,
    operation: str,
) -> None:
    """Fail every DP rank together on locally detectable packed-forward errors."""
    local_error = None
    try:
        sequences = experience.sequences
        attention_mask = experience.attention_mask
        if sequences.ndim != 2 or tuple(attention_mask.shape) != tuple(sequences.shape):
            raise ValueError(
                "Consistency forward requires aligned two-dimensional sequences and attention_mask"
            )
        valid_mask = attention_mask.bool()
        lengths = valid_mask.sum(dim=-1, dtype=platform.tensor_dtype.int32)
        lengths_cpu = lengths.tolist()
        if any(length < 2 for length in lengths_cpu):
            raise ValueError("Consistency forward requires at least two valid tokens per sequence")
        expected_mask = (
            platform.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not bool((valid_mask == expected_mask).all().item()):
            raise ValueError("Consistency forward requires contiguous right padding")
    except Exception as error:  # pylint: disable=W0718
        local_error = str(error)

    errors: list[Optional[str]] = [None] * group_size
    if group_size == 1:
        errors[0] = local_error
    else:
        platform.all_gather_object(errors, local_error, group)
    if any(error is not None for error in errors):
        raise RuntimeError(f"{operation} consistency forward preflight failed: errors={errors}")


def validate_pre_update_consistency(
    experience: Any,
    actor_log_probs: Any,
    *,
    expected_policy_version: int,
    expected_policy_fingerprint: Optional[str],
    group: Any,
    group_size: int,
) -> dict[str, float]:
    """Require rollout and pre-update Actor logprobs to be bit-exact on every DP rank."""
    try:
        rank = platform.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    record: dict[str, Any] = {
        "rank": rank,
        "error": None,
        "token_count": 0,
        "mismatch_count": 0,
        "abs_diff_sum": 0.0,
        "max_abs_diff": 0.0,
        "first_mismatch": None,
    }
    try:
        rollout_log_probs = experience.old_log_probs
        if rollout_log_probs is None:
            raise ValueError("Pre-update consistency requires rollout log-probabilities")
        if experience.worker_policy_version != expected_policy_version:
            raise ValueError(
                "Pre-update worker policy version mismatch: "
                f"expected={expected_policy_version}, actual={experience.worker_policy_version}"
            )
        if expected_policy_fingerprint is None:
            raise ValueError("Pre-update consistency requires a published policy fingerprint")
        if experience.worker_policy_fingerprint != expected_policy_fingerprint:
            raise ValueError(
                "Pre-update worker policy fingerprint mismatch: "
                f"expected={expected_policy_fingerprint}, "
                f"actual={experience.worker_policy_fingerprint}"
            )
        if tuple(actor_log_probs.shape) != tuple(rollout_log_probs.shape):
            raise ValueError(
                "Pre-update log-probability shape mismatch: "
                f"actor={tuple(actor_log_probs.shape)}, rollout={tuple(rollout_log_probs.shape)}"
            )
        if actor_log_probs.dtype != rollout_log_probs.dtype:
            raise ValueError(
                "Pre-update log-probability dtype mismatch: "
                f"actor={actor_log_probs.dtype}, rollout={rollout_log_probs.dtype}"
            )
        if actor_log_probs.dtype != platform.tensor_dtype.float32:
            raise ValueError(
                "Pre-update consistency requires FP32 raw log-probabilities, "
                f"got {actor_log_probs.dtype}"
            )
        mask = experience.loss_action_mask.bool()
        if tuple(mask.shape) != tuple(actor_log_probs.shape):
            raise ValueError(
                "Pre-update action mask shape mismatch: "
                f"mask={tuple(mask.shape)}, log_probs={tuple(actor_log_probs.shape)}"
            )
        token_count = int(mask.flatten().sum(dim=0).item())
        if token_count <= 0:
            raise ValueError("Pre-update consistency requires at least one valid action token")
        actor_values = actor_log_probs.detach().masked_select(mask)
        rollout_values = rollout_log_probs.detach().masked_select(mask)
        if not bool(actor_values.isfinite().all().item()):
            raise ValueError("Pre-update Actor log-probabilities contain non-finite values")
        if not bool(rollout_values.isfinite().all().item()):
            raise ValueError("Pre-update rollout log-probabilities contain non-finite values")

        actor_bits = actor_log_probs.detach().contiguous().view(platform.tensor_dtype.int32)
        rollout_bits = rollout_log_probs.detach().contiguous().view(platform.tensor_dtype.int32)
        mismatch_mask = actor_bits.ne(rollout_bits) & mask
        mismatch_count = int(mismatch_mask.flatten().sum(dim=0).item())
        absolute_diff = (actor_values - rollout_values).abs()
        record.update(
            {
                "token_count": token_count,
                "mismatch_count": mismatch_count,
                "abs_diff_sum": float(absolute_diff.sum(dim=0).item()),
                "max_abs_diff": float(absolute_diff.max().item()),
            }
        )
        if mismatch_count:
            first = mismatch_mask.nonzero(as_tuple=False)[0]
            row = int(first[0].item())
            column = int(first[1].item())
            trajectory = experience.trajectories[row] if experience.trajectories else None
            actor_value = float(actor_log_probs[row, column].item())
            rollout_value = float(rollout_log_probs[row, column].item())
            record["first_mismatch"] = {
                "rank": rank,
                "row": row,
                "trajectory_id": None if trajectory is None else trajectory.trajectory_id,
                "prompt_id": None if trajectory is None else trajectory.prompt_id,
                "response_offset": int(mask[row, : column + 1].sum().item()) - 1,
                "sequence_position": column + 1,
                "token_id": int(experience.sequences[row, column + 1].item()),
                "actor_value": actor_value,
                "rollout_value": rollout_value,
                "actor_bits": f"0x{int(actor_bits[row, column].item()) & 0xFFFFFFFF:08x}",
                "rollout_bits": f"0x{int(rollout_bits[row, column].item()) & 0xFFFFFFFF:08x}",
                "abs_diff": abs(actor_value - rollout_value),
            }
    except Exception as error:  # pylint: disable=W0718
        record["error"] = str(error)

    records: list[Optional[dict[str, Any]]] = [None] * group_size
    if group_size == 1:
        records[0] = record
    else:
        platform.all_gather_object(records, record, group)
    gathered = [item for item in records if item is not None]
    errors = [item for item in gathered if item["error"] is not None]
    mismatches = [item for item in gathered if item["mismatch_count"]]
    if errors or mismatches:
        raise RuntimeError(
            "Pre-update rollout/Trainer bit-exact gate failed: "
            f"errors={errors}, first_mismatches="
            f"{[item['first_mismatch'] for item in mismatches]}"
        )
    token_count = sum(int(item["token_count"]) for item in gathered)
    diff_sum = sum(float(item["abs_diff_sum"]) for item in gathered)
    return {
        "training/pre_update_exact_valid": 1.0,
        "training/pre_update_exact_tokens": float(token_count),
        "training/pre_update_mismatch_count": 0.0,
        "training/pre_update_max_abs_diff": max(
            float(item["max_abs_diff"]) for item in gathered
        ),
        "training/pre_update_mean_abs_diff": diff_sum / token_count,
    }


__all__ = [
    "validate_consistency_forward_inputs",
    "validate_pre_update_consistency",
]
