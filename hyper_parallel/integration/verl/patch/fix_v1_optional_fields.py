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
"""Direct method replacements for VeRL v0.9.1; call register_fix_v1_optional_fields() before training.

Optional fields may be absent when their TransferQueue producers are disabled.
References: https://github.com/verl-project/verl/pull/8461
            https://github.com/verl-project/verl/pull/8565
Field selection also guards ref_log_prob when the reference policy is disabled.
The method bodies are copied from v0.9.1 and need review when upgrading VeRL.
"""

from verl.trainer.ppo.v1.trainer_base import (
    PPOTrainer,
    KVBatchMeta,
    DataProto,
    TensorDict,
    np,
    tq,
    apply_kl_penalty,
    compute_advantage_for_multi_trajectories,
    compute_rollout_correction_and_add_to_batch,
    response_to_nested,
    get_metric_data_with_optional_routed_experts,
    compute_moe_lb_metrics,
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
    compute_variance_proxy_metrics,
    compute_spec_decode_metrics,
)


def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    """Compute the advantage of the batch."""
    # ==================== FIX BEGIN: PR #8461 ====================
    # Before: fields unconditionally included "values".
    # After: request "values" only when the critic produces it.
    fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs"]
    if self.use_reference_policy:
        fields.append("ref_log_prob")
    if self.use_critic:
        fields.append("values")
    # ===================== FIX END: PR #8461 =====================
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

    response_mask = data["response_mask"]
    data = DataProto(batch=data.to_padded_tensor())
    data.batch["token_level_scores"] = data.batch["rm_scores"]
    data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

    # 1. apply kl penalty to rewards
    if self.config.algorithm.use_kl_in_reward:
        data, kl_metrics = apply_kl_penalty(
            data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        data.batch["token_level_rewards"] = data.batch["token_level_scores"]

    # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
    # Only runs in decoupled mode (computes once per batch using stable π_old)
    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
    rollout_correction = (
        rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
    )
    if rollout_correction:
        data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
        metrics.update(is_metrics)

    # 3. compute advantages
    data = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=batch.keys,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        config=self.config.algorithm,
    )

    # 4. write nested advantages and returns back to TransferQueue
    fields = ["advantages", "returns"]
    if self.config.algorithm.use_kl_in_reward:
        fields.append("token_level_rewards")
    if rollout_correction:
        fields.append("response_mask")
        if "rollout_is_weights" in data.batch:
            fields.append("rollout_is_weights")

    output = {}
    for field in fields:
        output[field] = response_to_nested(data.batch[field], response_mask)
    output = TensorDict(output, batch_size=len(batch))

    batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    return batch


def _spec_decode_statistics(self, batch):
    """Read speculative-decoding counters only when rollout produces them."""
    spec_drafts = spec_accepts = spec_verifies = None
    mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
    if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
        spec_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = spec_data.pop("extra_fields").tolist()
        # The rollout omits the spec_* stats when the backend does not report
        # per-request spec-decode stats; leave all three as None in that case.
        if extra_fields and all(
            isinstance(extra_field, dict) and "spec_num_draft_tokens" in extra_field for extra_field in extra_fields
        ):
            spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
            spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
            spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

    return spec_drafts, spec_accepts, spec_verifies


def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
    """Compute PPO metrics using only fields produced by enabled features."""
    # 1. collect necessary fields from TransferQueue for computing metrics
    non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
    # =============== FIX BEGIN: PR #8461 / #8565 ===============
    # Before: fields unconditionally included "values" and "token_level_rewards".
    # After: append each field only when its producing feature is enabled.
    fields = [
        "prompts",
        "responses",
        "response_mask",
        "advantages",
        "returns",
        "rm_scores",
        "num_turns",
    ]
    if self.config.algorithm.use_kl_in_reward:
        fields.append("token_level_rewards")
    if self.use_critic:
        fields.append("values")
    # ================ FIX END: PR #8461 / #8565 ================
    moe_lb_metrics_interval = self.config.actor_rollout_ref.rollout.get("moe_load_balance_metrics_interval", 0)
    data = get_metric_data_with_optional_routed_experts(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=fields,
        moe_lb_metrics_interval=moe_lb_metrics_interval,
        global_steps=global_steps,
        accumulator=self._rollout_moe_lb_metrics_accumulator,  # pylint: disable=protected-access
        kv_batch_get=tq.kv_batch_get,
    )

    num_turns = np.array(data.pop("num_turns").tolist())
    prompt_length = data["prompts"].offsets().diff()
    response_length = data["responses"].offsets().diff()
    global_token_num = (prompt_length + response_length).tolist()
    min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
    max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

    spec_drafts, spec_accepts, spec_verifies = _spec_decode_statistics(self, batch)

    data = data.to_padded_tensor()
    data["token_level_scores"] = data["rm_scores"]
    if "token_level_rewards" not in data:
        data["token_level_rewards"] = data["rm_scores"]
    data["prompt_length"] = prompt_length.float()
    data["response_length"] = response_length.float()
    batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
    metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch
    # Expose per-row turn counts under the V0 schema (agent_loop writes
    # ``__num_turns__`` into the non-tensor batch) so compute_data_metrics
    # emits the V0-compatible ``num_turns/{mean,max,min}`` tags in addition
    # to the ``training/num_turns/*`` names computed below.
    num_turns_for_metrics = num_turns[non_padding_mask] if non_padding_mask.any() else num_turns
    metrics_batch.non_tensor_batch["__num_turns__"] = np.asarray(num_turns_for_metrics, dtype=np.int32)

    # 2. compute metrics
    metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
    metrics.update(
        compute_moe_lb_metrics(
            metrics_batch=metrics_batch,
            moe_lb_metrics_interval=moe_lb_metrics_interval,
            global_steps=global_steps,
            accumulator=self._rollout_moe_lb_metrics_accumulator,  # pylint: disable=protected-access
        )
    )
    metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    n_gpus = self._get_n_gpus_for_throughput()  # pylint: disable=protected-access
    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
    gradient_norm = metrics.get("actor/grad_norm", None)
    metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

    # 3. other auxiliary metrics
    if non_padding_mask.any():
        num_turns = num_turns[non_padding_mask]
    metrics.update(
        {
            "training/num_turns/mean": num_turns.mean(),
            "training/num_turns/max": num_turns.max(),
            "training/num_turns/min": num_turns.min(),
        }
    )

    # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
    # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
    metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

    # 5. off-policy staleness metrics
    #   global_steps is the model weight version (one update_weights per global_step), and
    #   min/max_global_steps are the versions a trajectory was generated across, so all quantities
    #   below are already in model-version units.
    #   - trajectory_spans: how many distinct model versions a single trajectory was
    #     generated across (1 == fully generated on a single version). This captures the
    #     within-trajectory policy inconsistency caused by partial rollout / continuation.
    #   - trajectory_staleness: how many model versions the trajectory lags behind the
    #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
    #     so the lag is a range: the freshest weights used give the lower bound
    #     (global_steps - max_global_steps) and the oldest weights the worst case
    #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
    trajectory_spans = max_global_steps - min_global_steps + 1
    trajectory_staleness = (global_steps - 1) - max_global_steps
    trajectory_staleness_worst = (global_steps - 1) - min_global_steps
    metrics.update(
        {
            "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
            "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
            "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
            "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
            "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
            "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
            "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
            "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
            "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
        }
    )


def register_fix_v1_optional_fields() -> None:
    """Replace the two V1 methods without changing VeRL source files."""
    PPOTrainer._compute_advantage = _compute_advantage  # pylint: disable=protected-access
    PPOTrainer._compute_metrics = _compute_metrics  # pylint: disable=protected-access
