# Optional OPD / MOPD

**English** | [简体中文](README.zh.md)

This example reuses the synchronous Trainer, agent trajectories, and student weight publication. When `distillation` is omitted or
`distillation.enabled: false`, no teacher client is created, no teacher requests are sent, and no distillation loss is computed.

## Models and Resources

The first recipe uses a Qwen3-4B student and frozen Qwen3-30B-A3B teacher. They must have identical
`tokenizer.json`, tokenizer configuration, and special-token configuration. The teacher is an independently managed external vLLM service;
It does not participate in the student's optimizer or full_gather publication.

The example uses one student training rank and a separate single-device TP1 rollout; allocate two additional TP2 devices for the teacher. A3B denotes active parameters,
but the teacher must load its entire checkpoint. Headroom depends on weights, caches, context, and concurrency; arbitrary configurations are not guaranteed to fit.
Keep the teacher model directory immutable. The service must load the configured checkpoint, not merely reuse the same service name.

Use the [pinned runtime](../../docker/README.md), ensuring the editable installation points to the current checkout.
Run from the repository root; the following paths must be visible inside the container:

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl${PYTHONPATH:+:$PYTHONPATH}"
ASCEND_RT_VISIBLE_DEVICES=4,5 HCCL_IF_BASE_PORT=62000 python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B --served-model-name opd-teacher \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 --logprobs-mode raw_logprobs \
  --host 127.0.0.1 --port 8450
```

Teacher and student processes need nonconflicting devices and HCCL ports. Across containers, use a reachable service address.
The teacher needs `v1/models` and a `v1/completions` echo/logprobs interface accepting raw token IDs.
Request neutral temperature/top-p/top-k and no penalties, extracting input probabilities rather than generating extra tokens.

## Single-Teacher Execution

Prepare GSM8K Parquet as in [Quick Start](../../README.md#快速开始), and update data/output paths in YAML.
Before launching the student, validate model identity, tokenizer compatibility, and raw-token probability alignment:

```bash
python -m examples.distillation.probe_teacher \
  --config hyper_parallel/rl/examples/distillation/configs/qwen3_4b_opd.yaml
ASCEND_RT_VISIBLE_DEVICES=0 HCCL_IF_BASE_PORT=62600 \
  bash hyper_parallel/rl/examples/distillation/scripts/run_distillation.sh \
  hyper_parallel/rl/examples/distillation/configs/qwen3_4b_opd.yaml
```

`qwen3_4b_opd.yaml` disables the task objective and optimizes distillation only. Task scoring remains for observation, without constructing GRPO group advantages
or Critic. `qwen3_4b_grpo_opd.yaml` retains GRPO and adds distillation.
The launcher does not rewrite training decisions in YAML. `NPROC_PER_NODE` must match the configured training topology.
Distillation recipes set `train.optimizer.fp32_main_params: true`, using existing main-project wrappers to retain FP32 master parameters and moment states.
BF16 parameters serve computation/publication; after gradient preparation, clipping uses FP32 master parameters.
Default CPU offload retains optimizer master parameters and states in host memory; plan capacity for the model and topology.
Checkpoints record master precision and sharding topology and reject mismatched resume configurations. BF16-only checkpoints cannot recover lost small updates.
Rank-local state saves master-parameter DTensors as local FP32 tensors, without serializing distributed layouts or rounding back to BF16.
Single-rank CPU offload also uses native FSDP, preventing the unsharded path from ignoring offload configuration.
Existing nondistillation configurations retain their precision behavior when this field is omitted.
See [precision regression tests](../../../../tests/ut/rl/policy/test_optimizer_precision.py) for small-learning-rate accumulation, actual master-gradient clipping, and recovery.
Student training defaults to NPU 0 and rollout to YAML `visible_devices: '1'`; teachers use additional devices.
The pinned image's DP=2 probability discrepancy was traced to nonpersistent RoPE frequency buffers not restored after meta materialization.
RL's [materialization adapter](../../rl/workers/engine/materialization.py) uses existing public interfaces
to register buffer reconstruction only for Qwen3 instances built by the RL builder. It evaluates native HF formulas on CPU, then transfers to the target device,
to match the native HF initialization formula. Recomputing frequencies directly on NPU introduces small FP32 differences
that may change BF16 cos/sin at larger positions.
Registration occurs before sharding and `to_empty`, covering checkpoint loading and random initialization. Repeated registration does not modify public classes or other instances.
No changes to `hyper_parallel/core` or `hyper_parallel/models` are required.
Instance-isolation and HF RoPE comparisons are in [RL materialization tests](../../tests/ut/test_materialization.py).
The distillation examples use native-vLLM and record rollout probabilities separately from training-side recomputation. The old bit-exact gate has been removed; teacher-probability validity checks still block invalid updates and publication.
Successful replay does not replace training, publication, and checkpoint acceptance. The lower-resource single-rank recipe is the default.

## Multiple Teachers

`qwen3_4b_mopd.yaml` uses separate Qwen3-30B-A3B and Qwen3-8B checkpoints.
Prepare actual paths, services, and routing data first. Multiple replicas inside one service increase capacity but do not constitute multiple teachers.

Start the second teacher in another independent process or container:

```bash
ASCEND_RT_VISIBLE_DEVICES=6 HCCL_IF_BASE_PORT=62300 python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-8B --served-model-name dense-teacher \
  --tensor-parallel-size 1 --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 --logprobs-mode raw_logprobs \
  --host 127.0.0.1 --port 8451
```

Training Parquet retains task prompts/labels and adds `teacher_key`, with values `large` and `dense` for the two teachers.
After configuring `data.metadata_columns: [teacher_key]`,
this field enters PromptRecord.metadata.
Task environments and grading rules must still match the data. This template inherits GSM8K; real code tasks require the existing code row adapter
and environment. Changing teachers alone does not make GSM8K rewards suitable for code. A minimal working multi-teacher experiment may instead use two math
expert checkpoints on different GSM8K subsets before extending to mixed tasks.

Select one teacher per sample without averaging teacher distributions. Missing/unknown routing, HTTP errors, oversized contexts, token mismatches,
and missing/nonfinite probabilities reject the current update. DP padding rows make no teacher requests and have zero loss.

## Mathematical and Algorithmic Boundaries

Follow verl's sampled k1 PG approach: recompute current student probabilities on every training forward and use
`stopgrad(log p_teacher - log p_student_current)` as an independent token advantage,
combined with clipped ratios and dual clipping.
ExperienceBatch.rollout_log_probs independently retains actual sampling probabilities; old_log_probs stores pre-update training-side recomputation.
With OPD enabled, use the rollout-anchor (bypass) path: both task RL and OPD ratios divide by original sampling probabilities,
fixed within the batch. Recomputed probabilities are diagnostic only and do not replace the sampling distribution; diagnostics explicitly compare both fields.
Missing sampling probabilities, invalid action probabilities, or scoring failures reject the update. Invalid observations and DP padding are excluded from probability checks/loss.
The initial version requires student temperature=1, top_p=1, and top_k disabled; tempered/truncated rollout correction is unavailable.

Task and OPD objectives are computed separately. OPD normalizes by valid action-token count across all DP; the GSPO task term remains sequence-normalized.
`loss.coefficient` applies to both pure distillation and joint training. k1 is unclipped by default; `loss_max_clamp` explicitly changes the signal.
GRPO/PPO/GSPO combinations share interfaces, but real NPU acceptance must be performed per recipe.

External tool tasks such as Search-R1 retain episode GRPO: add this example's `distillation` block to an existing configuration.
Score every call in its actual context; tool observations do not contribute action loss. Single-teacher mode needs no routing field; multiple teachers
receive their key through the existing row adapter/metadata_columns. Pure OPD currently supports only the internal runner;
this feature does not broaden external-program PPO/GSPO support. Teacher context limits must cover every complete call plus one extra token.
For long-context Search-R1, expand max_model_len in both teacher service and configuration; do not retain the short example's 2048 limit.
Validate training/inference probability consistency separately. Identical parameter versions do not guarantee bit-exact probabilities. This path uses sampling denominators directly, without a recomputation anchor plus extra IS weights; clipping remains a biased surrogate objective.

## Boundaries Compared with verl

| Item | verl reference | This implementation |
| --- | --- | --- |
| Distillation objective | Sampled KL PG and top-k forward KL/GKD | Sampled k1 PG with explicit stop-gradient |
| MOPD | One teacher selected by sample routing key | One teacher selected by teacher_key in PromptRecord.metadata |
| Teacher resources | Ray resource pools, async inference clients, multiple replicas | Externally managed frozen vLLM HTTP services; Trainer TP owner sends requests |
| Policy denominator | Optional rollout correction/bypass | Original rollout logprobs for both task and distillation terms when OPD is enabled |
| Loss normalization | Configured aggregation and global batch information | Distillation by valid action tokens across all DP; task retains algorithm normalization |
| Pure distillation | Disable task loss | No task advantages, Reference, or Critic; coefficient remains effective |

Older Hyper-RL distillation configuration, teacher clients, k1 objectives, and tests may be adapted and reused, but its complete RL tree contains older
model, agent, checkpoint, and publication code and must not overwrite the new mainline. verl's mathematics and routing contracts are useful
references; its Ray/TensorDict/worker code cannot directly replace this project's SyncTrainer/ExperienceBatch.

## Teacher Selection

For Qwen3-4B, prefer Qwen3 teachers or domain-finetuned models with the same tokenizer:

| Candidate | Purpose and validation requirements |
| --- | --- |
| Qwen3-30B-A3B | Default MoE teacher; load all approximately 30B weights, not memory sized for 3B active parameters |
| Qwen3-8B | Two-teacher functional acceptance recipe; distinct checkpoints do not automatically imply distinct domain experts |
| Qwen3-14B, Qwen3-32B | Larger dense candidates; strict tokenizer and probability probes still required before deployment |
| Math/code finetunes with the same tokenizer | Better suited to evaluating cross-domain benefits; independent task environments and held-out evaluation required |
| Qwen3.8-27B | Tokenizer incompatible with Qwen3; current sampled-token MOPD must reject it |

Qwen3 and Qwen3.8 have different vocabularies and token IDs. Copying tokenizer files, taking probabilities for the same numeric ID,
or simple re-tokenization cannot fix this. Cross-tokenizer distillation needs separately defined sequence/text probability objectives and alignment, outside this contract.
See official model descriptions for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B),
[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), and
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B).

## Validation and Recovery

`PYTHONPATH=hyper_parallel/rl python -m pytest -q hyper_parallel/rl/tests/ut/test_materialization.py`
checks native HF frequencies, cos/sin, repeated materialization, and instance isolation.

```bash
python -m pytest -q tests/ut/rl/distillation
python -m pytest -q tests/ut/rl
```

Key metrics are `distillation/loss`, `distillation/supervised_tokens`,
`distillation/teacher_samples/<key>`, and `timing_s/teacher`,
alongside existing gradient norm and policy version
and held-out quality. OPD loss may be negative and is not a standalone quality metric.

An optional in-service probability comparison decodes student tokens prefix by prefix and compares echo scoring:

```bash
python -m examples.distillation.probe_teacher \
  --config /path/to/config.yaml --decode-reference --atol 0 --rtol 0
```

`--reference-forward` separately loads the complete teacher for a CPU HF comparison, defaulting to BF16. It checks cross-backend numerical differences
and cannot be replaced by in-service comparisons. OPD's teacher distribution is the actual frozen service with validated raw logprobs.

The probability contract now uses rollout-anchor. Old-contract checkpoint fingerprints are incompatible, requiring new training or explicit migration.

Checkpoint manifests save distillation configuration, local model/tokenizer metadata, and content fingerprints for all teacher weight shards and indexes.
Every training process streams teacher weights once at startup validation with bounded memory. Saving reuses that identity without rescanning each step.
Resume recomputes identities and rejects mismatches. Older distillation checkpoints lacking weight digests require retraining or explicit migration, not direct resume.
Local hashes do not prove external service in-memory weight identity. Teachers must use matching read-only model directories and must not replace weights or reload during training.
Teacher failures propagate across training ranks before Actor updates, without falling back to another teacher or publishing partial scores.

CPU routing/gradient tests do not replace real multi-teacher, long-run learning, or complete benchmarks. With only one actual checkpoint,
OPD and MOPD routing contracts can be accepted, but real multi-teacher effects cannot be claimed.

Real NPU acceptance uses an independent ST entry. Missing resources cause skips, not passes:

```bash
example_root=hyper_parallel/rl/examples/distillation
export RL_OPD_ST_CONFIG="$example_root/configs/qwen3_4b_opd.yaml"
export RL_OPD_ST_NPROC=1
ASCEND_RT_VISIBLE_DEVICES=0 HCCL_IF_BASE_PORT=62600 python -m pytest -vv -s \
  hyper_parallel/rl/tests/st/test_distillation_st.py
```

ST requires at least two real updates, nonzero gradient norms, teacher-supervised tokens, and correct published versions.
Multi-teacher ST requires prepared real two-teacher configuration; replicas of one checkpoint cannot replace expert validation.
ST also checks pre-update rollout versus Actor log-probability differences: mean absolute difference defaults to at most 0.1,
and maximum absolute difference to at most 1.0. Set explicit tolerances through `RL_OPD_ST_MAX_MEAN_LOGPROB_DIFF` and
`RL_OPD_ST_MAX_LOGPROB_DIFF`; document the basis for tolerance changes in acceptance records.
Two-step ST validates execution and numerical thresholds. Long-term learning gains require independent held-out evaluation.

## Implementation and Test Navigation

- [Teacher client and routing][teacher]: `TeacherManager`, raw-token alignment, and service identity.
- [Distillation objective][objective]: `compute_opd_loss` and pure-OPD requirement selection.
- [Trainer][trainer]: `_score_teacher_rollout` and TP request ownership.
- [CPU contract and gradient tests][unit]: Routing, masks, failure propagation, algorithm combinations, and resume identity.
- [Real ST][system]: Explicit resources and supervision, gradient, and publication assertions.

[teacher]: ../../rl/distillation/teacher.py
[objective]: ../../rl/algorithm/distillation.py
[trainer]: ../../rl/trainer/sync.py
[unit]: ../../../../tests/ut/rl/distillation/test_distillation.py
[system]: ../../tests/st/test_distillation_st.py

## Pre-Update Gates and Distillation Monitoring

Implementation resides in the [distillation monitoring module](../../rl/distillation/monitoring.py);
update-rejection and mixed-length routing tests are in [monitoring contract tests](../../../../tests/ut/rl/distillation/test_monitoring.py).

With distillation enabled, every batch compares sampling and training-side recomputed probabilities before Actor/Critic updates and policy publication.
All DP ranks jointly aggregate valid action tokens; tool observations, padded tokens, and DP padding are excluded.
Nonfinite probabilities, positive log-probabilities, empty global action sets, or any exceeded metric threshold reject the whole update.

```yaml
distillation:
  probability_gate:
    max_mean_logprob_diff: 0.1
    max_logprob_diff: 1.0
    max_ratio_outside_fraction: 0.05
```

The ratio is `exp(logp_actor_before_update - logp_rollout)`; its allowed interval uses distillation
`loss.clip_ratio`. These three defaults target validated Qwen3 recipes and allow small numerical tail discrepancies,
not a claim that every token agrees. New models/backends need independent calibration; do not loosen thresholds to hide sharding errors.
ST tolerance environment variables also configure the pre-update gate; post-update checks only corroborate existing acceptance evidence.

`train/clip_fraction` retains the task algorithm's definition. Distillation has independent metrics:

| Metric | Meaning |
| --- | --- |
| `distillation/ratio_outside_fraction` | Fraction of action tokens whose ratio lies outside the PPO clipping interval |
| `distillation/clip_fraction` | Fraction where the PPO clipped branch is actually selected, distinct from merely being outside the interval |
| `distillation/dual_clip_fraction` | Fraction of negative advantages additionally triggering dual clipping |
| `distillation/ratio_mean`, `log_ratio_min/max` | Mean ratio and unclipped log-ratio tail range |
| `distillation/log_ratio_saturation_fraction` | Fraction where numerical protection clamps log-ratio to ±20 |
| `distillation/sampled_k1_mean` | Mean sampled current-student-minus-teacher log-probability signal; may be negative and is not exact KL |
| `distillation/teacher/<key>/token_fraction` | Each teacher's share of valid supervised tokens |
| `distillation/teacher/<key>/weighted_loss` | Each teacher's weighted contribution to global distillation loss, including coefficient |

Teacher sample counts remain in `distillation/teacher_samples/<key>`.
Equal sample counts do not imply equal token shares. The objective is token-normalized, without automatically averaging teachers equally.
Metrics aggregate across micro-batches and DP; repeated training epochs count supervised tokens again.

## Effectiveness Evaluation

Use the unified `scripts/run_distillation.sh` entry, select independent evaluation data through `data.test_path`,
and set `evaluation.enabled: true`. No extra comparative-experiment worker or configuration generator is required.
Keep initial student weights, data splits, seeds, and sampling budgets identical when comparing RL, OPD, and MOPD.
Inspect task rewards, response truncation, gradients, and each teacher's supervision contribution together.
Limited steps and small evaluation sets do not establish convergence or reliable improvement.
