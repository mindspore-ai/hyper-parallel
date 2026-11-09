# Hyper-RL Architecture

**English** | [简体中文](architecture.zh.md)

## System View

`main()` in `train_rl.py` calls `rl.config.loader.load_config()` to load configuration. `trainer/build.py` validates it and assembles resources, selecting the synchronous, disjoint asynchronous, or colocated asynchronous loop from YAML.
Builder objects exist only during initialization; loops directly own the components they need and do not inherit a shared role runtime.
The main project continues to own model loading, parallel plans, HSDP, optimizers, and DCP; RL owns algorithms, trajectories, rewards, and training scheduling.
`trainer/` contains training loops and control flow; `workers/` contains execution roles such as Actor/Critic,
with `engine/` providing shared model, optimizer, residency, and offload mechanisms.

| Responsibility | Code |
| --- | --- |
| Types and basic fields | `rl/config/schema.py` |
| HP configuration conversion, automatic capacity, and support matrix | `rl/config/resolve.py`, `limits.py`, `validate.py` |
| YAML loading and strict overrides | `rl/config/loader.py`; external entry point: `train_rl.py` |
| Checkpointing and evaluation | `rl/trainer/checkpoint.py`, `rl/trainer/evaluation.py` |
| Tool evidence, XML validation, and diagnostics | `rl/tool_protocol/protocol.py`, `xml.py`, `diagnostics.py` |
| Shared extension registry | `rl/utils/registry.py` |
| Sole assembly entry point | `rl/trainer/build.py` |
| Three training loops | `rl/trainer/sync.py`, `fully_async.py`, `colocate_async.py` |
| Progress, failure synchronization, and service cleanup | `rl/trainer/state.py`, `distributed.py`, `lifecycle.py` |
| Actor, Critic, and fixed training targets | `rl/workers/actor.py`, `critic.py`, `targets.py` |
| Models, optimizers, and residency | `rl/workers/engine/` |
| Token batching, DP reordering, and model adapters | `rl/workers/input/`, `adapters/` |
| Trajectories, prompts, episodes, and experience | `rl/data/` |
| Native-vLLM inference | `rl/rollout/` |
| Weight publication and IPC/HCCL transport | `rl/weight_sync/` |
| CPU producer, versioned samples, and partial service | `rl/async_runtime/` |

## Control Plane

Configuration validates algorithms, models, device topology, and paths before starting the distributed environment. Both startup and training failures use `RunLifecycle` to clean up owned resources while preserving the original exception.
`RewardScorer` and `TeacherScorer` perform actual scoring; evaluation callbacks capture these components, not the builder.

<!-- markdownlint-disable-next-line MD033 -->
<a id="master-模型接入"></a>

### Mainline Model Integration

`train.plan_overrides` must be declared explicitly in this run's RL YAML; an empty list means no component replacements.
Examples directly list shared HP replacement rules for RMSNorm, Attention, or MoE, without reading another training recipe at runtime or implicitly supplying family-specific rules.
`train.reference.plan_overrides` and `train.critic.plan_overrides` can replace each role's complete rule list; each is normalized independently without modifying the Actor plan.
The Qwen3 adapter handles only tied-parameter identity, materialization buffers, and loading provenance; it does not register a custom vLLM model.

Actor/Critic share `workers/engine/optimizer.py` for stream synchronization, master-parameter gradient preparation, clipping, stepping, and gradient cleanup.
Optimizer mini-batch membership, valid action-token normalization, and optimizer step counts remain unchanged; schedulers advance at each role update's established boundaries.
Dynamic batching reuses HP `FirstFitPackingSelector`, charging the actual padded rectangle against the budget. An over-budget row forms its own batch without truncation.
DP ranks align microbatch invocation counts without duplicating valid training samples.

Training CPU offload and generation-phase offload are configured separately. `TrainingResidency` reshards before/after publication, validates optimizer state, and synchronizes device streams.
`TrainingStateOffload` preserves parameter and optimizer object identity; transfers require cleared gradients. Buckets and caches are released only after IPC receipt acknowledgment, before waking rollout.
`sequence_parallel`, `loss_parallel`, `reshard_after_backward`, and `requires_grad_sync` map explicitly to HP configuration; the support matrix still rejects unvalidated combinations.

<!-- markdownlint-disable-next-line MD033 -->
<a id="边界合同"></a>

## Boundary Contracts

### Token-First Invariants

Trajectory token IDs are the authoritative training input; decoding and retokenizing them is prohibited.
`sequences`, `attention_mask`, and `action_mask` have shape `[batch, tokens]`; probabilities, advantages, and value targets align to next-token positions.
The loss mask is `action_mask[:, 1:]`. Prompts, padding, tool observations, and post-EOS tokens do not contribute to policy loss.
Episode/call IDs, worker policy versions, behavior probabilities, and complete contexts move together during reordering.

### Agent Boundary

The internal runner drives environments and tools; Codex/DeepSeek harnesses preserve multi-turn programs and per-call trajectories.
A complete episode reward may attach to multiple calls, but discontinuous model calls must not be joined into a fabricated token sequence.

### Agentic Harness

Task prompts, container workspaces, and scoring functions remain in examples. The generic Agent layer does not branch on task names.
Synchronous CPU preparation overlap does not change policy versions or prefetch across evaluation, checkpoint, or epoch boundaries.

<!-- markdownlint-disable-next-line MD033 -->
<a id="一个训练步"></a>

## One Training Step

1. Generate actual trajectories using the committed policy, complete RM/Teacher scoring, put rollout to sleep, and restore training residency.
2. `TargetPreparer` computes the Actor anchor, Reference probabilities, PPO values/bootstrap, and fixed targets according to algorithm requirements.
   With DP balancing enabled, compute targets on complete GRPO groups first, then reorder, then compute the corresponding anchor/reference. Advantages must not be recomputed on incomplete groups.
3. Update Actor and optional Critic; publish Actor weights and verify every worker's version before advancing global_step, recording metrics, evaluating, and saving.

`old_log_prob_source=actor` forces recomputation and freezing of pre-update probabilities; separate rollout probabilities remain available for diagnostics and distillation.
Optional diagnostics do not themselves change the training denominator. The old consistency profile and bit-exact gate have been removed; distillation probability-validity checks remain.

## State Ownership

Actor/Critic each own their model, optimizer, and scheduler. Progress belongs to the training loop; worker publication versions belong to the weight synchronization controller.
`TargetPreparer` holds only target-computation dependencies, `TrainingResidency` owns only storage lifecycles, and `PolicyPublication` controls the training-to-inference transition order.
`CheckpointResources` explicitly lists persistence resources without back-references to the trainer, rollout, or builder.

## Two Deployment Modes

Colocated deployment uses NPU IPC; disjoint deployment uses HCCL. Configuration validation determines allowed model/parallelism combinations.
Only native-vLLM and streaming full_gather are supported. `model_implementation=hyper`, `direct_reshard`, and top-level `consistency` are explicitly rejected.

## Weight Publication and Failure Atomicity

Publication prepares layouts, pauses admission, begins the update, transfers buckets, completes loading, and verifies the committed version, in that order.
Only one full-gather publication implementation remains. At most one bucket is in flight, and temporary storage is released after acknowledgment of synchronous copying.
A single parameter may exceed the bucket size and then occupies its own bucket. Metrics separately record the largest bucket, acknowledgments, and releases.
Failure at any stage prevents committing a new version or resuming sampling; transport strategies never switch implicitly.

<!-- markdownlint-disable-next-line MD033 -->
<a id="checkpoint-与恢复"></a>

## Checkpoints and Recovery

The new format uses `schema=2` and is incompatible with historical checkpoints. Missing or mismatched versions fail before model loading.
HP DCP saves the model and rank-local optimizer, scheduler, dataloader, and RNG; the completion marker is committed last.
Recovery preserves FP32 master-parameter precision, parallel topology, Critic ownership, and Teacher/RM identity checks; RNG is restored after other state.
Asynchronous completion markers bind sample-state checksums and configuration fingerprints. Samples use versioned fields without storing Python import paths for RL classes.
Recovery requires a drained producer, with no in-flight or selected uncommitted groups; corrupt or incompatible snapshots must not be silently accepted.

## Extension Points

Algorithm requirements determine optional roles and targets. Task extensions use existing Agent/environment interfaces.
New models require explicit validation of training mappings, native weight loading, complete trajectories, and the support matrix; there are no placeholder implementations for IsoExec, multiple backends, or deleted strategies.

## Evolution Boundaries (Planned)

IsoExec is outside this implementation. Runnable code and configuration do not establish device acceptance for new combinations or learning quality.

## Reading the Code

Start at `trainer/build.py`, enter the appropriate loop, then read `workers/targets.py`, Actor/Critic updates, `weight_sync/publication.py`, and checkpointing.
See [feature navigation](../../../docs/rl-navigation.md) for configuration, source, example, and test indexes.

## Rayless asynchronous GRPO

The disjoint asynchronous loop runs the optimizer and publication collectives on the training thread; a CPU producer owns complete candidate groups.
Updates complete before consumption credits are committed; policy publication then follows freshness requirements or the publication interval.
`policy_version = source_update + 1`; global_step cannot directly substitute for the publication version. Buffer capacity includes selected groups; whole-group admission, lag, and retry limits remain unchanged.
Existing GRPO, model, and topology support boundaries remain; this does not add asynchronous PPO.

<!-- markdownlint-disable-next-line MD033 -->
<a id="共卡异步-grpo"></a>

## Colocated Asynchronous GRPO

The colocated loop explicitly performs pause/drain, inference sleep, training residency, updates, publication, and partial resume.
Failures must not incorrectly commit leases or versions. Before Codex evaluation, the producer releases its gateway; afterward, the evaluation gateway closes and the producer restarts from the recovery snapshot.

<!-- markdownlint-disable-next-line MD033 -->
<a id="共卡-off-policy-校正"></a>

### Colocated Off-Policy Correction

Correction remains disabled by default. When enabled, it follows the existing algorithm support matrix and per-behavior-segment probability contract.
Rejected groups do not reach the optimizer, and training batches are released before rollout resumes.

### Colocated Publication and Recovery

Before saving, drain complete groups and freeze the source. Commit the model checkpoint and asynchronous snapshot at the same successful-update boundary.
After recovery, republish the actual Actor before releasing the producer; do not reuse inference-version state from the old process.
