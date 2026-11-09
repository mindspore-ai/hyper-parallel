# vLLM Rollout

**English** | [简体中文](vllm_rollout.zh.md)

## Scope

Only native-vLLM remains. HP constructs training models; the RL plugin does not register a replacement Qwen3 model.
The plugin entry point is `rl.rollout.vllm_plugin:install_runtime_hooks`, which installs weight synchronization, tool evidence, partial service, and required sampling fixes.
Version-sensitive private lifecycle hooks must match supported vLLM/vLLM-Ascend combinations; unknown versions fail explicitly.

## Qwen3-30B-A3B

MoE uses the HP replacement rules declared in this run's YAML and native rollout. Existing constraints such as GRPO, colocation, and disabled EPLB remain; MoE PPO and disjoint deployment are unsupported.
full_gather reconstructs per-expert checkpoint weights from GroupedExperts shards and updates target storage through native layerwise reload.
Receivers retain independent tensors to prevent dangling references after bucket release; acknowledgments must cover every DP×TP worker.

<!-- markdownlint-disable-next-line MD033 -->
<a id="moe-r3-路由重放"></a>

### MoE R3 Router Replay

`train.router_replay.mode: R3` enables expert capture and Actor replay; the default is `disabled`.
Existing eager/raw-logprob/topology restrictions remain; Agent budgets and sleep levels are unchanged.
See the [MoE guide](moe_code_agent.md#moe-r3) for detailed scope.

## Ownership

All training ranks participate in parameter gathering; the coordinator rank manages the shared HTTP control endpoint.
Workers acknowledge versions after actual loading completes; a DP engine return value cannot replace acknowledgment from every physical worker.
Configuration validation distinguishes colocated device mappings from disjoint device sets and verifies endpoint consistency across training ranks.

## Requests and Admission

Requests must carry a committed policy version. After pausing admission, wait for active requests to finish or pause them through the partial protocol before updating parameters.
Token IDs, raw logprobs, worker version, and optional routing traces return together; tool/observation tokens do not become model actions.
The sampler fix restores RNG offsets for discarded requests during partial prefill; it does not recreate the deleted consistency-model path.

## Policy Lifecycle

Rollout wakes only after training releases shard and optimizer residency. Publication failure neither wakes the KV cache nor commits a version.
Admission resumes only after loading, cache handling, and committed-version checks all succeed.
Restart recovery begins with the new process's actual state and republishes Actor; an old checkpoint version does not prove that new workers have loaded it.

## Weight Synchronization

`WeightPublisher` directly implements streaming full_gather, without a strategy-selection layer around a single implementation.
Colocation uses IPC; disjoint deployment uses HCCL. Transfer one bucket at a time and release it after acknowledgment; an oversized parameter occupies its own bucket.
Bucket plans cache only names and layouts, not fully gathered parameters; tied weights and MoE mappings retain their respective native loading semantics.
The old `direct_reshard` configuration and RPCs have been removed.

## Configuration

```yaml
rollout:
  engine: vllm
  vllm:
    model_implementation: native
    weight_sync:
      strategy: full_gather
      bucket_size_mb: 128
```

This illustrates fields only; launch with a complete example configuration declaring the model, data, algorithm, and devices.
Before running, install both the main project and `hyper_parallel/rl`, verify that `rl.__file__` resolves to the current checkout, and inspect plugin metadata for the new entry point.

## Validation Status

CPU unit tests validate transaction ordering, failure propagation, parameter content, single-bucket memory lifecycles, and recovery protocols.
NPU training, actual IPC/HCCL, MoE, vLLM graph capture, and long-term learning must each be accepted separately on explicitly allocated devices.
Historical device results do not automatically establish that device paths pass after this refactor.

## Change Gates

Do not make tests pass by disabling probability recording, relaxing version checks, or skipping worker acknowledgments.
Entry points, configuration, plugins, and receivers must migrate together; see the [architecture](architecture.md) for the complete contract.
