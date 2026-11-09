# MoE R3, Code, and Agent Usage and Validation

**English** | [简体中文](moe_code_agent.zh.md)

This guide collects support boundaries, configuration, and validation entry points for MoE router replay, single-turn code tasks, and external agents.
See the [Docker runtime guide](../docker/README.md) for the environment and each task example for data preparation and launch instructions.

<!-- markdownlint-disable-next-line MD033 -->
<a id="功能与支持边界"></a>

## Features and Support Boundaries

| Feature | Supported scope | Entry point |
| --- | --- | --- |
| Qwen3-30B-A3B | GRPO, colocated native vLLM, training TP1/TP2 with EP/EDP; EPLB disabled; the removed consistency field must be omitted | [MoE rollout](vllm_rollout.md#qwen3-30b-a3b) |
| MoE R3 | Qwen3-MoE Actor router replay; internal single-turn/continuous multi-turn and Codex per-call trajectories | [R3 configuration](#enable-r3), [GSM8K example](../examples/gsm8k/README.md) |
| Single-turn Python code | Private stdin/stdout tests, single TP-owner grading, independent SandboxFusion service | [Code example](../examples/code/README.md) |
| External agents | Actual prompt/action retained per call, episode GRPO, zero-loss DP padding | [Agent contract](agentic_rl.md), [Search-R1](search_r1.md) |
| Repository Code Agent | Isolated candidate workspaces, frozen artifacts, independent graders; local/remote execution | [SWE-Smith example](../examples/swe_smith/README.md) |

MoE does not support PPO, disjoint rollout, or Hyper-vLLM. R3 is not yet adapted to the DeepSeek runner
and cannot be combined with enabled distillation/OPD. Model-format errors and confirmed task budget exhaustion may count as task failures;
communication, execution-service, and unknown errors must propagate, never silently become zero scores or discarded candidates.

## MoE R3

Rollout Router Replay (R3) records sampled expert choices and replays them during Actor probability computation, training forward,
and activation-checkpoint backward recomputation. Discrete expert IDs are replayed; weights are still computed from current router probabilities,
with router/expert gradients participating normally in updates. Reference retains native routing, and `old_log_probs` retains original rollout values.

R3 data transport and micro-batch scopes are generic RL capabilities, independent of datasets, reward rules, and agent call budgets.
Only the Qwen3-MoE gate interface is currently adapted; arbitrary MoE models cannot enable it directly.
Adapters live within RL, requiring no changes to shared models, FSDP/HSDP, TP/EP, or optimizer code.

### Enable R3

Default configuration preserves existing behavior:

```yaml
train:
  router_replay:
    mode: disabled  # R3 enables rollout capture and Actor replay together.
  model_init_cpu_threads: null
rollout:
  vllm:
    sleep_level: 1
```

Start with the [four-device Qwen3-30B-A3B Codex configuration](../examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml).
In a training container with the current checkout installed and required weights/data mounted, run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml \
  --train.router_replay.mode=R3
```

The recipe presets training TP2/EP2/EDP2 and rollout TP4/DP1; R3 remains off by default.
Enabling R3 does not adjust candidate counts, tool calls, context, generation budgets, evaluation, or saving.
Do not configure `enable_return_routed_experts` separately; if set manually, it must agree with the R3 switch.

The pinned environment is Transformers 5.5.4 and vLLM 0.23.0 / vLLM-Ascend 0.23.0.post1.
The adapter distinguishes probability returns before Transformers 5.6 from logits returns starting at 5.6, preserving the native gate output contract.
Changing Transformers still requires adapter and full-training validation; matching class names are insufficient. The pinned real-device environment remains unchanged.
Current requirements:

- `model_implementation: native`, `deployment: colocated`, `enforce_eager: true`, `logprobs_mode: raw_logprobs`.
- Disable EPLB and omit the removed consistency field; training CP=PP=1.
- Rollout `data_parallel_size: 1`, with TP spanning the shared devices. This limitation comes from the Ascend capture DP-padding layout,
  not training DP. When enabling R3 from the single-turn MoE recipe, also change rollout DP2/TP2 to DP1/TP4.

Initialization threads and sleep level are independent: `--train.model_init_cpu_threads=8` temporarily changes only model-construction
intra-op threads. `--rollout.vllm.sleep_level=2` discards old inference weights to reduce CPU memory,
requiring complete weight publication before sampling resumes. Defaults remain `null` and `1`; R3 does not change them automatically.
See [initialization threads](../README.md#cpu-threads-for-model-initialization) and
[inference lifecycle](vllm_rollout.md#policy-lifecycle) for all conditions. Device-side gathering for `full_gather` is built in, without a new switch.

### Data and Execution Flow

```text
vLLM capture → GenerationResult → Trajectory → ExperienceBatch
             → Actor micro-batch RouterReplay.use → gate hook → expert / EP
             → clear replay state after backward completes
```

`RouterTrace` stores CPU integer `expert_ids[S,L,K]`, boolean `valid_mask[S]`, and stable `layer_ids`.
Routing aligns with model input token positions, without the next-token logprob shift.
For example, `[p0,p1,a0,a1]` usually has forward execution only for its first three inputs; the last sampled `a1` has no routing.
If the last valid action is at `m`, the causal prefix `0..m-1` must be valid. Capturing only action-mask positions is insufficient.

Unexecuted final tokens, terminal observations, and padding remain invalid. Invalid positions use native routing, not fabricated expert 0.
Traces remain on CPU with training rows and are padded/transferred only for micro-batch execution. Transport is compact binary; NPY decoding disables pickle.
Invalid shapes, layer maps, expert ranges, duplicate experts, or causal coverage fail explicitly without falling back to ordinary training.

`router_adapter.py` installs removable hooks only on gate instances in the constructed Actor,
gathers current probabilities using rollout IDs, and normalizes according to original model rules. Parameter names, state_dict, and optimizer references are unchanged.
The micro-batch scope covers forward/backward including checkpoint recomputation, and clears state on normal or exceptional exit.

### Multi-Turn Interaction

| Trajectory style | Trace handling | Reason |
| --- | --- | --- |
| Internal continuous context | Replace the old snapshot with each new complete trace; ordered expert IDs must match on overlapping valid prefixes | New prompts already contain historical actions/observations; concatenating traces would duplicate prefixes |
| Codex per-call training | Retain an independent trace for each actual prompt/action without deleting earlier calls | Harnesses may rewrite or trim context; the final call cannot replace historical calls |

Codex Gateway extracts routing from model responses; agents receive ordinary responses only. Routing is not duplicated in audit metadata.
Call identity, policy versions, episode rewards, and zero-loss DP padding continue following the [Agent contract](agentic_rl.md).
Additional calls such as Search-R1 citation corrections use the same path, without dataset-specific replay logic.

For new tasks, check whether original tokens, call records, or trajectory concatenation change. Changing only questions/rewards does not require copying R3;
new context truncation, call merging, or transport boundaries require corresponding validation.

### Implementation Locations and References

Paths below are relative to `hyper_parallel/rl/`:

| Responsibility | Files |
| --- | --- |
| Data contracts and training-row binding | `rl/data/contracts.py`, `rl/data/experience.py` |
| Capture decoding and TP transport | `rl/rollout/router_capture.py`, `rl/rollout/vllm.py` |
| Model-independent replay scope | `rl/workers/adapters/router_replay.py` |
| Qwen3-MoE gate adaptation and Actor integration | `rl/workers/adapters/router_adapter.py`, `rl/workers/actor.py` |
| Multi-turn call records and trajectories | `rl/agentic/codex/gateway.py`, `harness.py`, `rl/agentic/core/` |
| Configuration and assembly | `rl/config/validate.py`, `rl/trainer/sync.py` |

Implementation references the [R3 paper](https://arxiv.org/html/2510.11370v1) and these pinned revisions:

- [verl](https://github.com/verl-project/verl/tree/f95be0226a65d3f335901ee6703f974fe82f840d):
  Replay indices, weights computed from current probabilities, and forward/backward scope; router adaptation stays in RL.
- [Agent Lightning](https://github.com/microsoft/agent-lightning/tree/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9):
  Proxy routing extraction and per-call training records; overwritten old traces may be deleted only when aggregating continuous trajectories.
- [Uni-Agent](https://github.com/verl-project/uni-agent/tree/13f14f361912bf0c4cd53701bfe1e9d09510a3df):
  Continuous token chains use the latest complete routing. This optimization cannot be applied directly to this repository's independent Codex call rows.

## Unit Tests

UT resides in `tests/ut/rl/`. Core entry points below cover CPU logic and necessary mocks:

| Module | Representative tests (relative to UT root) |
| --- | --- |
| MoE configuration and synchronization | `trainer/test_moe_config.py`, `weight_sync/test_moe_weight_sync.py` |
| R3 positions, gradients, recomputation, capture | `policy/test_router_replay.py`, `rollout/test_router_capture.py` |
| Device gathering, bucket release, sleep 2 | `weight_sync/test_packed_weight.py`, `weight_sync/test_weight_sync_transaction.py` |
| Initialization CPU-thread restoration | `policy/test_actor_roles.py`, `trainer/test_config_runtime.py` |
| Code data, grading, trajectories | `data/test_code_data.py`, `agentic/test_code_judge.py`, `agentic/test_code_runner.py` |
| Agent calls, budgets, padding | `data/test_episodes.py`, `agentic/test_agent_program.py`, `agentic/codex/test_gateway_transport.py` |
| Repository tasks and launch flow | `agentic/swe_smith/` |

```bash
python -m pytest -q tests/ut/rl
```

## System Tests

ST resides in `hyper_parallel/rl/tests/st/`, explicitly launched by `test_feature_st.py`.
`_moe_train.py`, `_code_train.py`, `_agent_train.py`, and `_agent_dp.py` are subprocess workers;
existing dense and PPO cases in `test_rl_st.py` remain independent.

Two-process CPU/Gloo validation compares gradients and updated parameters before/after unequal-call padding, without NPUs:

```bash
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_agent_dp_padding
```

Real MoE/R3 training uses existing MoE recipes. Prepare models, data, and devices, enabling R3 as needed:

```bash
export RL_ST_MOE_CONFIG=/path/to/resolved-moe.yaml
export RL_ST_WORLD_SIZE=4
export RL_ST_RESULT_DIR=/path/to/new-results
export RL_ST_REQUIRED=1
python -m pytest -q 'hyper_parallel/rl/tests/st/test_feature_st.py::test_feature_training[moe]'
```

Configuration must match device count, and the results directory must be empty. Code and agent cases use `RL_ST_CODE_CONFIG`,
`RL_ST_AGENT_CONFIG`, and parameters `[code]`, `[agent]`. Tests retain two-step parameter-update and policy-version checks.
The R3 observer additionally verifies actual gate expert IDs, current weights, checkpoint recomputation, and router gradients.
Agent cases require real multiple calls and unequal-row padding; untriggered scenarios cannot be reported as passes.

Independent real SandboxFusion service tests require neither models nor NPUs; see the [code example](../examples/code/README.md) for entry points and resources.
Skipping for missing resources is not a pass. With `RL_ST_REQUIRED=1`, missing configuration fails.
Write all results to independent output directories excluded from version control.

## Validation Scope and Limitations

| Path | Existing functional coverage | Limits on conclusions |
| --- | --- | --- |
| Single-turn MoE R3 | Routing, gate weights, backward recomputation, two-step updates, and version publication | Adapted model and pinned runtime only |
| GSM8K Codex R3 | Tool context, per-call traces, budget termination, two-step loop | Does not establish convergence |
| Search-R1 Codex R3 | Search context, unequal-call DP padding, first-step replay/update publication, citation-correction capture | Complete two-step loop not finished; later capture does not establish successful training replay |
| Internal multi-turn / SWE-Smith R3 | CPU contracts for continuous-prefix checks / per-call routing handoff | Corresponding R3 multi-turn real-device loops not independently completed |
| Independent code sandbox | Isolation contracts plus dense-code two-step training, evaluation, saving | Does not establish all data labels are correct or checkpoint recovery passed |

Exact replay of expert choices does not imply training/inference logprob consistency. MoE still has pre-update probability differences at identical weights;
no bit-exact or learning-gain claim is made. Generic training/agent success does not prove every R3 combination has been validated.
After changing models, capture backends, parallel layouts, or trajectory structures, rerun affected contracts and real execution checks.
