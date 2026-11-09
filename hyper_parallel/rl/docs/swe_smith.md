# SWE-Smith Code Agent: Implementation and Limitations

**English** | [简体中文](swe_smith.zh.md)

See the [example README](../examples/swe_smith/README.md) for usage. This document maintains implementation boundaries, grading contracts, and acceptance requirements.

## Code Organization

Tasks integrate with the shared `SyncTrainer` through `program_callable`; task data, repository orchestration, and business scoring remain in examples.

| Location | Responsibility |
| --- | --- |
| `examples/swe_smith/agent.py` | Repository preparation, Codex lifecycle, call budgets, frozen submissions, grading, and per-call trajectories |
| `workspace.py` | Pin issue branches and restore trusted tests |
| `rl/agentic/codex/container_execution.py` | In-container session relay, CLI execution, and output recording shared with Search-R1 |
| `artifacts.py`, `reward.py` | Source patch projection, protected tests/configuration, independent pytest grading, and failure attribution |
| `prepare_data.py`, `prepare_images.py` | Subset/full-data conversion, task image mapping, architecture and identity checks |
| `infer.py`, `configs/` | Inference without training, model recipes, and fixed nine-case image/task identities |
| `launch.py`, `run_config.py` | Local/SSH sandbox selection, paths and data/image identity checks, trusted controller lifecycle |
| `rl/agentic/codex/`, `rl/agentic/envs/docker_workspace.py` | Generic Codex protocol, original sampling evidence, local/remote workspaces |
| `rl/workers/adapters/log_probs.py` | RL-side action-position probabilities, chunked vocabulary projection, and recomputation |
| `rl/workers/`, `rl/weight_sync/`, `rl/trainer/checkpoint.py` | Training construction, streaming publication, graph refresh, and complete recovery |

## Task and Grading Contract

Candidates check out the instance's problem branch, validate the “Bug Patch → Remove F2P Tests” structure, and resolve pinned commits.
Git history is then removed, while the controller retains the captured trusted problem baseline.
Containers have no network, Docker socket, or NPU. Management credentials never enter candidates, and the relay is bound only to its session.
Candidate, relay, and grader use the same explicit Docker endpoint; remote connection failure never switches to local execution.

Explanatory text and function calls in one Responses reply are combined into one assistant Chat message.
Source validation permits decoded text and engine text to differ only by one exact trailing EOS; original sampled tokens remain unchanged.
Unclosed XML is a nonexecutable model-format error; other source inconsistencies still reject the session.
Once delivery of a budget-termination response is acknowledged, the relay stops accepting new requests and the controller shuts it down. CLI retries cannot reopen the session.

After normal Codex completion or gateway-confirmed budget exhaustion, stop the candidate and export its actual code.
`artifacts.py` generates the patch on the controller, allowing ordinary Python/Python-stub source changes
while protecting existing tests, project configuration, special files, and execute permissions; candidate Git metadata is untrusted.
New non-Python reproduction inputs, backups, and other intermediate files are excluded. New Python tests/hooks are still checked against protected paths.
The grading version is `swe-smith-pytest-v3`; frozen artifacts can be regraded independently without regenerating model trajectories. <!-- codespell:ignore regraded -->
Task configuration controls artifact byte/file limits.

The grader starts in a fresh container, verifies the same problem/test commits, applies the frozen patch, and restores trusted tests.
Default `f2p_only=true` selects all F2P and P2P in the same test files; false checks the complete P2P set.
Reward 1 requires trusted statuses for all selected tests, all PASSED/XFAIL. Complete grading failure yields 0.
Independently confirmed submission or syntax errors may yield zero reward. If candidate source prevents collection,
pytest hooks must attribute the exception to modified source, and the original problem baseline must collect every selected test, before assigning zero reward.
Retain actual exceptions and reported statuses; do not fabricate FAILED for unexecuted nodes.
Sampling, container, transfer, unattributable collection, and other incomplete grading failures reject the update.

## Training Configuration and Resources

Supported models are Qwen3.5-9B / Qwen3.8-27B for text GRPO with training TP=CP=1,
colocated native vLLM and streamed `full_gather`, without the removed consistency field. No BF16 bit-exact claim is made.
See the [example README](../examples/swe_smith/README.md#one-training-step-and-new-process-recovery) for training and new-process resume commands.
Recovery requires identical model, recipe, and data/image manifest identities. Publish the restored version before resuming sampling and updates.
Full training checkpoint recovery is supported; ongoing Codex sessions and incomplete optimizer steps are not recoverable.

Short-budget recipes use one prompt per rank, two sessions per prompt, at most two calls per session, and 512 output tokens per call.
`train.response_mini_batch_size` counts per-call training rows. Larger call budgets increase recomputation context,
so check training batch size, device memory, and runtime together. Identical real rewards may produce zero gradients; learning acceptance requires real mixed rewards.
Enable `train.learning_gate` to check nonzero updates. Default scoring does not inject synthetic rewards.

The unified training entry defaults to `HCCL_IF_BASE_PORT=63200` and `HCCL_NPU_SOCKET_PORT_RANGE=63200-63299`;
rollout uses `63400-63499`. Override both training port variables together; the range must lie within `1024..65520`,
include the base port, and avoid overlap with rollout. The short 9B recipe explicitly reserves an 8 GiB KV cache to leave room for repeated publication.

### Sampling Groups and Training DP

The shared Trainer assigns prompts by training DP rank; `rollout.num_return_sequences` is the session count per prompt.
Within-group GRPO advantages are computed on that rank. Distributing sessions from one prompt group across training DP ranks is currently unsupported.
Inference TP is independent of training sampling groups: vLLM DP1 / TP2 is one two-device inference replica and does not reassign sampling groups automatically.

Two-device FSDP2 needs two assignable prompts. After creating duplicate-question records with `prepare_data --repeats 2`,
four sessions per rank form two independent four-sample groups, eight sessions total; eight per prompt yield sixteen globally.
Redistribution across training DP ranks belongs in shared RL, preserving unique episode identities, actual rewards,
policy versions, and zero-loss padding.

## Reference-Budget Inference

[qwen3_5_9b_infer.yaml](../examples/swe_smith/configs/qwen3_5_9b_infer.yaml)
provides a longer inference budget: 100 calls, at most 12,288 output tokens per call, 81,920 context tokens,
5,400 seconds, temperature 1.0, and thinking disabled. The tool protocol is Codex/Qwen3Coder.
`max_observation_chars=6000` retains the head and tail of observations delivered to the model and truncates by characters; raw tool evidence is unchanged.
The automatic compaction threshold is 65,536; compaction requests also consume call budget. CLI output limits remain effective.
This recipe does not claim identical prompts, observation wrappers, compaction strategies, or random trajectories to other agents.
The recipe controls independent grader resources and timeouts; external termination is not a solving failure.

## Graph Execution Configuration

Pinned vLLM 0.23.0 / vLLM-Ascend 0.23.0.post1 with rollout DP1 supports explicit FULL_DECODE_ONLY:
Set `enforce_eager=false`, add the following under `rollout.vllm`, and retain `max_num_seqs=4`.

```yaml
compilation_config:
  mode: 3
  cudagraph_mode: FULL_DECODE_ONLY
  cudagraph_capture_sizes: [1, 2, 4]
  max_cudagraph_capture_size: 4
```

This mode includes `torch.compile` and graph replay. After complete streamed full-gather publication and KV wake,
recapture decode graphs before admitting sampling; a refresh failure must not commit the version.

## Validation Scope and Regression

9B short-budget eager validation covers ARM64 Ascend controllers with local ARM64 or remote amd64 sandboxes,
including inference, training, complete saving, and new-process recovery. Task grading covers nine schema, boltons, and oauthlib cases,
checking the independent grader against failing problem baselines and successful trusted fixes. Zero-reward, zero-gradient loops prove execution only,
not effective learning.

9B graph inference has run. Repeated graph-mode publication/refresh, long-budget updates/recovery with real mixed rewards, complete 27B regression,
and full-dataset acceptance still require separate execution. Additional repositories require architecture, branch layout, test dependency, and resource checks.
Current patch scope includes only ordinary Python/Python stubs; other languages and tasks requiring build-configuration changes are outside acceptance scope.

Run CPU contract checks from the repository root:

```bash
python -m pytest -q tests/ut/rl/agentic/swe_smith \
  tests/ut/rl/agentic/codex tests/ut/rl/agentic/test_docker_workspace.py
```

Model mathematical contracts live in `tests/ut/auto_models/models/qwen3_5/`; training, publication, and recovery tests reside in corresponding `tests/ut/rl/` modules.
Shared code changes must also cover Search-R1, the text gateway, Hermes evidence, and complete checkpoint regression.
CPU success does not establish acceptance for new hardware combinations, long-budget learning, or the full dataset. Before real execution, check devices, image storage,
separate output/checkpoint directories, and budgets.
