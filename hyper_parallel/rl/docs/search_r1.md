# HotpotQA File Search

**English** | [简体中文](search_r1.zh.md)

See [prepare_data.py](../examples/search_r1/prepare_data.py), [training config](../examples/search_r1/configs/qwen3_4b_search_r1.yaml), and [run_search_r1.sh](../examples/search_r1/scripts/run_search_r1.sh).

Run `launcher.py` on the host to start the trusted training container and a sibling, unprivileged Codex container.
Docker-in-Docker is not used. CLI instruction files use the candidate container's `/codex-home` path and must not reference the training container's `/results` path.
Codex can read only the current question's candidate articles and search them through the shell.
Dataset labels, other sessions, NPUs, and the Docker socket are not mounted. Each session probes isolation inside its actual candidate container before starting Codex.
Training retains the `hyper-rl:v0.23.0.post1-code-agent-arm64` image; candidates default to the locally built
`hyper-rl-codex:0.152.1-arm64`. Use `--candidate-image` to explicitly select another verified CLI image.
Candidate images contain only Codex, Python, and search tools, without model-inference libraries.
Permissions and mounts remain independent; see the [image guide](../docker/README.md#independent-codex-candidate-containers-for-search-r1).

## Data and Training

The current dataset is the official `hotpotqa/hotpot_qa` **Distractor** split, retaining each question's original candidate articles and sentence-level citations.
Long-context data constructed for MemAgent is not used. Training questions are sampled from the 90,447 official training questions; held-out questions come from the independent validation source.
Preparation checks cross-source IDs and normalized-question overlap, records SHA256 hashes, and refuses to overwrite existing data.

The current eight-device, two-step recipe reuses Qwen3-4B training/inference capabilities accepted for GSM8K; the search combination still requires independent NPU acceptance:

| Item | Configuration |
| --- | --- |
| Data | Prepare 6,400 training questions and 128 held-out questions; two steps consume only the first 256 training questions |
| Per-step budget | 128 questions × 8 episodes, at most 10 LLM calls per episode including citation correction |
| Inference | Native vLLM TP2×DP4, 0.8 memory utilization, 256 sequences per replica, graph execution/prefix caching/chunked prefill enabled |
| Scheduling | Gateway limit 1,024; at most 128 episodes per rank; the host defaults to 128 concurrent candidate containers |
| Container resources | At most 2 CPUs and 2 GiB memory per candidate; use `--agent-workers` to change the concurrent-container limit |
| Call length | At most 1,024 output tokens per call and 8,192 tokens of complete model context |
| Training | DP shard8, TP1, gradient synchronization per microbatch, training-state offload during generation; Actor CPU offload is disabled during training itself |
| Probability computation | Actor recomputes old probabilities while sampling probabilities are retained; Actor/old/reference dynamic budgets are all 8,192, with selected_chunked chunks of 512 |
| Optimizer | At most 1,280 call rows per rank mini-batch, covering 128 episodes × 10 calls, with one update per step |
| Reward | Existing rule: 0.70×answer F1 + 0.20×answer EM + 0.08×supporting-sentence F1 + 0.02×format reward; citation checks are unchanged |
| Evaluation/saving | Disabled by default for the two-step flow; short experiments are not 50-step or validation-set acceptance |

`max_prompt_length=1024` constrains the initial question, not the combined CLI instructions, tools, and history.
Dynamic budgets govern ordinary microbatch grouping; an over-budget complete call forms its own batch. Single-sample memory capacity still requires acceptance at the context limit.
This recipe retains the 8,192 budget; chunked probability computation reuses the existing RL implementation without adding a training core.
CPU preparation overlap remains disabled: current search uses custom programs and container execution outside the supported native local CLI overlap path.
Candidate-container concurrency and vLLM sequence limits are separate resource constraints; 128 containers do not guarantee 256 inference sequences per replica.

Prepare data from the repository root after downloading the raw files:

```bash
python3 hyper_parallel/rl/examples/search_r1/prepare_data.py \
  --input /absolute/raw/train-00000-of-00002.parquet \
  --additional-input /absolute/raw/train-00001-of-00002.parquet \
  --validation-input /absolute/raw/validation-00000-of-00001.parquet \
  --train-samples 6400 --test-samples 128 --seed 42 \
  --output-dir /absolute/path/to/new-prepared-hotpotqa
```

Add `--preserve-test /absolute/old/test.parquet` to retain an existing held-out set. Both the training source and independent validation source are required:
the historical preparation tool downloaded validation by default, and splitting one source does not reproduce the official train/validation split.

From `hyper_parallel/rl/`, verify that eight devices are idle, then run:

```bash
python3 examples/search_r1/launcher.py --devices 0,1,2,3,4,5,6,7 --steps 2 \
  --model /absolute/path/to/Qwen3-4B --data /absolute/path/to/prepared-hotpotqa \
  --backend-privileged --agent-workers 128 --output /absolute/path/to/new-results
```

The launcher and `scripts/run_search_r1.sh DEVICE_IDS STEPS DATA_DIRECTORY --model MODEL_DIRECTORY` accept trailing YAML overrides.
Startup reuses RL plugin installation and the accepted sampler stream fix and DP coordinator cold-start timeout patch; mismatched dependencies fail.
Existing experiment directories are not overwritten. `container.json` records the actual training container ID so host monitoring can identify ownership of device processes.
`--backend-privileged` applies only to the trusted training backend, never to Codex candidate containers.
Training HCCL uses ports `62800-62900`; vLLM uses `62600-62799`. The experiment environment still owns device checks and utilization monitoring.

Default acceptance checks the target policy version, nonzero gradients, optimizer updates, per-session isolation, and actual contexts.
The legacy `--strict-acceptance` flag still checks retired exact-probability metrics alongside evaluation and checkpoint reload.
It is not a supported acceptance path for current native-vLLM; the removed Hyper inference implementation and consistency profile cannot be re-enabled.
`--resume /original/output/checkpoints/step_N` requires a complete checkpoint marker and identical data hashes.

## Prompts and Context

The original question comes from the dataset's `question`, and articles from `context`; reference answers and supporting-facts labels are not exposed to candidate processes.
The task template specifies evidence sources, absolute article paths, tool examples, and the `ANSWER/SOURCES` format without forcing a tool call in the first response
or prescribing a step-by-step search strategy. System instructions require answers grounded in actually read evidence. Reward formulas and citation checks remain unchanged.
With thinking enabled, use vLLM's existing `reasoning_parser: qwen3`; the existing Codex protocol carries reasoning and final answers separately,
keeping `<think>` content out of the two-line answer check. Original generated tokens remain intact; reward-format constraints are not relaxed to solve protocol issues.

Search-R1 sets `compact_harness_context: true` and `preserve_environment_context: true`:
it removes irrelevant skill-directory instructions while retaining Codex's original environment and permission messages, including the `/work` directory and shell.
Candidate articles are mounted at `/work/articles`; command examples use absolute paths to avoid appending articles again after changing workdir.
GSM8K's compact recipe retains its default filtering policy without sharing file-search environment-preservation requirements.

The tool allowlist retains only `exec_command` and `write_stdin`, preserving argument descriptions and historical calls/observations.
The last ordinary call and citation correction explicitly disable tools; the Gateway permits these requests to omit tool declarations.
Tool outputs pass unchanged to Codex and the model. The integration layer does not parse shell commands, exit codes, or error text,
add retry guidance, or terminate early for no matches or repeated commands. It records only observed tool output for citation checks.

## Correctness and Failure Boundaries

- Train each actual `P_i + A_i` separately; `training-audit.json` checks tokens, masks, and sampling logprobs.
  Adjacent calls need not have continuous prefixes; re-encoded history must not replace training inputs.
- The current native recipe records training/inference probability differences without claiming exact per-token equality; the former strict consistency profile has been removed.
- Training rewards and evaluation are aggregated by complete episode, without repeated weighting by model call count.
- Disable tools on the last ordinary model call and reserve one actual citation-correction call; retain all actions.
- Model output-token exhaustion, Gateway call-budget exhaustion, and explicit context-limit rejection reuse the generic Codex terminal-state checks.
  After validation, original trajectories are retained with zero reward and no further citation-correction call.
  Abnormal container exits, missing final messages without complete budget evidence, and unknown communication failures still reject updates.
- Confirmed model-budget exhaustion and invalid citations are task failures; actual actions are retained with zero reward.
  If no sampled actions exist, raise an error rather than fabricate an empty training sample.
- `failure_origin`, `failure_reason`, and `trainable` record responsibility, cause, and training eligibility respectively.
  Actual execution-service/container failures, HTTP/network errors, unattributed context overflows, and unknown failures abort the entire update.
  Do not substitute zero rewards, discard group members, or merely mask loss, which would corrupt GRPO advantages.
- Nonzero shell-command exits, path/argument errors, and no matches are tool observations; the model chooses subsequent actions within the overall budget.
  Exit codes in tool output are distinct from Codex process/container exit codes; the latter still require validation against actual execution results.
- A pinned vLLM plugin records Hermes evidence: actual generated tokens, decoded text, engine text, and parser input/output.
  Invalid tool JSON with consistent evidence receives explicit format feedback and resampling, consuming the total call budget. Do not execute or repair the original command.
  Successful recovery is still scored by final task outcome. Parser failure on valid source text or inconsistent evidence aborts the update.
- Format-recovery calls consume the original budget and may use the reserved citation-correction call; the budget is not expanded.
- `training-audit.json` retains original prefix-mismatch counts and up to three token-difference samples; training still uses actual per-call tokens.
- `training-acceptance.json` is produced only after the specified optimizer steps, weight synchronization, and all session audits pass.
  Citation existence/visibility does not establish semantic correctness; short-run success does not guarantee arbitrary long training is failure-free.

## File Responsibilities

- `workspace.py`: Export only the current question's candidate articles.
- `agent.py`: Task outcomes, citation checks/correction, and per-call training trajectories.
- `container_runtime.py`: host handoff, article isolation probes, and search-budget feedback; calls shared workspace and Codex execution components using only the standard library.
- `rl/agentic/envs/docker_workspace.py` and `rl/agentic/codex/container_execution.py`: shared container lifecycle, session forwarding, and CLI execution.
- `launcher.py`: Host training entry point, data verification, and training acceptance.
- Generic token/mask/logprob auditing lives in `rl/agentic/core/program_runner.py`.

Isolation checks still run before every production session; the standalone smoke entry point lives under tests.

Run the model-free isolation check from the host's `rl/` directory, without NPUs:

```bash
python3 tests/st/search_r1_isolation.py \
  --output output/search_r1/search_r1_isolation_check
```

The output directory must be new. This verifies container boundaries and Codex startup, not real model training.
`candidate-runtime.json` records the actual candidate image identity; each session's `execution-timing.json` records execution-stage durations.
Historical experiment details remain in logs under their respective output directories.
