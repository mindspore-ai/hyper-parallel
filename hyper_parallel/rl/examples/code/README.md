# Single-Turn Python Code Training

**English** | [简体中文](README.zh.md)

This example uses the internal runner to generate one Python stdin/stdout program, grades it in an independent SandboxFusion service,
and passes the reward to GRPO. See [qwen3_4b_code_vllm.yaml](configs/qwen3_4b_code_vllm.yaml).

## Integration Contract

```yaml
data:
  row_adapter: examples.code.prepare_data:adapt_row
agentic:
  module_path: examples.code.agent
  environment: code_stdio
  interaction_mode: single_turn
  max_turns: 1
  code:
    endpoint: http://127.0.0.1:18790
    runtime_version: sha256:5f5c291ba3d15d9ec20bc3b5695f87f107ee26d39c86817d92283fbd79a0f856
    run_timeout: 5
    request_timeout: 300
```

`row_adapter` preserves original multi-message inputs, stable `task_id`, source metadata, and structured `ground_truth`.
It is mutually exclusive with `prompt_column`, `answer_column`, and `prompt_instruction`. Private tests belong only in
`ground_truth.inputs/outputs`, never the prompt. Oversized messages raise an explicit error instead of silently truncating the problem.

The judge extracts the last Python code block or accepts an unfenced program. Extraction is only for sandbox execution; training retains original
action tokens and rollout logprobs. After every declared test completes, output is compared exactly after whitespace tokenization. All tests passing earns 1,
otherwise 0. The first failure is recorded as `wrong_answer`, `runtime_error`, `timeout`, or `output_limit`.
Empty programs, unsupported code fences, or code over 64 KiB yield `format_error`. Interactive tasks, special checkers, and floating-point tolerances are unsupported.

HTTP, decoding, protocol, and `SandboxError` failures propagate; they are not disguised as zero scores or implicitly retried.
Only the request owner in each Trainer TP group executes the environment; other ranks replay results. `finish_reason=length` records truncation.

## Data Preparation and Review

Run these commands from the repository root in the unified training container. Source is mounted at `/workspace/hyper-parallel`,
weights at `/models`, data at `/data`, and output at `/results`.
Preserve CANN initialization and plugin registration from the [runtime guide](../../docker/README.md), and verify the editable main package points to the current checkout.
Example modules use the existing launcher's source paths:

```bash
export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
python -m examples.code.prepare_data \
  --source-dir /data/eurus2-code-stdio/9776b13264b5/raw \
  --output-dir /results/eurus2-code-candidates \
  --tokenizer /models/Qwen3-4B \
  --revision 9776b13264b5aaa0b16495fcf086a0a8d86fd655 \
  --max-prompt-tokens 2048 --max-test-cases 64 \
  --max-train 128 --max-validation 32
```

Preparation reads only local raw Parquet, preserves test content/order, filters unsupported task types, over-budget tests, and duplicate questions,
and produces `manifest.json` with source/result hashes, filter counts, and stable task IDs.
`--manual-exclusions exclusions.json` accepts explicit exclusions such as `{"train:455277": "review reason"}`.

Successful preparation does not establish correct labels. Before training, run every original test for each question with an independent reference solution. Preserve mismatch evidence,
and exclude the entire question with `--manual-exclusions`; do not modify labels to obtain passing results.
Training configuration must point to reviewed prepared files; the candidate directory is not automatically trusted data.

## SandboxFusion Deployment

Use `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64`.
Pull and verify the image following [image download and verification](../../docker/README.md#sandboxfusion-image-for-single-turn-code).
It uses an independent Python slim environment, without inheriting the training base layer.
The training image is `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64`; run them separately.
Do not mount training source, weights, data, or the Docker socket into the sandbox.

This launch method targets Docker 18 / cgroup v1. Docker 18 lacks `--cgroupns`,
so `unshare` creates private cgroup/mount namespaces inside the container before mounting controllers for the original lite isolation.
Retain the image's `lite` isolation; bundled `tini` reaps exited children left after candidate timeout or completion.

```bash
docker network inspect hyper-rl-sandbox >/dev/null 2>&1 || docker network create hyper-rl-sandbox
docker run -d --name hyper-rl-code-sandbox \
  --runtime runc --network hyper-rl-sandbox \
  --cap-add SYS_ADMIN --cap-add NET_ADMIN \
  --security-opt no-new-privileges \
  --memory 2g --cpus 4 --pids-limit 128 \
  -p 127.0.0.1:18790:8080 \
  -e SANDBOX_CONFIG=hyper_lite \
  --entrypoint /usr/bin/tini \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64 \
  -- /usr/bin/unshare --cgroup --mount /bin/bash -c '
    set -e
    mount -t tmpfs tmpfs /sys/fs/cgroup
    mkdir /sys/fs/cgroup/memory /sys/fs/cgroup/cpu,cpuacct
    mount -t cgroup -o memory cgroup /sys/fs/cgroup/memory
    mount -t cgroup -o cpu,cpuacct cgroup /sys/fs/cgroup/cpu,cpuacct
    ln -s cpu,cpuacct /sys/fs/cgroup/cpu
    exec /opt/sandbox-server/bin/python -m uvicorn python_service:app \
      --host 0.0.0.0 --port 8080 --workers 1
  '
curl --fail http://127.0.0.1:18790/health
```

The training container needs host networking to reach the sandbox through this loopback endpoint.
Service-wide concurrency is 1, with a 1 MiB raw candidate stdout/stderr limit. HTTP timeout includes queueing time
and is separate from per-test `run_timeout`. The pinned image offers no per-request memory limit configuration; deployment and existing lite isolation constrain resources.
For `SandboxError`, inspect service logs. A successful `/health` alone does not prove overlay/cgroups or candidate execution work.

## Validation and Training

CPU contract tests never execute candidate code in the training process; HTTP cases use locally controlled service responses.

```bash
python -m pytest -q \
  tests/ut/rl/data/test_code_data.py \
  tests/ut/rl/agentic/test_code_judge.py \
  tests/ut/rl/agentic/test_code_runner.py
```

Real sandbox contract tests use fixed correct/incorrect programs to check runtime exceptions, timeouts, output limits, non-root/zero-effective-capability execution,
independent candidate dependencies, network isolation, and cross-request file isolation.
UT covers propagation of unreachable-service and corrupted-response failures. Configure resources explicitly; skipping due to missing resources is not a pass:

```bash
export RL_ST_SANDBOX_ENDPOINT=http://127.0.0.1:18790
export RL_ST_SANDBOX_RUNTIME_VERSION=sha256:5f5c291ba3d15d9ec20bc3b5695f87f107ee26d39c86817d92283fbd79a0f856
export RL_ST_REQUIRED=1
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_code_sandbox
```

### Launch Directly from the Host

Use the [Docker launcher](scripts/run_qwen3_4b_code_docker.sh). Deploy the v2 sandbox above first,
prepare the local training image, Qwen3-4B weights, and reviewed data, and confirm four devices are available. The script does not download them.

```bash
export HYPER_CODE_MODEL_ROOT=/path/to/models/Qwen3-4B
export HYPER_CODE_DATA_ROOT=/path/to/code-reviewed
export HYPER_CODE_DEVICES=0,1,2,3
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh --dry-run
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh
```

Replace example paths with local model and reviewed-data directories. The latter must contain `train.parquet` and `validation.parquet`.
Output defaults to repository `output/code/<timestamp>-<pid>/`,
overridable with `HYPER_CODE_OUTPUT_DIR`, and contains `train.log` and `checkpoints/`.
The script installs the current source's editable main package and RL plugin, passing the actual sandbox Image ID as `agentic.code.runtime_version`.

By default, it runs the YAML's two training steps, four candidates per question, final evaluation, and saving; generation budgets follow configuration.
Initialization CPU threads are `null`, and sleep level is `1`. Override existing fields explicitly at the end of the command, for example:

```bash
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh \
  --train.model_init_cpu_threads=8 --rollout.vllm.sleep_level=2
```

Default execution is unprivileged, maps only the selected four physical devices, and uses logical IDs `0,1,2,3` inside the container.
If the driver requires privileged access, explicitly set `HYPER_CODE_PRIVILEGED=1`; this mode uses selected physical device IDs.
See `--help` for image, sandbox service, device, and port configuration.

### Launch in an Existing Training Container

After confirming four devices are idle and data has passed the review above, run:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/code/configs/qwen3_4b_code_vllm.yaml \
  --data.train_path=/results/code-reviewed/train.parquet \
  --data.test_path=/results/code-reviewed/validation.parquet
```

This configuration uses dense Qwen3, TP2/DP2, and colocated native vLLM. Checkpoint saving triggers evaluation,
so both `evaluation.enabled` and `train.checkpoint.save_final` are enabled; model and optimizer states are saved at the end.
Before launch, verify total training/rollout device occupancy and ensure `runtime_version` matches the actual sandbox image.
Acceptance checks finite loss/gradients, actual parameter updates, increasing policy versions, and resampling after updates.
See [feature and validation documentation](../../docs/moe_code_agent.md#system-tests) for ST entry points and boundaries.
If reusing the same questions for train/validation to check execution quickly, record the overlap explicitly and do not report independent generalization evaluation.
