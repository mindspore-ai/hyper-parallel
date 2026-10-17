# 单轮 Python code 训练

本示例通过 internal runner 生成一次 Python stdin/stdout 程序，在独立 SandboxFusion 服务中判题，
再将奖励交给 GRPO。示例配置见 [qwen3_4b_code_vllm.yaml](configs/qwen3_4b_code_vllm.yaml)。

## 接入合同

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

`row_adapter` 保留原始多消息、稳定 `task_id`、来源 metadata 和结构化 `ground_truth`。
它与 `prompt_column`、`answer_column`、`prompt_instruction` 互斥。私有测试仅放在
`ground_truth.inputs/outputs`，不进入 prompt；超长消息明确报错，不能静默截断题意。

判题器提取最后一个 Python 代码块，也接受未加代码围栏的程序。提取仅用于沙箱执行，训练仍使用原始
action tokens 和 rollout logprobs。所有声明测试都执行完成后，按空白分词精确比较输出；全通过奖励为 1，
否则为 0。首次失败原因记录为 `wrong_answer`、`runtime_error`、`timeout` 或 `output_limit`；
空程序、不支持的代码围栏或超过 64 KiB 的代码记为 `format_error`。不支持交互题、特殊 checker 和浮点容差。

HTTP、解码、协议及 `SandboxError` 均向上传播，不伪装成零分，也不隐式重试执行。
Trainer TP 组仅 request owner 执行环境，其他 rank 重放结果；`finish_reason=length` 记录为截断。

## 数据准备和核验

以下命令在统一训练容器的仓库根目录执行。当前源码挂载为 `/workspace/hyper-parallel`，
权重根目录挂载为 `/models`，数据根目录挂载为 `/data`，输出目录挂载为 `/results`。
先按[运行环境](../../docker/README.md)保留 CANN 初始化和插件注册，并确认主包 editable 指向当前 checkout。
示例模块沿用现有 launcher 的源码路径：

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

准备脚本只读本地原始 parquet，保持测试内容与顺序，过滤不支持的题型、超预算测试和重复题目，
并生成包含源文件/结果哈希、过滤计数及稳定任务 ID 的 `manifest.json`。
`--manual-exclusions exclusions.json` 接受 `{"train:455277": "审核原因"}` 形式的明确排除记录。

准备成功不代表标签正确。训练前须用独立参考解运行每题全部原始测试；输入输出错配时保留失败证据，
通过 `--manual-exclusions` 排除整题，不修改标签来获得通过结果。
训练配置应指向审核后的 prepared 文件，不直接把候选目录视为可信数据。

## SandboxFusion 部署

使用 `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64`。
先按[镜像获取与校验](../../docker/README.md#单轮-code-的-sandboxfusion-镜像)拉取并核对镜像身份；
镜像基于独立 Python slim 环境，不继承训练基础层。
训练镜像是 `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64`，两者分开运行。
沙箱不挂载训练源码、权重、数据或 Docker socket。

以下启动方式适用于 Docker 18 / cgroup v1。Docker 18 没有 `--cgroupns`，
因此在容器内部使用 `unshare` 建立私有 cgroup/mount namespace，再为原始 lite 隔离挂载控制器；
保持镜像的 `lite` 隔离；镜像内的 `tini` 回收候选超时或结束后遗留的已退出子进程。

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

训练容器须使用 host 网络才能通过上述 loopback endpoint 访问沙箱。
服务全局并发为 1，候选 stdout/stderr 原始输出上限为 1 MiB；HTTP 超时包含排队时间，
与单测试 `run_timeout` 分开。当前固定镜像不提供逐请求内存限制配置，资源由部署与原有 lite 实现约束。
如果返回 `SandboxError`，检查服务日志；仅 `/health` 成功不能证明 overlay/cgroup 和候选执行可用。

## 验证与训练

CPU 合同测试不会在训练进程执行候选代码；HTTP 用例使用本地受控服务响应。

```bash
python -m pytest -q \
  tests/ut/rl/data/test_code_data.py \
  tests/ut/rl/agentic/test_code_judge.py \
  tests/ut/rl/agentic/test_code_runner.py
```

真实沙箱合同测试使用固定正确/错误程序，验证运行异常、超时、输出限制，以及非 root/零有效权限、
候选依赖独立、网络和跨请求文件隔离；
服务不可达与响应损坏的失败传播由 UT 覆盖。资源须显式配置，缺失时跳过不算通过：

```bash
export RL_ST_SANDBOX_ENDPOINT=http://127.0.0.1:18790
export RL_ST_SANDBOX_RUNTIME_VERSION=sha256:5f5c291ba3d15d9ec20bc3b5695f87f107ee26d39c86817d92283fbd79a0f856
export RL_ST_REQUIRED=1
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_code_sandbox
```

### 从宿主机直接启动

使用 [Docker 启动脚本](scripts/run_qwen3_4b_code_docker.sh)。先部署上面的 v2 沙箱，
准备本地训练镜像、Qwen3-4B 权重和已审核数据，并确认四卡可用。脚本不会下载这些资源。

```bash
export HYPER_CODE_MODEL_ROOT=/path/to/models/Qwen3-4B
export HYPER_CODE_DATA_ROOT=/path/to/code-reviewed
export HYPER_CODE_DEVICES=0,1,2,3
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh --dry-run
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh
```

将示例路径替换为本机的模型及已审核数据目录，后者须包含 `train.parquet` 和 `validation.parquet`。
输出默认写到仓库 `output/code/<时间>-<进程号>/`，
可用 `HYPER_CODE_OUTPUT_DIR` 指定；包含 `train.log` 和 `checkpoints/`。
脚本安装当前源码的 editable 主包和 RL 插件，并将实际沙箱 Image ID 传给 `agentic.code.runtime_version`。

默认执行 YAML 的两步训练、每题四候选、最终评测和保存；生成预算以配置为准。
初始化 CPU 线程配置为 `null`，sleep level 为 `1`。既有配置项可在脚本末尾显式覆盖，例如：

```bash
bash hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh \
  --train.model_init_cpu_threads=8 --rollout.vllm.sleep_level=2
```

默认普通权限，只映射所选四张物理卡，并在容器内使用逻辑编号 `0,1,2,3`。
驱动确需特权访问时，可显式设置 `HYPER_CODE_PRIVILEGED=1`；此模式使用所选物理卡编号。
`--help` 可查看镜像、沙箱服务、设备和端口等配置入口。

### 在已有训练容器中启动

确认 4 张卡空闲、数据经过上述审核后执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/code/configs/qwen3_4b_code_vllm.yaml \
  --data.train_path=/results/code-reviewed/train.parquet \
  --data.test_path=/results/code-reviewed/validation.parquet
```

本配置为 dense Qwen3、TP2/DP2、colocated native vLLM。保存 checkpoint 时会触发评估，
因此同时启用 `evaluation.enabled` 和 `train.checkpoint.save_final`；最终写入模型与优化器状态。
启动前核对训练和 rollout 的总设备占用，`runtime_version` 须与实际沙箱镜像身份一致。
验收检查有限 loss/梯度、真实参数更新、策略版本递增及更新后的再次采样。
系统测试入口及边界见[功能与验证说明](../../docs/moe_code_agent.md#系统测试)。
为快速检查链路而复用同一批题作为 train/validation 时，必须明确记录重叠，不报告为独立泛化评估。
