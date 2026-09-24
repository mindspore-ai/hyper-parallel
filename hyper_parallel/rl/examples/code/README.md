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
    runtime_version: sha256:1ac76247ee612e1ac4a07ae10192056f061d93438f7c16578c7b169938b24131
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

准备成功不代表标签正确。现有 125 条训练数据尚未全部通过独立参考解核验；已发现
`eurus2:9776b13264b5:taco:train:455277` 的 nth-prime 输入输出错配。应保留失败证据并排除整题，
不能修标签来获得通过结果。其余题也须先以独立参考解运行全部原始测试。
训练配置应指向审核后的 prepared 文件，不直接把候选目录视为可信数据。

## SandboxFusion 部署

使用 `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v1-arm64`。
先按[镜像下载与校验](../../docker/README.md#单轮-code-的-sandboxfusion-镜像)拉取并核对镜像身份；
镜像公开可读，无需登录，已有训练基础层由 Docker 自动复用。
训练镜像是 `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-unified-arm64`，两者分开运行。
沙箱不挂载训练源码、权重、数据或 Docker socket。

下面是当前 Docker 18 / cgroup v1 环境验证过的启动方式。Docker 18 没有 `--cgroupns`，
因此在容器内部使用 `unshare` 建立私有 cgroup/mount namespace，再为原始 lite 隔离挂载控制器；
没有修改镜像，也没有将隔离模式改为 `none`。

```bash
docker run -d --name hyper-rl-code-sandbox \
  --cap-add SYS_ADMIN --cap-add NET_ADMIN \
  --security-opt no-new-privileges \
  --memory 2g --cpus 4 --pids-limit 128 \
  -p 127.0.0.1:18790:8080 \
  -e SANDBOX_CONFIG=hyper_lite \
  --entrypoint /usr/bin/unshare \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v1-arm64 \
  --cgroup --mount /bin/bash -c '
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

真实沙箱合同测试使用固定正确/错误程序，验证运行异常、超时和输出限制；
服务不可达与响应损坏的失败传播由 UT 覆盖。资源须显式配置，缺失时跳过不算通过：

```bash
export RL_ST_SANDBOX_ENDPOINT=http://127.0.0.1:18790
export RL_ST_SANDBOX_RUNTIME_VERSION=sha256:1ac76247ee612e1ac4a07ae10192056f061d93438f7c16578c7b169938b24131
export RL_ST_REQUIRED=1
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_code_sandbox
```

在已有 NPU 训练容器中，确认 4 张卡空闲、数据经过上述审核后执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/code/configs/qwen3_4b_code_vllm.yaml \
  --data.train_path=/results/code-reviewed/train.parquet \
  --data.test_path=/results/code-reviewed/validation.parquet
```

本配置为 dense Qwen3、TP2/DP2、colocated native vLLM。现有训练器在保存 checkpoint 时触发评估，
因此本配置同时启用 `evaluation.enabled` 和 `train.checkpoint.save_final`；最终会写入模型与优化器状态。启动前检查训练和 rollout 的总设备占用，避免与其他任务冲突。`runtime_version` 要与实际镜像身份一致。
验收检查有限 loss/梯度、真实参数更新、策略版本递增及更新后的再次采样。Dense 与 MoE+code 的四卡两步/评估
在迁移阶段均曾通过；更新代码或运行环境后需重新验收，不能把历史结果当作当前入口的新运行结果。
系统测试入口及边界见[功能与验证说明](../../docs/moe_code_agent.md#系统测试)。
为快速检查链路而复用同一批题作为 train/validation 时，必须明确记录重叠，不报告为独立泛化评估。
