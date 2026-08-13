# Qwen3.5 vLLM 兼容与实验基线

本文记录 Hyper-RL 接入 vLLM 时必须保留的既有实验边界。历史精度证据来源于上游 Hyper-Parallel revision
`4c766bdd` 及对应 vLLM 验证脚本；当前实现不能在没有新证据时放宽这些要求。

训练和推理联合启动的完整故障现象、排查证据、修复过程、文件索引和两步门禁状态见
[vLLM 在线 Rollout 故障记录与验收状态](vllm_online_rollout_status.md)。

## 固定环境

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3.5-0.8B-Base，text-only |
| Torch / torch-npu | 2.10.0 / 2.10.0 |
| Transformers | 5.5.4 |
| vLLM / vLLM-Ascend | 0.22.1 / 0.22.1rc1 |
| 开发镜像 | `hyper-parallel/unified-e2-dev:v0.22.1rc1` |
| 开发镜像 ID | `sha256:1b479d5b01788d77acf3d6c7f0d8636f983b7f6014fc9130421782c5cc4f7348` |
| 基础镜像 digest | `sha256:9008b47081282612abfe4d28069ce34436752c980fd06f7599343213205ce64d` |
| dtype / execution | BF16 / eager |
| TP | 1 和 2 |
| PP / PCP / DCP | 1 / 1 / 1 |
| worker | `spawn` |
| 数值 profile | `functional` |

当前工作区默认使用：

```text
models/Qwen3.5-0.8B-Base
data/gsm8k/train.parquet
data/gsm8k/test.parquet
```

该开发镜像当前没有 registry `RepoDigest`，镜像 ID 是本轮可重复标识；基础镜像 digest 不能替代可运行开发镜像 ID。
Docker 必须允许 `--privileged`，并挂载 launcher 中声明的宿主机 Ascend driver、`dcmi` 和 `npu-smi`。

模型关键文件哈希：

```text
config.json                                      b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
model.safetensors-00001-of-00001.safetensors    c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c
model.safetensors.index.json                    ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a
tokenizer.json                                  fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927
```

GSM8K 文件必须匹配历史实验：

```text
train.parquet  89cd3cb8d28e5274e7f0bf71ff541ea5654ac9e30589ac4b5d19c3f783a3858c  7473 rows
test.parquet   e801bce6b15925630ea9976dc14419e49735a90fd54d267911e6701ebdc0d489  1319 rows
```

提交实验前可以在 workspace 根目录校验：

```bash
sha256sum \
  models/Qwen3.5-0.8B-Base/config.json \
  models/Qwen3.5-0.8B-Base/model.safetensors-00001-of-00001.safetensors \
  models/Qwen3.5-0.8B-Base/model.safetensors.index.json \
  models/Qwen3.5-0.8B-Base/tokenizer.json \
  data/gsm8k/train.parquet \
  data/gsm8k/test.parquet
```

两个 parquet 文件必须直接包含 `prompt` 和 `extra_info` 列。

## 插件要求

统一 launcher 会在容器内显式清除 `VLLM_PLUGINS`，让 vLLM 同时加载 Ascend platform plugin 和全部 Ascend
general plugins。不要只在宿主机执行 `unset`，因为宿主变量不会覆盖镜像内的 `ENV`。
如果部署必须使用 allowlist，至少包含：

```text
ascend
ascend_kv_connector
ascend_model_loader
ascend_service_profiling
ascend_model
hyper_parallel_models
hyper_rl_models
```

遗漏 Ascend general plugins 会使 spawned TP worker 缺少 NPU memory API patch。当前包同时提供
`hyper_parallel_models` 兼容入口和 `hyper_rl_models` 新入口，用于处理固定实验镜像中的旧 metadata。

## 已有证据

历史修正后的 native Hyper adapter 已通过：

- TP1 与 TP2 eager prefill/decode；
- 两请求、多 token decode 和有限 selected-token logprob；
- TP1/TP2 的 4/4 token 一致，最大 selected-logprob 差 `0.0220201612`；
- 22 个 adapter contract；
- deterministic TP2 三步训练 loss、local-gradient SHA256 和 grad-norm byte gate；
- 专用 parity profile 下完整 8,792 条 GSM8K prefill/scoring exact gate。

完整 GSM8K exact 结果是历史 precision evidence，不代表多步 paged decode、在线权重更新或 GRPO 闭环。
旧架构下的 graph、MTP 和多模态结果也不能继承到当前 native ownership adapter。

## 当前扩展门禁

进程边界参考了以下已有实现和官方约束：

- RL2 `RL2/workers/rollout.py`：Trainer 只保留 rollout 代理，SGLang server 作为独立进程，通过 HTTP 控制；
- veRL `verl/workers/rollout/vllm_rollout/vllm_async_server.py`：在构造 vLLM 前为 server actor 固定 Ascend 可见卡；
- vLLM [Python Multiprocessing](https://docs.vllm.ai/en/latest/design/multiprocessing.html)：accelerator runtime 初始化后不能使用 `fork`，library 场景必须明确管理 `spawn` 和 main-process 边界；
- vLLM-Ascend `examples/rl/rlhf_http_hccl.py`：外部 `vllm serve`、HTTP 控制面和 stateless HCCL 权重数据面；
- vLLM-Ascend `tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py`：NPU0/NPU1 分卡的官方两卡端到端门禁。

RL2 本身使用 CUDA/NCCL + SGLang，不包含 vLLM-Ascend 或 HCCL 实现，因此只借鉴其服务进程边界，不复制其
`fork`、CUDA IPC 或设备 API。

Hyper-RL 在上述 functional envelope 上新增在线权重同步。分卡/disjoint HCCL 路径的每次策略更新必须满足：

1. Trainer 固定使用 NPU0；独立 `vllm serve` 子进程在启动解释器前固定使用 NPU1；
2. 子进程不继承 torchrun rank、world-size、master rendezvous 和 TorchElastic store 身份；
3. 生成通过 loopback OpenAI completion API 返回 token IDs 和 sampled-token log-probs；
4. gather 完整 Actor state，并在训练 NPU 上保持 contiguous tensors；
5. `/pause` 清理 running requests 和 prefix cache；
6. vLLM-Ascend `backend="nccl"` 建立独立 stateless HCCL group，Trainer 为 rank 0，推理 worker 从 rank 1 开始；
7. 每个 vLLM worker 使用严格 Hyper `load_weights()` 完成原子 reload，再执行 `/finish_weight_update` 和 `/resume`；
8. 所有步骤成功后才推进 `policy_version`，下一轮 rollout 必须使用新权重。

`backend="nccl"` 在 Ascend plugin 中实际映射为 HCCL；不能写成 `backend="hccl"`。`backend="ipc"` 只适用于训练与
推理位于同一物理 NPU 的场景，且要求 insecure serialization，因此不用于当前分卡闭环。CPU safetensors refit 仍保留为
独立 adapter correctness gate，不再作为 GRPO 在线数据面。

共卡/colocated 路径使用 FSDP DP2 和一 rank 一 TP1 vLLM replica，权重传输 backend 为 `ipc`。它要求 CPU offload、
forward 后 reshard、vLLM sleep level 1，并在全部 replica 完成 refit、KV wake 和 resume 后才推进 policy version。

2026-08-11 已完成 Hyper/native TP1 一致性、单次 CPU refit、共卡 FSDP DP2 + rollout DP2/TP1 的两步真实学习门禁，
以及两种实现各 20 步的 NPU IPC/sleep-mode correctness soak。当前在线 HCCL refitter仍限制为单训练 rank；多 rank
分卡/TP rollout 需要独立扩展 rank mapping 后再开放。完整结果和并行 HCCL 端口隔离要求见在线 rollout 状态文档。

## 重复实验

统一脚本会自动从仓库同级的 `models/` 和 `data/` 目录选择既有 artifact：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh rollout-tp1
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh rollout-tp2
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh refit
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-smoke
```

可通过 `HYPER_VLLM_MODEL_ROOT`、`HYPER_VLLM_DATA_ROOT`、`HYPER_VLLM_IMAGE` 和
`HYPER_VLLM_RESULT_ROOT` 覆盖自动发现值。
`HYPER_VLLM_TIMEOUT_SECONDS` 控制主动终止时间。大多数 action 默认 600 秒，`production-benchmark` 和
`grpo-production-canary` 默认 1,800 秒，`grpo-production` 默认 259,200 秒。
