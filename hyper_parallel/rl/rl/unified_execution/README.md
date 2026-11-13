# Unified Execution

`unified_execution` 让 HyperParallel 训练与 vLLM 推理共用模型定义、并行策略和关键数值实现，
在已验证的不同 batch，以及 DP/FSDP、TP、SP、EP 配置之间实现训推前向一致。
HyperParallel 负责模型构建、参数切分和重分布；
vLLM 负责请求调度、paged KV cache 和推理图执行。

一致性比较以相同权重、token 前缀和有效位置为前提。训练 padding、推理扁平 token 与 SP 分片
需要先对齐到同一逻辑坐标；不要求不同拓扑下的训练梯度或完整 RL 轨迹逐位相同。

## 核心实现

### 共享模型与 Attention

vLLM plugin 注册 HP 模型入口，worker 通过已有 HP builder 应用训练侧完整的
`train.plan_overrides`，复用模型替换、切分和通信边界。执行契约携带配置指纹，两端校验实际组件、
参数形状和别名，避免只有一端安装替换。

Attention 分成两层：`paged_attention.py` 连接 HP 的 QKV、RoPE、输出投影与 vLLM cache/调度；
`attention_backend.py` 调用两端共用的 FlashAttention NPU V4 数值接口。训练使用 dense/varlen
接口，推理适配分页 KV 元数据；prefix caching、chunked prefill 和 `FULL_AND_PIECEWISE` 保留。

### 消除 batch 变化带来的差异

训练进程与每个推理 worker 分别初始化并核验 `torch_npu.npu.set_deterministic_level(3)`。
当前默认设置 `torch.utils.deterministic.fill_uninitialized_memory=False`，关闭未初始化内存的
自动填充，保留 Level 3。已有填充开关对照验证了相同数值结果，并观察到 prefill/decode 的性能收益；
需要填充对照时可显式设置为 `true`。这不改变序列 padding 或 MoE 固定容量缓冲区的有效性掩码。
两端使用相同策略，不同时叠加 vLLM-Ascend 的全局 batch-invariant 补丁。

Attention 还需要已验证的 V4 canonical rowsum 补丁，使完整 KV tile 与尾块使用一致的 softmax
规约顺序；仅开启 Level 3 不能替代这一处理。补丁和加载方式见 [运行环境](#运行环境)。

### 消除 TP/SP 变化带来的差异

PIK 将矩阵乘法的全局 K 维划分为固定数量的 leaf，TP 改变时只改变 leaf 的归属。
每个 rank 用 Level 3 `bmm` 批量计算本地 leaf，Triton 按固定二叉树进行 FP32 规约，
最终转换为 BF16；通信搬运部分结果，保持同一逻辑规约树。两端必须固定相同的 leaf 数和算术选项。

PIK 通过普通 row-parallel 组件接入：Dense Qwen3 覆盖 attention `o_proj` 和 MLP `down_proj`，
Qwen3-MoE 当前只覆盖 attention `o_proj`。SP 输出布局及一阶 autograd 由算子和 HP 适配共同处理，
非 SP 的训练反向保留 HP 原有的下游梯度 SUM。

### MoE 训推一致与图捕获

MoE 两端共用 HP 模型的 router、专家权重、GroupedExperts 和 Level 3 策略，保留同一套
top-k 选择、路由权重和专家计算。这样，从路由到专家输出都沿用共享计算路径。

固定容量 EP 进一步统一结果合并：通信后按专家 ID 选取每条路由的唯一 owner 输出，
再按 token 的 top-k 槽位顺序，以 FP32 加权累加并在最后转换 dtype；不沿 rank 维做浮点求和。
因此 EP 分布改变时，结果仍按相同的逻辑路由顺序合并。

图捕获使用固定形状的计算/通信缓冲区承载动态路由。容量覆盖各 rank 的本地 token 行数，
填充行在专家计算前后被掩码排除，超界报错、不丢 token。该路径同时用于训练和推理，
保留 HP 专家参数的 unshard/reshard 与训练梯度。专家内部 PIK/ETP 尚未接入。

## 代码组织

| 位置                                                                                            | 职责                                                |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [contract.py](contract.py)、[execution_plan.py](execution_plan.py)、[validation.py](validation.py) | 执行契约、显式配方解析、实际模型检查                |
| [batch_invariant.py](batch_invariant.py)                                                         | 进程级 Level 3 初始化与状态核验                     |
| [ops/](ops/)                                                                                     | V4 Attention、PIK、固定容量 EP 数值实现             |
| [components/](components/)                                                                       | 将算子绑定到 HP 模型与并行边界                      |
| [models/](models/)                                                                               | 模型类型、已有构建工厂、投影和 router 描述          |
| [runtimes/hyper_parallel/](runtimes/hyper_parallel/)                                             | 复用 HP builder，接入训练 mask、位置和数值策略      |
| [runtimes/vllm/](runtimes/vllm/)                                                                 | 模型注册入口、编译、mesh、分页 Attention 和权重加载 |

接入新模型时，先在 `models/` 描述已有 HP 模型，复用公共组件；特殊计算放入 `ops/`，
绑定逻辑放入 `components/`，cache 或调度适配放入对应 runtime，避免复制整套模型 forward。

## 使用方式与边界

从[四层 Qwen3-MoE 两步 RL 示例](../../examples/consistency/README.md)开始。
[完整 YAML](../../examples/consistency/configs/qwen3_moe_4l_rl.yaml) 显式声明模型替换与并行策略：
训练 `TP4/SP/EP4`，推理 `TP2/DP2/无 SP/EP4`，共用四张卡；两端采用 Level 3、V4 补丁、
固定容量 EP 与 attention `o_proj` PIK（8 leaves）。推理开启图、prefix cache 和 chunked prefill。

示例复用正常训练入口的 builder 和 RL 循环，每步在 optimizer 更新前检查实际采样 token 的
训推 log-prob 最大差值为零；两步训练及权重发布完成后写入 `consistency.json`。
改变并行配置时保留两端相同的 `train.plan_overrides` 和算术规则，按 YAML 调整两端布局。

当前覆盖 Qwen3 Dense 与 Qwen3-MoE。在已验证场景中，训练侧 FSDP/TP/SP/EP 与推理侧
DP/TP/SP/EP 的不同配置实现了训推前向逐位一致，包括八卡 `TP4/FSDP2/SP/EP4` 训练配置、
非均匀 DP 请求，以及两端 TP/SP 不同的真实 RL 流程。MoE 数值与梯度验收使用四层、128 专家、
top-k8 模型。CP/PP 和专家内部 PIK/ETP 后续接入；当前 `TP1/DP2/EP2 + PIK` 推理配方仍有重复绑定限制。

配置、组件与 autograd 的 CPU 检查位于 [tests/ut/rl](../../../../tests/ut/rl/)；
设备测试集中在 [PIK](../../tests/st/test_pik.py)、[V4 attention](../../tests/st/test_unified_attention.py)
和 [MoE 图捕获](../../tests/st/test_moe_graph.py)，覆盖算子前向、梯度、图回放和分页缓存。
模型级跨并行前向、正常 RL 流程与权重发布验收统一使用上述两步 RL 示例。
扩展模型或并行配置时，需要同时验证有效 token 的前向一致性、与自身前向匹配的训练梯度和真实参数更新。

## 运行环境

本地使用 Linux ARM64、Ascend 910B 和配套驱动。已准备的镜像为：

```text
swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:consistency-v0.30.0-cann9.2.1-arm64
```

发布 digest：`sha256:ee0e756bd5905e762039b0bac5339e28756d350da356251ff9f1c71b634c8f7d`。
镜像中已配套安装以下版本，无需在宿主机逐项安装：

| 组件 | 版本 |
| --- | --- |
| Python / CANN | 3.12.16 / 9.2.1 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post7 |
| Triton Ascend | 3.6.0.post2 |
| vLLM / vLLM-Ascend | 0.30.0 / 0.30.0 |
| Transformers | 5.14.1 |
| NumPy / PyArrow | 1.26.4 / 24.0.0 |
| batch-invariant 设备算子与 Python 扩展 | 2.0.0（本配方使用 Level 3） |
| FlashAttention NPU | 镜像内 0.5.0；运行时加载 V4 `0.5.0+hpbi1` 补丁 wheel |

V4 补丁尚未固化到基础镜像；启动器校验 wheel SHA256 后安装到独立目录 `/opt/hp-v4`，
并让训练进程、vLLM 服务与 worker 使用同一路径。
[canonical rowsum 源码补丁](../../docker/patches/flash-attn-npu-v4-canonical-rowsum.patch)
统一完整 tile 与尾块的规约顺序。使用已验证的 Python 3.12 / ARM64 wheel：

```text
flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl
SHA256: 5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9
```

没有该 wheel 时，可用[源码构建脚本](../../docker/README.md#构建-v4-补丁-wheel)在现有镜像中生成，
无需修改基础镜像或通过 Git 传递二进制。使用重建产物时，同时设置 `V4_WHEEL` 和 `V4_WHEEL_SHA256`。

镜像已保留采样器 `q.record_stream(torch.npu.current_stream())` 修复。
依赖兼容调整采用 FastAPI 0.136.0、Starlette 1.0.1、OpenCV headless 4.11.0.86；
启动时执行 `pip check`。本示例走文本模型，vLLM 视频读取的上游问题另行处理。

本机模型放在 `../models/Qwen3-30B-A3B`、GSM8K 数据放在 `../data/gsm8k`，
补丁 wheel 放在 `../wheels`（相对于仓库根目录）。直接执行：

```bash
bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
```

默认使用空闲卡 `0,1,2,3`，结果目录 `/tmp/hyper-rl-consistency`。
首次运行自动裁剪四层权重，保留 128 专家与 top-k8；后续核验校验和后复用。
路径、卡号与 YAML 的修改方式见[示例说明](../../examples/consistency/README.md)。
