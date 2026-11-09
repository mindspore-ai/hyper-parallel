# 可选 OPD / MOPD

[English](README.md) | **简体中文**

该示例复用同步 Trainer、Agent 轨迹和学生权重发布。默认不配置 `distillation` 或设置
`distillation.enabled: false` 时，不创建教师客户端、不发送教师请求、不计算蒸馏损失。

## 模型与资源

首个配方使用 Qwen3-4B 学生与冻结的 Qwen3-30B-A3B 教师。二者必须具有相同的
`tokenizer.json`、tokenizer 配置和特殊 token 配置。教师是外部管理的独立 vLLM 服务；
不参与学生 optimizer 或 full_gather 发布。

示例配置学生单 rank、独立单卡推理 TP1；教师建议另外两卡 TP2。A3B 是激活参数量，
教师仍需装载整个 checkpoint。具体余量取决于权重、缓存、上下文与并发，不保证任意配置可运行。
教师模型目录应保持不可变，服务必须装载配置指定的 checkpoint，不能只复用同一个服务名称。

使用 [固定运行环境](../../docker/README.zh.md)，确保 editable 安装指向当前 checkout。
从仓库根目录运行；下面路径在容器内可见：

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl${PYTHONPATH:+:$PYTHONPATH}"
ASCEND_RT_VISIBLE_DEVICES=4,5 HCCL_IF_BASE_PORT=62000 python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B --served-model-name opd-teacher \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 --logprobs-mode raw_logprobs \
  --host 127.0.0.1 --port 8450
```

教师和学生进程必须使用互不冲突的设备与 HCCL 端口。跨容器时使用可访问的服务地址；
教师需要 `v1/models` 和支持原始 token IDs 的 `v1/completions` echo/logprobs 接口。
请求 neutral temperature/top-p/top-k、不加 penalty，提取输入概率而非额外生成 token。

## 单教师运行

准备 GSM8K parquet，见 [快速开始](../../README.zh.md#快速开始)，修改 YAML 中的数据与输出路径。
启动学生前先验证模型身份、tokenizer 和原始 token 概率对齐：

```bash
python -m examples.distillation.probe_teacher \
  --config hyper_parallel/rl/examples/distillation/configs/qwen3_4b_opd.yaml
ASCEND_RT_VISIBLE_DEVICES=0 HCCL_IF_BASE_PORT=62600 \
  bash hyper_parallel/rl/examples/distillation/scripts/run_distillation.sh \
  hyper_parallel/rl/examples/distillation/configs/qwen3_4b_opd.yaml
```

`qwen3_4b_opd.yaml` 关闭任务目标，只优化蒸馏；保留任务评分供观察，但不构建 GRPO 分组优势
或 Critic。`qwen3_4b_grpo_opd.yaml` 保留原 GRPO 目标并叠加蒸馏。
启动脚本不改写 YAML 中的训练决策，`NPROC_PER_NODE` 必须与配置的训练拓扑匹配。
蒸馏配方使用 `train.optimizer.fp32_main_params: true`，通过主项目现有封装保存 FP32 主参数和矩状态，
BF16 参数用于模型计算与发布；梯度准备之后按 FP32 主参数裁剪。
默认 CPU offload 将 optimizer 主参数与状态保留在主机内存，实际容量需求仍须按模型和拓扑规划。
checkpoint 记录主参数精度与分片拓扑，恢复时拒绝不匹配配置；不能从 BF16-only checkpoint 恢复丢失的小更新。
rank 本地状态将主参数 DTensor 转为其本地 FP32 张量保存；不序列化分布式布局，不重新舍入到 BF16。
单 rank 开启 CPU offload 时同样使用原生 FSDP，避免未分片路径忽略 offload 配置。
原有非蒸馏配置未指定此字段时保持原精度行为。
小学习率累积、实际主梯度裁剪与恢复见 [精度回归测试](../../../../tests/ut/rl/policy/test_optimizer_precision.py)。
默认学生训练使用 NPU 0，rollout 使用 YAML 中的 `visible_devices: '1'`；教师另外分配设备。
指定镜像中的 DP=2 概率偏差已定位到 meta materialization 后未恢复的非持久 RoPE 频率缓冲区。
RL 的 [materialization 适配](../../rl/workers/engine/materialization.py) 通过已有公共接口，
仅为 RL builder 构建的 Qwen3 实例注册缓冲区重建：在 CPU 调用原生 HF 公式，再传到目标设备，
与原生 HF 初始化公式一致；直接在 NPU 重算频率会产生 FP32 微小差异，
并可能在较大位置改变 BF16 cos/sin。
注册在分片与 `to_empty` 前完成，覆盖 checkpoint 加载与随机初始化；重复注册不会改变公共类或其他模型实例。
无需修改 `hyper_parallel/core` 或 `hyper_parallel/models`。
相关实例隔离与 HF RoPE 对照测试位于 [RL 内的 materialization 测试](../../tests/ut/test_materialization.py)。
蒸馏示例使用 native-vLLM，并独立记录 rollout 与训练端重算概率。旧 bit-exact 门禁已删除；Teacher 概率有效性门禁继续阻止无效更新与发布。
重放通过不替代训练、发布与 checkpoint 验收，默认使用资源需求更低的单 rank 配方。

## 多教师

`qwen3_4b_mopd.yaml` 使用 Qwen3-30B-A3B 与 Qwen3-8B 两个独立 checkpoint。
先准备真实路径、服务和路由数据；每个服务内部的多个副本仅用于扩容，不构成多个教师。

在另一独立进程或容器启动第二个教师：

```bash
ASCEND_RT_VISIBLE_DEVICES=6 HCCL_IF_BASE_PORT=62300 python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-8B --served-model-name dense-teacher \
  --tensor-parallel-size 1 --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 --logprobs-mode raw_logprobs \
  --host 127.0.0.1 --port 8451
```

训练 parquet 保留原任务 prompt/label，并增加 `teacher_key` 列：对应两个教师的值为 `large`、`dense`。
配置 `data.metadata_columns: [teacher_key]` 后，
该字段进入 PromptRecord.metadata。
任务环境与判题规则仍须匹配数据；此模板继承 GSM8K 环境，真实代码任务应使用已有 code row adapter
与环境，不能仅更换教师就把 GSM8K reward 用于代码。一个可运行的最小多教师实验也可让两个数学
专家 checkpoint 指导 GSM8K 的不同子集，再扩展到混合任务。

每条样本选一个教师，不平均教师分布。多教师缺失/未知路由、HTTP 错误、超长上下文、token 不匹配、
缺失或非有限概率都拒绝当前更新。DP 补齐行不请求教师且损失为零。

## 数学与算法边界

采用 verl 的 sampled k1 PG 路线：每次训练前向重算当前学生概率，使用
`stopgrad(log p_teacher - log p_student_current)` 作为独立 token 优势，
配合 clipped ratio 和 dual clipping。
ExperienceBatch.rollout_log_probs 独立保留真实采样概率，old_log_probs 保存更新前训练侧重算概率。
启用 OPD 时采用 rollout-anchor（bypass）路径：任务 RL 和 OPD 的概率比分母均为原始采样概率，
当前 batch 内固定；重算概率仅用于训推诊断，不替代采样分布。诊断显式比较这两个字段。
缺失采样概率、非法动作概率或评分失败拒绝更新；无效观察与 DP padding 不参与概率检查或 loss。
初版要求学生 temperature=1、top_p=1、关闭 top_k，不提供 tempered/truncated rollout 校正。

任务目标与 OPD 分别计算，OPD 按全 DP 有效动作 token 数归一化；GSPO 的任务项仍按序列归一化。
`loss.coefficient` 在纯蒸馏和联合训练中均有效。默认不截断 k1；可用 `loss_max_clamp` 显式改变信号。
GRPO/PPO/GSPO 的组合共享接口，但真实 NPU 验收应按配方分别进行。

外部 Search-R1 等工具型任务 继续采用原 episode GRPO：将本例 `distillation` 块加入已有配置。
按每次调用的真实上下文评分，工具观察不贡献动作 loss。单教师无需添加数据路由字段；多教师
通过原 row adapter/metadata_columns 填写教师 key。纯 OPD 当前只支持 internal runner，
本特性不扩展外部程序的 PPO/GSPO 支持范围。教师上下文上限必须覆盖每次完整调用及一个额外 token；
Search-R1 的长上下文需要同步扩大教师服务和配置中的 max_model_len，不能直接沿用短序列示例的 2048。
训推概率一致性应单独验证；参数版本相同不保证概率 bit-exact，本路径直接使用采样分母，不采用重算锚点加额外 IS 权重；clipping 仍是有偏代理目标。

## 与 verl 的实现边界

| 项目 | verl 参考实现 | 本实现 |
| --- | --- | --- |
| 蒸馏目标 | sampled KL PG 与 top-k forward KL/GKD | sampled k1 PG，显式 stop-gradient |
| MOPD | 根据样本 routing key 选择一个教师 | 根据 PromptRecord.metadata 中的 teacher_key 选择一个教师 |
| 教师资源 | Ray 资源池、异步推理客户端、多个副本 | 外部管理的冻结 vLLM HTTP 服务，Trainer TP owner 发请求 |
| 策略分母 | 可选 rollout correction/bypass | 启用 OPD 时任务项与蒸馏项都使用原始 rollout logprobs |
| 损失归一化 | 配置的 aggregation 与 global batch 信息 | 蒸馏项按全 DP 有效动作 token 数；任务项保留原算法归一化 |
| 纯蒸馏 | 关闭 task loss | 不构建任务优势、Reference 或 Critic，coefficient 仍有效 |

旧 Hyper-RL 的蒸馏配置、教师客户端、k1 目标和测试可以适配复用；其整个 RL 目录包含旧版
模型、Agent、checkpoint 和权重发布代码，不能整体覆盖新主干。verl 的数学目标和路由合同可作为
参考，其 Ray/TensorDict/worker 代码不能直接替换本项目的 SyncTrainer/ExperienceBatch。

## Teacher 选择

对于 Qwen3-4B 学生，优先选择使用相同 tokenizer 的 Qwen3 教师或同 tokenizer 的领域微调模型：

| 候选 | 用途与验证要求 |
| --- | --- |
| Qwen3-30B-A3B | 默认 MoE 教师；需要装载全部约 30B 权重，不能按 3B 激活参数计算内存 |
| Qwen3-8B | 双教师功能验收配方；不同 checkpoint 不自动代表不同领域专家 |
| Qwen3-14B、Qwen3-32B | 更大的 dense 候选；上线前仍需严格 tokenizer 与概率探针校验 |
| 同 tokenizer 的数学/代码微调模型 | 更适合验证跨领域收益；还需要独立任务环境和 held-out 评估 |
| Qwen3.8-27B | 与 Qwen3 的 tokenizer 不兼容，当前 sampled-token MOPD 必须拒绝 |

Qwen3 与 Qwen3.8 的词表和 token ID 不同，不能通过复制 tokenizer 文件、对同一个 ID 取概率或
简单重分词修复。跨 tokenizer 蒸馏需要另行定义序列/文本概率目标和对齐方法，超出当前合同。
模型官方说明见 [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)、
[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) 和
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)。

## 验证与恢复

`PYTHONPATH=hyper_parallel/rl python -m pytest -q hyper_parallel/rl/tests/ut/test_materialization.py`
检查原生 HF 频率、cos/sin、重复 materialization 与实例隔离。

```bash
python -m pytest -q tests/ut/rl/distillation
python -m pytest -q tests/ut/rl
```

关键指标为 `distillation/loss`、`distillation/supervised_tokens`、
`distillation/teacher_samples/<key>` 和 `timing_s/teacher`，
结合原 gradient norm、policy version
与 held-out 质量检查。OPD loss 数值可以为负，不能单独作为质量指标。

可选的服务内概率对照逐前缀 decode 学生 token，并与 echo 评分比较：

```bash
python -m examples.distillation.probe_teacher \
  --config /path/to/config.yaml --decode-reference --atol 0 --rtol 0
```

`--reference-forward` 另行加载完整教师做 CPU HF 对照，默认 BF16；它检查跨后端数值差异，
不能用服务内对照替代。OPD 的教师分布定义为已验证 raw-logprobs 的实际冻结服务。

概率合同升级为 rollout-anchor；旧合同的 checkpoint fingerprint 不兼容，需要新训练或明确迁移。

Checkpoint manifest 保存蒸馏配置、本地模型/tokenizer 元数据和全部教师权重分片及索引的内容 fingerprint。
每个训练进程在启动校验时流式读取一次教师权重，内存开销固定；保存时复用该身份，不逐步扫描。
恢复进程重新计算身份并拒绝不一致；旧版未包含权重摘要的蒸馏 checkpoint 需要重新训练或明确迁移，不能直接恢复。
本地摘要不证明外部服务的内存权重身份；教师必须使用对应的只读模型目录，训练期间不得替换权重或重新加载服务。
教师服务失败会在进入 Actor 更新前跨训练 rank 传播；不回退到其他教师，不发布部分评分结果。

CPU 路由与梯度测试不代替真实多教师、长程学习或完整 benchmark。只有一个真实 checkpoint 时
可验收 OPD 和 MOPD 路由合同，不能宣称真实多教师效果已验证。

真实 NPU 验收使用独立 ST 入口，资源未准备时会跳过，不能算作通过：

```bash
example_root=hyper_parallel/rl/examples/distillation
export RL_OPD_ST_CONFIG="$example_root/configs/qwen3_4b_opd.yaml"
export RL_OPD_ST_NPROC=1
ASCEND_RT_VISIBLE_DEVICES=0 HCCL_IF_BASE_PORT=62600 python -m pytest -vv -s \
  hyper_parallel/rl/tests/st/test_distillation_st.py
```

ST 要求至少两个真实更新、非零 gradient norm、教师监督 token 和正确的发布版本。
多教师 ST 使用准备好的真实双教师配置，不能用单 checkpoint 的副本替代专家验证。
ST 还检查更新前 rollout 与 Actor 的 log-probability 偏差：默认平均绝对偏差不超过 0.1，
最大绝对偏差不超过 1.0。可通过 `RL_OPD_ST_MAX_MEAN_LOGPROB_DIFF` 和
`RL_OPD_ST_MAX_LOGPROB_DIFF` 明确设置数值容差；改变容差须在验收记录中说明依据。
两步 ST 验证训练链路与数值门槛，长期学习收益仍需独立 held-out 评估。

## 实现与测试定位

- [教师客户端与路由][teacher]：`TeacherManager`、原始 token 对齐及服务身份。
- [蒸馏目标][objective]：`compute_opd_loss`、纯 OPD 需求选择。
- [Trainer][trainer]：`_score_teacher_rollout` 与 TP 请求所有权。
- [CPU 合同与梯度测试][unit]：路由、mask、失败传播、算法组合及恢复身份。
- [真实 ST][system]：显式资源入口及监督、梯度、发布断言。

[teacher]: ../../rl/distillation/teacher.py
[objective]: ../../rl/algorithm/distillation.py
[trainer]: ../../rl/trainer/sync.py
[unit]: ../../../../tests/ut/rl/distillation/test_distillation.py
[system]: ../../tests/st/test_distillation_st.py

## 更新前门禁与蒸馏监控

实现位于 [蒸馏监控模块](../../rl/distillation/monitoring.py)，
拒绝更新与混合长度路由的测试位于 [监控合同测试](../../../../tests/ut/rl/distillation/test_monitoring.py)。

启用蒸馏时，每一批训练均在 Actor、Critic 更新和策略发布之前检查采样概率与训练侧重算概率。
所有 DP rank 共同汇总有效动作 token；工具观察、补齐 token 和 DP padding 不参与统计。
非有限概率、正的 log-probability、空的全局动作集合或者任何超限指标都会拒绝整批更新。

```yaml
distillation:
  probability_gate:
    max_mean_logprob_diff: 0.1
    max_logprob_diff: 1.0
    max_ratio_outside_fraction: 0.05
```

概率比为 `exp(logp_actor_before_update - logp_rollout)`；越界区间使用蒸馏的
`loss.clip_ratio`。这三个默认阈值用于已验证的 Qwen3 配方，允许少量数值尾部偏差，
不能据此宣称每个 token 都一致。新模型或后端需要独立校准；不可通过放宽阈值掩盖分片错误。
ST 的容差环境变量也会写入更新前门禁，更新后检查只用于核对已有验收证据。

`train/clip_fraction` 保留任务算法的定义。蒸馏使用独立指标：

| 指标 | 含义 |
| --- | --- |
| `distillation/ratio_outside_fraction` | 概率比越过 PPO 裁剪区间的动作 token 比例 |
| `distillation/clip_fraction` | PPO 裁剪分支实际选中的比例，区别于仅越界 |
| `distillation/dual_clip_fraction` | 负优势进一步触发 dual clip 的比例 |
| `distillation/ratio_mean`、`log_ratio_min/max` | 概率比均值与未截断 log-ratio 尾部范围 |
| `distillation/log_ratio_saturation_fraction` | 数值保护将 log-ratio 截断到 ±20 的比例 |
| `distillation/sampled_k1_mean` | 当前学生减教师的采样 log-probability 信号均值，可为负，不等于精确 KL |
| `distillation/teacher/<key>/token_fraction` | 每个 Teacher 的有效监督 token 占比 |
| `distillation/teacher/<key>/weighted_loss` | 每个 Teacher 对全局蒸馏损失的加权贡献，包含 coefficient |

Teacher 的样本数仍在 `distillation/teacher_samples/<key>` 中记录。
相同样本数不保证相同 token 占比；本目标按 token 归一化，不自动改成 Teacher 等权平均。
指标跨微批与 DP 汇总，重复训练 epoch 的监督 token 会重复计数。

## 效果评估

使用统一训练入口 `scripts/run_distillation.sh`，通过 `data.test_path` 指定独立评估集，
并设置 `evaluation.enabled: true`。无需额外的对照实验 worker 或配置生成器。
比较 RL、OPD 与 MOPD 时保持初始学生权重、数据划分、种子和采样预算一致。
同时检查任务奖励、响应截断比例、梯度和各 Teacher 的监督贡献；
有限步数和小评估集不能证明收敛或稳定提升。
