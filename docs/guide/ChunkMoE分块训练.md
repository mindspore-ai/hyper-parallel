# Chunk MoE 分块训练

原 common `MoE` 可以通过 `chunk_size` 启用固定源 token 分块训练。在专家展开张量或反向重建工作区过大时，该方式提供一个显式的显存与时间权衡选项。它保留完整一阶梯度，不截断 chunk 间的模型反向传播。

## 使用方式

```python
import torch
from hyper_parallel.components.modules.moe import MoE

moe = MoE(
    dim=128,
    hidden_dim=256,
    num_experts=8,
    top_k=2,
    score_before_experts=False,
    use_grouped_mm=True,
    chunk_size=2048,
).to("npu", dtype=torch.bfloat16)

optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)
x = torch.randn(1, 8192, 128, device="npu", dtype=torch.bfloat16)
loss = moe(x).float().square().mean()
loss.backward()
optimizer.step()
```

须先正确安装并初始化 PyTorch、torch-npu、CANN 和 Triton-Ascend。CPU 可用于参考执行，此时关闭 `use_grouped_mm` 并使用普通浮点张量。

已有 `MoE` 可设置 `moe.chunk_size = C`。`None` 关闭分块；`C` 必须是正整数，表示将 `[batch, sequence]` 展平后的源 token 数。末块可以短于 C。该配置不创建或替换模型参数，不改变 state_dict 的键名和参数布局；配置本身应由训练配置保存。`eval()` 使用原整段执行路径。

[完整示例](../../examples/torch/chunk_moe/train.py)支持单进程 CPU/NPU 和一个 EP 域的 NPU 训练：

```bash
python examples/torch/chunk_moe/train.py
python examples/torch/chunk_moe/train.py --device npu
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  examples/torch/chunk_moe/train.py --device npu --chunk-size 64 --tokens 129
```

EP 沿用 `ExpertParallel().apply(moe.experts, ep_mesh)`，在创建 optimizer 前完成原有并行分片。启用分块本身保持分片后的参数对象不变。DeepSeek-V3 的显式模型接入见下文；本接口不会自动修改其他 packed MoE 模块或启用 `HP_EP_DISPATCH_CHUNKS`。

## 有界通信覆盖

`chunk_overlap` 默认是 `"none"`，保留同步行为；可在创建模块时传入，或与 `chunk_size` 一起设置：

```python
moe.chunk_size = 2048
moe.chunk_overlap = "both"
```

| 值 | 执行方式 | 新增暂存 |
| --- | --- | --- |
| `none` | 同步逐块 | 无额外流水槽 |
| `forward` | 两块一组，第二块 dispatch 与第一块专家计算、第一块 combine 与第二块专家计算交错 | 当前 pair 的输入/输出收发缓冲 |
| `backward` | 当前块 p/G 已就绪后，发起下一逆序块 X，覆盖当前 FC1/SwiGLU 重建及专家梯度计算 | 下一块 X 的一个发送/接收槽 |
| `both` | 同时启用上述前向和反向调度 | 两阶段分别计费，不能直接相加单项收益 |

该选项需要启用 `chunk_size`。所有 EP rank 必须使用同一个调度，进入 payload 前会校验；不要按每个rank的本地负载独立切换。奇数块、短尾和空接收仍遵守相同集合通信顺序。每次调用独立持有原路由计划，反向调度在该次前向时确定；已有待反向调用不受后续配置切换影响。

覆盖不增加从前向保存到反向的高维专家激活，但会增加当前层工作区。按实际接收量计入send、raw receive、重排后的receive及共存专家临时量；rank负载不均时不能只按平均kC估算。未来X可能延迟当前dX，通信与GMM也可能争用资源。因此建议保留同步选项并在完整训练步上选择，不保证覆盖始终更快。

使用已有EP通信后端；没有新增跨层队列、专家迁移或独立路由流水。前向pair的两块路由/元数据先准备好，不包含“dispatch0覆盖router1”。CPU只验证数学与生命周期，不代表设备并行。

示例可添加 `--chunk-overlap both`，DeepSeek配置替换器接受同名参数。父模块FSDP边界及外层checkpoint限制与同步模式相同。

## 前向保存与反向执行

1. 每个源 token 块运行原 router、dispatch、grouped SwiGLU、combine 和可选 shared expert；默认同步，可选上述有界调度。
2. 保留本次输入引用与原始 expert IDs、排列、counts、splits，完整调用结束时只提交一次原 token 统计。
3. 外层 `autograd.Function` 在反向逆序消费这些计划。重建可微 router 分数和 shared 分支，使用原 IDs，不重新做 top-k 决策。
4. NPU 使用最小 SwiGLU VJP，得到输入、路由概率和全部专家参数梯度；专家梯度在当前反向私有 FP32 缓冲累加，最后交给原参数。CPU 使用逐块 autograd 参考。
5. 当前块计算完后释放高维临时量。每次 forward 的计划独立，因此支持正常 microbatch 梯度累积；同一 forward 只消费一次 backward。

最小 VJP 利用 `U = G @ W2` 同时计算 `dp = sum(U * A)` 和 `dA = p * U`，避免为 dp 再次重建 FC2 输出。该等价变换改变 BF16 舍入顺序，数值验收使用完整梯度误差，不能承诺与整段实现逐位一致。

## 统计与梯度归约

可使用原 `TokenChoiceTopKRouter` 的 sigmoid/softmax、group-limited 选择、expert bias、route scale，及原 `FeedForward` shared expert。统计只计原前向，反向复用计划不会重复累加 `tokens_per_expert`。

aux loss 使用完整调用的计数与分数和；若提供 `sequence_partition_group`，按原 MoE 语义在该域累计计数并计算全局归一化。`MoEAuxLossAutoScaler` 的梯度缩放继续生效，`last_aux_loss` 与输出上的 `_load_balance_loss` 用于日志，无需再次加进 loss。bias 更新仍由调用者按训练步骤完成。

执行器只交付本 rank 的梯度，不自行决定训练器的 DP/CP 归约策略：

- router、shared 和其他共享参数按其数据副本域归约。
- 专家参数在同一 EP 域内已经收集了各来源的 token 贡献；只在相同专家的副本域归约，不能再跨不同专家 owner 求和。
- 示例采用各 rank 独立源序列、一个 EP 域、loss 除以 world size，非专家梯度 SUM；其他拓扑须按其 loss 归一化定义配置。

## 显存与性能边界

设本地 token 数为 N，源块大小为 C，top-k 为 k，EP size 为 P，隐藏宽度为 H，专家中间宽度为 I。当前 owner 的实际接收量 R 由各源 rank 的路由共同决定。均衡时 R 通常接近 kC，安全粗上界为 PkC。

高维工作区主要随当前块的 `R×H` 和 `R×I` 变化。固定 C 不能保证 owner 接收量恒定，也不能消除持续负载倾斜。完整输入/输出、层间残差和低维路由计划仍随 N 或 Nk 增长；参数、optimizer、梯度及临时打包权重另占显存。因此，这不是“整个训练显存与序列长度无关”的实现。

层数也影响峰值。下表只比较每层前向后仍存活的主要 MoE 张量，不含各方式共有的参数、optimizer、残差和 attention 激活。R_total 表示该 owner 在整层的接收路由数，区别于上文当前块的 R。

| 执行边界 | 跨层保留的主要张量 | 当前反向工作区 |
| --- | --- | --- |
| 仅 experts checkpoint，dispatch 在外 | 接收后的 `R_total×H` 专家输入，以及边界外的路由/合并计算图 | 整段专家重建 |
| 完整 MoE checkpoint | `N×H` 层输入；普通 Python 属性持有的 dispatch 元数据并不会自动被 checkpoint 清除 | 整段 MoE 重建，包含展开及专家中间量 |
| 当前 ChunkMoE | `N×H` 层输入和整层低维路由计划 | 当前源块对应的展开及专家中间量，加当前层参数梯度累加缓冲 |

因此，固定 C 只约束当前块的高维工作集，不使模型总峰值与层数无关。当前 EP 路由计划包含三份长度 Nk 的 int64 源端索引、两份总长度 R_total 的 int64 接收映射，以及每块本地专家 counts。跨层保存张量须按实际存储去重；`save_for_backward` 中的参数引用不是额外参数副本。完整 MoE checkpoint 与分块的显存先后关系还取决于层数、checkpoint 边界及峰值发生阶段，不能只由单层或四层测量推断。

较小 C 通常降低专家工作区，但增加 GEMM、排列和通信调用次数。默认使用同步通信；启用有界覆盖后，需将新增缓冲和并发竞争一起计入。请在真实模型和训练步骤上比较时间、allocated 与 reserved 峰值，不将模块显存变化直接换算为整网收益。没有通用最优 C。

## 首版支持范围

| 项目 | 支持与限制 |
| --- | --- |
| 模块 | 原 common MoE、GroupedExperts；router 可提供模型语义兼容的 scores_for_selected；可选 FeedForward shared |
| 数值 | CPU 参考；NPU BF16、grouped SwiGLU、score-after、无 autocast |
| 专家结构 | 原无 bias SwiGLU 参数；不支持量化或任意自定义专家 |
| EP | 本地参数或现有一维专家 `Shard(0)`；专家均匀分配到 rank |
| 输入 | 通信域内相同本地 token 数、C、E、k及覆盖模式；支持短尾和空接收 owner；不支持零长度本地输入 |
| 反向 | 完整一阶梯度；冻结参数、正常 microbatch 累积；不支持高阶梯度和同图重复 backward |
| checkpoint | 可 checkpoint 其他独立子模块；不要在分块 MoE 或包含它的整个 decoder 外再包 checkpoint |
| FSDP | 可管理整个 MoE 或其父模块，覆盖前向和显式反向；禁止单独管理被执行器绕过的 experts/router/shared 子模块 |
| 其他包装 | TP、外部 main_grad、torch.compile/CUDA graph 尚未支持 |
| 静态计划 | 禁止 `HP_EP_STATIC_SPLITS=1`，不同 chunk 必须持有自己的真实通信计划 |
| 验证范围 | 单节点 Ascend 910B；单卡、EP4、EP8、EP2 两副本、父模块 FSDP2×EP2；不据此承诺跨节点性能 |

输入合同检查涉及小型集体通信和同步。所有参与 rank 必须启用相同执行策略，并以相同顺序进入本层；混合不同策略的 rank 不能依靠局部报错防止所有等待。

## DeepSeek-V3 模型入口

[模型适配器](../../hyper_parallel/models/deepseek_v3/adapter/conversion/chunked_moe.py)复用 common experts 和分块执行器，并保留 HF DeepSeek-V3 的 FP32 sigmoid gate、bias 修正后的组内 top2 求和、选中概率归一化、route scale 和 shared experts。直接调整 common router 配置不能保证这些语义相同。

替换必须发生在创建 optimizer 之前：源 batched `gate_up_proj` 拆成 w1/w3 参数，down/router/shared 复用源参数；适配器声明可逆权重转换。分块开关自身仍不替换参数。

[训练入口](../../examples/torch/chunk_moe/deepseek_v3.py)运行真实 DeepSeek 解码器、因果语言模型 loss 和 AdamW；默认是小型随机模型，`--config` 只读取本地 HF 配置，不下载或加载预训练权重。已验证的 Transformers 版本为 5.5.4，要求 batched SwiGLU 专家与 shared experts。

```bash
python examples/torch/chunk_moe/deepseek_v3.py --steps 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  examples/torch/chunk_moe/deepseek_v3.py --device npu --fsdp-size 2 --chunk-overlap both --steps 3
```

该 FSDP2×EP2 示例先分配 EP 专家，再将整个模型交给 FSDP，开启真实前向后重分片；使用 FP32 参数、BF16 计算、FP32 FSDP 梯度归约及原生 AdamW。dense 梯度另在 EP 域 SUM，专家只在相同 owner 的 FSDP 域归约。loss 除以 EP size，FSDP 按其副本域平均。这里的整模型 FSDP 边界用于功能演示，并非最佳大模型参数驻留策略。

[配置片段](../../hyper_parallel/models/deepseek_v3/recipes/chunk_moe.yaml)通过已有 module replacement 和模型安全重计算区域接线。当前通用 Trainer 的 FSDP2Manager 自动创建独立 experts FSDP 单元；该边界会被分块执行器绕过，因此仍不支持。不要把显式父模块示例通过理解为通用 Trainer 的任意自动包装已通过。

## 外层 checkpoint 的边界

ChunkMoE 已通过 `autograd.Function` 控制逐块重建和反向。若外面再包整个 decoder 的 non-reentrant checkpoint，反向恢复保存张量时会重新进入 decoder 和 MoE；当前 MoE 在循环末尾保存输入/参数，可能因此重跑整次分块前向，包括路由、通信、专家计算和统计。随后仍要运行逐块反向，产生重复工作。一般 checkpoint 的 early-stop 不等于可以自动绕过此 MoE。

reentrant checkpoint 首次前向使用 no_grad，当前训练入口会明确拒绝。无需为这一版扩展任意组合：将重计算放在独立 attention 等子模块，保持 MoE 位于重计算区域之外。

DeepSeek-V3 模型安全区域是 `model.layers.*.self_attn`。只有配置选择 `model_adapter_safe_regions` 才采用该策略；它不会自动纠正用户另外手工包裹的 decoder checkpoint。示例直接对每层 attention 使用 non-reentrant checkpoint。

## 公共接收排列与新增元数据

整段专家执行和 ChunkMoE 共用 EP 的接收排列。接收端从 rank 顺序变为 expert 顺序是一一映射，输出恢复及其反向可使用逆索引 gather，无需高维 scatter 或求和。token 与概率分支复用同一组映射；普通 autograd 路径也采用逆排列反向。

每次 dispatch 新增一份长度为实际接收路由数 `R` 的 int64 逆索引，约 `8R` 字节。ChunkMoE 在前向保存各块映射，反向按块消费后释放，跨层保存量会随总接收路由数增长。此优化不使整个模型显存与序列长度无关，也不改变源端 top-k 多对一归约。

这项优化同时用于整段及分块基线，不改变路由、专家算子或 collective 次序。小块可能由提交和通信固定开销主导；`chunk_size` 仍需结合实际显存预算选择，不能假设分块始终快于专家 checkpoint。可选覆盖使用前向两块和反向一槽，不启用更深队列。

## 必要测试

```bash
pytest -q tests/ut/components/modules/test_chunked_moe.py tests/ut/components/modules/test_moe.py
pytest -q tests/ut/models/deepseek_v3/adapter/test_deepseek_chunked_moe.py
pytest -q tests/torch/expert_parallel/test_chunked_moe.py
```

系统测试 launcher 不初始化计算后端；实际测试由 torchrun worker 执行。覆盖原 route、完整输出/输入/参数梯度、shared/aux、空接收、短尾及 EP2 专家副本归约。没有以固定均衡路由代替正常路由测试。

## 可选的 NPU 接收索引构造

普通 EP 和分块路径共用接收排列。已知接收行数至少为 8192、整数输入连续且
Triton-Ascend 后端可导入时，直接生成排列，省去展开偏移量的中间张量。
门槛按实际接收量判断，不按 EP 编号选择。小接收量、短尾、未知接收长度、
非 NPU 及缺少可选后端时继续使用 Torch 实现；编译或执行错误会正常报告。
单分段本身是恒等排列，直接生成行号。这只描述排列构造的回退；
NPU 专家最小 VJP 的既有 Triton 依赖不因此消失。

这项优化同时作用于普通执行和分块执行，不改变路由、通信顺序、反向保存边界
或流水深度。当前门槛来自 Ascend 910B / CANN 9.0 / Triton-Ascend 3.2 的测量，
不是跨硬件的最优分界；更换后端或形状后仍应比较完整训练步。
