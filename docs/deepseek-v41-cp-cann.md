# DeepSeek V4.1 Flash：CP 与融合算子适配

本实现保留当前 V4.1 的 Full/Reindex/Reuse 和 KL teacher 定义。
本轮聚焦预训练，不优化 candidate pool；保留其原有行为。减层模型验收显式设置 `v41_candidate_source_layer_id=-1`。
主 attention 继续使用 Omni sparse attention；普通 Indexer TopK 和满足条件的 KL 接入 ops-transformer。
CPU、未安装新算子包以及不满足能力约束的输入使用参考路径；已选择的 kernel 执行异常会直接上报。

修改原因、减层模型实测收益和精度结果见 [V4.1 分支 README](../README_V41.md)。

## 执行路径

| 阶段 | 实现与边界 |
|---|---|
| packed 准备 | runtime/模型入口创建边界快照；多层和同一 prepared batch 的重复 forward 复用 positions、样本起点、压缩下界、SWA 索引和 Indexer 分段 |
| SWA raw KV | 连续、等长 CP 分片默认用 `all_to_all_single` 的可变 split 交换左 halo；不需要新增 YAML 开关 |
| main KV / Indexer K | 仍按 source 全局 all-gather；同步和异步 AG 的 backward 都用 SUM reduce-scatter，保持输入 dtype，无额外 CP 平均 |
| main attention | Omni `npu_sparse_flash_attention_enhance`，使用修复后的 scalar sink bridge |
| 普通 Indexer TopK | CANN LightningIndexerV2；r1/r2、packed 和非末 CP rank 的 Q/K 前缀坐标显式适配 |
| candidate 生成 / restricted Reindex | 保留参考实现；普通 TopK 不能替代 block-max candidate 算法 |
| KL teacher | Q 与 selected KV 分块的小算子，保持每 head 的 selected main+sink softmax、跨 head 求和后归一化并 detach |
| KL student | 完整选集行用 SparseLightningIndexerKLLossGrad；partial/empty 行使用修正后的参考实现 |

halo 仅交换当前 rank 需要的左侧 `window-1` 行，支持窗口跨多个 owner。异步发送与后续局部计算重叠，消费前 `wait()`；handle 保留发送 storage。反向执行逆向 all-to-all-v 并 SUM 到 owner；rank 0 即使没有接收数据，也保留反向通信边。

prepared 对象是只读几何快照。原始边界通过正常原地操作修改后，版本计数使下一次 `prepare()` 重建；旧消费者继续持有旧快照。不同 batch/CP 区间使用不同对象。禁止绕过版本计数用 `.data` 或外部共享 storage 修改边界；无版本计数的 inference tensor 每次重新快照。KV、TopK 和带梯度的共享状态均不跨 forward 缓存。910B 上，当前 LI 与 NoMask KL 的调用不传 `seqused_q/k`，使用 host tiling，省略可选 metadata。其它设备保留生成；未来带设备端动态长度的接口不能复用这个省略条件。不能将此实现描述为消除了所有 forward 准备开销。

## 数值修复

Omni bridge 将 sink 质量分配到合法、value 为零的虚拟条目，通过额外 score 坐标表达 sink，而不从 softmax 总质量中减去 dummy 概率。FP32 sink 使用 BF16 分量和小残差校正；物理 KV 容量大于选集宽度，避免短 K 快捷路径改变 sparse 地址语义。独立 oracle 覆盖负 logits、padding、空选集以及 dSink。

参考 KL 的 teacher/student 计算显式关闭外层 autocast，使用 FP32。令归一化后的 teacher 为 `p`、student 为 `y`，手写 score 梯度为：

```text
L = coeff / (B * Q) * sum(p * (log(p) - log(y)))
dL/dscore = coeff / (B * Q) * (sum(p) * y - p)
```

正常行 `sum(p)=1`；teacher 全部下溢为零时，该式返回零梯度，与标量 loss 一致。所有实现继续应用 autograd 上游 seed；teacher 和 attention/sink 不从此辅助目标接收梯度。

当前 CANN KL `NoMask` 对 `-1` 槽位采用零 logit，未从 softmax 分母排除。适配器仅将全有效行交给该模式，partial/empty 行回退，并按各子集 query 数加权还原全局 `B*Q` 分母。`cmp_ratio=1` 表示在显式选集上计算 KL；并未把模型 r2 改成 r1，r2 因果已由 Indexer 选集表达。融合算子返回的概率若下溢，仅为标量 loss 对对应行重算稳定 `log_softmax`，不随意 clamp 正 teacher 对应的概率。

LI 首版要求 NPU BF16/FP16、D128、head 数 1–64、ratio 1–128、TopK 1–2048 或不超过 8192 的 1024 整数倍；KL 额外要求 head 数为 8/16/32/64、TopK 不超过 key 数、key 数不超过 524288、batch 不超过 256。KL 超过 8192 个 Q 时分调用。TP 的 head reduction 以及不符合这些条件的输入保留参考实现；本次实际 NPU 验证以 BF16 为主。

### 有界临时张量

普通 TopK 选集是离散结果，参考选集路径关闭无用 autograd 记录；不对 Indexer K 分块。
需要规范化顺序的 key ID，仅在包含 sentinel 的最大值不超过 `2**24` 时转 FP32 排序，之后恢复整数 dtype；超界保留整数排序。

teacher 的 FP32 bank 转换同时受 64 MiB 上限和 gather 临时张量大小约束，长序列不无条件转换整个 bank。
teacher 按 selected KV 和 head 切块，完整保留各 head 的选集 softmax 分母；参考 student KL 及时释放上一 Q chunk 的大临时张量。
64 MiB 是分块和 bank 转换的选择预算，不是整模型显存上限，也不保证多个同时存活张量之和小于该值。

## 算子依赖

实测环境：Ascend 910B3、PyTorch / torch-npu 2.10.0、CANN 9.0.0。
ops-transformer 源码固定为 `0b08075141913daeaf1e856faf7480ed393c8949`；不能只根据 Python wheel 的 `1.0.0` 判断是否支持这些接口。

所需 kernel 与 metadata：

```text
lightning_indexer_v2
lightning_indexer_v2_metadata
sparse_lightning_indexer_kl_loss_grad
sparse_lightning_indexer_kl_loss_grad_metadata
```

本 PR 不要求构建 SparseFlashMla/SparseFlashMlaGrad，也不替换现有 Omni 包。Python 适配器延迟导入标准包 `cann_ops_transformer` 或官方定向构建包名 `cann_ops_transformer_custom`，不依赖本机 vendor 名称。必须同时安装匹配的 kernel vendor，加载 vendor 环境，并保留 Omni 的库路径。

该源码版本的定向 Python 打包需要两处修复，见 [配套包装补丁](patches/ops-transformer-selected-wrapper.patch)：选中模块导出名可能与目录名不同，builder 也需要相对导入以支持定向包名。只修改 Python 包装，不修改 kernel 算法。在 ops-transformer checkout 应用补丁后可定向构建：

```bash
git apply /path/to/hyper-parallel/docs/patches/ops-transformer-selected-wrapper.patch
bash build.sh --pkg --soc=ascend910b \
  --ops=lightning_indexer_v2,lightning_indexer_v2_metadata,sparse_lightning_indexer_kl_loss_grad,sparse_lightning_indexer_kl_loss_grad_metadata \
  -j128
TORCH_EXTENSION_VENDOR= \
TORCH_EXTENSION_OPS=lightning_indexer_v2,sparse_lightning_indexer_kl_loss_grad \
  python -m pip wheel --no-deps --no-build-isolation ./torch_extension --wheel-dir ./wheels
```

先完成该版本 ops-transformer 要求的 CANN/构建依赖配置，再安装生成的 vendor 和 wheel。以上是最小构建范围；验证机器使用此前定向构建的四类 kernel vendor，其中额外两类 V4 attention kernel 不参与本实现。

当前组合环境观察到 custom AICPU 与设备网络服务初始化顺序依赖：先运行 metadata，再首次打开 HCCL 所用网络服务，后续 custom AICPU 可能报参数错误。最小单卡复现不需要 Indexer 计算或分布式通信，空 AICPU 入口也失败，因此没有证据表明是 LI V2 数学 kernel 的缺陷。
910B 上本适配器省略已验证可选的 LI / NoMask KL metadata，避免这两个 AICPU 调用；这没有修复 runtime/driver 的通用初始化问题。其它设备仍保留 metadata。分布式测试用一次 `dist.barrier()` 实际完成 HCCL 初始化，不在每层插入 barrier；多 group/FSDP 生产组合尚需验收。

带 `seqused_q/k` 的 arch22 KL 必须提供 metadata，不能由上述 NoMask 结论推广。本版不纳入仍在实验的 causal partial KL；partial/empty 行保持参考计算。

## 验证入口

从已 editable 安装的 HyperParallel checkout 执行；NPU 测试前配置匹配的 CANN、Omni 与新算子包。薄 launcher 不在收集阶段导入 Torch。

```bash
python -m pytest -q tests/ut/components/functional/test_compressed_indexer.py \
  tests/ut/components/modules/test_shared_compressed_dsa_attention.py \
  tests/ut/core/context_parallel/test_sequence_halo.py
python -m pytest -q tests/ut/auto_models/models/deepseek_v41
python -m pytest -q tests/torch/context_parallel/test_v41_cann.py
python -m pytest -q tests/torch/context_parallel/test_sequence_halo.py -k gloo
python -m pytest -q tests/torch/context_parallel/test_sequence_halo.py -k hccl
```

CP8 原语测试覆盖零收发、多 owner、非连续 tensor、多 sequence 轴、FP32/BF16、远端 loss 和重复梯度 SUM。共享链覆盖 r0/r1/r2、mixed、Full/Reindex/Reuse、多 source、packed。Qwen3-MoE 覆盖 CP1/旧 AR/新 RS、FP16/BF16、GQA 8:2、S1024/2048。

共享链的 CPU 梯度以 CP1 为 oracle。NPU 要求 CP1 输出与 CP8 输出、AG 与 halo 的输出及梯度逐位一致；CP1 对 CP8 的 BF16 参数梯度另报范数误差，不把它写成逐元素通过。原因是 CP8 先舍入局部 GEMM 再归约，与 CP1 的全序列 GEMM 不同；同输入旧 AR 路径也在近零梯度处不满足 dtype 默认逐元素检查。公共 RS 的正确性还由原语梯度和 Qwen 旧 AR/新 RS 回归检查，未用自定 3% 门槛替代。

最终模型测试为六层、59,390,788 参数、D512、index D128/heads32、TopK512、S4096、CP8、三步带 momentum 的 SGD；关闭 candidate 生成及 restricted Reindex，保留普通 Full/Reindex/Reuse，并轮换 packed 边界。使用确定性模式、相同初值和显式 aux seed `1/CP`，比较 AG 参考、参考重复、halo 参考、halo 融合及融合重复。记录实际 kernel 调用次数和分组梯度误差，并使用 `torch.testing.assert_close` 的 dtype 默认容差检查模型输出、loss、梯度、最终参数；不引入统一“3%”门槛。

验证范围包括模块级与减层模型训练，不等于完整 Trainer/FSDP/TP/GAS 组合验收。测试各方案统一使用可用的 reference mHC post；BF16 SGD 中极小的 Indexer 梯度不一定造成可见参数更新，因此“最终参数一致”不能代替 Indexer 梯度测试或长程收敛验证。

## 保留事项

- GAS 与 token weighting 的辅助梯度缩放仍按当前 Trainer 合同处理，本 PR 未修复这项独立问题。
- candidate/restricted Reindex 和 TP Indexer/KL 未全融合。
- partial 行拆分有额外同步、gather 和调用开销，不保证所有选集形态都有耗时收益。
- teacher 仍由分块小算子计算；SMLAG 默认 full-SWA teacher 与当前目标不同，不能直接替换。
- 未做大规模训练与长程收敛测试；不声明正式大模型或完整 Trainer 训练的加速比例。
