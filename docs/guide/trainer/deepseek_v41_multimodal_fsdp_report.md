# DeepSeek-V4.1 多模态 FSDP Wrapper 分析与验证报告

## 结论

原有 FSDP wrapper 对标准 Hugging Face 文本 decoder 基本合理，但不适合作为多模态模型的通用发现机制。
它只根据 `gradient_checkpointing` 容器和子模块树推断 transformer block，无法可靠区分 decoder、ViT、
aligner 等不同执行分支，也无法表达模型真实的 forward 顺序和同一逻辑模块内的混合 mesh 所有权。

本次优化后，框架保留原有 decoder block 与 routed-expert 自动发现，同时允许模型 adapter 声明：

- 额外 FSDP child unit；
- 不应被当成 decoder 容器分析的子树；
- 所有 child unit 的真实 first-forward 顺序。

DeepSeek-V4.1 adapter 负责提供视觉、Engram 等模型结构语义，通用 FSDP manager 只做解析、合法性校验、
mesh 选择、由深到浅的嵌套包装和 prefetch 配置。这个边界适合继续扩展视觉、音频等非 decoder 分支，
无需再在框架中增加 DeepSeek 类名判断。

当前结论是：**wrapper 的结构扩展性已经达到多模态训练验证所需水平；正式全量模型的显存与吞吐仍需专项
profile，不能用本次裁剪模型结果代替全量模型性能验收。**

## 原实现的问题

### 1. 仅靠 `gradient_checkpointing` 猜测模块边界

旧逻辑遍历带 `gradient_checkpointing` 属性的模块，再把其有子节点的 child 当成 block 容器。文本模型通常有
规则的 `model.layers`，因此能够工作；加入 `model.vision` 和 `model.aligner` 后，这个启发式会把视觉层级误认为
decoder 层级，或者使视觉参数落到过大的 root 单元。

实际 16 卡调试中出现过视觉 RMSNorm 激活维度为 1024，而参数仍是 FSDP local shard 维度 64 的错误。这不是
视觉算法问题，而是视觉参数没有在其 forward 边界前完成正确 unshard。

### 2. 整座 ViT 作为单一边界不适合正式模型

把整个 `model.vision` 声明为一个 FSDP 单元可以让单层裁剪模型运行，但正式 V4.1 有 32 个 vision blocks。
整塔边界会在一次 forward 前 all-gather 全部 ViT 参数，放大峰值显存并降低通信与计算重叠空间。

当前 adapter 改为逐 `model.vision.blocks.N` 包装；patch embedding 和末端 norm 体积较小，由 root 管理；
参数较大的 aligner 单独包装。裁剪 smoke 因此使用一个 vision-block 单元，完整模型会自然生成 32 个单元。

### 3. 模块注册顺序不等于多模态执行顺序

裁剪模型的 Python 注册顺序是 decoder 后注册 vision，但 forward 先执行 vision/aligner，再进入 decoder；
Engram 又是在 decoder layer 构建后追加，而实际在该层 attention/MLP 之前执行。按 `named_modules()` 配置
prefetch 会产生无效或过早的 all-gather。

`ModelAdapterSpec.fsdp_execution_order` 现在由模型 adapter 返回完整 child-unit 排列，框架要求它与发现结果
严格一一对应，不允许漏项、重复或未知 FQN。V4.1 的验证顺序是：

```text
vision blocks -> aligner
-> layer 0 -> layer 0 experts
-> layer 1 -> layer 1 Engram -> Engram wkv -> layer 1 experts
-> layer 2 -> layer 2 experts
-> layer 3 -> layer 3 experts
```

### 4. Engram 表会污染父 decoder 的 mesh 选择

Engram `embed.weight` 在 EP 轴按行切分，但原有 expert-unit 发现只识别 FQN 中的 `.experts.`。如果不单独建立
边界，含有 Engram 的整个 decoder layer 会同时看到 dense source metadata 和 expert source metadata；只要
存在一个 expert-mesh 参数，整层就可能选择 `edp_shard` mesh。在本配置 `EP=16, edp_shard=1` 下，这会让该层
大量稠密参数失去 FSDP16 优化器分片。

直接包装 `engram.embed` 也不正确。Engram 的稀疏路由使用 `F.embedding(local_ids, self.embed.weight)`，不会调用
`nn.Embedding.forward`，所以 child FSDP hook 不会触发。本次 16 卡负向试验准确复现了 DTensor weight 与本地
Tensor ids 不匹配的异常。

最终边界采用嵌套多 mesh 结构：

- `model.layers.N.engram`：expert mesh，拥有 EP-sharded 表和很小的 q/k 参数，模块 forward hook 可正常触发；
- `model.layers.N.engram.wkv`：嵌套 dense FSDP 单元，较大的 WKV 矩阵与优化器状态继续按 FSDP16 切分；
- `model.layers.N`：排除已经被嵌套单元拥有的参数后，仍使用 dense FSDP16；
- `model.layers.N.mlp.experts`：保持现有 routed-expert mesh。

FSDP 本身允许有意的父子嵌套，且不同层级可选择不同 mesh。框架只禁止同一个模块对象被不同 FQN 重复声明，
包装时按 FQN 深度从深到浅执行。

### 5. 预包装 owner map 与真实 managed parameters 可能漂移

旧 `_build_managed_source_shard_info` 依赖包装前计算的 owner map。深层 child 完成 `fully_shard` 后，其参数会从
父单元的实际 managed set 中移除，旧 map 仍会过度包含这些参数。

当前实现改为每次包装时调用与 `fully_shard` 相同的 `get_managed_modules_parameters((owner,))`，再为精确参数集
构建 source metadata。该修正是视觉、Engram、expert 多层嵌套能同时工作的必要条件。

## 当前 FSDP 参数所有权

本次 recipe 为 `TP=1, CP=1, EP=16, dense FSDP=16, edp_shard=1`。`{tp: replicate}` 只描述进入 FSDP 前的
TP source layout，不表示参数绕过 FSDP。

| 参数区域 | source layout | FSDP 单元与 mesh | 单卡优化器状态 |
| --- | --- | --- | --- |
| Vision block、aligner | TP replicate | 独立 dense FSDP16 | 16 路切分 |
| Decoder dense 参数、mHC、DSA | TP replicate | decoder/root dense FSDP16 | 16 路切分 |
| Engram WKV | TP replicate | 嵌套 dense FSDP16 | 16 路切分 |
| Engram q/k | TP replicate | Engram expert parent | EP rank 间复制；体积很小 |
| Engram embedding 表 | EP shard(0) | Engram expert parent，edp=1 | 每卡只持有自己的 EP 表分片 |
| Routed experts | EP shard | `.experts` expert unit，edp=1 | 每卡只持有本地 expert 分片 |

因此，EP 表和 routed experts 不再额外做 dense FSDP16 是正确的：它们已经通过 EP16 切分。需要避免的是让一个
EP 参数把整个 dense decoder layer 拉到 edp=1；本次嵌套边界正是对此的修复。

## 多模态端到端验证

### 输入与环境

- 模型配置：`/home/ma-user/work/y00512198/DeepSeek-V4.1-Flash/config.json`
- 配置 SHA256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- Transformers：5.13.0，按 `current_hf_model_environment.md` 准备
- 裁剪：decoder 40 -> 4 层，vision 32 -> 1 层，routed experts 384 -> 16
- 保留：hidden size、head、patch、aligner、top-k、词表和共享压缩注意力维度
- 数据：Online OpenAI messages JSONL，512 条训练样本、128 条验证样本，来自本地导出的 ChartQA/DocVQA 图片
- 训练 JSONL SHA256：`4728da3cc64ce744b6921f368cd306d87703482dd8620471e455f536bbc85a0f`
- 序列长度：4096；global batch 16；micro batch 1

### 最终结果

执行入口：

```bash
RUN_NAME=vlm_tp1_ep16_100steps \
    bash examples/training_demo/run_deepseek_v41_vlm_online.sh \
    /home/ma-user/work/y00512198/DeepSeek-V4.1-Flash \
    --training.train_iters=100
```

最终日志与 marker：

- `output/training_demo/deepseek_v41/run_vlm_tp1_ep16_100steps.log`
- `output/training_demo/deepseek_v41/vlm_tp1_ep16_100steps.success`

验证结果：

- adapter 声明 4 个额外单元：vision block、aligner、Engram、Engram WKV；
- 合计 12 个 child units + root：4 decoder + 4 routed-expert + 4 adapter units；
- 100 个 forward、backward、gradient clip、optimizer step 和结束 barrier 均完成；
- loss 和 grad norm 在全部 step 中均为有限值；
- 前 10 / 后 10 step 平均 loss 为 `11.193749 / 4.690136`；
- grad norm 范围为 `33.2585 ～ 128.487`；
- 排除首步后的平均 step 时间为 `4.9602s`；
- `device_max_allocated_gb=38.8125`；
- `device_max_reserved_gb=48.502`。

本次随机初始化裁剪模型可作为 Online 多步训练和数值稳定性 smoke，但不能代替正式 checkpoint 的精度对齐
或完整 40 层模型的性能结论。日志中的整数 `ArgSort` AiCPU 提示来自现有通用 EP dispatcher 路径，不是视觉
FSDP wrapper。完整曲线和结果见 `deepseek_v41_flash_hyperparallel_100step_report.md`。

单元与聚焦回归共 36 项通过，覆盖视觉梯度、VLM packed batch、DeepSeek 配置/骨架和 adapter FSDP 发现、嵌套
边界、执行顺序。

## 仍需完成的正式训练验证

1. 用完整 32 层 vision tower 和正式 checkpoint 运行，不再使用随机初始化裁剪模型。
2. 做多步 warmup/profile，采集 FSDP all-gather/reduce-scatter 与 vision/decoder compute overlap timeline。
3. 比较 vision-block 粒度、block 分组粒度以及 prefetch depth，确定完整模型的峰值显存和吞吐最优点。
4. 对纯文本 batch、单图 batch、多图 batch分别验证。一个视觉 block 在多图循环中会被重复调用，静态
   first-forward 顺序只能描述第一次调用，后续可考虑按相同 grid 分组/批量化视觉编码。
5. 当前 VLM `get_batch` 明确限制 `TP=CP=PP=1`；需要单独接入并验证多模态 TP2、CP 和 PP stage 数据传递。
6. 按离线固定样本、参考实现对齐、有限 loss/grad 的精度流程完成正式 numerical acceptance。
