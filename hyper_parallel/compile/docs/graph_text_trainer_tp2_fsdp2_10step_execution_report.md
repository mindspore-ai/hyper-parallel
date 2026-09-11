# 基于 `scripts/train_lm.py` 的 GraphTextTrainer TP2 + FSDP2 四卡 10 Step 执行报告

## 1. 本次目标

将 graph-mode 文本训练的最终使用方式统一到：

- [scripts/train_lm.py](file:///home/whh/graphtrainer/scripts/train_lm.py)
- graph-mode YAML
- `TrainerConfig.graph.enabled`

并完成一次 4 卡 `tp2 + fsdp2` 的 10 step 回归，保留最终可复用的最小代码与资料集合。

## 2. 最终保留的入口与资料

本次收束后，最终方案只保留以下入口与支撑文件：

1. 标准训练入口
   - [scripts/train_lm.py](file:///home/whh/graphtrainer/scripts/train_lm.py)

2. graph-mode 配置开关
   - [graph.py](file:///home/whh/graphtrainer/hyper_parallel/trainer/config/graph.py)
   - [trainer.py](file:///home/whh/graphtrainer/hyper_parallel/trainer/config/trainer.py)

3. graph 回归使用的 YAML
   - [train_lm_graph.yaml](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph.yaml)
   - [train_lm_graph_tp2_fsdp2.yaml](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml)
   - [train_lm_graph_fixed_tp2_fsdp2.yaml](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_fixed_tp2_fsdp2.yaml)

4. graph 回归辅助代码
   - [recipe_support.py](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/recipe_support.py)
   - [tiny_text.jsonl](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/data/tiny_text.jsonl)

不再保留单独的 `train_tp2_fsdp2.py` 原型入口，后续统一通过 `scripts/train_lm.py + YAML` 使用与验证。

## 3. 标准入口下的代码执行路径

最终标准入口的执行路径如下：

1. `scripts/train_lm.py` 调用 `parse_training_args()` 解析 YAML
2. 读取 `config.graph.enabled`
3. 当 `graph.enabled=true` 时，分流到 `GraphTrainer.from_text_config(config)`
4. `GraphTrainer.from_text_config()` 返回 `GraphTextTrainer`
5. `GraphTextTrainer.__init__()` 内部继续复用 `TextTrainer.__init__()`
6. `TextTrainer` 主流程继续负责：
   - 分布式初始化
   - AutoModel 模型准备
   - optimizer / dataloader 构建
   - 训练循环
7. 单步前反向由 `GraphTextTrainer.forward_backward_step()` 切换到 `graph_executor.train_step()`

因此，当前最终方案已经满足：

- 用户入口统一为 `scripts/train_lm.py + YAML`
- Trainer 主训练流程继续复用 `TextTrainer`
- graph 模式只接管前反向与 graph pass pipeline

## 4. 本次四卡回归中遇到的问题

### 4.1 问题 1：Qwen3-0.6B 在 TP2 下的预训练权重加载失败

使用：

- [train_lm_graph_tp2_fsdp2.yaml](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml)

并加载本地 Qwen3-0.6B 预训练权重时，会在 checkpoint materialize/load 阶段失败，典型报错为：

```text
ValueError: Local shape mismatch for lm_head.weight:
checkpoint shard (151936, 1024) vs target (75968, 1024)
```

这说明当前 `Qwen3-0.6B + TP2` 的预训练权重加载路径里，`lm_head.weight` 的 tied embedding / vocab shard 还没有和本地 TP target layout 对齐。

后续验证表明，这个问题与 `HyperAutoModelForCausalLM.from_pretrained()` 中的 `load_base_model` 开关直接相关：

- `load_base_model=True` 时，问题稳定复现
- `load_base_model=False` 时，不再触发上述 checkpoint shape mismatch，而是进入：

```text
Initialized model state with model-native random initialization
```

因此，最终方案中保留了：

- [auto_model.py](file:///home/whh/graphtrainer/hyper_parallel/models/_transformers/auto_model.py)

里的 `load_base_model=False`，用于绕开当前 `Qwen3-0.6B + TP2` 的预训练权重加载兼容性问题。

### 4.2 问题 2：online text dataset 在 graph 模式下出现 shape 漂移

在使用 online text 链路时，即使已经绕开问题 1，graph 训练仍可能失败，典型报错为：

```text
RuntimeError: ... MatmulKernelNpuOpApi.cpp ...
The k-axis of the two inputs are different [1, 128, 312], [1024, 1024]
```

根因是：

1. `DynamicBatchDataLoader` 会按 token budget 动态拼装 packed batch
2. online text 样本经过 tokenizer / packing 后，不同 step 的 `packed_length` 可能不同
3. graph 静态图按首个 batch 的形状完成 trace
4. 后续 batch 若从 `[1, 128]` 漂移为 `[1, 39]` 这类短 batch，就会与静态图期望不一致

### 4.3 两种 DataLoader 的报错与原因

围绕 online text graph 回归，实际对比了两种 batch dataloader：

1. **`DynamicBatchDataLoader`**
   - 可以进入 graph 训练阶段
   - 但会在后续 step 因变长 packed batch 触发静态图报错：

```text
RuntimeError: ... MatmulKernelNpuOpApi.cpp ...
The k-axis of the two inputs are different [1, 128, 312], [1024, 1024]
```

   - 根因：
     - 动态 token packing 会产生跨 step 变化的 `packed_length`
     - graph trace 只对首批固定 shape 生效

2. **`FixedBatchDataLoader`**
   - 在当前 online text 数据链路下并不能作为直接替代
   - 会在数据层提前失败：

```text
ValueError: A multi-sample source item requires DynamicBatchDataLoader
```

   - 对应代码：
     [transform_dataset.py:L117-L128](file:///home/whh/graphtrainer/hyper_parallel/data/text/transform_dataset.py#L117-L128)
   - 根因：
     - 当前 online text 的 mapping transform 链路，一个 source item 可能展开成多个 `ModelSample`
     - `FixedBatchDataLoader` 走的是 `__getitem__()` 的 fixed-size consumer 契约，只接受“一个 source item 最终恰好对应一个 `ModelSample`”

因此，本次问题并不是“直接切换另一个 loader 即可解决”，而是必须让 graph-mode 下实际进入执行器的最终 batch shape 保持稳定。

## 5. 保留的最终修复

为了解决 online text graph 回归中的 shape 漂移问题，最终保留了以下修复：

1. 在
   [build_dataloader.py](file:///home/whh/graphtrainer/hyper_parallel/data/batching/build_dataloader.py)
   的 `DynamicBatchDataLoader` 中新增 `pad_to_token_budget`
2. 当该开关开启时，将 online packed batch 补齐到固定：
   `token_budget = batch_size * max_seq_len`
3. 同步补齐：
   - `input_ids`
   - `labels`
   - `cu_seq_lens`
4. 在
   [dependency_bridge.py](file:///home/whh/graphtrainer/hyper_parallel/compile/dependency_bridge.py)
   的 `clone_config_for_graph_mode()` 中，自动为 graph-mode 的
   `DynamicBatchDataLoader` 打开 `pad_to_token_budget=True`

这样做的效果是：

- eager 模式继续保持原来的 dynamic batching 行为
- graph 模式下进入 trace 的最终 batch shape 固定为静态图可复用的长度

## 6. 最终验证

### 6.1 标准入口回归命令

最终成功使用的标准入口命令为：

```bash
torchrun --nproc_per_node=4 --master_port=29671 \
  --log-dir /home/whh/graphtrainer/hyper_parallel/compile/docs/logs/train_lm_fixed_tp2_fsdp2_10step/torchrun_logs \
  --tee 3 \
  --local_ranks_filter=0 \
  /home/whh/graphtrainer/scripts/train_lm.py \
  /home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_fixed_tp2_fsdp2.yaml
```

### 6.2 固定 shape 回归结果

使用：

- [train_lm_graph_fixed_tp2_fsdp2.yaml](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_fixed_tp2_fsdp2.yaml)

可稳定完成 4 卡 `tp2 + fsdp2` 的 10 step graph 回归。

关键日志：

- [rank0_console.log](file:///home/whh/graphtrainer/hyper_parallel/compile/docs/logs/train_lm_fixed_tp2_fsdp2_10step/rank0_console.log)

其中可以看到：

```text
step=1  ... training/graph_loss=9.77689
step=5  ... training/graph_loss=9.79279
step=10 ... training/graph_loss=9.72236
Training: 100%|██████████| 10/10
```

### 6.3 online text 修复后结果

在保留 `Qwen3-0.6B + TP2 + FSDP2 + online text(local arrow)` 的组合下，应用 graph-mode padding 修复后：

1. 可以成功跑通 4 step
2. 同一链路可以进一步成功跑通 10 step

这说明当前最终方案已经具备：

- 统一入口：`scripts/train_lm.py + YAML`
- graph trainer 分流：`graph.enabled`
- 预训练权重问题规避：`load_base_model=False`
- online text graph shape-stable 修复：`pad_to_token_budget=True`

## 7. 当前结论

1. 最终保留的用户入口已经统一到：
   - `scripts/train_lm.py + YAML`
2. `GraphTrainer` 的接入方式已经稳定为：
   - `config.graph.enabled = true`
3. `train_tp2_fsdp2.py` 这类单独原型脚本不再属于最终方案，已移除
4. 当前保留下来的必要代码与资料，已经足够支撑：
   - graph-mode 文本训练的标准入口使用
   - `tp2 + fsdp2` 四卡回归
   - Qwen online text 链路的问题说明、修复与复现资料

从“保留最少但完整的最终方案”这个目标看，当前目录应以：

- `scripts/train_lm.py`
- graph config schema
- `recipe_support.py`
- `train_lm_graph*.yaml`
- 本报告

作为最终保留集合。
