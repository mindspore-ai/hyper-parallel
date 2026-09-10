# GraphTextTrainer -> TextTrainer 路线下 TP2 + FSDP2 10 Step 执行报告

## 1. 本次目标

将 `tp2 + fsdp2` 图模式样例重新对齐到当前的：

- `GraphTrainer.from_text_config()`
- `GraphTextTrainer`
- `TextTrainer`

这一条结构上，并完成一次 4 卡、10 step 的成功回归。同时保存 `rank0` 终端训练日志。

## 2. 代码执行路径与逻辑

### 2.1 样例入口

入口文件：

- [train_tp2_fsdp2.py](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py)

主函数入口在：

- [train_tp2_fsdp2.py:L340-L360](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L340-L360)

当前入口代码是：

```python
trainer = GraphTrainer.from_text_config(
    build_config(),
    pass_plan=create_simple_sharding_plan(),
)
trainer.train()
```

这意味着外部使用方式是 `GraphTrainer`，但实际实例化的是 `GraphTextTrainer`。

### 2.2 GraphTrainer 如何对接 GraphTextTrainer

桥接入口定义在：

- [compile/trainer.py:L313-L328](file:///home/whh/graphtrainer/hyper_parallel/compile/trainer.py#L313-L328)

`GraphTrainer.from_text_config()` 的行为很直接：

1. 接收标准 `TrainerConfig`
2. 延迟导入 `GraphTextTrainer`
3. 返回 `GraphTextTrainer(config, **kwargs)`

因此当前设计里：

- `GraphTrainer` 是统一入口
- `GraphTextTrainer` 是文本图模式运行时

### 2.3 GraphTextTrainer 如何复用原 TextTrainer

`GraphTextTrainer` 定义在：

- [compile/text_trainer.py:L38-L99](file:///home/whh/graphtrainer/hyper_parallel/compile/text_trainer.py#L38-L99)

它直接继承：

- `hyper_parallel.trainer.text_trainer.TextTrainer`

初始化逻辑是：

1. `clone_config_for_graph_mode(config)` 复制配置并关闭 eager layer compile
2. `build_pass_config_from_trainer_config(...)` 投影 graph pass 配置
3. `super().__init__(graph_config)` 进入原 `TextTrainer` 初始化主流程
4. 构造 `self.graph_executor = GraphExecutionEngine(...)`

其中真正替换掉 eager 路径的只有一处：

- [compile/text_trainer.py:L86-L99](file:///home/whh/graphtrainer/hyper_parallel/compile/text_trainer.py#L86-L99)

`GraphTextTrainer.forward_backward_step()` 会：

1. 调 `self.base.get_batch(data_iterator)` 取 batch
2. 统计 token count
3. 调 `self.graph_executor.train_step(model_inputs, loss_inputs)`

所以关系可以概括为：

- `TextTrainer` 继续负责训练主循环、optimizer step、scheduler step、callback/logging
- `GraphTextTrainer` 只替换前反向执行

### 2.4 TextTrainer 主流程如何被复用

原始 `TextTrainer.train()` 在：

- [trainer/text_trainer.py:L284-L324](file:///home/whh/graphtrainer/hyper_parallel/trainer/text_trainer.py#L284-L324)

其关键行为包括：

1. 按 epoch 驱动训练
2. 每个 epoch 调用 `iter(train_dataloader)`
3. 每个 step 调 `self.train_step(data_iterator)`
4. 在 `train_step()` 内继续调用 `self.forward_backward_step(...)`

因为 `GraphTextTrainer` 重写了 `forward_backward_step()`，所以 TextTrainer 训练主流程无需改动，就能把 eager 前反向切换成 graph-mode 前反向。

### 2.5 如何与 AutoModel 动态图模式对接

桥接逻辑在：

- [dependency_bridge.py:L26-L64](file:///home/whh/graphtrainer/hyper_parallel/compile/dependency_bridge.py#L26-L64)

这层主要做两件事：

1. `clone_config_for_graph_mode()`  
   - 复制原始 `TrainerConfig`
   - 关闭 eager `compile`
2. `build_model_for_graph_mode()`  
   - 保留 AutoModel 的 distributed setup、mesh 构建、TP planner/apply 等模型准备能力
   - 将 `distributed_setup.strategy_config` 置空，跳过 eager FSDP2 wrap

因此当前链路的职责划分是：

- `AutoModel` 继续负责模型准备
- `TextTrainer` 继续负责训练主流程
- `GraphTrainer/GraphTextTrainer` 接管图模式前反向与 graph pass

## 3. 模型结构、并行策略与入口配置

### 3.1 模型结构

模型构建函数在：

- [train_tp2_fsdp2.py:L127-L155](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L127-L155)

底层使用：

```python
HyperAutoModelForCausalLM.from_config(LlamaConfig(...))
```

本次使用的小规格 Llama 配置为：

- `vocab_size = 128`
- `hidden_size = 64`
- `intermediate_size = 128`
- `num_hidden_layers = 1`
- `num_attention_heads = 4`
- `num_key_value_heads = 2`
- `max_position_embeddings = 64`
- `tie_word_embeddings = False`

### 3.2 并行策略

训练配置在：

- [train_tp2_fsdp2.py:L246-L298](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L246-L298)

关键参数：

- `AcceleratorConfig(tp_size=2)`
- `FSDP2Config(dp_shard_size=2)`
- `TrainingConfig(train_iters=10, global_batch_size=4, micro_batch_size=1)`

实际运行日志显示：

```text
Built device mesh (2, 1, 2), dense FSDP mesh (1, 2, 2)
```

也就是：

- TP = 2
- FSDP shard = 2
- world size = 4

### 3.3 样例 dataloader 入口

本次为了适配 `TextTrainer` 的 epoch 训练方式，在样例中新增了一个可重建 dataloader 包装器：

- [train_tp2_fsdp2.py:L82-L124](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L82-L124)

对应 builder 在：

- [train_tp2_fsdp2.py:L223-L243](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L223-L243)

它的作用是：

1. 每次 `iter(train_dataloader)` 都新建底层 `TorchDataLoader`
2. 如果 batch sampler 支持 `set_epoch()`，则继续透传

这样就能在不修改 `TextTrainer` 主流程的情况下，稳定支撑多 epoch、10 step 的 smoke test。

## 4. 如何基于 GraphTrainer 启动训练

当前推荐方式就是样例中的：

```python
trainer = GraphTrainer.from_text_config(
    build_config(),
    pass_plan=create_simple_sharding_plan(),
)
trainer.train()
```

对应位置：

- [train_tp2_fsdp2.py:L344-L348](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py#L344-L348)

本次容器内实际执行命令：

```bash
export PYTHONPATH=/home/whh/graphtrainer:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_NPU_SOCKET_PORT_RANGE=29640-29679
export GRAPH_TRAIN_ITERS=10
torchrun --nproc_per_node=4 --master_port=29631 \
  --log-dir /home/whh/graphtrainer/hyper_parallel/compile/docs/logs/texttrainer_tp2_fsdp2_10step_final/torchrun_logs \
  --tee 3 \
  --local_ranks_filter=0 \
  /home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py
```

## 5. 运行结果简单概括

### 5.1 运行成功

本次 4 卡 `tp2 + fsdp2` 训练已成功跑满 `10 step`。

`rank0` 日志保存路径：

- [rank0_console.log](file:///home/whh/graphtrainer/hyper_parallel/compile/docs/logs/texttrainer_tp2_fsdp2_10step_final/rank0_console.log)

完整 per-rank `torchrun` 日志目录：

- [torchrun_logs](file:///home/whh/graphtrainer/hyper_parallel/compile/docs/logs/texttrainer_tp2_fsdp2_10step_final/torchrun_logs)

### 5.2 关键结果

1. AutoModel 的 TP 模型准备成功
   - 日志包含：
     - `Running ShardingPlanner.plan(tp=2, cp=1, ep=1, ...)`
     - `Sharding plan applied; source_shard_info keys=12`

2. graph FSDP pass 成功执行
   - 日志包含：
     - `Running with fsdp_degree=2, world_size=4`
     - `Identified 12 FSDP parameter nodes out of 14 total state inputs`

3. 训练成功达到 10 step
   - rank0 关键日志：

```text
step=1  ... training/graph_loss=9.77689
step=5  ... training/graph_loss=9.67864
step=10 ... training/graph_loss=9.57254
Training: 100%|██████████| 10/10
tp2+fsdp2 graph prototype finished
```

4. 参数形状符合 TP2 + graph-FSDP2 预期
   - 训练前：
     - `model.embed_tokens.weight = [64, 64]`
     - `q_proj.weight = [32, 64]`
   - 训练后：
     - `model.embed_tokens.weight = [32, 64]`
     - `q_proj.weight = [16, 64]`

### 5.3 结论

当前样例已经完成对齐，并验证了以下链路是成立的：

- 外部入口使用 `GraphTrainer.from_text_config()`
- 实际训练运行时是 `GraphTextTrainer -> TextTrainer`
- AutoModel 继续负责动态图模式下的模型准备与 TP 处理
- graph pass 接管 FSDP 图模式处理
- 在 4 卡 `tp2 + fsdp2` 配置下，样例能够稳定跑满 `10 step`
