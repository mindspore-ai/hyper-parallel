# GraphTextTrainer TP2 + FSDP2 运行记录

## 1. 目标

验证最新 `GraphTrainer` / `GraphTextTrainer` 路径下，是否可以打通以下混合流程：

- 复用 `AutoModel` 的模型准备能力
- 复用 `TextTrainer` 的主要训练流程
- 在 Graph 模式下启用 `tp=2 + fsdp_shard=2`
- 由 graph pass 接管静态图 FSDP 处理

本次运行环境使用 `whh_titan` 容器。

## 2. 本次代码更改

### 2.1 Graph 模式桥接层增强

文件：
- [dependency_bridge.py](file:///home/whh/graphtrainer/hyper_parallel/compile/dependency_bridge.py)

改动：
- `build_pass_config_from_trainer_config()` 改为默认根据 `TrainerConfig.fsdp_config` 自动开启 graph FSDP。
- 新增 `build_model_for_graph_mode()` / `wrap_model_target_for_graph_mode()`：
  - Graph 模式仍复用 `AutoModel` 的 mesh 构建、TP plan/apply 等模型准备能力
  - 但会清掉 `distributed_setup.strategy_config`
  - 从而避免 eager `FSDP2Manager` 和 graph `FSDPPass` 同时接管 FSDP

### 2.2 TP2 + FSDP2 最小样例

文件：
- [train_tp2_fsdp2.py](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py)

新增内容：
- 构造 `tp_size=2`、`dp_shard_size=2` 的 `TrainerConfig`
- 继续通过 `HyperAutoModelForCausalLM.from_config()` 构造模型
- 使用 `GraphTrainer.from_text_config(...)`
- 显式传入 `create_simple_sharding_plan()`，确保 graph FSDP pass 覆盖模型参数
- 在训练前后打印每个 rank 的：
  - `dp_rank/tp_rank`
  - `graph_fsdp_enabled/fsdp_degree`
  - 代表性参数形状

## 3. 运行命令

在容器 `whh_titan` 中执行：

```bash
export PYTHONPATH=/home/whh/graphtrainer:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_NPU_SOCKET_PORT_RANGE=29640-29679
torchrun --nproc_per_node=4 --master_port=29631 \
  /home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py
```

## 4. 运行结果

### 4.1 第一次尝试

第一次 4 卡运行失败在 HCCL 端口初始化阶段：

- 报错关键字：
  - `Failed to enable listening for the NPU network adapter socket`
  - `port 16666 have already been bound`

处理方式：

- 增加环境变量：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE=29640-29679
```

### 4.2 第二次尝试：`train_iters=2`

第二次运行已经成功进入真实训练链路，并验证了以下事项。

#### A. AutoModel 模型准备能力已复用

日志显示：

- `Built device mesh (2, 1, 2)`
- `Running ShardingPlanner.plan(tp=2, cp=1, ep=1, sequence_parallel=False, loss_parallel=False)`
- `Sharding plan applied; source_shard_info keys=12`

说明：

- `AutoModel` 侧 mesh 构建成功
- TP=2 的 planner/apply_sharding_plan 已生效
- 模型在进入 Graph 模式前已经完成 TP 本地化准备

#### B. TextTrainer 主流程已复用

日志显示：

- `Rank0 Start training. Global step: 0. Train iters: 2`
- 其余 rank 也进入了相同训练流程

说明：

- 当前不是独立 demo loop
- 已经复用 `TextTrainer.train()` 主流程

#### C. Graph FSDP pass 已实际执行

日志显示：

- `Running with fsdp_degree=2, world_size=4`
- `Identified 12 FSDP parameter nodes out of 14 total state inputs`
- 多个参数被切分，例如：
  - `model.embed_tokens.weight: [64, 64] -> [32, 64]`
  - `model.layers.0.self_attn.q_proj.weight: [32, 64] -> [16, 64]`
  - `lm_head.weight: [64, 64] -> [32, 64]`

说明：

- graph 侧 `FSDPPass` 已经接管参数切 shard
- 且切分是发生在 TP 本地参数基础上

#### D. 训练首步已经跑完

rank0 日志记录到：

```text
step=1 ... training/graph_loss=9.71014 ... training/grad_norm=5576.86
```

说明：

- 首步 forward/backward/optimizer step 至少已执行到能产出 loss 和 grad_norm
- 也说明 `AutoModel + TextTrainer + Graph pass` 这一条主链已经基本打通

### 4.3 第三次尝试：`GRAPH_TRAIN_ITERS=1`

为了区分“训练链路未打通”和“继续迭代时触发 runtime 问题”，我又做了一次单步运行：

```bash
export GRAPH_TRAIN_ITERS=1
```

这次运行可以完整退出，关键日志如下：

```text
step=1 ... training/graph_loss=9.70772 ... training/grad_norm=nan
[after_train_local] rank=0 param_shapes={'model.embed_tokens.weight': [32, 64], ...}
tp2+fsdp2 graph prototype finished
```

这说明：

1. 单步 `tp2 + graph-fsdp2` 训练是可以完整跑通并正常退出的
2. 训练后本地参数形状已经从 TP local 形态进一步变为 graph FSDP shard 形态

以 rank0 为例：

- 训练前：
  - `model.embed_tokens.weight = [64, 64]`
  - `q_proj.weight = [32, 64]`
- 训练后：
  - `model.embed_tokens.weight = [32, 64]`
  - `q_proj.weight = [16, 64]`

这里非常直观地反映出：

- AutoModel 先做了 TP=2 的本地化
- Graph FSDP 再基于 TP-local 参数做 `fsdp_degree=2` 的进一步切分

## 5. 训练中断位置（2 step 运行）

第二次运行没有完整完成 2 step，而是在首步之后遇到 NPU 内核异常：

- 关键报错：
  - `vector core exception`
  - `MTE accesses an invalid GM address or the cross-device memory access times out`
  - `AclrtSynchronizeStreamWithTimeout`

触发位置：

- `TextTrainer.train_step()` 内部执行 `loss.item()` 时同步流失败

对应现象：

- 训练首步日志已经打印
- 之后在 rank0 做 host 同步取标量时暴露设备端异常

## 6. 结论

本次 `tp2 + fsdp2` 样例的结论是：

### 已确认打通的部分

1. `GraphTextTrainer` 已经能够复用 `TextTrainer` 主要流程
2. Graph 模式已经能够复用 `AutoModel` 的模型准备能力
3. `tp=2` 的模型准备已经生效
4. graph `FSDPPass` 已经在 `fsdp_degree=2` 下生效
5. 单步运行可以完整执行并正常退出
6. 两步运行时，首个训练 step 已经实际执行并产出 loss

### 当前未完成的部分

1. 2 step 完整收敛流程未跑完
2. 当前中断原因是 NPU runtime / kernel 级异常，不再是训练接线层问题

## 7. 当前判断

从架构验证角度，本次任务可以认为：

**“Graph 模式复用 AutoModel + TextTrainer 主流程，并在 tp2+fsdp2 下执行 graph FSDP pass” 已经成立，且单步样例可完整跑通。**

当前下一步更像是：

- 针对 `tp + graph fsdp` 混合场景做 runtime/kernel 级排障
- 优先检查 graph FSDP 后的参数/梯度布局和 NPU kernel 对齐问题
- 必要时缩小到 1 step + 更小 hidden size + selective parameter wrapping 继续定位

## 8. 关联文件

- [compile/trainer.py](file:///home/whh/graphtrainer/hyper_parallel/compile/trainer.py)
- [compile/text_trainer.py](file:///home/whh/graphtrainer/hyper_parallel/compile/text_trainer.py)
- [compile/dependency_bridge.py](file:///home/whh/graphtrainer/hyper_parallel/compile/dependency_bridge.py)
- [compile/tracer/graph_tracer.py](file:///home/whh/graphtrainer/hyper_parallel/compile/tracer/graph_tracer.py)
- [compile/examples/automodel_text_graph/train_tp2_fsdp2.py](file:///home/whh/graphtrainer/hyper_parallel/compile/examples/automodel_text_graph/train_tp2_fsdp2.py)
