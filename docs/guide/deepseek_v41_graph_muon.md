# DeepSeek V4.1 静态图 Muon 适配与验证

本文合并静态图 Muon 的问题分析、TorchTitan 参考、实现和验证记录。
适用配置为八卡、四层、16 个专家及裁剪 Engram，使用 FSDP4＋TP2/SP＋EP2、expert EDP4，
序列长度 2304、micro batch=1、global batch=8，开启 packing、Compressor、Indexer 和全重计算。

## 问题与整体方案

原 GraphCompiler/FSDP pass 直接切片普通 `Parameter.data`。完整布局单独保存在编译器中，
而 eager Muon 依赖 DTensor 的全局尺寸、mesh 和 placements 判断哪些矩阵需要通信。
因此直接切换优化器虽然能运行，却把分片矩阵当作完整矩阵独立进行 Newton–Schulz（NS）更新。
此前真实首步 `q_a_proj` 更新相对完整矩阵参考误差约为 51.6%。

另一个问题是 FP32 主参数在 lazy compile 之前创建。模型参数被 FSDP 切片后，
主参数仍保留旧形状，BF16 训练首步出现梯度 `(16160,5120)` 与主参数 `(64640,5120)` 不匹配。

解决方式是在首次编译完成物理分片后，给优化器绑定稳定的 DTensor 参数视图。
图继续读取普通本地参数，优化器在 eager 中执行：

```text
首次：联合图编译与参数切片 → 导出最终布局 → 绑定优化器参数
每步：图内 forward/backward → 梯度累积、归一化、裁剪
      → DTensor 梯度 → eager Muon/AdamW → 共享存储更新或主参数回写
```

FP32 视图与原图参数共享本地存储，不复制整份权重；BF16 的 FP32 主参数只持有最终本地 shard。
反向梯度归约仍由原图负责，视图绑定不重复进行归约。

## 对 TorchTitan 的参考

参考目录中的 `torchtitan-npu-master` 固定依赖 `torchtitan==0.3.0`。
其 DSV4 GraphTrainer 并行化先应用 TP/EP 和 SimpleFSDP，再启用图编译。
SimpleFSDP 在图外保留 DTensor 参数；tracer 保存 subclass/layout 信息，图运行时解包本地 Tensor，
返回梯度时恢复原分布式表示。
参见上游 [SimpleFSDP](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/experiments/graph_trainer/simple_fsdp.py)
及 [tracer](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/experiments/graph_trainer/make_fx_tracer.py)。

NPU DSV4 的 `_dsv4_muon_profile()` 显式兼容 `dp_shard`、`dp_shard_cp` 和 GraphTrainer 的 `fsdp` 轴，
将 storage layout 与 `Owned`、`BlockShard` 等 compute policy 分开。
对应 [DistMuon](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/distributed/flex_shard/dist_muon.py)
继续读取 DTensor 信息决定矩阵处理范围。
NPU DSV4 Muon 文档当时只验证 TP=1、PP=1，不能据此假定 TP2 已验证。

HyperParallel 借鉴的是图边界保留分布式参数信息的设计，继续复用自己的 FSDP pass 和 eager Muon。
本次没有移植 SimpleFSDP/DistMuon，也没有将 Muon 放入图中。

## 实际代码实现

### 最终逻辑布局

[GraphCompiler](../../hyper_parallel/compile/compiler.py) 新增 `optimizer_parameter_layouts`，
返回参数到 `(Layout, valid_local_shape)` 的映射。它保留 FSDP/EDP 与源 TP/EP 布局，
从切片前尺寸恢复逻辑矩阵，并排除 FSDP padding；原 `parameter_layouts` 继续描述裁剪所需的存储布局。

| 参数 | 图中本地形状 | 优化器逻辑形状 | 布局 |
| --- | --- | --- | --- |
| q_a_proj | `(320,5120)` | `(1280,5120)` | FSDP Shard(0)，TP 复制 |
| q_b_proj | `(4096,1280)` | `(32768,1280)` | FSDP StridedShard(0,2)＋TP Shard(0) |
| o_b_proj | `(1280,4096)` | `(5120,8192)` | FSDP Shard(0)＋TP Shard(1) |
| expert down_proj | `(2,5120,2304)` | `(16,5120,2304)` | EP/EDP 切 expert 维 |

二维矩阵被切开时需要恢复相应矩阵；仅切 expert/batch 维时，本地已有完整 NS 矩阵。
不能把激活的 SP 布局直接当成另一条参数分片轴。

### 参数、梯度和主参数绑定

新增 [GraphOptimizer](../../hyper_parallel/compile/optimizer.py)，包装既有 `ChainedOptimizer`：

- `bind()` 使用 `DTensor.from_local_with_layout()` 创建参数视图，保留原图参数身份，
  再重建 Muon 的分组、shard metadata 和 owner 缓存。视图只绑定一次，不每步重建参数对象。
- `prepare_grads()` 包装已经归约并完成归一化、裁剪的梯度，使 momentum 的 `zeros_like` 继承相同布局。
- `step_with_ready_grads()` 调用原优化器；FP32 直接更新共享存储，FP32 主参数更新后回写 BF16，
  无效 padded 行保持为零。
- `zero_grad()` 清理原图参数与优化器视图两侧的梯度，避免跨步重复累积。
- 参数名称与双向映射保持稳定，支持编译后 `state_dict/load_state_dict` 和主参数状态往返。
  storage/layout 变化会明确报错，要求重建优化器并恢复状态。

[BaseTrainer](../../hyper_parallel/trainer/base.py) 在静态图选择 Muon 或 FP32 主参数时启用包装器，
在首次更新准备阶段绑定最终布局。叶优化器与 scheduler 的参数组保持稳定；
AdamW fallback 保留本地逐元素更新，普通 FP32 Adam 路径保留原流程。

## 配置和复现入口

资源要求与启动变量见 [README](../../hyper_parallel/compile/examples/dsv41_graph/README.md)。
[Muon YAML](../../hyper_parallel/compile/examples/dsv41_graph/train_deepseek_v41_graph_muon.yaml)
使用 lr=1e-4、momentum=0.95、Nesterov、5 步 asym5 NS，weight decay=0；
AdamW fallback 的 lr=1e-4、weight decay=0。训练使用以下入口：

```bash
CONFIG=hyper_parallel/compile/examples/dsv41_graph/train_deepseek_v41_graph_muon.yaml \
bash hyper_parallel/compile/examples/dsv41_graph/run_deepseek_v41_graph.sh
```

BF16 在该 YAML 上覆盖：

```bash
CONFIG=hyper_parallel/compile/examples/dsv41_graph/train_deepseek_v41_graph_muon.yaml \
bash hyper_parallel/compile/examples/dsv41_graph/run_deepseek_v41_graph.sh \
  --model.torch_dtype=bfloat16 --model_init_dtype=bfloat16 \
  --fsdp_config.mix_precision.param_dtype=bfloat16 \
  --fsdp_config.mix_precision.reduce_dtype=float32 \
  --fsdp_config.mix_precision.output_dtype=bfloat16 \
  --fsdp_config.mix_precision.cast_forward_inputs=true \
  --optimizer.fp32_main_params=true
```

CLI 覆盖按顺序校验，先设置 `reduce_dtype=float32`，再开启 FP32 主参数。

CPU 回归与八卡分片检查：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/compile/test_optimizer.py tests/ut/compile/test_compiler.py \
  tests/ut/trainer/test_text_trainer.py \
  tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py \
  tests/ut/core/optimizer/test_muon.py
python -m pytest -q tests/torch/compile/test_graph_optimizer.py
```

数值参考与性能采样都直接构造原生 `TextTrainer`，避免 Codegen sidecar 影响任务退出。
它们接受 YAML 和与训练入口相同的 dotted overrides。先设置 README 中匹配的本地模型、tokenizer、
数据和 Engram 资源；模型配置须包含有效的 Engram assets 路径。
在同一 shell 设置所需的 CANN/HCCL 环境，例如：

```bash
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HCCL_OP_EXPANSION_MODE=HOST HCCL_DETERMINISTIC=true
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_EXEC_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

实际训练梯度的独立参考检查：

```bash
METRICS_DIR=output/muon_verify python -m torch.distributed.run \
  --nproc_per_node=8 --master_port=29744 \
  hyper_parallel/compile/examples/dsv41_graph/verify_muon_training.py \
  hyper_parallel/compile/examples/dsv41_graph/train_deepseek_v41_graph_muon.yaml \
  --model.pretrained_model_name_or_path="$MODEL_DIR" \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path="$TOKENIZER_DIR" \
  --dataset.data_path="$WIKITEXT_TRAIN_FILE" --training.train_iters=3
```

性能测量将上面的入口替换成 `benchmark_training.py`，设置独立 `METRICS_DIR`，运行 30 步。
eager 对照增加 `--compile.enabled=false --compile.use_joint_graph=false`；
BF16 对照增加前述精度覆盖参数。数值参考会增加计算、同步和显存，不能用于性能结论。
采样文件包含每个 rank 的 optimizer wall time、current/peak allocated 和 reserved 字节数。

## 验证结果

整理提交时（2026-11-19）在基于 `f0f6c8ca` 的隔离工作树重新运行上述 CPU 回归，
共 103 项通过；确认可编辑安装指向该工作树，辅助入口可导入，ST launcher 不导入重型训练依赖。
FP32/BF16 配置及 CLI 覆盖解析、文档相对链接通过；AutoGit 检查通过，保留非阻断建议项。
当时全部 NPU 被其他训练任务占用，因此本轮没有重复启动八卡训练或重新测量性能。

以下训练和性能记录来自 **2026-11-14**，运行基线为 `2abad6ce` 加本次 Muon 改动及当时全重计算工作区改动。
整理提交基于后续分支最新代码；历史表格不冒充整理当天重新测量的性能。
当前提交的入口和测试可用于在其他软件栈/模型资源上复测。

### 正确性

- 78 项 CPU 回归测试、pylint、AGENTS catalog 检查通过。
- FP32 和 BF16＋FP32 主参数各完成 30 步八卡训练。
- 在两种精度下连续检查 3 步、8 个 rank 的 q_a、q_b、o_b、router 和 expert down 更新。
  q_a 相对完整矩阵参考误差最大约 0.0025%，q_b 同量级，o_b 最大约 0.00404%；
  所有实际持有 momentum 的检查项与参考一致，主参数按模型 dtype 回写后误差为零。
- 专家矩阵与 batch=2 的 BF16 参考初始相差约 0.55%。另记录 FP32 首步原生 NS 实际 batch=4，
  相同 batch 形状下八 rank 最大误差降至 0.00268%，将残差定位到 BF16 batched matmul 的批大小差异。
- 八卡小矩阵覆盖 FP32/BF16 的 padding、空 shard、TP 列分片、同轴 StridedShard 和 head-wise reshape；
  每例 3 步与完整矩阵参考权重一致，编译后优化器状态往返继续更新一致。
- Adam 回归完成 10 步，loss 从 12.7633 降至 8.49075，与此前日志显示值最大差约 1e-5；
  该显示精度不用于声称 bit-wise 一致。

### 显存和性能

每项运行 30 步。步耗时取原生日志 rank 0 第 6–30 步中位数；优化器前后同步 NPU，
取每步最慢 rank 后再取中位数。峰值取八 rank 最大值，allocator 在 Trainer 构造后重置，
包含首次编译/执行。所有模式使用相同采样 harness。

| 模式 | 秒/步 | 优化器耗时 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 原 FP32 静态图（错误更新） | 2.908 | 217 ms | 17.80 GiB | 27.12 GiB |
| 修复后 FP32 静态图 | 2.964 | 275 ms | 17.76 GiB | 26.61 GiB |
| 正确 FP32 eager 参考 | 2.961 | 273 ms | 19.73 GiB | 25.36 GiB |
| BF16＋FP32 主参数 | 2.090 | 280 ms | 13.97 GiB | 18.32 GiB |

相对原错误更新版本，总耗时增加约 1.94%，优化器部分增加约 58 ms，包含恢复正确矩阵聚合/NS 的成本；
allocated 峰值基本不变。相对正确 eager 参考，FP32 图模式基本同速，allocated 少约 1.97 GiB，
但 reserved 高约 1.25 GiB。BF16 的更低耗时/显存属于模型精度配置收益，不能单独归为视图适配收益。
这些是单次短跑统计，不代表长期收敛或跨多次试验的性能保证。

原始逐 rank 日志、JSONL 和配置保留在当时验证机器的 `output/muon_graph_impl_20261114/`，
不随源码提交。两次独立初始化的 eager/graph loss 没有被直接当作端到端精度对齐证据。

## 支持边界

优化器状态保存/加载要求已经编译并绑定最终布局。编译后状态往返已验证，
首批编译前通过 `checkpoint.restore_from` 冷恢复的训练入口流程尚未接入，会明确报错。
参数 storage/layout 替换后需要重建优化器并恢复状态。
当前结果适用于上述裁剪模型和并行配置，不代表完整模型、其他拓扑或长期收敛已经验证。
