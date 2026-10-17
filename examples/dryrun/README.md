# HyperModels LLM Dry-run

Dry-run 通过现有模型构造和并行切分流程构建 meta 模型，不加载权重，并在
FakeTensor/FakeProcessGroup 上执行 forward、loss、backward 和 optimizer step。
每个 rank 生成 `output_dir/rank_<n>/rank_<n>_memory.pickle`，可用 PyTorch
MemoryViz 打开。快照包含原生格式的 allocation/free 事件、调用栈及逻辑显存峰值，
不按模型结构、训练阶段或张量用途给内存分组。

基础实现支持 TP、CP、EP、FSDP/HSDP 及其合法组合。PP adapter 支持 PP 与
TP、CP、FSDP/HSDP 的组合，并通过 `dry_run.pipeline_stage_builder` 在正常
`config.model.build(...)` 构造链路中切分 stage；它不读取权重或建立真实 P2P 传输。

## 运行示例

从仓库根目录生成仅含配置的模型目录，然后运行 FSDP2 smoke：

```bash
python -m examples.dryrun.prepare_model --output-dir outputs/dryrun/models
torchrun --standalone --nproc_per_node=2 \
  --module examples.training_demo.train_text examples/dryrun/smoke_llama.yaml
```

PP2 × FSDP2 使用相同的模型配置目录：

```bash
torchrun --standalone --nproc_per_node=4 \
  --module examples.training_demo.train_text examples/dryrun/pp_fsdp.yaml
```

PP adapter 模拟 scheduler 顺序、stage-local 张量生命周期与 Tensor/DTensor
boundary；它不传输 payload、不估算通信时间。当前不支持 EP、sequence parallel、loss
parallel、activation swap 或 tied embeddings。正式 Trainer PP 装配完成后，应以该 adapter
作为需要审查和替换的范围。

快照中的 segment 为按逻辑分配合成的占位信息；workspace、缓存分配器碎片、
异步流、通信运行时常驻内存以及执行时间不在统计范围。不能仅凭逻辑峰值判断
真实设备是否能够运行。未配置或无法推导的值依赖会报错，不会凭空生成梯度。

`target_device` 只决定快照采用的目标设备类型，不会在验证配置阶段初始化真实设备。
若 PyTorch 未注册该设备的 FakeTensor 支持，Runner 会退回 CPU FakeTensor，并在
metadata 中分别记录 `target_device` 与 `simulation_device`。

`value_dependencies.rules` 描述 attention、MoE routing、TP cross-entropy 或模型
分支中会影响 shape、通信量和内存生命周期的值。未配置的 FakeTensor 值依赖会报告
模块 FQN、ATen 算子和源码位置。项目可控的分支应使用 `handler: branch`；
`handler: operator_debug` 仅用于临时定位问题。
