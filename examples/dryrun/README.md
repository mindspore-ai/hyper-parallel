# HyperModels LLM Dry-run

Dry-run 从本地 Hugging Face `config.json` 构建 meta 模型，不读取模型权重，
并使用 FakeTensor 执行一次完整的 forward、loss、backward 和 optimizer step。
每个 rank 会输出 JSON 摘要和内存生命周期 CSV。

基础实现支持 TP、CP、EP、FSDP/HSDP 及其合法组合。PP adapter 支持 PP 与
TP、CP、FSDP/HSDP 的组合，并通过 `dry_run.pipeline_stage_builder` 在正常
`config.model.build(...)` 构造链路中切分 stage；它不读取权重或建立真实 P2P 传输。

## PP adapter 示例

先生成本地仅含配置的模型目录，再以 PP2 × FSDP2 启动：

```bash
python -m examples.dryrun.prepare_model --output-dir outputs/dryrun/models
torchrun --standalone --nproc_per_node=4 \
  --module examples.training_demo.train_text examples/dryrun/pp_fsdp.yaml
```

PP adapter 模拟 scheduler 顺序、stage-local 参数/激活/梯度生命周期与 Tensor/DTensor
boundary；它不传输 payload、不估算通信时间。当前不支持 EP、sequence parallel、loss
parallel、activation swap 或 tied embeddings。正式 Trainer PP 装配完成后，应以该 adapter
作为需要审查和替换的范围。

## EP explicit 示例

该配置默认启用 Dry-run，并使用显式 MoE routing 统计：

```bash
torchrun --standalone --nproc_per_node=4 \
  --module examples.training_demo.train_text examples/dryrun/ep_explicit.yaml
```

要通过相同 Trainer 入口运行真实训练和 profiler，可覆盖 Dry-run 开关：

```bash
torchrun --standalone --nproc_per_node=4 \
  --module examples.training_demo.train_text examples/dryrun/ep_explicit.yaml \
  --dry_run.enabled=false
```

Profiler trace 写入 YAML 配置的 `profiling.trace_dir`。真实训练会读取模型权重，
需要 NPU/CUDA、对应分布式后端，以及配置目录中的 checkpoint。

`target_device` 只决定报告采用的目标设备类型，不会在验证配置阶段初始化真实设备。
若 PyTorch 未注册该设备的 FakeTensor 支持，Runner 会退回 CPU FakeTensor，并在
metadata 中分别记录 `target_device` 与 `simulation_device`。

`value_dependencies.rules` 描述 attention、MoE routing、TP cross-entropy 或模型
分支中会影响 shape、通信量和内存生命周期的值。未配置的 FakeTensor 值依赖会报告
模块 FQN、ATen 算子和源码位置。项目可控的分支应使用 `handler: branch`；
`handler: operator_debug` 仅用于临时定位问题。
