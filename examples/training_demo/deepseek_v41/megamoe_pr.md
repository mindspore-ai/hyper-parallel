# DSV4.1 声明式 MegaMoe 适配

本分支基于 `trainer_dev` 的 `22fb803b`。该基线已包含 master 的 multicore 更新，
底层只额外引入两项能力：

- [PR #855](https://github.com/mindspore-ai/hyper-parallel/pull/855) 的 `cf2486f0`：
  clipped SwiGLU 前反向，保留 DSV4.1 的 `limit=10`。
- `megamoe-push-pull` 的外部专家权重接口（原提交 `01839418`）：
  `create_parameters=False` 和 `forward(..., expert_weights=...)`。

不引入动态 token kernel、push/pull 切换、Muon 或 Indexer 优化。
PR #855 与先前 PR #847 的 kernel 实现不同，必须重新构建 native payload，
旧分支的设备精度和性能结果不代表本分支通过。

## 声明式配置

复用现有文本和 VLM 训练入口及 YAML。两份 YAML 默认仍走原 native EP，
不新增 `TrainerConfig.megamoe` 或顶层 `megamoe` 开关。
启用 MegaMoe 时，在 `plan_overrides` 中添加如下专家替换规则，并将原 MoE
计算规则替换为第二项。不要同时保留原 native MoE compute 项。

```yaml
plan_overrides:
  # 保留其他 mHC、Engram、attention 和参数分片规则。
  - match: "*.mlp.experts"
    module_type: transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts
    replace_module:
      _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts
      local_num_tokens: 4096
      expert_capacity_factor: null

  - match: "*.mlp"
    region_dispatch: false
    local_compute_fn:
      _target_: >-
        hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_megamoe_compute_fn
```

两项都不设置 `when: ep`：结构替换不能依赖并行拓扑条件，EP1 也需要绑定计算函数。
启用时要求 TP=CP=PP=1，EP 覆盖整个 world，rank 顺序与全局一致。
原计算入口 `deepseek_v41_ep_compute_fn` 保留 native EP 行为；错误的模块/计算函数
搭配会在准备或绑定阶段失败。

`local_num_tokens` 是正数且为 128 的倍数。它显式指定每卡固定专家执行长度，
至少覆盖真实输入长度；Trainer 不再读取数据配置推导或改写该值。
对于 packing，容量至少覆盖 `max_seq_len` 与实际 token budget 的较大值；
未单独设置 budget 时还应考虑 micro-batch 累积在一次模型调用中的 token 数。
`expert_capacity_factor: null` 预留最坏路由容量，显存成本较高；有限 factor 可以降低
预留量，但真实路由及补位路由都计入接收量，超限会报错，不截断真实 token。

## 模块与生命周期

`DeepseekV41MegaMoeExperts` 在结构替换阶段转换专家权重布局：
`[E,2I,H]` / `[E,H,I]` 转为 `[E,H,2I]` / `[E,I,H]`。
保留 `gate_up_proj`、`down_proj` 名称及可逆 checkpoint 转换，之后由框架完成 EP
切分、FSDP 包装、meta materialization 和 optimizer 参数管理。

并行化阶段的 `deepseek_v41_megamoe_compute_fn` 绑定 EP group 和 Top-K。
前向保留原 router、路由权重缩放、shared experts 以及 VLM 的 `image_mask` / `bias_vl`。
调用仍经过 `module.experts(...)`，保留嵌套 FSDP hooks。
每次从当前参数取得本地权重传给 executor，不注册第二套参数，也不缓存外部权重视图。
SwiGLU limit 从原模块读取，并要求 routed/shared experts 一致。

本轮使用固定 token kernel。短 batch 仅在专家入口补零 hidden states、零路由权重和
均衡的虚拟 expert ID，执行后裁掉补位输出。真实 token、图像位置、attention packed
边界和 loss 输入保持原样。容量 4096、真实 224/240 时仍按 4096 行执行专家，
因此本轮没有动态 token 计算量收益。超过容量时应调大配置，不能截断输入。

通用 `ModelRuntimeModule` / `ModelRuntimeResources` 接口负责构建阶段的资源准备：

- 共用模型构建完成后、第一次前向前准备显式参与的模块；MegaMoe 在这里共享兼容层的 workspace。
- 准备失败会回滚；需要提前关闭时，调用方仍可显式关闭资源并重试失败的参与者。
- 普通模块不参与；Trainer 不导入 DSV4.1 或 multicore，也不根据 MegaMoe 类型分支。

MegaMoe 运行时自动在 WORLD / SHMEM Root 通信组关闭前清理资源，并在正常进程退出时
通过 `atexit` 尝试清理。Base/Text/VLM Trainer 不再增加释放资源的 `try/finally`，
训练异常原样传播。GC 仅登记成员退出，不触发 collective；自动清理仍要求各 rank 同序进入，
且没有在途调用或待反向图，通信故障和强制退出不保证安全释放。
详见 [Multicore 生命周期说明](../../../hyper_parallel/core/multicore/README.md)。
非 Trainer 调用方应在并行绑定后创建并准备 `ModelRuntimeResources(model)` 以共享 workspace；
通常无需显式关闭，需要提前回收时可调用其 `close()`。

## 验证

当前 CPU 回归覆盖外部权重更新、输出与全部梯度、固定补位、空 batch、真实 VLM router
及图像 token 梯度、checkpoint 往返、meta 初始化、YAML Target 解析与实际替换、
native EP 默认行为，以及资源共享、自动关闭边界、异常传播、准备回滚和关闭重试。
CPU 数值测试替换了 native executor，只验证适配语义。

```bash
python -m pytest -q \
  tests/ut/core/multicore \
  tests/ut/auto_models/models/deepseek_v41 \
  tests/ut/auto_models/test_runtime_resources.py \
  tests/ut/trainer \
  tests/ut/dual_mode_dtensor/test_module_replacement.py
```

当前结果为 **175 passed、393 subtests passed**。新增适配和生命周期文件 pylint 通过。
`BaseTrainer` 全文件检查保留 5 个 `no-member` 告警；已对照未经修改的 trainer_dev
确认同样存在（组合式运行时提供的 `data_transform`、`get_batch` 和 `num_micro_batches`），
本轮不修改这些基线诊断。
使用 Torch 2.9.0、torch_npu 2.9.0.post6、CANN 9.1 从本分支重新构建 native payload 成功。
外部权重提交在当前基线的 UT 缺少 `arg_mark` 导入，适配提交补齐该测试依赖。

设备测试入口如下；需先按 multicore 构建文档激活当前构建的 `set_env.bash`：

```bash
python -m pytest -v tests/torch/multicore/test_deepseek_v41_megamoe.py
```

该测试为 EP8/E48、H512/I128、Top-K6、limit10、固定容量 256。
覆盖每卡 224/240、不等长含空 rank、全空和满容量输入，对照直接 MegaMoe 的显式
补位实现，比较输出、输入梯度、路由权重梯度和两份专家权重梯度，
阈值保持 `rtol=2e-2, atol=2e-3`。另外记录独立 FP32 oracle 误差及 clamp 触发数量；
FP32 数据属于独立诊断，不将两条相同 kernel 路径的一致性视为独立数学精度验收。
本轮设备测试已完成：8 个 rank、四组输入全部通过，输出和全部梯度相对直接 MegaMoe
的最大绝对差均为 0，进程退出码为 0。非空输入确实触发了 clamp。
独立 FP32 诊断的最大相对 L2 误差分别为：输出 0.337%、输入梯度 5.530%、
路由权重梯度 0.317%、gate_up 梯度 6.060%、down 梯度 0.333%。
两条 BF16 路径的误差完全相同；这证明适配层没有引入额外差异，
不等价于通过纯 FP32 逐元素验收。clamp 边界与 BF16 中间舍入的具体贡献尚未在本分支
逐算子拆分。整网文本/VLM 尚未复测。

## 整网精度与性能复测

从现有 `train_deepseek_v41_online.yaml` 或 `train_deepseek_v41_vlm_online.yaml`
配置上述两个 Target，其他设置与 native EP 对照保持一致。
例如在准备好的配置和数据下运行 EP8/E48 文本训练：

```bash
torchrun --standalone --nproc-per-node=8 \
  -m examples.training_demo.train_text "$TRAIN_YAML" \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --training.global_batch_size=8 \
  --model.num_routed_experts=48
```

VLM 使用原 `scripts/train_vl.py` 入口。模型、tokenizer、Engram assets、数据路径
通过既有配置字段设置。当前 CLI 不支持替换 `_target_`，应在 YAML 中声明目标。

先做单层固定权重/路由精度及至少三步整网训练，对照 loss、梯度范数、完整选定专家梯度
和 optimizer 更新；有差异时用实际算子输入对照独立 FP32 参考，不能放宽阈值。
固定 canonical checkpoint、数据、随机种子、学习率、模型尺寸和任务队列配置后，
用独立进程 ABBA 测 native EP 与 MegaMoe，分别报告整步吞吐和 routed MoE 前反向合计。
MoE 加速比为 native MoE 时间 / MegaMoe 时间，单独诊断计时，避免同步污染整步计时。
同时记录 Torch allocator peak 与整卡 HBM 采样峰值。
本分支目前不声明性能收益，也不沿用旧分支的整网加速比。
