# Dense MXFP8 在线训练

本指南描述 HyperParallel 的第一阶段低精度训练路径：在 Ascend A5 上，对选中的 Dense `nn.Linear` 在线量化输入和权重，并使用 MXFP8 GEMM 完成前向、输入梯度和权重梯度计算。

## 范围与限制

- 仅支持 Ascend 950PR/950DT（A5）及提供相应算子的 `torch_npu` 运行时。
- 当前格式固定为 `mxfp8_e4m3`，缩放固定为 `mx_block`，块大小为 32。
- 仅转换精确类型为 `nn.Linear` 的 Dense 层；`nn.Linear` 子类会明确报错，避免静默改变其 `forward` 语义。
- 输入维度和输出维度都必须是 32 的倍数。
- 路由专家容器不由 Dense converter 处理，MoE 需要独立的 plan/apply 实现。
- 参数和优化器状态保持 BF16/FP32；MXFP8 是 GEMM 边界上的在线表示，不是持久化参数格式。

## YAML 配置

`TrainerConfig.low_precision` 使用 `LowPrecisionConfig`。最小配置如下：

```yaml
low_precision:
  enabled: true
  format: mxfp8_e4m3
  scaling: mx_block
  include_fqns:
    - "model.layers.*.mlp.*_proj"
  exclude_fqns:
    - "lm_head"
```

`include_fqns` 和 `exclude_fqns` 使用 FQN glob。启用后，任意 include pattern 未命中、选中层未对齐、选中 Linear 子类或路由专家都会在模型变异前失败，而不是跳过。

## 调用链

```text
TrainerConfig.low_precision
  -> HyperAutoModel.from_pretrained(..., low_precision_config=...)
  -> apply_model_infrastructure()
  -> validate_npu_runtime()
  -> apply_low_precision(model, config)
  -> NpuQuantLinear
  -> _NpuQuantLinearFn.forward/backward
  -> dynamic MX quant + npu_quant_matmul
```

转换发生在并行切分和 FSDP 包裹之前。`NpuQuantLinear` 保留原始 `weight`、`bias` Parameter 对象，只替换计算边界，因此既有 optimizer 参数引用不会失效。

`_NpuQuantLinearFn` 在 autograd 中实现三个 GEMM 方向：

| 阶段 | layout | 结果 dtype |
| --- | --- | --- |
| forward | `NT` | 输入 dtype |
| dgrad | `NN` | `grad_output` dtype |
| wgrad | `TN` | weight dtype |

量化器仅保留反向所需的 row-wise/column-wise表示；不再需要的方向在 forward/backward 后释放。

## 运行示例

仓库提供独立示例配置：

```bash
python examples/training_skeleton/main.py \
  examples/training_skeleton/train_low_precision.yaml
```

实际 NPU 多卡启动仍应使用环境对应的 launcher，并确保 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 已设置。NPU 运行时检查会验证动态 MX 量化、dual-axis 量化、量化矩阵乘和 E8M0 scale dtype 是否存在。

## 验证

不依赖 NPU 的转换和 autograd 契约测试：

```bash
python -m pytest \
  tests/ut/trainer/test_low_precision_core.py \
  tests/ut/trainer/test_low_precision_example.py -q
```
