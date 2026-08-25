# Qwen3-VL-MoE 训练接入

本文只覆盖 `qwen3_vl_moe`。当前这条线的重点是视觉塔独立并行、保存恢复和训练模板。

## 关键配置

推荐在模型侧显式写 `vision_parallel`：

```yaml
model:
  name: qwen3_vl_moe
  vision_parallel:
    dp_shard: 1
    cp: 2
    ulysses_degree: 1
    async_cp: false
train:
  accelerator:
    dp_shard: 1
    cp: 2
```

含义很直接：

- `dp_shard: 1` 让视觉塔参数保持复制，文本侧仍可按全局 `train.accelerator.dp_shard` 工作。
- `cp: 2` 只作用于视觉 encoder；使用时需要和 `train.accelerator.cp` 匹配。
- `ulysses_degree: 1` 表示 Pure Colossal，`ulysses_degree: 2` 表示 Pure Ulysses。
- `async_cp: true` 时启用视觉 encoder 的异步 CP 路径。

## 保存恢复

训练器已经支持 DCP 保存和恢复，配置里只要把 checkpoint 打开即可：

```yaml
train:
  checkpoint:
    output_dir: outputs/qwen3_vl_moe
    save_steps: 50
    load_path: null
    save_hf_weights: false
```

恢复时把 `load_path` 指向上一次保存出来的 checkpoint 目录即可。当前回调会恢复模型、优化器、学习率调度器、RNG 和 dataloader 状态。

## 示例模板

仓库里的示例模板在 `examples/qwen3_vl_30b_a3b_instruct/train.yaml`。它可以直接作为：

- 100 step 训练模板
- 保存恢复模板
- 视觉 encoder DP/CP/async CP 模板

## 测试入口

相关测试都在 `tests/torch/qwen3_vl_moe`：

- `test_qwen3_vl_moe_vision_parity.py`：CPU 视觉塔前向语义回归。
- `test_qwen3_vl_moe_vl_trainer.py`：Ascend 2 卡 VL trainer smoke、视觉 DP/CP 和 first-step loss 自洽。

## 验证结果

### 功能与精度

- DeepStack CP 切片校验：3 个检查全部通过。
- Qwen3-VL-MoE 视觉塔 CPU parity：通过。
- 1 卡 baseline、2 卡 baseline DP、2 卡 visual DP1 均通过。
- 2 卡 visual CP Pure Colossal、Pure Ulysses 和 async CP Pure Colossal 均通过。
- 上述 5 种模式的首步 loss 全部为 `11.931214332580566`，在
  `rel_tol=1e-6`、`abs_tol=1e-5` 下完成对齐。
- same-sample 100 step 对齐通过，最大绝对误差为 `0.00095845`，平均绝对误差为
  `0.00026391`，首步误差为 `0`，末步误差为 `0.00037098`，满足 `0.005`
  容差要求。

这些结果覆盖了视觉 DP、视觉 CP、Pure Ulysses、异步 CP 以及 DeepStack 视觉特征
注入路径，说明本次修改没有破坏原有视觉 encoder 前向和训练 loss 计算。

### 两卡性能快照

本次验收得到以下 tokens/s 结果：

| 模式 | 吞吐 |
| --- | ---: |
| baseline DP | 1934.418 |
| visual DP1 | 2675.551 |
| visual CP Pure Colossal | 1930.730 |
| visual CP Pure Ulysses | 1908.070 |
| visual async CP Pure Colossal | 1831.948 |

在该短序列条件下，CP 通信开销较明显，因此 CP 吞吐低于 visual DP1 属于当前
测试规模下的正常现象。

这条任务线不需要扩到 `qwen3_5` 或 `qwen3_5_moe`。
