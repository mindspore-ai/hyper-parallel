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
    reuse_dp_shard_mesh: true
    share_samples_across_dp: true
```

含义很直接：

- `dp_shard: 1` 让视觉塔参数保持复制，文本侧仍可按全局 `train.accelerator.dp_shard` 工作。
- `cp: 2` 只作用于视觉 encoder。
- `reuse_dp_shard_mesh: true` 表示视觉 CP 可以复用 `dp_shard` mesh。
- `share_samples_across_dp: true` 只用于验证和自洽对齐，不建议作为常规训练模式。

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
- 视觉 encoder DP/CP 验证模板

## 测试入口

相关测试都在 `tests/torch/qwen3_vl_moe`：

- `test_qwen3_vl_moe_vision_parity.py`：CPU 视觉塔前向语义回归。
- `test_qwen3_vl_moe_vl_trainer.py`：Ascend 2 卡 VL trainer smoke、视觉 DP/CP 和 first-step loss 自洽。

这条任务线不需要扩到 `qwen3_5` 或 `qwen3_5_moe`。
