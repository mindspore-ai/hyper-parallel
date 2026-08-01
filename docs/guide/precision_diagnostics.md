# 可选精度诊断

`hyper-low-precision-observer` 是独立 wheel，负责低精度比较指标、采样上下文和离线报告；HyperParallel 仅提供 adapter，把诊断绑定到已转换的 `NpuQuantLinear` 与 Trainer 生命周期。

## 安装与构建

常规训练不依赖 observer。需要诊断时安装额外依赖：

```bash
pip install 'hyper-parallel[precision-debug]'
```

源码构建会在 `dist/` 同时生成 HyperParallel 和 observer wheel：

```bash
./build.sh
```

若启用诊断但 observer wheel 不可导入，adapter 会给出明确的安装错误，不会静默退化。

## YAML 配置

诊断配置位于 `low_precision.precision_debug`。输出目录不允许由 YAML 指定，Trainer 固定写入 `training.train_url/precision_debug`。

```yaml
training:
  train_url: ./outputs/run_001

low_precision:
  enabled: true
  include_fqns: ["model.layers.*.mlp.*_proj"]
  precision_debug:
    sections:
      - name: selected_fprop
        select:
          module_name_regex: "model\\.layers\\..*\\.mlp"
          gemm_roles: [fprop]
        observe:
          operands: [lhs, rhs]
          schedule:
            every_n_steps: 10
            start_step: 0
```

公共配置接受 `fprop`、`dgrad`、`wgrad` 三种 role；当前 HP Dense MXFP8 adapter 只发出 `fprop`。请求未实现 role 或 selector 未命中任何转换后的 Linear 会在安装阶段报错。

## 训练时行为

1. `FinetuneRecipe.setup()` 在模型转换完成后安装 session。
2. Trainer 每个 optimizer step 调 `set_step()`，在 step 结束时调用 `flush()`。
3. 验证阶段通过 `session.paused()` 停止采样，避免验证前向污染训练窗口。
4. `NpuQuantLinear` 只在 selector 和 schedule 都命中时重建反量化候选值。

诊断失败会清理已累计 moments 并禁用 session，但不会改变模型前向、反向或 optimizer step 的执行结果。

## 产物与续跑限制

每个 rank 在 `raw_moments/rank_<rank>.jsonl` 写入可合并原始 moments，并在 `statistics/` 写 rank 本地标量日志。当前明确**不支持 resume**：如果目标 rank artifact 已存在，首次 flush 会抛出 `FileExistsError`。续跑诊断必须使用新的 `training.train_url`。

离线聚合：

```bash
hyper-low-precision-report --root ./outputs/run_001/precision_debug
```

该命令读取所有 rank artifact，按 step 合并原始 moments，并输出全局 JSONL；TensorBoard 可通过 `--no-tensorboard` 关闭。

## 验证

```bash
python -m pytest tests/ut/trainer/test_low_precision.py -q
```

