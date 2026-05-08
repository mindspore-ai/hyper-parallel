# Llama3 并行场景精度测试

本目录在 **HyperParallel Torch 后端** 上，将 `examples/torch/llama3` 中的 Llama3 风格 demo
跑在并行场景下，并与同进程内的**单卡基线**逐步比对训练 loss，作为并行栈的精度回归。

每个并行 case 复用 `examples/torch/llama3` 的 `model.py` / `parallelize.py` 设计（在本目录有
本地自包含副本 `model.py` / `parallelize.py`，避免依赖 `examples` 的 `sys.path` 注入）。

## 文件清单

| 文件 | 说明 |
|------|------|
| `model.py` | `examples/torch/llama3/model.py` 的本地副本：`Llama3DemoConfig` / `Llama3Model` / `Llama3BshdSdpaCore` 等，便于 `ContextParallel` 在 BSHD SDPA 子模块上挂钩。 |
| `parallelize.py` | `examples/torch/llama3/parallelize.py` 的本地副本：`parallelize_llama3` 提供 TorchTitan 风格的 TP+SP 计划。 |
| `_test_llama3_accuracy.py` | torchrun 触发的 worker；包含 `test_single_card_baseline`、`test_tp_fully_shard_matches_single_card`、`test_tp_cp_fully_shard_matches_single_card` 三个用例。 |
| `test_llama3_accuracy.py` | pytest 入口；使用 `arg_mark` 与 `parallel_run` / `torchrun_case` 在 1 / 4 / 8 卡场景下启动对应 worker。 |

## 精度对齐策略

每个用例都跑 `_STEPS = 10` 步训练，并对**所有步**做严格的 loss 比对（容差 `rtol=1e-3, atol=1e-3`）。让分布式与单卡逐步严格一致依靠下面六点：

1. **相同初始化**：所有 rank 在同一 `_INIT_SEED` 下构造 `Llama3Model`，并由 rank 0 把参数 / 缓冲区广播到其他 rank（`broadcast_state_dict_from_rank0`），保证起点参数完全一致。
2. **相同全局 batch**：在每个 worker 内按 `_DATA_SEED` 生成 `(tokens, targets)` 后由 rank 0 广播，保证所有 rank 看到的全局 batch 完全一致。
3. **`reduction="sum"` 的交叉熵**：把 CE 改为 sum 形式，每个 rank 在自己的 `(B/dp, S/cp)` 切片上计算 partial sum-CE，沿 `(dp, cp)` 平面 `all_reduce(SUM)` 后即可重建出与单卡 full-batch 完全相同的全局 sum-loss。
4. **`set_reduce_op_type("sum")`**：让 FSDP 的 gradient reduce 用 SUM，与 sum-loss 反向把各 rank partial gradient 求和聚合的语义一致。
5. **TP 反向归一化**：TP plan 让 logits 是 Replicate（`use_local_output=True`），每个 TP rank 都计算了一份相同的 scalar loss；若不归一化，反向时 grad 会在 TP 维上被求和 `tp_size` 次。我们把 `1.0 / tp_size` 作为 `loss.backward(...)` 的 gradient seed，正好抵消（与 `tests/torch/fully_shard/_test_tp_fully_shard_e2e.py` 中的做法一致）。
6. **CP+RoPE 对齐**：每个 CP rank 把 `rope_seq_start = cp_rank * (seq_len / cp_size)` 传入 `Llama3Model.forward`，让 `freqs_cis` 切片对齐到该 CP 窗口的全局位置。

## 用例覆盖

| 用例 | 进程数 | mesh | 说明 |
|------|--------|------|------|
| `test_single_card_baseline` | 1 | — | 单卡参考；为后续并行用例提供同进程内的可重复 loss 轨迹。 |
| `test_tp_fully_shard_matches_single_card` | 4 | `(dp=2, tp=2)` | TP + FSDP；DP 切 batch，TP 在 `mesh["tp"]` 上跑 `parallelize_llama3`，FSDP 在 `mesh["dp"]` 上分片。 |
| `test_tp_cp_fully_shard_matches_single_card` | 8 | `(dp=2, cp=2, tp=2)` | TP + CP + FSDP；TP 与 FSDP 与上一用例相同，并对每一层的 `attention.sdpa_core` 在 `mesh["cp"]` 上挂 Colossal `ContextParallel(ulysses_degree=1)`。 |

## 运行方式

进入本目录后再执行（`tests/torch/utils.py:torchrun_case` 拼出的命令对 worker 文件名是相对路径，需要 pytest 的 `cwd` 与 launcher 同目录，与仓库其他多卡 launcher 用法一致）：

```bash
cd tests/torch/accuracy

# 单卡基线（1 卡）
pytest -s test_llama3_accuracy.py::test_llama3_single_card_baseline

# TP + FSDP（4 卡）
pytest -s test_llama3_accuracy.py::test_llama3_tp_fully_shard_accuracy

# TP + CP + FSDP（8 卡）
pytest -s test_llama3_accuracy.py::test_llama3_tp_cp_fully_shard_accuracy
```

精度比较的容差为 `rtol=1e-3, atol=1e-3`；任一步重建出的全局 loss 偏离单卡基线即会断言失败。
