# examples/distributed — components/distributed 独立示例

零依赖训练流程的双模式 DTensor 示例集（gloo/CPU 可跑，torchrun 启动）。
每个示例都与单卡参考做数值对拍，覆盖 production 与 validate 双模式。

| 示例 | 并行 | 演示点 | 启动 |
|---|---|---|---|
| [tp.py](tp.py) | TP=2 | 零配置自动推导 + 应用 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/tp.py` |
| [cp.py](cp.py) | CP=2 | `shard_batch_for_cp` 数据管道、内置 `"sdpa_hf"` wrapper（启发式分派）、D-04 causal offset mask、D-07 本地 chunk 输出对拍 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/cp.py` |
| [ep.py](ep.py) | TP=2×EP=2 | HF 原生 MoE 零配置：D-09 per-expert 堆叠 + D-10 TP-extend-EP + 内置 `_hf_native_ep_compute` | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/ep.py` |
| [custom_local_compute_fn.py](custom_local_compute_fn.py) | TP=2 | 自研 MoE：`plan_overrides` 手写 spec + `local_compute_fn` 注入自定义 compute（骨架边界通信保留） | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_local_compute_fn.py` |
| [custom_inner_wrapper.py](custom_inner_wrapper.py) | CP=2 | 自研 attention：`inner_target` 指定非标准子模块 + `CP_WRAPPER_REGISTRY` 注册命名 wrapper + `_resolved_inner_wrapper` 回写 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_inner_wrapper.py` |

使用教程：`docs/components_distributed_tutorial.md`
设计文档：`docs/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`
