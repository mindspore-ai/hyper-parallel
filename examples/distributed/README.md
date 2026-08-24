# examples/distributed — components/distributed 独立示例

零依赖训练流程的双模式 DTensor 示例集（gloo/CPU 可跑，torchrun 启动）。
每个示例都与单卡参考做数值对拍，覆盖 production 与 validate 双模式。

| 示例 | 并行 | 演示点 | 启动 |
|---|---|---|---|
| [tp.py](tp.py) | TP=2 | 零配置自动推导 + 应用 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/tp.py` |
| [cp.py](cp.py) | CP=2 | `shard_batch_for_cp` 数据管道、plan_overrides glob merge 显式声明 `"sdpa_hf"` wrapper（显式注入，无启发式分派）、D-04 causal offset mask、D-07 本地 chunk 输出对拍 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/cp.py` |
| [ep.py](ep.py) | TP=2×EP=2 | HF 原生 MoE：D-09 per-expert 堆叠 + D-10 TP-extend-EP 参数分片（planner 推导）+ `local_compute_fn` 显式注入仓内默认工厂 `hf_native_ep_compute_fn`（Target 形态） | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/ep.py` |
| [tp_cp_ep.py](tp_cp_ep.py) | TP=2×CP=2×EP=2 | 三维组合：mesh (cp=2, tp=2) + ep=2（TP-extend-EP）；CP/EP compute 双显式注入；cp-major 序列嵌套布局（每 rank 持 S/(cp·tp) 连续 token 段）；plan 内省断言 | `PYTHONPATH=. torchrun --nproc_per_node=4 examples/distributed/tp_cp_ep.py` |
| [nested_local_map.py](nested_local_map.py) | TP=2（嵌套） | D-14 嵌套 spec：外层 local_map（根 fqn `""` 整 LM I/O 契约）+ 内层 validate 孤岛（策略传播校验）；双模式对拍 + 不变式 3 探针 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/nested_local_map.py` |
| [multimodal_encoder_dp.py](multimodal_encoder_dp.py) | ViT dp=4 + LLM dp=2×tp=2×ep=4 | 多模态双 mesh：encoder_dp ViT（params={} 纯 FSDP 公民，dp 语义由 vit_mesh + 数据分配表达——plan 坐标系 = 单 dp 切片）+ 桥接边界 all-gather（out 边界 `Shard(0)→Replicate`，plan_overrides 注入）+ LLM 独立 plan/apply（EP compute 显式注入）；双模式对拍 + gather 探针 | `PYTHONPATH=. torchrun --nproc_per_node=4 examples/distributed/multimodal_encoder_dp.py` |
| [custom_local_compute_fn.py](custom_local_compute_fn.py) | TP=2 | 自研 MoE：glob override 注入字段-only（merge 继承推导契约）+ `local_compute_fn` 注入自定义 compute（骨架边界通信保留） | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_local_compute_fn.py` |
| [custom_inner_wrapper.py](custom_inner_wrapper.py) | CP=2 | 自研 attention：`inner_target` 指定非标准子模块（glob override + merge 继承契约）+ `INNER_WRAPPER_REGISTRY` 注册命名 wrapper + `_resolved_inner_wrapper` 回写 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_inner_wrapper.py` |
| [custom_autograd_function.py](custom_autograd_function.py) | TP=2 | 自定义 autograd.Function：第三方宿主（不可改）裸调用 `A.apply` → 子类化 + `__class__` 实例级替换 + `FunctionModule` 挂载 + plan_overrides 边界（入口 all-gather）；含版本锁定/smoke test 纪律，对拍即探针 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_autograd_function.py` |
| [plan_overrides_demo.py](plan_overrides_demo.py) | TP=2 | **plan_overrides 全场景**（[plan_overrides_demo.yaml](plan_overrides_demo.yaml)）：merge 注入 / 契约 DSL 覆盖 / 显式空 `{}` / `when` 条件跳过 / insert 完整自声明（模板外模块，契约驱动真实通信）；「不写继承，写了照办」语义 + plan 内省逐场景断言 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/plan_overrides_demo.py` |
| [perf_replacement.py](perf_replacement.py) | TP=2 | **YAML 性能替换双通道**：`plan_overrides` + `_target_` 把朴素 attention（eager S×S scores）/ 分解 silu 替换为用户高性能 kernel（[perf_kernels.py](perf_kernels.py)）——变体 2 走 `local_compute_fn` 工厂（[perf_replacement.yaml](perf_replacement.yaml)，骨架托管契约），变体 3 走 `inner_wrapper` 原地替换（[perf_replacement_inner_wrap.yaml](perf_replacement_inner_wrap.yaml)，自负双模式容错）；`resolve_component`→`entries_to_plan_overrides` 全链路 + 脱糖结果打印；kernel 计数器证明替换生效 + 双模式对拍 + 同条件计时 | `PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/perf_replacement.py` |

> **显式注入约定**（改造后）：CP/EP 的 compute 不再自动注入——planner 只推导
> 参数分片与 I/O 契约，`inner_wrapper`（通用 inner forward 织入；内置四路为
> CP 方案，需活跃 cp 轴）/ `local_compute_fn`（local-region 契约通道，EP 与
> 性能替换共用）必须显式声明；缺失时 apply 阶段 fail-fast 并给出可粘贴的
> 配置片段。统一注入接口为
> `ShardingPlanner(plan_overrides={...})`（exact/glob key，命中推导边界时
> **「不写继承，写了照办」**（未声明字段继承推导，显式 `{}` = 清空）的
> merge 语义；未命中且全部未声明则报错）；YAML
> `plan_overrides` 是其传输形态（trainer 侧 `entries_to_plan_overrides()`
> 脱糖后传入同一接口），也可 plan 后直接给 spec 赋值。仓内默认实现：
> `hyper_parallel.auto_models.components.distributed.cp_wrappers`（四个 CP wrapper +
> `INNER_WRAPPER_REGISTRY`）与
> `hyper_parallel.auto_models.components.distributed.ep_compute.hf_native_ep_compute_fn`。

使用教程：`docs/components_distributed_tutorial.md`
设计文档：`docs/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`
