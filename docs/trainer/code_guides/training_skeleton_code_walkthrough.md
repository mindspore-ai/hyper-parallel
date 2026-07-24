# 训练流程骨架代码详解说明书

> 范围：`hyper_models/recipes/`（base_recipe + llm/train_ft）+ `hyper_models/components/training/`（5 个源文件）+ `hyper_models/components/loss/` 运行时 + `hyper_models/components/optim/` 运行时扩展 + 各类桩模块（`_transformers/`、`models/common/`、`checkpoint/`、`data/`、`distributed/infrastructure.py`），共约 2600 行。
> 读法：本文严格按**调用时序**组织——先配置层（TrainerConfig），再 setup 期（①–⑲ 组件构建），再运行期（训练主循环 → 单步优化 → 前向反向 → 验证），最后按模块详解训练组件与桩。
> 每个函数给出功能、输入/输出与代码链接；关键设计（loss 归一化、Callback 混合方案、状态追踪）附推导与示例。仅阅读本文即可掌握当前实现。
> 设计文档对应关系：03（训练循环详细设计，本文档主体）、04 §8（checkpoint 故障恢复）、01 §6-8（HF 兼容层桩）、02（数据管道桩）、06（分布式基础设施桩）。

---

## 0. 总览：文件地图与全局调用时序

### 0.1 文件地图

| 文件 | 职责 | 阶段 | 实现状态 |
|---|---|---|---|
| [hyper_models/recipes/base_recipe.py](../../../hyper_models/recipes/base_recipe.py) | BaseRecipe：`__state_tracked` 状态追踪 + save/load_checkpoint | 骨架核心 | 完整 |
| [hyper_models/recipes/llm/train_ft.py](../../../hyper_models/recipes/llm/train_ft.py) | FinetuneRecipe：setup ①–⑲ + 训练主循环 + 单步优化 + 前向反向 + 验证 | 骨架核心 | 完整 |
| [hyper_models/recipes/__init__.py](../../../hyper_models/recipes/__init__.py) | RECIPE_REGISTRY（`"FinetuneRecipe"` → 类） | 入口 | 完整 |
| [hyper_models/trainer/config.py](../../../hyper_models/trainer/config.py) | TrainerConfig 扩展：step_scheduler/checkpoint/wandb/dataset 等规划字段落地 | 配置 | 完整 |
| [components/training/step_scheduler.py](../../../hyper_models/components/training/step_scheduler.py) | StepScheduler + StepSchedulerConfig：训练节奏控制 | runtime | 完整 |
| [components/training/callback.py](../../../hyper_models/components/training/callback.py) | StepState / TrainingCallback / CallbackManager / 7 个内置 Callback | runtime | 完整 |
| [components/training/grad_accum.py](../../../hyper_models/components/training/grad_accum.py) | 梯度累积辅助 + 通信/MFU/symlink 等工具函数 | runtime | 完整（PP 钩子除外） |
| [components/training/signal_handler.py](../../../hyper_models/components/training/signal_handler.py) | DistributedSignalHandler：SIGTERM 分布式协调 | runtime | 完整 |
| [components/training/rng.py](../../../hyper_models/components/training/rng.py) | StatefulRNG：rank 感知种子 + checkpoint 状态 | runtime | 桩 |
| [components/loss/masked_ce.py](../../../hyper_models/components/loss/masked_ce.py) | MaskedCrossEntropy：fp32 recast 掩码 CE | runtime | 完整 |
| [components/loss/utils.py](../../../hyper_models/components/loss/utils.py) | calculate_loss dispatcher + calculate_mtp_loss | runtime | 完整 |
| [components/loss/linear_ce.py](../../../hyper_models/components/loss/linear_ce.py) | FusedLinearCrossEntropy（融合 lm_head + CE） | runtime | 桩（分离实现） |
| [components/loss/__init__.py](../../../hyper_models/components/loss/__init__.py) | LossConfig typed config + 公共导出 | 配置 | 完整 |
| [components/optim/optimizer.py](../../../hyper_models/components/optim/optimizer.py) | Optimizer.Config.build / OptimizerFromFactoryConfig / 参数分组 / OptimizerInit | runtime | 完整 |
| [components/optim/lr_scheduler.py](../../../hyper_models/components/optim/lr_scheduler.py) | LRSchedulerConfig / RatioBasedLRSchedulerConfig | runtime | 桩（torch 原生调度器暂代 OptimizerParamScheduler） |
| [components/checkpoint/config.py](../../../hyper_models/components/checkpoint/config.py) + [checkpointing.py](../../../hyper_models/components/checkpoint/checkpointing.py) | CheckpointingConfig + Checkpointer | runtime | 桩（torch.save 暂代 DCP） |
| [components/distributed/infrastructure.py](../../../hyper_models/components/distributed/infrastructure.py) | MeshContext / DistributedSetup / initialize_distributed 等 | build | 桩 |
| [hyper_models/_transformers/](../../../hyper_models/_transformers/) | HyperAutoModelForCausalLM 等 HF 兼容入口 | build | 桩 |
| [components/models/common/model_utils.py](../../../hyper_models/components/models/common/model_utils.py) | build_model：模型构建 + OptimizerInit 导出 | build | 桩 |
| [hyper_models/data/build_dataloader.py](../../../hyper_models/data/build_dataloader.py) | build_dataloader / build_validation_dataloader | build | 桩（dummy 数据） |

> 测试基线：138 例位于 [`tests/components/training/`](../../../tests/components/training/)（step_scheduler 24 + callback 24 + grad_accum 29 + loss 21）与 [`tests/hyper_models/recipes/`](../../../tests/hyper_models/recipes/)（base_recipe 24 + finetune_recipe 16）。

### 0.2 全局调用时序

```
main()
  │  cfg = parse_training_args(...)                [TrainerConfig，§1]
  │  recipe = RECIPE_REGISTRY[cfg.recipe]()        [默认 FinetuneRecipe]
  │
  ├─ setup 期：recipe.setup(cfg) ────────────────────────────────────── [§3]
  │   ① initialize_distributed("nccl")            进程组
  │   ② setup_logging + apply_cache_compatibility_patches
  │   ③ StatefulRNG(seed=cfg.training.seed, ranked=True)
  │   ④ create_distributed_setup_from_config → mesh + dp_cp_mesh 展平
  │   ⑤ setup_magi（可选）                        ⑥ build_callback_manager
  │   ⑦ cfg.loss.build()                          ⑧ _configure_pp
  │   ⑨ peft_config = cfg.peft                    ⑩ cfg.checkpoint.build(...)
  │   ⑪ build_model(cfg.model, peft, distributed_setup) → (model, optimizer_init)
  │   ⑫ cfg.optimizer.build(model, device_mesh=, optimizer_init=, is_peft=) → list[Optimizer]
  │   ⑬ build_dataloader(...) → (dataloader, tokenizer)
  │   ⑭ build_validation_dataloader(...) → dict[str, DataLoader]
  │   ⑮ cfg.step_scheduler.build(dataloader, dp_size, local_bs)
  │   ⑯ cfg.lr_scheduler.build(optimizer, step_scheduler)（可 None）
  │   ⑰ register_state × 6（model/optimizer/lr_scheduler/rng/dataloader/train_state）
  │   ⑱ load_checkpoint(cfg.checkpoint.restore_from)
  │   ⑲ AutoMFU.from_config(model_parts[0]) + 模型信息打印
  │
  └─ runtime：recipe.run_train_validation_loop() ────────────────────── [§4]
     for epoch in step_scheduler.epochs:
       step_scheduler.set_epoch(epoch)
       for batches in step_scheduler:             [按 grad_acc_steps 分组，§8.1]
         metrics = _run_train_optim_step(batches, max_grad_norm)   [核心，§5]
           Phase 1  全局 label token 统计（_dp_cp_all_reduce_sum）
           Phase 2  for microbatch: _forward_backward_step          [§6]
                    （FSDP sync 外层 prepare_* / 内层 get_sync_ctx 双层管理）
           Phase 3  scale_grads_and_clip_grad_norm（唯一除 N 点）
                    → opt.step/zero_grad → lr_scheduler.step
                    → loss 聚合 + tps/mfu/lr
         callback_manager.on_step_end(StepState)   [外围，§8.2]
           CheckpointCallback / EvaluateCallback / LoggingCallback
           TqdmCallback / WandbCallback / GCCallback / SIGTERMHandler
     save_checkpoint(..., is_final_checkpoint=True)  [final save，先于 close]
     finally: callback_manager.on_train_end() + checkpointer.close()
     destroy_process_group()
```

### 0.3 骨架三条核心原则

- **核心显式、外围回调**：forward/backward/optimizer step 在 Recipe 中显式编排（可断点、可推理）；checkpoint/验证/日志/GC/WandB/tqdm/SIGTERM 全部由 `CallbackManager.on_step_end(StepState)` 驱动。时序标记（is_ckpt_step 等）由 StepScheduler **唯一**计算，经 frozen StepState 透传，Callback 只执行不判断。
- **typed `.build()` 构建**：optimizer/lr_scheduler/step_scheduler/loss/checkpoint 五个组件均为强类型 Config，`cfg.xxx.build(**runtime_deps)` 产出运行时对象，无 `_target_` 桥接层；model 走 `build_model`（HF 兼容入口），dataloader 走独立构建函数。
- **`__state_tracked` 自动状态追踪**：setup ⑰ 用 `register_state(name, kind)` 声明需要 checkpoint 的组件，save/load 两侧**遍历同一注册表**按 kind 分发（kind 与 04 `_state_path` 对齐），save/load 天然对称，新增有状态组件只需一行注册。

---

## 1. 配置层：TrainerConfig 扩展

[hyper_models/trainer/config.py:L78-119](../../../hyper_models/trainer/config.py#L78-L119)

设计文档 03 §5.2/§13 的"规划字段"本次全部落地为正式 dataclass 字段：

| 字段 | 类型 | 消费者 |
|---|---|---|
| `recipe` | `str = "FinetuneRecipe"` | main() 经 RECIPE_REGISTRY 解析（03 §13） |
| `step_scheduler` | `StepSchedulerConfig` | setup ⑬⑭⑮ + build_callback_manager（gc_every_steps） |
| `checkpoint` | `CheckpointingConfig` | setup ⑩⑱ + save_checkpoint 的 checkpoint_dir |
| `wandb` | `WandbConfig`（新增 [L69-75](../../../hyper_models/trainer/config.py#L69-L75)：enabled/project/entity） | build_callback_manager |
| `dataset` / `dataloader` / `packed_sequence` | `Optional[Any]` | build_dataloader（02 落地前弱类型） |
| `magi` | `Optional[Any]` | setup ⑤ setup_magi |
| `peft` | `Optional[Any]` | setup ⑨⑪⑫ |
| `training.seed` | `int = 42`（TrainingConfig 新增字段） | setup ③⑬⑭ |

两个配套调整：

- **`StepSchedulerConfig.local_batch_size`**（[step_scheduler.py:L50-53](../../../hyper_models/components/training/step_scheduler.py#L50-L53)）：不参与 StepScheduler 构造（构造参数由 `build()` 调用方注入），但作为 YAML 配置键保留——setup ⑬⑭⑮ 三处读取同一属性，保证 dataloader 批大小与 grad_acc_steps 计算**同源不分叉**。
- **循环 import 规避**：callback.py 的 `TrainerConfig` 仅用于类型标注，移入 `TYPE_CHECKING`（[callback.py:L20-26](../../../hyper_models/components/training/callback.py#L20-L26)）——否则 `trainer.config → components.training(__init__) → callback → trainer.config` 成环。

> **用例**：[`tests/ut/config/test_resolver.py`](../../../tests/ut/config/test_resolver.py)（配置解析回归，含 §9.3 新 build 契约用例）；[`tests/ut/trainer/test_config.py`](../../../tests/ut/trainer/test_config.py)。

---

## 2. BaseRecipe —— 自动状态追踪（03 §3 + 04 §8）

[hyper_models/recipes/base_recipe.py](../../../hyper_models/recipes/base_recipe.py)

### 2.1 注册机制

**`__init__`** [L47-58](../../../hyper_models/recipes/base_recipe.py#L47-L58)：三个成员——`__state_tracked: list[tuple[name, kind]]`（双下划线触发 name mangling，存为 `_BaseRecipe__state_tracked`，子类**必须**用 `register_state()` 访问）、`__state_names` 反查表（防重复注册）、`callback_manager = None`（setup ⑥ 赋值）。

**`register_state(name, kind)`** [L61-72](../../../hyper_models/recipes/base_recipe.py#L61-L72)：同名忽略。kind ∈ `{"model","optimizer","lr_scheduler","rng","dataloader","train_state"}`，与 04 `_state_path` 的 kind 一一对应。另有只读视图 property `_state_tracked` [L75-77](../../../hyper_models/recipes/base_recipe.py#L75-L77)（测试/子类检查用）。

### 2.2 rank / group size 查询

[L83-97](../../../hyper_models/recipes/base_recipe.py#L83-L97)：`_get_dp_rank/_get_tp_rank/_get_pp_rank/_get_dp_group_size` 全部委托 `self.mesh`（MeshContext，setup ④ 赋值）；无 mesh 时兜底 `0/0/0/1`——单进程与测试路径不需要分布式环境。

### 2.3 外围关注点辅助

- **`log_val_metrics(val_losses)`** [L101-110](../../../hyper_models/recipes/base_recipe.py#L101-L110)：仅 rank 0 输出（`_is_rank_0()`），EvaluateCallback 在 is_val_step 调用。
- **`_maybe_collect_garbage()`** [L113-117](../../../hyper_models/recipes/base_recipe.py#L113-L117)：`gc.collect()` + CUDA 可用时 `empty_cache()`，GCCallback 在 is_gc_step 调用。

### 2.4 `save_checkpoint` —— 遍历注册表按 kind 分发

[L124-191](../../../hyper_models/recipes/base_recipe.py#L124-L191)

```python
path = f"{checkpoint_dir}/epoch_{epoch}_step_{step}/"
```

逐 `(name, kind)`：`getattr(self, name, None)` 为 None → 跳过（注册未赋值不报错）；否则按 kind 分发：

| kind | 落地位置 | 实现 |
|---|---|---|
| `model` | `{path}/model` | `checkpointer.save_model(obj, ...)`；首个 model 记为 `model_ref` |
| `optimizer` | `{path}/optimizer` | `checkpointer.save_optimizer(model_ref, obj, ...)`——obj 为 **list[Optimizer] 原样不拆包**（canonical，03 §3.1） |
| `lr_scheduler` | `{path}/scheduler.pt` | 聚合 dict `{f"sch_{i}": s.state_dict()}` 一次 torch.save |
| `rng` / `dataloader` | `_state_path` per-rank 子目录（`rng/rng_dp_rank_{r}.pt`） | 先 `os.makedirs(dirname)` 再 torch.save |
| `train_state` | `{path}/extra_state.json` | `**obj.state_dict()` 先展开、显式键居后——`epoch/global_step/train_loss/val_losses` 不被 state_dict 覆盖 |
| 其他 | `{path}/{name}.pt` | 有 `state_dict` 兜底保存；无则 warning 跳过 |

末尾 `_update_latest_symlink(checkpoint_dir, path)`（[grad_accum.py:L270-281](../../../hyper_models/components/training/grad_accum.py#L270-L281)）原子更新 `LATEST` 软链（写**相对路径**，临时文件 + `os.rename` 保证原子性）。

`is_final_checkpoint=True` 仅用于训练循环末尾的 final save（04 §5.2 要求，触发 consolidated 导出等收尾）；周期保存与 CheckpointCallback 保持默认 False。

### 2.5 `load_checkpoint` —— 04 §8 canonical 恢复流程

[L195-226](../../../hyper_models/recipes/base_recipe.py#L195-L226)

```
restore_from=None          → 直接返回
restore_from="LATEST"      → _resolve_latest_symlink(cfg.checkpoint.checkpoint_dir)
                             无 LATEST → log info，from scratch
路径不存在                  → log warning，from scratch
否则                        → _validate_checkpoint_compatibility（桩，§2.6）
                             for name, kind in sorted(__state_tracked):
                               path = _state_path(restore_from, name, kind)
                               os.path.exists(path) → _load_state_by_kind(name, kind, path)
```

**`_load_state_by_kind`** [L228-251](../../../hyper_models/recipes/base_recipe.py#L228-L251)：model → `checkpointer.load_model(model_parts, model_path=path)`；optimizer → `load_optimizer(model_parts, obj, path)`；lr_scheduler → 逐 `sch_{i}` load_state_dict（与 save 聚合对称）；train_state → JSON 读入后 `obj.load_state_dict(extra)`（StepScheduler 恢复 step/epoch/start_epoch，断点续训生效点）；rng/dataloader → torch.load + load_state_dict。

**`_state_path(root, name, kind)`** [L254-263](../../../hyper_models/recipes/base_recipe.py#L254-L263)：save/load 共用的路径规则——model/optimizer → `{root}/{kind}`；rng/dataloader → `{root}/{kind}/{kind}_dp_rank_{dp_rank}.pt`；train_state → `extra_state.json`；lr_scheduler → `scheduler.pt`；其他 → `{name}.pt`。

### 2.6 模块级辅助

- **`_resolve_latest_symlink(checkpoint_dir)`** [L282-295](../../../hyper_models/recipes/base_recipe.py#L282-L295)：读 `LATEST` 软链，**拼回 checkpoint_dir 再判 exists**（写侧存相对路径，直接对 readlink 结果判 exists 会依赖 CWD）；非软链退化到 `_maybe_load_latest_marker`（[L298-306](../../../hyper_models/recipes/base_recipe.py#L298-L306)，纯文本 marker，兼容无符号链接 FS）。
- **`_validate_checkpoint_compatibility`** [L266-277](../../../hyper_models/recipes/base_recipe.py#L266-L277)：**桩**——仅 debug 日志；完整实现需读 extra_state.json / .dtensor_metadata.json 对比 DP/TP/PP size（04 §7.2）。
- **`_is_stateful(obj)`** [L309-313](../../../hyper_models/recipes/base_recipe.py#L309-L313)：setup 期辅助判断（nn.Module / Optimizer / LRScheduler / DataLoader / 有 state_dict）。

> **用例**：[`tests/hyper_models/recipes/test_base_recipe.py`](../../../tests/hyper_models/recipes/test_base_recipe.py)（24 例：注册/去重、无 mesh 兜底、六种 kind 的 save 分发与文件落位、missing attr 跳过、LATEST 软链更新与解析、save→load 对称恢复 train_state、`_state_path` 全 kind、log_val_metrics rank 守卫、GC 辅助）。

---

## 3. `FinetuneRecipe.setup()` —— 组件构建 ①–⑲（03 §5.3）

[recipes/llm/train_ft.py:L72-241](../../../hyper_models/recipes/llm/train_ft.py#L72-L241)

两类构建方式：**typed `.build(**runtime_deps)`**（loss/checkpoint/optimizer/step_scheduler/lr_scheduler）与**独立构建函数 / from_pretrained**（model/dataloader/tokenizer）。逐步详解：

### ①② 分布式初始化 + 日志补丁

[L91-104](../../../hyper_models/recipes/llm/train_ft.py#L91-L104)：`initialize_distributed("nccl")`（infrastructure 桩：未初始化则 `dist.init_process_group`）。`dist_env` 当前即 `torch.distributed` 模块，故 **device / world_size 在此派生并缓存**为 `self._device` / `self._world_size`，供数据搬运（§6 Step 1、§7）与 MFU 计算使用。随后 `setup_logging()` + `apply_cache_compatibility_patches()`。

### ③④ RNG + 分布式拓扑

[L106-139](../../../hyper_models/recipes/llm/train_ft.py#L106-L139)：`StatefulRNG(seed=cfg.training.seed, ranked=True)`；`create_distributed_setup_from_config(cfg)` → `self.mesh = distributed_setup.mesh_context`。

**dp_cp_mesh 推导**（DP+CP 联合 all-reduce 通信域，统计全局 label token / val loss 用）：
- `mesh.device_mesh is None`（stub MeshContext 兜底）→ `self.dp_cp_mesh = None`，后续 `_dp_cp_all_reduce_sum` 退化为全局 all-reduce 或 no-op；
- 有 mesh 时按轴名优先级取子 mesh：`("dp_shard_cp","cp")` 二维 > `"dp_shard_cp"` > `("dp","cp")`/`"dp"` > `("dp_replicate","cp")`/`"dp_replicate"` > 全 mesh 兜底；
- 多维子 mesh **setup 期一次性 `_flatten("dp_cp")`** 为 1D——`DeviceMesh.get_group()` 仅对 1D 语义明确，且避免每步重建 group。

> 为什么必须纳入 CP 维：cp_size>1 时全局 token 数 = 各 cp rank 持有段之和，只用 DP 维会少算 cp_size 倍（03 §5.3 ④ 注）。

### ⑤⑥ MagiAttention + Callback 管理器

[L140-147](../../../hyper_models/recipes/llm/train_ft.py#L140-L147)：`cfg.magi` 为 None → `self.magi = None`；否则 `setup_magi(cfg, device_mesh)`（未安装 magi_attention 时 warning + None）。`build_callback_manager(self, cfg, pbar_total=max_steps>0 ? max_steps : None)`（§8.2.8）。

### ⑦⑧⑨ Loss + PP + PEFT

[L149-157](../../../hyper_models/recipes/llm/train_ft.py#L149-L157)：`self.loss = cfg.loss.build()`（LossConfig → MaskedCrossEntropy 或 `_target_` 实例，§9.3）；`self.pp_enabled = mesh.pp_size > 1` → `_configure_pp`（[L244-254](../../../hyper_models/recipes/llm/train_ft.py#L244-L254)，桩：pp>1 时 warning——PP backward 钩子未落地，`prepare_for_final_backward` 会显式 NotImplementedError）；`self.peft_config = cfg.peft`。

### ⑩⑪⑫ Checkpointer + Model + Optimizer

[L159-182](../../../hyper_models/recipes/llm/train_ft.py#L159-L182)：

```python
self.checkpoint_config = cfg.checkpoint            # load_checkpoint 读取 checkpoint_dir
self.checkpointer = cfg.checkpoint.build(
    dp_rank=..., tp_rank=..., pp_rank=...,
    moe_mesh=getattr(self.mesh, "moe_mesh", None)) # 06 D-10：恒 None
self.model, self.optimizer_init = build_model(
    cfg.model, self.peft_config, distributed_setup=self.distributed_setup)
self.model_parts = self.model.parts if hasattr(self.model, "parts") else [self.model]
self.optimizer = cfg.optimizer.build(
    self.model, device_mesh=self.mesh.device_mesh,
    optimizer_init=self.optimizer_init,            # 复用 build_model 导出的 param 分组
    is_peft=self.peft_config is not None)          # → list[Optimizer]（canonical）
```

### ⑬⑭ DataLoader + Validation DataLoader

[L184-211](../../../hyper_models/recipes/llm/train_ft.py#L184-L211)：调 02 的独立构建函数（当前为 dummy 数据桩）。关键同源约束：`local_batch_size/global_batch_size/max_steps/val_check_interval` 全部读 `cfg.step_scheduler.*`——与 ⑮ 的 grad_acc_steps 计算读同一 YAML 键。

### ⑮⑯ StepScheduler + LR Scheduler

[L213-225](../../../hyper_models/recipes/llm/train_ft.py#L213-L225)：`cfg.step_scheduler.build(dataloader, dp_world_size, local_batch_size)`；`cfg.lr_scheduler` 为 None 时 `self.lr_scheduler = None`（下游 Phase 3 有守卫，§5.3）。

### ⑰⑱⑲ 状态注册 + 断点恢复 + MFU

[L227-242](../../../hyper_models/recipes/llm/train_ft.py#L227-L242)：六次 `register_state`（顺序：model/optimizer/lr_scheduler/rng/dataloader/step_scheduler-as-train_state）→ `load_checkpoint(cfg.checkpoint.restore_from)`（§2.5）→ `AutoMFU.from_config(model_parts[0])` → `_log_model_and_optimizer_details()`（[L257-271](../../../hyper_models/recipes/llm/train_ft.py#L257-L271)：参数量/trainable/优化器数/flops_per_token 一行日志）。

> **用例**：[`tests/hyper_models/recipes/test_finetune_recipe.py`](../../../tests/hyper_models/recipes/test_finetune_recipe.py)（16 例：mock 全部分布式依赖后的**构建顺序断言**（①–⑲ 时序）、各组件构建、`_state_tracked` 六项注册、load_checkpoint 调用、训练循环 epoch 迭代与 callback 驱动、final save 先于 close）。

---

## 4. 训练主循环（03 §6）

[recipes/llm/train_ft.py:L274-345](../../../hyper_models/recipes/llm/train_ft.py#L274-L345)

```python
for mp in self.model_parts: mp.train()
self.callback_manager.on_train_begin()
train_metrics = None; self._last_val_losses = None   # 零迭代场景 final save 守卫
try:
    for epoch in self.step_scheduler.epochs:
        self.step_scheduler.set_epoch(epoch)          # sampler shuffle 种子 + 重置 epoch ckpt 标记
        for batches in self.step_scheduler:           # 按 grad_acc_steps 分组
            train_metrics = self._run_train_optim_step(batches, max_grad_norm=...)
            sigterm = self.step_scheduler.sigterm_received   # 每步只查一次（内含 all_gather）
            self.callback_manager.on_step_end(StepState(...))
    self.save_checkpoint(..., is_final_checkpoint=True)      # final save
finally:
    self.callback_manager.on_train_end()
    self.checkpointer.close()                                # final save 必须先于 close
destroy_process_group()
```

三个易错点（设计文档明确约束）：

1. **StepState.is_final_step** = `_max_steps_reached or sigterm`——最终步/SIGTERM 步由循环末尾的 final save 统一保存，CheckpointCallback 对 is_final_step 跳过（§8.2.2），避免同一步重复保存（周期+final 两次，SIGTERM 叠加可达三次）。
2. **sigterm_received 每步只查询一次**并复用——其内部是 all_gather 集合通信，StepState 构造期间多次调用会放大通信量甚至死锁。
3. **final save 先于 `checkpointer.close()`**——04 close() 会销毁异步保存进程组，顺序颠倒在 is_async=True 下保存失败。

> **用例**：`test_train_loop_epochs / test_train_loop_callback_driver / test_train_loop_final_save`（[`tests/hyper_models/recipes/test_finetune_recipe.py`](../../../tests/hyper_models/recipes/test_finetune_recipe.py)）。

---

## 5. 单步优化器步进（03 §7）

[recipes/llm/train_ft.py:L413-519](../../../hyper_models/recipes/llm/train_ft.py#L413-L519)

### 5.1 Phase 1：全局 token 统计

[L425-433](../../../hyper_models/recipes/llm/train_ft.py#L425-L433)：逐 microbatch 累加 `(labels != -100).sum()`，再 `_dp_cp_all_reduce_sum(num_label_tokens, self.dp_cp_mesh)` 得 **N_global**（DP+CP 联合；CP 切序列，token 需全量计数）。

### 5.2 Phase 2：梯度累积

[L435-457](../../../hyper_models/recipes/llm/train_ft.py#L435-L457)：外层 `prepare_for_grad_accumulation`（关 FSDP 同步）→ 逐 microbatch：FSDPModule `set_requires_gradient_sync(is_last)`，最后一个 microbatch 前 `prepare_for_final_backward`（开同步 + PP 钩子）→ `_forward_backward_step`（§6）→ 首个 microbatch 后 `prepare_after_first_microbatch`（预热 unshard 缓存）。外层/内层双层管理不互相覆盖（§8.3）。

### 5.3 Phase 3：裁剪 + 步进 + 聚合

[L459-519](../../../hyper_models/recipes/llm/train_ft.py#L459-L519)：

```python
grad_norm = scale_grads_and_clip_grad_norm(
    self.model_parts, max_grad_norm,
    num_label_tokens=num_label_tokens if (token_weighted and not pp_enabled) else None)
self.checkpointer.maybe_wait_for_staging()
for opt in optimizers: opt.step(); opt.zero_grad()
for sch in schedulers: sch.step()          # lr_scheduler=None → 空列表守卫
global_loss = _dp_cp_all_reduce_sum(sum(loss_buffer), dp_cp_mesh) / max(N_global, 1)
tps = N_global / step_time
mfu = calculate_mfu(tps, mfu_calc.flops_per_token, mfu_calc.peak_tflops, world_size)
lr  = schedulers[-1].get_last_lr()[0] if schedulers else optimizer.param_groups[0]["lr"]
```

**除 N 的唯一性**（03 §10.1 推导，本骨架最关键的数值正确性约束）：

```
calculate_loss 返回 raw ce_sum_local（不除 N）
backward: (ce_sum_local * dp_size).backward()   # 乘 dp_size 抵消 FSDP2 的 DP-mean
scale_grads: p.grad /= N_global                  # ← token-mean 归一化的唯一除法点
日志:     global_loss = Σ ce_sum_local(all-reduce) / N_global
```

- `num_label_tokens` **仅在 token_weighted 且非 PP 时传入**：rank_average 的 mean 尺度 loss 不能再除 N；PP 场景由 PP runtime 平衡（中间 stage 的 N 不准确）。
- CP 不乘因子：每个 cp rank 处理不同序列段（非冗余计算），N_global 也不除 cp_size。
- 梯度累积天然正确：分子分母跨 microbatch 同步累加，等价于拼成大 batch 的 token-mean。

返回 `{"loss","grad_norm","lr","step_time","tps","mfu","num_tokens"}`，供 StepState 构造。

---

## 6. 前向 + 反向传播（03 §8）

[recipes/llm/train_ft.py:L522-606](../../../hyper_models/recipes/llm/train_ft.py#L522-L606)

六个 Step：

1. **数据 → GPU** [L531-536](../../../hyper_models/recipes/llm/train_ft.py#L531-L536)：tensor 项 `.to(self._device, non_blocking=True)`。
2. **CP batch 准备** [L538-545](../../../hyper_models/recipes/llm/train_ft.py#L538-L545)：`cp_size > 1` 时优先 `model.prepare_model_inputs_for_cp(**batch)`，否则 `shard_batch_for_cp(batch, self.mesh.cp_mesh)`（05 canonical，contiguous chunk 切分 + seq_lens 重算）。**K/V all-gather 不在训练循环**——由 apply_sharding_plan 编译期注入的 CP inner-attention wrapper 在 forward 内部完成，循环无需任何 CP context/hook。
3. **分离 labels** [L547-548](../../../hyper_models/recipes/llm/train_ft.py#L547-L548)：`batch.pop("labels", None)`。
4. **前向** [L550-562](../../../hyper_models/recipes/llm/train_ft.py#L550-L562)：`get_sync_ctx(is_optim_step=True, defer=idx != num_batches-1)` 切 FSDP sync 开关（§8.3）→ `filter_forward_kwargs(model, batch)` 过滤签名不接受的 kwarg → `model(**filtered_batch)`。
5. **Loss** [L564-591](../../../hyper_models/recipes/llm/train_ft.py#L564-L591)：`calculate_loss(loss, logits=, labels=, model=, num_label_tokens=, loss_aggregation=, hidden_states=, lm_weight=)`（dispatcher，§9.4）→ 有 `mtp_per_depth_logits` 时追加 `calculate_mtp_loss` → `loss_buffer.append(local_loss.detach())`。
6. **反向** [L594-597](../../../hyper_models/recipes/llm/train_ft.py#L594-L597)：`(local_loss * dp_group_size).backward()`——乘 dp_size 抵消 FSDP2 的 DP-mean 除法（§5.3 推导）。

---

## 7. 验证流程（03 §6.1）

[recipes/llm/train_ft.py:L348-410](../../../hyper_models/recipes/llm/train_ft.py#L348-L410)

`_run_validation_epoch(val_dl)`：`model_parts.eval()` → `torch.no_grad()` 逐 batch（数据上卡 → CP 准备（与训练一致）→ 分离 labels → filter_forward_kwargs → forward）→ 恢复 `train()`（finally 保证）。

**聚合口径**（第六轮 P1 修复点，易错）：

```python
# 每 microbatch：累加 raw ce_sum（不除 N）与本 rank token 数
total_loss_sum += local_ce_sum.detach().item()
total_label_tokens += num_tok
# 末尾：分子分母分别 DP+CP all-reduce SUM 后再相除
global_val_loss = all_reduce(total_loss_sum) / max(all_reduce(total_label_tokens), 1)
```

旧实现 `local_ce/global_tok` 每步相除再 DP-mean，少算 dp_size×cp_size 倍。返回 `{"loss": float, "num_tokens": int}`（num_tokens 供加权聚合），由 EvaluateCallback 收进 `_last_val_losses` 并 `log_val_metrics`。

---

## 8. 训练组件详解（hyper_models/components/training/）

### 8.1 StepScheduler —— 训练节奏控制（03 §4/§4.1）

[step_scheduler.py](../../../hyper_models/components/training/step_scheduler.py)

**构造** [L104-154](../../../hyper_models/components/training/step_scheduler.py#L104-L154)：`grad_acc_steps = max(global // (local * dp), 1)` + **整除防御**（不整除抛 ValueError——floor division 截断会使每步样本数与配置不符）。安装 `DistributedSignalHandler`（`__enter__` 替换 SIGTERM handler）。

**迭代协议**：
- **`epochs` property** [L157-162](../../../hyper_models/components/training/step_scheduler.py#L157-L162)：`range(start_epoch, num_train_epochs)`，每 epoch 后检查 `_max_steps_reached` 提前退出。
- **`__iter__`** [L178-200](../../../hyper_models/components/training/step_scheduler.py#L178-L200)：按 grad_acc_steps 攒 batch_buffer，**step 在 yield 之前自增**——循环体读到的 step 是"当前正在训练的步"（1 起）。若 yield 后才自增，冷启动首步 step=0 会让 `step % interval == 0` 判断（is_ckpt/is_log）在 step 0 误触发；断点续训首步也会与 checkpoint 已完成的 step 重号。余量（drop_last 时不应到达）同样先自增再 yield。每组 yield 后检查 `_max_steps_reached or sigterm_received` 即返回。
- **`_max_steps_reached`** [L204-208](../../../hyper_models/components/training/step_scheduler.py#L204-L208)：`max_steps > 0 and step >= max_steps`——**max_steps ≤ 0（默认 -1）表示 epoch 驱动不限步**，必须排除，否则 `step >= -1` 恒真导致首步即退出且 is_ckpt_step 每步为 True。

**步类型判断**（StepState 的唯一来源）：

| property | 逻辑 | 备注 |
|---|---|---|
| `is_ckpt_step` [L213-230](../../../hyper_models/components/training/step_scheduler.py#L213-L230) | `step % ckpt_every == 0` 或最终步或 sigterm 或（save_every_epoch 且未标记） | epoch 语义 = 每 epoch **开头**保存一次（迭代协议感知不到 epoch 末尾），保存后由 `mark_epoch_ckpt_saved()` 关闭 |
| `is_log_step` [L239-241](../../../hyper_models/components/training/step_scheduler.py#L239-L241) | `step % log_remote_every == 0` | is_log_remote_step 的别名 |
| `is_gc_step` [L243-247](../../../hyper_models/components/training/step_scheduler.py#L243-L247) | gc_every_steps 非 None 且整除 | None → 恒 False |
| `is_val_step` [L250-254](../../../hyper_models/components/training/step_scheduler.py#L250-L254) | val_every_steps 非 None → 整除；None → 退化为 is_ckpt_step | |
| `sigterm_received` [L257-266](../../../hyper_models/components/training/step_scheduler.py#L257-L266) | `any(sig_handler.signals_received())`，带 `_sigterm_flag` 缓存 | **内含 all_gather 集合通信**——所有 rank 必须在 `__iter__` 同一位置同步调用，否则死锁 |

**状态序列化** [L273-284](../../../hyper_models/components/training/step_scheduler.py#L273-L284)：`state_dict() → {"step","epoch"}`（AutoModel 兼容键名）；`load_state_dict` 兼容旧键名 `global_step/current_epoch`，并**同步 `start_epoch = epoch`**——否则 `range(start_epoch, ...)` 仍从 0 重启，断点续训失效。

**`set_epoch`** [L165-171](../../../hyper_models/components/training/step_scheduler.py#L165-L171)：sampler 有 `set_epoch` 则调用（shuffle 种子），并重置 `_epoch_ckpt_saved`。**`cleanup()`** [L287-294](../../../hyper_models/components/training/step_scheduler.py#L287-L294)：恢复原始 SIGTERM handler，Recipe 训练结束时调用。

**`StepSchedulerConfig.build`** [L55-95](../../../hyper_models/components/training/step_scheduler.py#L55-L95)：注入运行时依赖（dataloader/dp_world_size/local_batch_size/start_step/start_epoch）；`global_batch_size=None` 时退化为 `local * dp`（grad_acc=1）。

> **用例**：[`tests/components/training/test_step_scheduler.py`](../../../tests/components/training/test_step_scheduler.py)（24 例：grad_acc 计算/整除防御/下限 1、迭代分组 [3,3,3,1]/step 自增/断点起始/max_steps 停止与负数不限、五种步标记、state_dict 新旧键名、epochs 范围与提前退出、set_epoch、sigterm 停止、Config.build 注入）。

### 8.2 Callback 系统 —— 混合方案（03 §4.2）

[callback.py](../../../hyper_models/components/training/callback.py)

#### 8.2.1 StepState [L35-57](../../../hyper_models/components/training/callback.py#L35-L57)

**frozen dataclass**——接收方只读（修改抛 `FrozenInstanceError`）。16 字段三组：步信息（step/epoch/is_final_step）、时序标记（is_ckpt/is_val/is_log/is_gc/sigterm_received，全部由 StepScheduler 计算）、训练指标（loss/grad_norm/lr/tps/mfu/num_tokens）。

#### 8.2.2 内置 Callback（7 个）

接口只有 3 个回调点（`on_train_begin / on_step_end / on_train_end`，[L63-79](../../../hyper_models/components/training/callback.py#L63-L79)）；`CallbackManager` [L84-103](../../../hyper_models/components/training/callback.py#L84-L103) 按注册顺序依次调用，空管理器安全。

| Callback | 触发条件 | 行为 |
|---|---|---|
| `CheckpointCallback` [L108-125](../../../hyper_models/components/training/callback.py#L108-L125) | is_ckpt_step 且**非** is_final_step | `recipe.save_checkpoint(...)` + `mark_epoch_ckpt_saved()`；最终步跳过（由循环末尾 final save 统一处理，防重复） |
| `EvaluateCallback` [L128-141](../../../hyper_models/components/training/callback.py#L128-L141) | is_val_step 且 val_dataloaders 非空 | 逐 val_dl 跑 `_run_validation_epoch` → `_last_val_losses` + `log_val_metrics` |
| `LoggingCallback` [L144-157](../../../hyper_models/components/training/callback.py#L144-L157) | is_log_step | 一行 INFO（step/loss/lr/grad_norm/tps/mfu） |
| `TqdmCallback` [L160-187](../../../hyper_models/components/training/callback.py#L160-L187) | 每步（rank 0） | pbar 延迟到 `on_train_begin` 创建（此时 step_scheduler 已存在，可读到断点起始步 initial） |
| `WandbCallback` [L190-205](../../../hyper_models/components/training/callback.py#L190-L205) | is_log_step | `wandb.log`（函数内 import，wandb 为可选依赖） |
| `GCCallback` [L208-217](../../../hyper_models/components/training/callback.py#L208-L217) | is_gc_step | `recipe._maybe_collect_garbage()` |
| `SIGTERMHandler` [L220-236](../../../hyper_models/components/training/callback.py#L220-L236) | sigterm_received | `step_scheduler.cleanup()` + `max_steps = state.step`（令迭代器下次取数即退出）；**不在此保存**——SIGTERM 步保存由 final save 覆盖 |

#### 8.2.3 `build_callback_manager` 工厂 [L239-258](../../../hyper_models/components/training/callback.py#L239-L258)

固定注册前 4 个 + SIGTERMHandler（共 5 个）；`cfg.wandb.enabled` → 追加 WandbCallback；`cfg.step_scheduler.gc_every_steps` 非空 → 追加 GCCallback（共至多 7 个）。

> **用例**：[`tests/components/training/test_callback.py`](../../../tests/components/training/test_callback.py)（24 例：frozen 契约、注册顺序调用、空安全、7 个内置 Callback 的触发/跳过分支、工厂默认 5 个与全量 7 个）。

### 8.3 grad_accum —— 梯度累积辅助（03 §7.1）

[grad_accum.py](../../../hyper_models/components/training/grad_accum.py)

**FSDP 梯度同步双层管理**：

| 层级 | 位置 | 函数 |
|---|---|---|
| 外层 | microbatch 循环边界 | `prepare_for_grad_accumulation` [L59-64](../../../hyper_models/components/training/grad_accum.py#L59-L64)（关 sync + 标 `_grad_accum_state="deferred"`）/ `prepare_for_final_backward` [L67-74](../../../hyper_models/components/training/grad_accum.py#L67-L74)（开 sync + 标 "final"；多 part 时 `_attach_pp_backward_hooks` **桩——直接 NotImplementedError**，PP>1 落地前不可用） |
| 内层 | 每个 microbatch 前向期间 | `get_sync_ctx` [L38-56](../../../hyper_models/components/training/grad_accum.py#L38-L56)——所有分支返回 `nullcontext()`，副作用只是切 sync 开关：非 optim step 不碰；defer=True 调 `set_requires_gradient_sync(False)`；最后 microbatch 不动作（外层已开） |

FSDP2 的 DP all-reduce 由 `set_requires_gradient_sync(True)` 在本轮 backward 末尾自动触发，无需显式上下文管理器。

**其他函数**：

- `prepare_after_first_microbatch` [L89-95](../../../hyper_models/components/training/grad_accum.py#L89-L95)：`reset_lazy_init()` 预热 unshard 缓存 + `_first_microbatch_done`。
- `set_requires_gradient_sync` [L98-104](../../../hyper_models/components/training/grad_accum.py#L98-L104)：批量开关。
- **`scale_grads_and_clip_grad_norm(model_parts, max_norm, num_label_tokens=None)`** [L107-126](../../../hyper_models/components/training/grad_accum.py#L107-L126)：**token-mean 归一化的唯一除法点**——`num_label_tokens` 非 None 且 >0 时逐参数 `p.grad.detach_().div_(N)`（detach_ 防御 FSDP2 hook），再 `clip_grad_norm_` 返回裁剪前总范数。零值保护：N≤0 跳过除法。
- **通信/工具**（Recipe 与验证流程共用）：
  - `_dp_cp_all_reduce_sum` [L129-143](../../../hyper_models/components/training/grad_accum.py#L129-L143)：标量自动 wrap tensor；mesh None 时退化全局 all-reduce（dist 未初始化则 no-op）；
  - `_dp_all_reduce_avg` [L146-163](../../../hyper_models/components/training/grad_accum.py#L146-L163)：纯 DP 维 mean（val loss 用；CP 非冗余不参与）；
  - `calculate_mfu` [L166-180](../../../hyper_models/components/training/grad_accum.py#L166-L180)：`(tps × flops_per_token) / (peak_tflops × world × 1e12)`，clamp 到 [0,1]，peak≤0 返回 0；
  - `filter_forward_kwargs` [L183-190](../../../hyper_models/components/training/grad_accum.py#L183-L190)：`inspect.signature(model.forward)` 过滤；签名不可内省（如 C 扩展）→ 返回全部 batch 兜底；
  - `AutoMFU.from_config` [L229-243](../../../hyper_models/components/training/grad_accum.py#L229-L243)：`flops_per_token = 6 × 参数量`；peak_tflops 由 `_infer_peak_tflops`（[L246-260](../../../hyper_models/components/training/grad_accum.py#L246-L260)：H100/H800→989、A100/A800→312、H20→148、V100→125、4090→330、默认 200）；
  - `_update_latest_symlink` [L270-281](../../../hyper_models/components/training/grad_accum.py#L270-L281)（§2.4）、`_is_rank_0` [L263-267](../../../hyper_models/components/training/grad_accum.py#L263-L267)、`setup_magi` [L213-226](../../../hyper_models/components/training/grad_accum.py#L213-L226)、`calculate_mtp_loss` [L193-210](../../../hyper_models/components/training/grad_accum.py#L193-L210)（§9.5，loss/utils.py 有同名 canonical 版本，Recipe 从 loss 包导入）。

> **用例**：[`tests/components/training/test_grad_accum.py`](../../../tests/components/training/test_grad_accum.py)（29 例：FakeFSDP 验证四个 prepare/get_sync_ctx 的开关序列与状态标记、scale_grads 除法/跳过/零保护、filter 双分支、MFU 公式/零峰/截断、GPU 名映射矩阵、symlink 原子更新、all_reduce mock）。

### 8.4 DistributedSignalHandler（03 §11）

[signal_handler.py:L29-60](../../../hyper_models/components/training/signal_handler.py#L29-L60)

`__enter__` 保存并替换 SIGTERM handler（`_handler` 只置位 `_signal_received`，不做 IO 以外的任何事）；`signals_received()` 把标志搬到当前设备做 `all_gather`——任一 rank 收到 → 全体 True。**dist 未初始化时直接返回本地标志**（单进程/测试路径安全）。选型注记：用当前 CUDA 设备走默认 NCCL group，而非新建 gloo group（单后端少一个生命周期管理对象，每步 1 个 int32 开销可忽略）。

### 8.5 StatefulRNG（桩）

[rng.py:L24-52](../../../hyper_models/components/training/rng.py#L24-L52)：`ranked=True` 且 dist 已初始化时 `seed += rank`（rank 感知种子）；`state_dict/load_state_dict` 保存 generator 状态（checkpoint kind="rng" 对称恢复）。完整的分布式 RNG 对齐待落地。

---

## 9. Loss 组件（03 §10）

### 9.1 MaskedCrossEntropy —— fp32 recast 掩码 CE

[masked_ce.py:L24-80](../../../hyper_models/components/loss/masked_ce.py#L24-L80)

构造参数：`fp32_upcast=True / ignore_index=-100 / reduction="sum"`。forward 流程：labels 设备对齐 → view 展平 → 可选 mask（0 位填 ignore_index）→ **fp32_upcast**（bf16 logits 直接 CE 会因 log_softmax + nll_loss 大数值累加损失精度）→ `F.cross_entropy(reduction=self.reduction)`。

`num_label_tokens` 参数：仅 reduction="sum" 时可用（否则 ValueError）；N=0 → 返回 `loss * 0.0`（零保护，保留计算图）；否则 `loss / N`。**注意**：主训练路径不向它传 N——除 N 唯一点在 scale_grads（§5.3），此参数服务自定义 loss 场景。

### 9.2 FusedLinearCrossEntropy（桩）

[linear_ce.py:L28-72](../../../hyper_models/components/loss/linear_ce.py#L28-L72)：分离实现暂代融合 kernel（`matmul(hidden, lm_weight.T)` → CE sum）。`lm_weight=None` 时 `hidden_states` 直接视为预计算 logits——对应 dispatcher 的 logits 降级路径。生产版依赖 cut_cross_entropy。

### 9.3 LossConfig —— typed config

[loss/__init__.py:L34-53](../../../hyper_models/components/loss/__init__.py#L34-L53)：

```python
@dataclass
class LossConfig:
    _target_: Optional[type] = None
    loss_aggregation: str = "token_weighted"   # 训练循环读取（_token_weighted 判定）
    kwargs: dict = field(default_factory=dict)
    def build(self): return MaskedCrossEntropy() if self._target_ is None else self._target_(**self.kwargs)
```

`build_loss_config(factory, **kwargs)` 归一化入口（_target_ factory + kwargs → LossConfig）。

### 9.4 `calculate_loss` dispatcher（03 §10）

[loss/utils.py:L46-109](../../../hyper_models/components/loss/utils.py#L46-L109)

两条路径：

- **路径 A（FusedLinearCrossEntropy）**：`hidden_states + lm_weight` 齐备 → 融合调用；缺 hidden_states → 降级 logits 路径（shift 后调 `loss_fn(logits_flat, labels_flat)`，lm_weight=None 分支）；两者皆无 → ValueError。
- **路径 B（标准 logit-based）**：`logits[..., :-1, :]` vs `labels[..., 1:]` **自回归 shift**（causal LM 恒需要）→ 调 `loss_fn(logits_flat, labels_flat)`。

**返回值恒为 raw ce_sum（不除 N）**——token_weighted 与 rank_average 的差异由 loss_fn 自身的 reduction（sum vs mean）与下游 scale_grads 是否传 N 共同体现（§5.3）；`num_label_tokens` kwarg 被 pop 保留但不在 dispatcher 内使用（留给自定义 loss 归一化）。

### 9.5 `calculate_mtp_loss`

[loss/utils.py:L112-131](../../../hyper_models/components/loss/utils.py#L112-L131)：Multi-Token-Prediction 辅助 loss（Qwen3.5 等）——逐 depth shift + CE sum 累加，与主 loss 同尺度（raw sum，fp32 起点）。

> **用例**：[`tests/components/training/test_loss.py`](../../../tests/components/training/test_loss.py)（21 例：sum/mean 数值对拍、ignore_index 默认与自定义、fp32 upcast 开/关（mock F.cross_entropy 捕获 dtype）、全屏蔽得 0、num_label_tokens 归一/零保护/reduction 冲突、dispatcher shift 正确性、两种 loss_aggregation、融合路径与降级路径、无输入报错）。

---

## 10. Optimizer 与 LR Scheduler（03 §9）

### 10.1 Optimizer.Config.build —— typed 构建契约

[optimizer.py:L31-45](../../../hyper_models/components/optim/optimizer.py#L31-L45)：基类声明新契约 `build(model, *, optimizer_init=None, device_mesh=None, is_peft=False) -> list[Optimizer]`（**list 为 canonical**，nemo_automodel 惯例；04 OptimizerState 接受 list）。`max_grad_norm: float = 1.0` 字段由训练循环消费（§5.3）。

**`AdamW.Config.build`** [L61-88](../../../hyper_models/components/optim/optimizer.py#L61-L88)：逐 `model.parts`（无 parts 则 `[model]`）——优先复用 `optimizer_init.param_groups`（ShardingPlan 推导的分组，weight_decay 已写入组内，不重复覆盖），否则现场 `_build_param_groups(part, self.weight_decay)`。

**`OptimizerFromFactoryConfig`** [L91-118](../../../hyper_models/components/optim/optimizer.py#L91-L118)：外部优化器 escape hatch（YAML `_target_: dion.Muon` 场景）——同口径复用 param_groups，`weight_decay` 从 kwargs 剔除后透传 factory。

**`build_optimizer_config(target, kwargs)`** [L125-145](../../../hyper_models/components/optim/optimizer.py#L125-L145) 归一化入口：Config 实例直返 → 字符串查 `OPTIMIZER_CONFIG_REGISTRY`（`"adamw"`/`"torch.optim.adamw"`）或 `_import_from_path` → Config 子类实例化 → 其他 callable 包成 FromFactory。

**参数分组（§9.5）**：`_build_param_groups` [L156-180](../../../hyper_models/components/optim/optimizer.py#L156-L180)（dedup by id、跳过 requires_grad=False，decay/no_decay 两组）+ `_is_no_decay` [L183-188](../../../hyper_models/components/optim/optimizer.py#L183-L188)（`bias/norm/rmsnorm/layernorm/ln_` 模式）。

**`OptimizerInit`** [L197-227](../../../hyper_models/components/optim/optimizer.py#L197-L227)：build_model 导出的 param 分组描述（`param_groups/device_mesh/is_peft/tp_grad_info`），`from_distributed_setup` 类方法现场推导。

### 10.2 LRSchedulerConfig（桩）

[lr_scheduler.py:L49-125](../../../hyper_models/components/optim/lr_scheduler.py#L49-L125)：step-based 字段与 AutoModel 对齐（lr_warmup_steps/lr_decay_steps/lr_decay_style/init_lr/max_lr/min_lr + WD 调度 + WSD）。`build(optimizer, step_scheduler)`：未设置字段从 step_scheduler.max_steps 与 optimizer param_groups 推断（init_lr/max_lr 取 `param_groups[0]["lr"]`——torch AdamW 无 "initial_lr" 键）。**当前桩用 torch 原生 LinearLR→CosineAnnealingLR 的 SequentialLR 暂代**；完整实现需 port nemo_automodel 的 `OptimizerParamScheduler`（03 §9.6 注明不得 import nemo_automodel，需 port 进本仓）。

**`RatioBasedLRSchedulerConfig`** [L128-141](../../../hyper_models/components/optim/lr_scheduler.py#L128-L141)：ratio 便利 wrapper——`warmup_steps_ratio/min_lr_ratio` 在 build 时换算为绝对步数后走父类。

---

## 11. 桩模块清单（接口已冻结，实现待落地）

| 模块 | 已冻结接口 | 桩行为 | 落地依赖 |
|---|---|---|---|
| [distributed/infrastructure.py](../../../hyper_models/components/distributed/infrastructure.py) | `MeshContext`（dp/tp/cp/pp/ep size+rank、`pp_enabled`、`dp_cp_mesh` property）、`DistributedSetup`、`initialize_distributed`、`create_distributed_setup_from_config`、`_is_rank_0`、`setup_logging`、`apply_cache_compatibility_patches`、`destroy_process_group` | MeshContext 默认全 1（device_mesh=None）；create_* 仅从 `cfg.accelerator` 填 dp/tp size | 06：DeviceMesh 构建 |
| [_transformers/auto_model.py](../../../hyper_models/_transformers/auto_model.py) | `HyperAutoModelForCausalLM/ImageTextToText/SequenceClassification.from_pretrained/from_config`（HF 兼容签名） | `_build_model` 编排骨架完整（meta 判定 → `_init_model` → `apply_model_infrastructure`）；FSDP2Manager/AutoPipeline 按 strategy_config/pp_size 实例化 | 01 §6-8 + 06 §4 |
| [_transformers/infrastructure.py](../../../hyper_models/_transformers/infrastructure.py) | `instantiate_infrastructure → (sharding_planner, fsdp2_manager, autopipeline)`；`apply_model_infrastructure`（01 §8.3 canonical Step 3-11） | ShardingPlanner 真实可用；plan/apply 完整传入 cp/ep/sequence_parallel/loss_parallel；FSDP2Manager stub 保留 `parallelize` 入口 | 01 §8 + 05 §4 |
| [components/distributed/fsdp2.py](../../../hyper_models/components/distributed/fsdp2.py) | `FSDP2Manager` + `_instantiate_fsdp2` | 桩：`parallelize` 记录参数并原样返回 model | 06 §4 |
| [components/distributed/pipelining.py](../../../hyper_models/components/distributed/pipelining.py) | `AutoPipeline` + `_instantiate_pipeline` | 桩：`build()` 仅日志，不改动 model | 06 §8.2 |
| [models/common/model_utils.py](../../../hyper_models/components/models/common/model_utils.py) | `build_model(model_cfg, peft_config, distributed_setup) -> (model, OptimizerInit)` | Path A（HyperAutoModel）/ Path B（HF 原生 + 手动 apply infra）分派完整 | 01 §6.7 |
| [checkpoint/config.py](../../../hyper_models/components/checkpoint/config.py) + [checkpointing.py](../../../hyper_models/components/checkpoint/checkpointing.py) | `CheckpointingConfig`（04 §4 全字段）+ `Checkpointer.save_model/save_optimizer/load_model/load_optimizer/save_on_dp_ranks/maybe_wait_for_staging/close` | torch.save/load 暂代 DCP；async 相关 no-op | 04：DCP + stateful_wrappers |
| [data/build_dataloader.py](../../../hyper_models/data/build_dataloader.py) | `build_dataloader(...) -> (DataLoader, tokenizer)`；`build_validation_dataloader(...) -> dict[str, DataLoader]` | dummy TensorDataset（train 100 样本 drop_last / val 20 样本不 drop）+ warning 日志 | 02：数据管道 |

> 零依赖边界注记：训练骨架（`hyper_models/recipes/`）是上述组件的**消费方**；`components/distributed/` 组件不反向 import `hyper_models/recipes`（由 [`test_s5_zero_dep_lint.py`](../../../tests/components/distributed/test_s5_zero_dep_lint.py) 守护）。

---

## 12. 端到端走查：max_steps=2 的单进程训练

把 §3-§7 串起来（对应 [`test_finetune_recipe.py::test_train_loop_epochs`](../../../tests/hyper_models/recipes/test_finetune_recipe.py)，全 mock 单进程）：

```
cfg: step_scheduler(max_steps=2, local_batch_size=1, global_batch_size=1),
     loss=LossConfig(), checkpoint=CheckpointingConfig(restore_from=None)

setup(cfg):
  ① dist 未初始化环境 → initialize_distributed 被 mock（真实环境 init_process_group）
  ④ DistributedSetup(mesh_context=MeshContext())  → dp_cp_mesh=None（all-reduce 退化 no-op）
  ⑦ LossConfig.build() → MaskedCrossEntropy(reduction="sum")
  ⑪ build_model → (nn.Linear(4,4), None)         → model_parts=[model]
  ⑫ optimizer.build → [AdamW(param_groups, lr=1e-3)]
  ⑬ build_dataloader → 4 个 {input_ids, labels} batch
  ⑮ StepScheduler: grad_acc = 1//(1×1) = 1       → 每 batch 一个 optim step
  ⑰ register_state × 6
  ⑱ restore_from=None → 跳过

run_train_validation_loop:
  epoch 0: set_epoch(0)
    batch 1 → step=1: _run_train_optim_step
      Phase 1: N_global = labels≠-100 计数（dp_cp_mesh=None → 不通信）
      Phase 2: 单 microbatch；is_last=True → prepare_for_final_backward
               forward → calculate_loss（shift + CE sum，fp32）
               (loss × dp_size=1).backward()
      Phase 3: grads /= N_global → clip → opt.step/zero_grad → scheduler.step
               global_loss = ce_sum / N_global
    on_step_end(StepState(step=1, is_final_step=False, ...))
    batch 2 → step=2: 同上；_max_steps_reached=True → 迭代器返回
    on_step_end(StepState(step=2, is_final_step=True, ...))
      → CheckpointCallback 看到 is_final_step → 跳过（防与 final save 重复）
  save_checkpoint(ckpt_dir, epoch=0, step=2, ..., is_final_checkpoint=True)
    → epoch_0_step_2/{model,optimizer,scheduler.pt,rng/...,dataloader/...,extra_state.json}
    → LATEST → epoch_0_step_2
  finally: on_train_end + checkpointer.close
```

---

## 13. 速查：常用入口与典型用法

```python
from hyper_models.recipes import RECIPE_REGISTRY                    # {"FinetuneRecipe": FinetuneRecipe}
from hyper_models.recipes.llm.train_ft import FinetuneRecipe
from hyper_models.trainer.config import TrainerConfig

# 1. 配置（YAML 经 parse_training_args 或手工构造）
cfg = TrainerConfig(model=..., ...)
cfg.step_scheduler.max_steps = 1000
cfg.step_scheduler.local_batch_size = 2
cfg.step_scheduler.global_batch_size = 32
cfg.checkpoint.checkpoint_dir = "outputs/run0"
cfg.checkpoint.restore_from = "LATEST"                # 断点续训

# 2. 训练
recipe = RECIPE_REGISTRY[cfg.recipe]()
recipe.setup(cfg)                                     # ①–⑲
recipe.run_train_validation_loop()                    # 主循环 + final save

# 3. 自定义 Callback
from hyper_models.components.training import TrainingCallback
class MyCallback(TrainingCallback):
    def on_step_end(self, state):                     # state: frozen StepState
        if state.is_log_step: ...
recipe.callback_manager.register(MyCallback())

# 4. 新增有状态组件（一行接入 checkpoint save/load 对称）
recipe.my_component = ...
recipe.register_state("my_component", "rng")          # kind 决定 _state_path
```

**排障速查**：

| 症状 | 首先检查 |
|---|---|
| 首步就保存/打日志 | StepScheduler `__iter__` 的 step 必须 yield 前自增（§8.1） |
| 第一步即退出、每步 is_ckpt_step | `max_steps=-1` 未被 `_max_steps_reached` 排除（§8.1） |
| 断点续训从 epoch 0 重启 | `load_state_dict` 未同步 `start_epoch`（§8.1） |
| 梯度数值差 N 倍 | 除 N 是否只在 scale_grads 一处（§5.3）；rank_average 是否误传 num_label_tokens |
| val loss 少 dp×cp 倍 | 分子分母须分别 all-reduce 后再相除（§7） |
| SIGTERM 死锁 | sigterm_received 内含 all_gather，所有 rank 须同步调用（§8.1） |
| final 步保存两次 | CheckpointCallback 必须跳过 is_final_step（§8.2.2） |
| is_async 保存失败 | final save 必须先于 checkpointer.close()（§4） |
| setup 循环 import | TrainerConfig 仅在 TYPE_CHECKING 下引入 callback（§1） |
