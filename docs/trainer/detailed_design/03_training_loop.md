# Hyper-Parallel 训练循环详细设计

> 参考实现：[AutoModel `recipes/llm/train_ft.py`](../../../auto_model/Automodel/nemo_automodel/recipes/llm/train_ft.py)
> 上下文设计：[dual_mode_dtensor_parallel_strategy.md](../dual_mode_dtensor_parallel_strategy.md)

---

## 0. 模块 import 约定

本文档代码片段假定如下 import（实际分散在各组件文件顶部）：

```python
import os
import json
import time
import signal
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

# FSDP2（torch >= 2.4）
from torch.distributed.fsdp import FSDPModule          # FSDP2 包裹器
# DeviceMesh
from torch.distributed.device_mesh import DeviceMesh

from hyper_models.components.checkpoint.config import CheckpointingConfig

logger = logging.getLogger(__name__)
```

> 注：`FSDPModule` 统一从 `torch.distributed.fsdp` 导入（torch >= 2.4；
> torch 2.13 实测无 `torch.distributed.fsdp2` 模块，不设 fallback 分支）。
> `nullcontext` 用于 `get_sync_ctx` 的各分支统一返回 ContextManager（§7.1）。

---

## 1. 模块职责

提供完整的训练 Recipe——组件构建、训练主循环、梯度累积、优化器步进、损失计算。

### 核心文件

| 文件 | 职责 |
|------|------|
| `recipes/base_recipe.py` | `BaseRecipe` — `__state_tracked` 自动状态追踪 + checkpoint 保存/恢复 + `CallbackManager` |
| `recipes/llm/train_ft.py` | `FinetuneRecipe` — LLM 训练 recipe（setup + 训练循环） |
| `recipes/vlm/finetune.py` | `FinetuneRecipeForVLM` — VLM 训练 recipe |
| `hyper_models/components/training/step_scheduler.py` | `StepScheduler` — 训练节奏控制 + SIGTERM 响应 |
| `hyper_models/components/training/callback.py` | `TrainingCallback` / `StepState` / `CallbackManager` — 混合 Callback 系统 |
| `hyper_models/components/training/rng.py` | `StatefulRNG` / `ScopedRNG` |
| `hyper_models/components/training/signal_handler.py` | `DistributedSignalHandler` — SIGTERM 分布式协调 |
| `hyper_models/components/training/grad_accum.py` | 梯度累积 + FSDP sync 管理 |
| `hyper_models/components/optim/optimizer.py` | Optimizer 构建（AdamW + ChainedOptimizer） |
| `hyper_models/components/optim/lr_scheduler.py` | LR Scheduler 构建 |
| `hyper_models/components/loss/masked_ce.py` | `MaskedCrossEntropy` |
| `hyper_models/components/loss/utils.py` | Loss 计算工具 |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/trainer/base.py` (大部分) | `recipes/base_recipe.py` + `hyper_models/components/training/` |
| `hyper_parallel/trainer/llm_trainer.py` | `recipes/llm/train_ft.py` |
| `hyper_parallel/trainer/vl_trainer.py` | `recipes/vlm/finetune.py` |
| `hyper_parallel/trainer/utils/loss.py` | `hyper_models/components/loss/` |
| 旧的 `_make_micro_batch_iterator` | `StepScheduler` 迭代器 |
| 旧的 callback 架构（`hyper_parallel/trainer/callbacks/`） | `hyper_models/components/training/callback.py` — 混合方案：核心训练显式编排，`on_step_end` 回调驱动外围关注点 |

---

## 2. 总入口调用时序：从 `main()` 到训练循环

训练流程的全部工作在 `main()` → `recipe.setup()` → `recipe.run_train_validation_loop()` 三个入口完成。以下是完整的调用树。

> 编号约定：④.x 采用 01 §4.1 canonical 编号（④.4=model、④.5=loss_fn、④.7=checkpointer、
> ④.8=optimizer、④.9=dataloader…），与 01/02 文档对齐；树中缩进顺序为实际执行顺序，
> 编号非单调属预期（如 ④.5 loss_fn 先于 ④.4 model 构建）。

```
main()                                                               # 01_hf_compatibility_layer.md §4
│
├─① cfg = load_yaml_config("train.yaml")                             # 01 §2: YAML → ConfigNode
├─② cfg = RecipeConfig(cfg)                                          # 01 §3: 类型化桥接
│
├─③ recipe = FinetuneRecipe()
├─④ recipe.setup(cfg)                                                # 01 §4: 构建所有组件
│   │
│   ├─④.1 self.dist_env = initialize_distributed("nccl")             # 分布式初始化
│   ├─④.2 self.rng = StatefulRNG(seed=..., ranked=True)              # RNG
│   ├─④.3 self.distributed_setup = create_distributed_setup_from_config(cfg)
│   ├─④.3a self.callback_manager = build_callback_manager(cfg)       # §4.2: Callback 管理器
│   │
│   ├─④.5 self.loss_fn = cfg.loss_fn.build()                         # §10: typed .build()
│   │   → LossConfig → MaskedCrossEntropy() / FusedLinearCrossEntropy()
│   │
│   ├─④.6 self.peft_config = cfg.peft.instantiate()  if configured   # untyped .instantiate()
│   │
│   ├─④.7 self.checkpointer = cfg.checkpoint.build(                  # 04_checkpoint.md: typed .build()
│   │       dp_rank=..., tp_rank=..., pp_rank=...)
│   │
│   ├─④.4 self.model, self.optimizer_init = build_model(               # 01 §6: 构建分片模型
│   │       cfg.model, self.peft_config,
│   │       distributed_setup=self.distributed_setup)
│   │   └─ from_pretrained() → _build_model() → 返回已分片模型
│   │   └─ self.model_parts = self.model.parts or [self.model]
│   │
│   ├─④.8 self.optimizer = cfg.optimizer.build(                      # §9: typed .build()
│   │       model, device_mesh=self.mesh.device_mesh)
│   │   └─ OptimizerConfig → 参数分组 → AdamW(param_groups, ...)
│   │
│   ├─④.9 self.dataloader, self.tokenizer = build_dataloader(...)    # 02_data_pipeline.md
│   ├─④.10 self.val_dataloaders = build_validation_dataloader(...)
│   │
│   ├─④.11 self.step_scheduler = cfg.step_scheduler.build(           # §4: typed .build()
│   │        self.dataloader, dp_size, local_batch_size)
│   │
│   ├─④.12 self.lr_scheduler = cfg.lr_scheduler.build(               # §9.6: typed .build()
│   │        self.optimizer, self.step_scheduler)
│   │
│   ├─④.13 self.load_checkpoint(restore_from)                        # 断点续训恢复
│   └─④.14 self.mfu_calc = AutoMFU.from_config(model)
│
└─⑤ recipe.run_train_validation_loop()                               # §6: 训练主循环
    │
    ├─⑤.0 self.callback_manager.on_train_begin()                     # Callback: 训练开始
    │
    ├─ for epoch in self.step_scheduler.epochs:                       # §4: StepScheduler 控制节奏
    │   └─ self.step_scheduler.set_epoch(epoch)                      # sampler shuffle 种子
    │
    └─ for batches in self.step_scheduler:                            # §4: 按 grad_acc_steps 分组
        │
        ├─⑤.1 train_metrics = self._run_train_optim_step(            # §7: 单步优化（核心：显式）
        │       batches, max_grad_norm)
        │   │
        │   ├─⑤.1.1 统计全局 token 数 (DP all-reduce)
        │   │
        │   ├─⑤.1.2 梯度累积循环 (for each microbatch):
        │   │   └─ self._forward_backward_step(batch)                 # §8: 前向+反向
        │   │       ├─ batch → GPU (non_blocking)
        │   │       ├─ CP batch 准备 (if cp_size > 1)
        │   │       ├─ labels 分离
        │   │       ├─ model(**filtered_batch)                        # 前向传播
        │   │       │   └─ PrecompiledBoundary → DTensor redistribute
        │   │       ├─ calculate_loss(...)                            # §10: dispatcher
        │   │       └─ (loss * dp_size).backward()                    # 反向传播
        │   │
        │   └─⑤.1.3 scale_grads + clip_grad_norm + optimizer.step() + lr_scheduler.step()
        │
        └─⑤.2 self.callback_manager.on_step_end(StepState(...))      # §4.2: Callback（外围）
            ├─ [CheckpointCallback]  is_ckpt_step → save_checkpoint
            ├─ [EvaluateCallback]    is_val_step  → _run_validation_epoch
            ├─ [LoggingCallback]     is_log_step  → 日志输出
            ├─ [TqdmCallback]                     → 进度条更新
            ├─ [WandbCallback]       is_log_step  → 远程日志
            ├─ [GCCallback]          is_gc_step   → 垃圾回收
            └─ [SIGTERMHandler]      sigterm      → 优雅退出
```

**与 01、02 文档的时序衔接**：

```
main()
├─① load_yaml_config()           # 01 §2
├─② RecipeConfig(cfg)            # 01 §3
└─④ recipe.setup(cfg)            # 01 §4
    ├─④.4  model = ...           # 01 §6 (from_pretrained → _build_model)
    ├─④.9  dataloader = ...      # 02_data_pipeline.md §3 (build_dataloader)
    ├─④.8  optimizer = ...       # 本文档 §9
    ├─④.11 step_scheduler = ...  # 本文档 §4
    ├─④.5  loss_fn = ...         # 本文档 §10
    └─④.12 lr_scheduler = ...    # 本文档 §9.6
└─⑤ run_train_validation_loop()  # 本文档 §6
    └─⑤.1 _run_train_optim_step  # 本文档 §7
        └─⑤.1.2 _forward_backward_step  # 本文档 §8
```

---

## 3. BaseRecipe —— 自动状态追踪

> **调用位置**: 时序树 ⑤.3 — `save_checkpoint()` 遍历 `__state_tracked` 自动保存所有组件

### 3.1 核心机制

```python
# recipes/base_recipe.py

class BaseRecipe:
    """训练 Recipe 基类。

    通过 register_state() 显式将组件注册到 __state_tracked
    （(name, kind) 元组列表，kind 对齐 04 `_state_path` 的 state kind）。
    save 侧由 04 Checkpointer 的 `save_model`/`save_optimizer`/`_state_path`
    per-rank 子目录落地；load 侧 canonical `load_checkpoint` 定义在 04 §8
    （Recipe 方法，1 参 `restore_from`，迭代 `__state_tracked`）。
    """

    def __init__(self):
        # 注册表：list[tuple[name, kind]]，kind ∈ {"model","optimizer",
        # "lr_scheduler","rng","dataloader","train_state"}，与 04 `_state_path`
        # kind 一致。
        #
        # 注意：__state_tracked 使用双下划线前缀触发 Python name mangling
        # （实际存储为 _BaseRecipe__state_tracked）。这可以防止子类意外覆盖，
        # 但也意味着若 BaseRecipe 被用作 mixin 且另一个父类也定义了同名属性，
        # name mangling 会为每个类生成独立的属性名，不会冲突。
        # 若未来需要跨类共享此属性，考虑改为单下划线 _state_tracked。
        #
        # 子类注意：在子类中直接访问 self.__state_tracked 会解析为
        # _SubClass__state_tracked（而非 _BaseRecipe__state_tracked），
        # 导致访问到空列表。子类应始终使用 register_state() 方法，
        # 不要直接操作 self.__state_tracked。
        self.__state_tracked: list[tuple[str, str]] = []
        # 反查表，避免重复注册同名状态
        self.__state_names: set[str] = set()
        # Callback 管理器（延迟初始化，在 setup() 中由 build_callback_manager 赋值）
        self.callback_manager: CallbackManager | None = None

    def register_state(self, name: str, kind: str) -> None:
        """显式注册一个需要 checkpoint 追踪的组件。

        name: Recipe 上的属性名（如 "model" / "optimizer"）。
        kind: 04 `_state_path` 所用的 state kind，取值：
              "model" / "optimizer" / "lr_scheduler" / "rng" / "dataloader"
              / "train_state"（如 §5.3 ⑰ 注册 ("step_scheduler", "train_state")）。
        同名重复注册将被忽略。
        """
        if name in self.__state_names:
            return
        self.__state_tracked.append((name, kind))
        self.__state_names.add(name)

    # ── rank / group size 查询（第六轮 P1 修复：补全此前调用点存在但未定义的方法） ──
    # 委托给 self.mesh（MeshContext，见 06 §2）；mesh 在 setup 中由
    # self.distributed_setup.mesh_context 赋值（§4.8 ⑤）。
    def _get_dp_rank(self) -> int:
        mesh = getattr(self, "mesh", None)
        return mesh.dp_rank if mesh is not None else 0

    def _get_tp_rank(self) -> int:
        mesh = getattr(self, "mesh", None)
        return mesh.tp_rank if mesh is not None else 0

    def _get_pp_rank(self) -> int:
        mesh = getattr(self, "mesh", None)
        return mesh.pp_rank if mesh is not None else 0

    def _get_dp_group_size(self) -> int:
        mesh = getattr(self, "mesh", None)
        return mesh.dp_size if mesh is not None else 1

    # ── 外围关注点辅助方法（Callback 调用，定义于此避免悬空引用） ──

    def log_val_metrics(self, val_losses: dict) -> None:
        """记录验证指标（仅 rank 0 输出，EvaluateCallback 在 is_val_step 调用）。

        val_losses 形如 {"validation": {"loss": float, "num_tokens": int}}
        （_run_validation_epoch 的返回结构，见 §6.1）。
        """
        if not _is_rank_0():
            return
        for name, metrics in val_losses.items():
            loss = metrics["loss"] if isinstance(metrics, dict) else metrics
            logger.info("validation/%s loss=%.4f", name, loss)

    def _maybe_collect_garbage(self) -> None:
        """手动触发垃圾回收（GCCallback 在 is_gc_step 调用）。"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Checkpoint save ──
    #
    # save 侧整体委托 04 Checkpointer 走 `_state_path` per-rank 子目录规则
    # （`rng/rng_dp_rank_{r}.pt`、`optimizer/...`、`model/...` 等），
    # 保证 save/load 同源。本方法仅负责遍历 `__state_tracked` 分发。

    def save_checkpoint(self, checkpoint_dir: str, epoch: int, step: int,
                        train_loss: float, val_losses: dict | None = None,
                        is_final_checkpoint: bool = False) -> None:
        """遍历 __state_tracked，按 kind 委托 Checkpointer 保存。

        model / optimizer 走 `self.checkpointer.save_model` / `save_optimizer`
        （per-rank 子目录）；scheduler.pt / extra_state.json 落在 checkpoint
        根目录；dataloader / rng 等通过 `self._state_path`（Recipe 方法）
        落到对应 per-rank 子目录（与 04 load 同源）。

        is_final_checkpoint: 训练结束后的 final save 传 True（04 §5.2 要求，
        用于触发 save_consolidated=final 的 consolidated 权重导出等收尾行为）；
        周期保存保持默认 False。
        """
        path = f"{checkpoint_dir}/epoch_{epoch}_step_{step}/"
        os.makedirs(path, exist_ok=True)

        model_ref = None
        for name, kind in self.__state_tracked:
            obj = getattr(self, name, None)
            if obj is None:
                continue

            if kind == "model":
                if model_ref is None:
                    model_ref = obj
                self.checkpointer.save_model(obj, f"{path}/model")
            elif kind == "optimizer":
                # 【canonical】self.optimizer 为 list[Optimizer]（nemo_automodel
                # 惯例，见 §9.3 build 返回类型），此处**原样**（不拆包）传给
                # checkpointer；04 OptimizerState 接受 list[Optimizer]（以本节
                # 为准，04 侧同步支持 list）。
                # optimizer 子目录名与 04 `_state_path(kind=="optimizer")` 对齐
                self.checkpointer.save_optimizer(model_ref, obj, f"{path}/optimizer")
            elif kind == "lr_scheduler":
                # lr_scheduler build 返回 list[OptimizerParamScheduler]，
                # 逐个保存 state_dict 到 scheduler.pt（聚合 dict）。
                # 04 `_state_path(kind=="lr_scheduler")` 返回 `{path}/scheduler.pt`，
                # load 侧遍历 list 逐个 load_state_dict（见 04 §8）。
                torch.save(
                    {f"sch_{i}": s.state_dict() for i, s in enumerate(obj)},
                    f"{path}/scheduler.pt",
                )
            elif kind == "rng":
                rng_path = self._state_path(path, name, kind)
                torch.save(obj.state_dict(), rng_path)
            elif kind == "dataloader":
                dl_path = self._state_path(path, name, kind)
                torch.save(obj.state_dict(), dl_path)
            elif kind == "train_state":
                # 训练元信息（epoch/step/loss）——与 04 load `kind=="train_state"`
                # 分支对称：load 侧读 extra_state.json 恢复 step_scheduler。
                # 文件名 extra_state.json 与 04 `_state_path` 该 kind 对齐。
                # state_dict 先展开，显式键居后，确保 epoch/global_step 不被 state_dict 覆盖；
                # train_loss/val_losses 一并落盘（否则签名接收的 val_losses 会被丢弃）
                extra = {
                    **obj.state_dict(),
                    "epoch": epoch,
                    "global_step": step,
                    "train_loss": train_loss,
                    "val_losses": val_losses,
                }
                with open(f"{path}/extra_state.json", "w") as f:
                    json.dump(extra, f)
            elif hasattr(obj, "state_dict"):
                torch.save(obj.state_dict(), f"{path}/{name}.pt")
            else:
                logger.warning("Skipping %s: no state_dict method", name)

        # 更新 LATEST symlink（_update_latest_symlink 为模块级函数，见 §7.1）
        _update_latest_symlink(checkpoint_dir, path)

    # ── Checkpoint load ──
    #
    # load_checkpoint 的 canonical 实现见 04 §8（定义在 BaseRecipe 上）：
    #   def load_checkpoint(self, restore_from: str | None) -> None: ...
    # 为 Recipe 方法（1 参 `restore_from`），内部 `self._resolve_latest_symlink`
    # + `self._validate_checkpoint_compatibility` + 迭代 `self.__state_tracked`
    # 调 `self._state_path` + `self._load_state_by_kind`（两者均为 Recipe 方法，
    # 非 `self.checkpointer.*`——与 04 canonical 对齐）。
    # 03 不重复定义 `load_checkpoint`，仅通过继承获得；Recipe.setup() 只负责
    # 通过 register_state() 注册需要恢复的状态。


def _is_stateful(obj: Any) -> bool:
    """判断对象是否需要 checkpoint 追踪（仅用于 setup 期辅助判断）。"""
    return isinstance(obj, (
        nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler,
        DataLoader,
    )) or hasattr(obj, "state_dict")
```

---

## 4. StepScheduler —— 训练节奏控制

> **调用位置**: 时序树 ④.11 — `cfg.step_scheduler.build()` 创建；⑤ — `epochs`/`__iter__` 控制训练循环节奏

```python
# hyper_models/components/training/step_scheduler.py

class StepScheduler:
    """训练节奏控制——梯度累积、checkpoint/validation 步判断、SIGTERM 响应。

    替代旧的 _make_micro_batch_iterator + callback 判断逻辑。
    """

    def __init__(
        self,
        dataloader: DataLoader,
        global_batch_size: int,
        local_batch_size: int,
        dp_world_size: int,
        max_steps: int,
        ckpt_every_steps: int = 500,
        val_every_steps: int | None = None,
        save_checkpoint_every_epoch: bool = False,
        log_remote_every_steps: int = 10,
        loss_average_window_steps: int = 100,
        gc_every_steps: int | None = None,
        num_train_epochs: int = 1,
        start_step: int = 0,
        start_epoch: int = 0,
    ):
        self.dataloader = dataloader
        self.grad_acc_steps = max(
            global_batch_size // (local_batch_size * dp_world_size), 1
        )
        # 防御：global_batch_size 必须能被 (local_batch_size * dp_world_size) 整除
        # 否则每个 optimizer step 处理的样本数与配置不符（floor division 截断）
        if global_batch_size % (local_batch_size * dp_world_size) != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"local_batch_size * dp_world_size ({local_batch_size * dp_world_size})"
            )

        self.max_steps = max_steps
        self.ckpt_every_steps = ckpt_every_steps
        self.val_every_steps = val_every_steps
        self.save_checkpoint_every_epoch = save_checkpoint_every_epoch
        self.log_remote_every_steps = log_remote_every_steps
        self.loss_average_window_steps = loss_average_window_steps
        self.gc_every_steps = gc_every_steps
        self.num_train_epochs = num_train_epochs

        # 断点续训起始位置
        self.start_epoch = start_epoch  # 冷启动即设，确保 epochs property 可用
                                       # （load_state_dict 会同步覆盖为断点 epoch）
        self.step = start_step      # 注：键名为 "step" 非 "global_step"（与 AutoModel 兼容）
        self.epoch = start_epoch    # 注：键名为 "epoch" 非 "current_epoch"

        # SIGTERM 处理
        self.sig_handler = DistributedSignalHandler().__enter__()
        self._sigterm_flag = False

        # Epoch 级别 checkpoint：每个 epoch 只触发一次（避免 is_ckpt_step 每步为 True）
        self._epoch_ckpt_saved = False

    @property
    def epochs(self):
        """Epoch 迭代器。"""
        for epoch in range(self.start_epoch, self.num_train_epochs):
            self.epoch = epoch
            yield epoch
            if self._max_steps_reached:
                break

    def set_epoch(self, epoch: int) -> None:
        """设置 sampler epoch（shuffle 种子），并重置 epoch checkpoint 标记。"""
        if hasattr(self.dataloader, "sampler") and hasattr(
            self.dataloader.sampler, "set_epoch"
        ):
            self.dataloader.sampler.set_epoch(epoch)
        self._epoch_ckpt_saved = False  # 新 epoch 允许再次触发 save_checkpoint_every_epoch

    @property
    def global_step(self) -> int:
        """兼容别名（内部使用 self.step）。"""
        return self.step

    def __iter__(self):
        """迭代 dataloader，按 grad_acc_steps 分组 yield micro-batch 列表。

        step 在 yield **之前**自增：训练循环体（含 on_step_end 回调）读到的
        self.step 是"当前正在训练的步"（1 起）。若 yield 后才自增，冷启动
        首个 optimizer step 会以 step=0 执行，`step % interval == 0` 的
        判断（is_ckpt_step/is_log_step）会在 step 0 误触发一次保存/日志；
        断点续训时首个 step 也会与 checkpoint 中已完成的 step 重号。
        """
        batch_buffer = []
        for batch in self.dataloader:
            batch_buffer.append(batch)
            if len(batch_buffer) >= self.grad_acc_steps:
                self.step += 1
                yield batch_buffer
                batch_buffer = []

                if self._max_steps_reached or self.sigterm_received:
                    return

        # 余量（drop_last 时不应到达这里）
        if batch_buffer and not self.sigterm_received:
            self.step += 1
            yield batch_buffer

    @property
    def _max_steps_reached(self) -> bool:
        """是否达到 max_steps。max_steps <= 0（如默认 -1）表示不按步数限制
        （epoch 驱动），必须排除——否则 `step >= -1` 恒真，会导致第一步即
        退出、且 is_ckpt_step 每步为 True。"""
        return self.max_steps > 0 and self.step >= self.max_steps

    # ── 步类型判断 ──

    @property
    def is_ckpt_step(self) -> bool:
        """是否需要保存 checkpoint。

        save_checkpoint_every_epoch 的实际语义：epoch 切换后（set_epoch 重置
        _epoch_ckpt_saved）第一次检查即触发，效果为**每个 epoch 开头**保存
        一次（迭代协议无法感知 epoch 末尾边界）；保存后由 CheckpointCallback
        调 mark_epoch_ckpt_saved() 关闭本 epoch 的触发。
        最终步（_max_steps_reached）与 SIGTERM 也计入本标记，但对应的保存
        由训练循环末尾的 final save 统一处理——CheckpointCallback 对
        is_final_step 跳过（§4.2.4），避免同一步重复保存。
        """
        return (
            self.step % self.ckpt_every_steps == 0
            or self._max_steps_reached
            or self.sigterm_received
            or (self.save_checkpoint_every_epoch
                and not self._epoch_ckpt_saved)
        )

    @property
    def is_log_remote_step(self) -> bool:
        """是否需要远程日志。"""
        return self.step % self.log_remote_every_steps == 0

    # 别名：is_log_step 供 Callback/StepState 使用，与 is_log_remote_step 等价
    @property
    def is_log_step(self) -> bool:
        return self.is_log_remote_step

    @property
    def is_gc_step(self) -> bool:
        """是否需要垃圾回收。"""
        if self.gc_every_steps is None:
            return False
        return self.step % self.gc_every_steps == 0

    @property
    def is_val_step(self) -> bool:
        """是否需要验证。"""
        if self.val_every_steps is None:
            return self.is_ckpt_step
        return self.step % self.val_every_steps == 0

    @property
    def sigterm_received(self) -> bool:
        """任意 rank 收到 SIGTERM → 全体响应。

        警告：signals_received() 内部执行 all_gather 集合通信。所有参与同一
        process group 的 rank 必须同步调用此 property，否则会死锁。当前设计
        保证所有 rank 在 __iter__ 循环的同一位置（yield 后）调用此 property，
        满足同步条件。若未来在非对称代码路径中调用，需额外同步。
        """
        if not self._sigterm_flag:
            self._sigterm_flag = any(self.sig_handler.signals_received())
        return self._sigterm_flag

    def mark_epoch_ckpt_saved(self) -> None:
        """标记当前 epoch 的 checkpoint 已保存，防止 is_ckpt_step 重复触发。"""
        self._epoch_ckpt_saved = True

    def state_dict(self) -> dict:
        return {
            "step": self.step,        # AutoModel 兼容键名
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        # 兼容两种键名：AutoModel 的 "step"/"epoch" 和旧版 "global_step"/"current_epoch"
        self.step = state.get("step", state.get("global_step", 0))
        self.epoch = state.get("epoch", state.get("current_epoch", 0))
        # 同步 start_epoch，使 epochs 属性从断点 epoch 起算
        # （否则 range(self.start_epoch, ...) 仍从 0 重启，断点续训失效）
        self.start_epoch = self.epoch

    def cleanup(self) -> None:
        """清理资源——恢复原始 SIGTERM handler。

        调用 `self.sig_handler.__exit__` 将 signal handler 恢复为 __enter__
        前保存的原始处理器。Recipe 应在训练结束（正常完成/异常退出）时调用
        此方法，确保进程退出后不再拦截 SIGTERM。
        示例：`finally: self.step_scheduler.cleanup()`。
        """
        self.sig_handler.__exit__(None, None, None)
```

### 4.1 StepSchedulerConfig

```python
@dataclass
class StepSchedulerConfig:
    """StepScheduler typed config —— RecipeConfig.step_scheduler 的返回类型。

    与 StepScheduler 构造参数一一对应，.build() 负责注入运行时依赖
    （dataloader, dp_world_size, local_batch_size）。
    """
    max_steps: int = -1
    ckpt_every_steps: int = 500
    val_every_steps: int | None = None
    save_checkpoint_every_epoch: bool = False
    log_remote_every_steps: int = 10
    loss_average_window_steps: int = 100
    gc_every_steps: int | None = None
    num_train_epochs: int = 1
    # global_batch_size 为正式字段（§12 YAML 示例直接配置它）。
    # None 时退化为 local_batch_size * dp_world_size（即 grad_acc_steps=1）。
    # 注意：本字段与 §5.3 ⑬ 传给 build_dataloader 的
    # `step_scheduler.global_batch_size` 读取的是同一 YAML 键（同源），
    # grad_acc_steps 计算与 dataloader 的 global_batch_size 不会出现口径分叉。
    global_batch_size: int | None = None

    def build(self, dataloader, dp_world_size, local_batch_size,
              start_step=0, start_epoch=0):
        # 字段已声明，直接读取；None 时按 local*dp 退化（与 dataloader 同源）
        global_batch_size = (
            self.global_batch_size
            if self.global_batch_size is not None
            else local_batch_size * dp_world_size
        )
        return StepScheduler(
            dataloader=dataloader,
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            dp_world_size=dp_world_size,
            max_steps=self.max_steps,
            ckpt_every_steps=self.ckpt_every_steps,
            val_every_steps=self.val_every_steps,
            save_checkpoint_every_epoch=self.save_checkpoint_every_epoch,
            log_remote_every_steps=self.log_remote_every_steps,
            loss_average_window_steps=self.loss_average_window_steps,
            gc_every_steps=self.gc_every_steps,
            num_train_epochs=self.num_train_epochs,
            start_step=start_step,
            start_epoch=start_epoch,
        )
```

---

## 4.2 Callback 系统 —— 混合方案

> **设计原则**：核心训练流程（前向/反向/优化器步进）**显式**编排在 Recipe 中；外围关注点（checkpoint/验证/日志/监控/GC）通过 **Callback** 处理。Callback 只负责"收到通知后执行操作"，不做时序判断——由 `StepScheduler` 统一计算 `is_ckpt_step` / `is_val_step` / `is_log_step` / `is_gc_step` 等标记，通过 `StepState` 透传给 callback。

### 4.2.1 StepState —— 每步状态契约

```python
# hyper_models/components/training/callback.py

@dataclass(frozen=True)
class StepState:
    """每步结束后传递给回调的状态快照。
    
    Frozen dataclass —— 接收方只读，不可修改。
    所有时序标记由 StepScheduler 统一计算，通过此结构透传。
    """
    # ── 步信息 ──
    step: int
    epoch: int
    is_final_step: bool

    # ── 时序标记（由 StepScheduler 决定）──
    is_ckpt_step: bool          # 需要保存 checkpoint
    is_val_step: bool           # 需要运行验证
    is_log_step: bool           # 需要远程日志
    is_gc_step: bool            # 需要垃圾回收
    sigterm_received: bool      # 收到 SIGTERM 信号

    # ── 训练指标 ──
    loss: float
    grad_norm: float | None
    lr: float
    tps: float
    mfu: float
    num_tokens: int


### 4.2.2 TrainingCallback 接口

```python
class TrainingCallback:
    """训练生命周期回调。
    
    只提供 3 个回调点，不渗透到训练内部细节：
    - on_step_end: 每步核心训练结束后，用于 checkpoint/验证/日志/监控
    - on_train_begin: 训练开始前，用于资源初始化
    - on_train_end: 训练结束后，用于资源清理
    """

    def on_step_end(self, state: StepState) -> None:
        """每步结束后调用。state 包含所有训练指标和时序标记。"""
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass
```

### 4.2.3 CallbackManager

```python
class CallbackManager:
    """管理所有注册的 callback，按注册顺序依次调用。"""

    def __init__(self):
        self._callbacks: list[TrainingCallback] = []

    def register(self, callback: TrainingCallback) -> None:
        self._callbacks.append(callback)

    def on_step_end(self, state: StepState) -> None:
        for cb in self._callbacks:
            cb.on_step_end(state)

    def on_train_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_train_begin()

    def on_train_end(self) -> None:
        for cb in self._callbacks:
            cb.on_train_end()
```

### 4.2.4 内置 Callback 实现

```python
class CheckpointCallback(TrainingCallback):
    """在 is_ckpt_step 时保存 checkpoint（周期保存）。

    最终步（达到 max_steps 或 SIGTERM）跳过：该场景由训练循环末尾的显式
    final save（is_final_checkpoint=True，触发 04 §5.2 的 consolidated
    导出）统一处理。若此处不跳过，最终步会被保存两次（周期 + final），
    SIGTERM 步甚至叠加 SIGTERMHandler 保存达三次。
    """
    def __init__(self, recipe: BaseRecipe):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_ckpt_step or state.is_final_step:
            return
        self.recipe.save_checkpoint(
            self.recipe.cfg.checkpoint.checkpoint_dir,
            state.epoch, state.step, state.loss,
            val_losses=self.recipe._last_val_losses,
        )
        self.recipe.step_scheduler.mark_epoch_ckpt_saved()


class EvaluateCallback(TrainingCallback):
    """在 is_val_step 时运行验证。"""
    def __init__(self, recipe: BaseRecipe):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_val_step or not self.recipe.val_dataloaders:
            return
        val_losses = {}
        for name, dl in self.recipe.val_dataloaders.items():
            val_losses[name] = self.recipe._run_validation_epoch(dl)
        self.recipe._last_val_losses = val_losses
        self.recipe.log_val_metrics(val_losses)


class LoggingCallback(TrainingCallback):
    """在 is_log_step 时输出训练日志。"""
    def __init__(self, recipe: BaseRecipe):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_log_step:
            return
        logger.info(
            "step=%d loss=%.4f lr=%.2e grad_norm=%.4f tps=%.0f mfu=%.2f%%",
            state.step, state.loss, state.lr,
            state.grad_norm or 0.0, state.tps, state.mfu * 100,
        )


class TqdmCallback(TrainingCallback):
    """每步更新 tqdm 进度条（仅 rank 0）。

    pbar 创建延迟到 on_train_begin()，此时 recipe.step_scheduler 已存在，
    可以正确读取断点续训的起始步数。
    """
    def __init__(self, recipe: BaseRecipe, total: int | None = None):
        self.recipe = recipe
        self.total = total
        self.pbar = None

    def on_train_begin(self) -> None:
        if not _is_rank_0():
            return
        from tqdm import tqdm
        initial = getattr(self.recipe, "step_scheduler", None)
        initial_step = initial.step if initial is not None else 0
        self.pbar = tqdm(
            total=self.total, initial=initial_step,
            desc="Training", unit="step", dynamic_ncols=True,
        )

    def on_step_end(self, state: StepState) -> None:
        if self.pbar is None:
            return
        self.pbar.set_postfix(loss=f"{state.loss:.4f}", lr=f"{state.lr:.2e}")
        self.pbar.update(1)

    def on_train_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WandbCallback(TrainingCallback):
    """在 is_log_step 时记录训练指标到 WandB。"""
    def __init__(self, recipe: BaseRecipe, project: str = ""):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_log_step:
            return
        wandb.log({
            "loss": state.loss, "lr": state.lr,
            "grad_norm": state.grad_norm,
            "tps": state.tps, "mfu": state.mfu,
            "step": state.step,
        })


class GCCallback(TrainingCallback):
    """在 is_gc_step 时触发垃圾回收。"""
    def __init__(self, recipe: BaseRecipe):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_gc_step:
            return
        self.recipe._maybe_collect_garbage()


class SIGTERMHandler(TrainingCallback):
    """收到 SIGTERM 时触发优雅退出（不在此保存 checkpoint）。

    去重设计：SIGTERM 步的保存由两条既有路径覆盖——
    ① 本步 is_ckpt_step 为 True（is_ckpt_step 含 sigterm_received），
       但因 is_final_step 同为 True，CheckpointCallback 按约定跳过；
    ② 迭代器退出后，训练循环末尾的显式 final save
       （is_final_checkpoint=True）保存最终态。
    本 callback 只负责让迭代器尽快退出并恢复信号处理器。
    """
    def __init__(self, recipe: BaseRecipe):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.sigterm_received:
            return
        logger.warning("SIGTERM received at step %d, exiting gracefully", state.step)
        self.recipe.step_scheduler.cleanup()
        # 令 StepScheduler.__iter__ 在下一次取数据时返回，退出训练循环；
        # final save 由 §6 循环末尾统一执行
        self.recipe.step_scheduler.max_steps = state.step
```

### 4.2.5 内置 Callback 注册工厂

```python
# hyper_models/components/training/callback.py

def build_callback_manager(
    recipe: BaseRecipe,
    cfg: RecipeConfig,
    pbar_total: int | None = None,
) -> CallbackManager:
    """构建默认的 CallbackManager，注册所有内置 callback。"""
    manager = CallbackManager()
    manager.register(CheckpointCallback(recipe))
    manager.register(EvaluateCallback(recipe))
    manager.register(LoggingCallback(recipe))
    manager.register(TqdmCallback(recipe, total=pbar_total))
    if cfg.get("wandb.enabled", False):
        manager.register(WandbCallback(
            recipe, project=cfg.get("wandb.project", ""),
        ))
    # gc_every_steps 是 StepSchedulerConfig 的字段，YAML 中位于
    # step_scheduler 节下（§4.1/§13），不能读顶层键
    if cfg.get("step_scheduler.gc_every_steps"):
        manager.register(GCCallback(recipe))
    manager.register(SIGTERMHandler(recipe))
    return manager
```

### 4.2.6 混合方案设计决策

**为什么只保留 3 个回调点？**

| 回调点 | 用途 | 理由 |
|--------|------|------|
| `on_train_begin()` | 资源初始化（wandb 初始化、profile 启动） | 训练开始前，一次性的 |
| `on_step_end(state)` | 所有步级操作（checkpoint/验证/日志/GC/监控） | 核心训练结束后，不再需要其他步级时机 |
| `on_train_end()` | 资源清理（tqdm 关闭、wandb 结束、profile 写出） | 训练结束后，一次性的 |

**不设 `on_step_begin` / `on_epoch_begin` / `on_epoch_end` 的理由**：
- automodel 的实践证明最常用的回调时机就是"每步结束后"
- 更少的回调点降低推理链长度，提升可读性
- Epoch 级别的操作（如 sampler shuffle）在 Recipe 的 `for epoch` 循环中显式处理，无需回调

**时序标记集中管理的优势**：
- `StepScheduler` 是唯一计算 `is_ckpt_step` / `is_val_step` 等标记的地方
- 每个 callback 不需要独立计算"当前步数 % 保存间隔"——避免分散逻辑、提升一致性
- `StepState` 是 frozen dataclass，callback 只能读取，不能修改——避免副作用

**核心 vs 外围的边界**：

| 归属 | 操作 | 理由 |
|------|------|------|
| **核心（显式）** | forward / backward / loss 计算 | 直接参与梯度计算 |
| | 梯度裁剪 / optimizer.step / lr_scheduler.step | 消费梯度 |
| | CP 数据切分 | 影响 forward 输入 |
| | FSDP 梯度同步（set_requires_gradient_sync） | 控制梯度通信 |
| | 梯度累积循环（microbatch 迭代） | 核心训练节拍 |
| **外围（Callback）** | checkpoint 保存/加载 | 不参与梯度流 |
| | validation / evaluation | 不参与梯度流 |
| | 日志（loss/lr/tps/MFU） | 纯监控 |
| | tqdm 进度条 | 纯 UI |
| | WandB / MLflow | 远程日志 |
| | Profiling | 性能分析 |
| | GC（garbage collection） | 资源管理 |
| | SIGTERM 处理 | 系统级 |

---

## 5. Recipe.setup() 完整实现

> **调用位置**: 时序树 ④ — `main()` 中 `recipe.setup(cfg)` —— 组件构建顺序

### 5.1 两条 `_target_` 使用路径

AutoModel 区分两类组件，使用不同的 `_target_` 消费方式：

| 路径 | 适用组件 | 调用方式 | 原因 |
|------|---------|---------|------|
| **直接 `.instantiate()`** | model, dataset, dataloader, tokenizer, collate, peft | `cfg.xxx.instantiate(**runtime_kwargs)` | 参数全部可在 YAML 中声明 |
| **两层 `.build()`** | optimizer, lr_scheduler, step_scheduler, loss_fn, checkpoint | `RecipeConfig` 先提取 `_target_`→类型化Config，再 `.build(**runtime_deps)` | 依赖运行时对象（model、optimizer、device_mesh 等） |

### 5.2 RecipeConfig 桥接：YAML → 类型化 Config

> **canonical 定义归 01 §3.3**（属性全集、`_callable_and_kwargs` /
> `_section_kwargs` 辅助函数、`get()` 语义以 01 为准，01 侧同步补齐）。
> 本节不重复完整实现，仅保留 03 消费侧视图：哪些属性是 typed（两层
> `.build()`）、setup() 如何取用。与 01 §3.3 冲突的描述（含此前本节
> `get()` 的 docstring 细节）一律删除。

```python
# recipes/_typed_config.py（实现见 01 §3.3；以下为 03 消费侧视图）

class RecipeConfig:  # canonical: 01 §3.3
    """将 YAML ConfigNode 桥接到强类型配置 Dataclass。

    03 消费的两类属性：
    - typed（有 .build()，setup() 注入运行时依赖）:
        optimizer   -> OptimizerConfig        （§9.2/§9.3）
        lr_scheduler -> LRSchedulerConfig     （§9.6）
        step_scheduler -> StepSchedulerConfig （§4.1；过滤
            local_batch_size/dp_size/dataloader 等运行时键后构造）
        loss_fn     -> LossConfig             （§10.0）
        checkpoint  -> CheckpointingConfig    （补 model_repo_id/is_peft）
    - untyped（直接 .instantiate() 或独立构建函数，__getattr__ 透传原始 ConfigNode）:
        model, peft, dataset, tokenizer, collate
      注：dataloader 不走 .instantiate()，而是通过 02_data_pipeline.md 的
      build_dataloader() 独立函数构建（cfg.dataset + cfg.dataloader +
      cfg.model + cfg.packed_sequence 等作为参数传入）
    另有 get(dot_path, default) 供 setup() 读取嵌套标量（语义见 01 §3.3）。
    """
```

setup() 中的典型取用方式（与 §5.3 对应）：

```python
self.loss_fn       = self.cfg.loss_fn.build()                       # ⑦
self.checkpointer  = self.cfg.checkpoint.build(dp_rank=..., ...)    # ⑩
self.optimizer     = self.cfg.optimizer.build(                      # ⑫
    self.model, device_mesh=..., optimizer_init=..., is_peft=...)
self.step_scheduler = self.cfg.step_scheduler.build(                # ⑮
    self.dataloader, dp_size, local_batch_size)
self.lr_scheduler  = self.cfg.lr_scheduler.build(                   # ⑯
    self.optimizer, self.step_scheduler)
# 嵌套标量：cfg.get("step_scheduler.local_batch_size", 1) 等（⑬⑭⑮）
```

### 5.3 Recipe.setup() 实现

```python
# recipes/llm/train_ft.py

class FinetuneRecipe(BaseRecipe):
    """LLM 微调 Recipe。"""

    def setup(self, cfg: RecipeConfig) -> None:
        """按依赖顺序构建训练组件。

        两类构建方式：
        - cfg.<typed>.build(**runtime_deps) → optimizer, lr_scheduler, step_scheduler, loss, checkpoint
        - cfg.<untyped>.instantiate(**runtime_kwargs) → model, peft, dataset, dataloader, tokenizer

        03 步骤编号 → 01 §4.1 canonical 编号映射（canonical 以 01 §4.1 时序树为准）：
          03 ①=01 ④.1, 03 ②=01 ④.x（日志，canonical 未单列）, 03 ③=01 ④.2,
          03 ④=01 ④.3, 03 ⑤=01 ④.x（MagiAttention）,
          03 ⑥=01 ④.x（日志器，canonical 未单列；§2 树记为 ④.3a）,
          03 ⑦=01 ④.5（Loss）, 03 ⑧=01 ④.x（PP）, 03 ⑨=01 ④.6（PEFT）,
          03 ⑩=01 ④.7（Checkpointer）, 03 ⑪=01 ④.4（Model）,
          03 ⑫=01 ④.8（Optimizer）, 03 ⑬=01 ④.9（DataLoader）,
          03 ⑭=01 ④.10（Val DataLoader）, 03 ⑮=01 ④.11（StepScheduler）,
          03 ⑯=01 ④.12（LR Scheduler）, 03 ⑰=01 ④.x（注册追踪状态）,
          03 ⑱=01 ④.13（load_checkpoint）, 03 ⑲=01 ④.14（MFU）
        """
        self.cfg = cfg

        # ① 分布式初始化
        self.dist_env = initialize_distributed("nccl")

        # ② 日志 + 兼容性补丁
        setup_logging()
        apply_cache_compatibility_patches()

        # ③ RNG
        self.rng = StatefulRNG(seed=cfg.get("seed", 42), ranked=True)

        # ④ 分布式策略
        self.distributed_setup = create_distributed_setup_from_config(cfg)
        self.mesh = self.distributed_setup.mesh_context
        # DP+CP 联合子 mesh：用于 DP/CP 维度的联合 all-reduce（统计全局
        # label token、val loss 聚合等）。取二维子 mesh ("dp_shard_cp","cp")，
        # 确保 CP 维纳入 all-reduce（cp_size>1 时全局 token 数 = 各 cp rank
        # 持有段之和，单维索引会少算 cp_size 倍）。
        # 退化分支：无 "cp" 轴（cp_size==1）时退化为纯 DP mesh。
        mesh = self.mesh.device_mesh
        dim_names = mesh.mesh_dim_names
        # FSDP2 mesh（"dp_shard_cp" + "cp" 轴）：取二维子 mesh 纳入 CP 维
        if "cp" in dim_names and "dp_shard_cp" in dim_names:
            self.dp_cp_mesh = mesh[("dp_shard_cp", "cp")]
        elif "dp_shard_cp" in dim_names:
            self.dp_cp_mesh = mesh["dp_shard_cp"]            # cp_size==1
        elif "dp" in dim_names:
            # DDP / Megatron mesh（轴如 ("dp",) 或 ("dp","pp","tp","cp")）
            self.dp_cp_mesh = mesh[("dp", "cp")] if "cp" in dim_names else mesh["dp"]
        elif "dp_replicate" in dim_names:
            self.dp_cp_mesh = mesh[("dp_replicate", "cp")] if "cp" in dim_names else mesh["dp_replicate"]
        else:
            # 单 rank 兜底
            self.dp_cp_mesh = mesh
        # 多维子 mesh 预展平为 1D：_dp_cp_all_reduce_sum 需要 1D group
        # （DeviceMesh.get_group() 仅对 1D mesh 语义明确，多维 mesh 无参
        # get_group() 行为依 torch 版本而异）。在 setup 期展平一次并缓存，
        # 避免每步 all_reduce 时重复建 group
        if self.dp_cp_mesh.ndim > 1:
            self.dp_cp_mesh = self.dp_cp_mesh._flatten("dp_cp")

        # ⑤ MagiAttention（可选）
        self.magi = setup_magi(cfg, self.mesh.device_mesh) if cfg.get("magi") else None

        # ⑥ Callback 管理器 —— 注册所有内置 callback
        # 由 build_callback_manager 根据 cfg 自动创建 CheckpointCallback、
        # EvaluateCallback、LoggingCallback、TqdmCallback、WandbCallback 等
        self.callback_manager = build_callback_manager(
            self, cfg,
            pbar_total=cfg.get("step_scheduler.max_steps", None),
        )

        # ⑦ Loss —— typed: .build()
        self.loss_fn = self.cfg.loss_fn.build()

        # ⑧ PP 配置
        self.pp_enabled = self.mesh.pp_size > 1
        self._configure_pp(cfg)

        # ⑨ PEFT —— untyped: .instantiate()
        self.peft_config = self.cfg.peft.instantiate() if cfg.get("peft", None) else None

        # ⑩ Checkpoint —— typed: .build(dp_rank=..., ...)
        checkpoint_config = self.cfg.checkpoint  # CheckpointingConfig 实例
        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=getattr(self.mesh, "moe_mesh", None),
            # 06 D-10 口径：MeshContext 无 moe_mesh 字段（主 mesh 不含 EP 轴，
            # expert mesh 由 apply_sharding_plan 期派生），此处 getattr 恒为 None。
            # MoE 模型的 consolidated 导出需要派生 expert mesh 时，需由 sharding
            # 层暴露（当前代码未导出，属已知缺口，见 04 §5.1 Checkpointer.__init__ 的 moe_mesh 注）。
        )

        # ⑪ Model —— untyped: .instantiate(**runtime_kwargs)
        self.model, self.optimizer_init = build_model(
            cfg.model, self.peft_config,
            distributed_setup=self.distributed_setup,
        )
        self.model_parts = self.model.parts if hasattr(self.model, "parts") else [self.model]

        # ⑫ Optimizer —— typed: .build(model, device_mesh=...)
        #     返回 list[Optimizer]（canonical，nemo_automodel 惯例；04
        #     OptimizerState 接受 list[Optimizer]，以本节/§9.3 为准）
        self.optimizer = self.cfg.optimizer.build(
            self.model, device_mesh=self.mesh.device_mesh,
            optimizer_init=self.optimizer_init,      # 传入 build_model 导出的 param 分组（01 §4.2 / §6.2）
            is_peft=self.peft_config is not None,
        )

        # ⑬ DataLoader —— 调用 02_data_pipeline.md::build_dataloader()
        #     global_batch_size 与 StepSchedulerConfig.global_batch_size（§4.1）
        #     读同一 YAML 键 step_scheduler.global_batch_size（同源，不分叉）
        self.dataloader, self.tokenizer = build_dataloader(
            cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
            cfg_db=cfg.get("dynamic_batching", None),
            cfg_transform=cfg.get("transform", None),
            cfg_multisource=cfg.get("multisource", None),
            seed=cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=cfg.get("step_scheduler.max_steps", None),
            val_check_interval=cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            cp_size=self.mesh.cp_size,
            model=self.model_parts[0],
        )

        # ⑭ Validation DataLoader（实现见 02_data_pipeline.md §3.3，
        #    复用 build_dataloader 但 drop_last=False / shuffle=False / 不 packing /
        #    不维护 sampler state；返回 dict[str, DataLoader]，03 仅调用）
        self.val_dataloaders = build_validation_dataloader(
            cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
            cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=cfg.get("step_scheduler.global_batch_size", 1),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            cp_size=self.mesh.cp_size,
            model=self.model_parts[0],
        )

        # ⑮ StepScheduler —— typed: .build(dataloader, dp_size, local_batch_size)
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            cfg.get("step_scheduler.local_batch_size", 1),
        )

        # ⑯ LR Scheduler —— typed: .build(optimizer, step_scheduler)
        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )

        # ⑰ 注册 checkpoint 追踪状态（kind 与 04 `_state_path` 对齐）
        #     load 由 04 §8 canonical `load_checkpoint(self, restore_from)`
        #     迭代 __state_tracked 完成，03 不再自己调 checkpointer.load。
        self.register_state("model", "model")
        self.register_state("optimizer", "optimizer")
        self.register_state("lr_scheduler", "lr_scheduler")
        self.register_state("rng", "rng")
        self.register_state("dataloader", "dataloader")
        # step_scheduler 以 "train_state" kind 注册：save 侧写 extra_state.json
        # （epoch/step），load 侧 04 `_state_path(kind=="train_state")` 读回
        # extra_state.json 恢复 self.epoch/self.step。确保 save/load 对称。
        self.register_state("step_scheduler", "train_state")

        # ⑱ 断点续训（继承自 04 §8 的 Recipe.load_checkpoint，1 参 restore_from）
        self.load_checkpoint(cfg.get("checkpoint.restore_from", None))

        # ⑲ MFU 计算器 + 模型信息打印
        self.mfu_calc = AutoMFU.from_config(self.model_parts[0])
        self._log_model_and_optimizer_details()
```

---

## 6. 训练主循环

> **调用位置**: 时序树 ⑤ — `recipe.run_train_validation_loop()`
>
> **混合方案**：核心训练（`_run_train_optim_step`）显式可见；外围关注点（checkpoint/验证/日志/GC）通过 `callback_manager.on_step_end` 驱动。
> `StepState` 将 `StepScheduler` 计算的时序标记统一透传，callback 只负责执行，不做判断。

```python
def run_train_validation_loop(self) -> None:
    """训练主循环 —— 核心显式 + 外围 Callback 混合方案。

    核心流程（forward/backward/optimizer step）显式在 Recipe 中编排；
    外围关注点（checkpoint、验证、日志、GC、WandB、tqdm）通过
    callback_manager.on_step_end(StepState) 驱动。
    """
    for mp in self.model_parts:
        mp.train()

    # ── Callback: 训练开始 ──
    self.callback_manager.on_train_begin()

    # 预绑 None：零迭代场景下 final save 需守卫
    train_metrics: dict | None = None
    self._last_val_losses: dict | None = None

    try:
        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)

            for batches in self.step_scheduler:
                # ── 核心训练：显式可见 ──
                train_metrics = self._run_train_optim_step(
                    batches,
                    max_grad_norm=self.cfg.optimizer.max_grad_norm,
                )

                # sigterm_received 内部是 all_gather 集合通信，每步只查询
                # 一次并复用结果（避免 StepState 构造期间多次 all_gather）
                sigterm = self.step_scheduler.sigterm_received

                # ── 外围关注点：Callback 统一驱动 ──
                state = StepState(
                    step=self.step_scheduler.step,
                    epoch=epoch,
                    is_final_step=(
                        self.step_scheduler._max_steps_reached or sigterm
                    ),
                    is_ckpt_step=self.step_scheduler.is_ckpt_step,
                    is_val_step=self.step_scheduler.is_val_step,
                    is_log_step=self.step_scheduler.is_log_step,
                    is_gc_step=self.step_scheduler.is_gc_step,
                    sigterm_received=sigterm,
                    loss=train_metrics.get("loss", 0.0),
                    grad_norm=train_metrics.get("grad_norm"),
                    lr=train_metrics.get("lr", 0.0),
                    tps=train_metrics.get("tps", 0.0),
                    mfu=train_metrics.get("mfu", 0.0),
                    num_tokens=train_metrics.get("num_tokens", 0),
                )
                self.callback_manager.on_step_end(state)

        # ── 正常结束：最终 checkpoint ──
        # 顺序约束（与 04 对齐）：final save 必须先于 checkpointer.close()——
        # 04 close() 会 destroy _saving_pg（异步保存进程组），close 之后再
        # 保存在 is_async=True 配置下失败。
        # is_final_checkpoint=True（04 §5.2 要求，触发 final consolidated 导出）。
        # 最终步/SIGTERM 的保存统一由本处完成：Callback 的 on_step_end 在循环
        # 结束后不再被调用，且 CheckpointCallback 对 is_final_step 跳过（§4.2.4）
        self.save_checkpoint(
            self.cfg.checkpoint.checkpoint_dir,
            self.step_scheduler.epoch,
            self.step_scheduler.global_step,
            (train_metrics or {}).get("loss", 0.0),
            self._last_val_losses if (self.val_dataloaders and self._last_val_losses) else None,
            is_final_checkpoint=True,
        )
    finally:
        # ── Callback: 训练结束 + checkpointer 资源清理（正常/异常路径都执行） ──
        self.callback_manager.on_train_end()
        self.checkpointer.close()

    destroy_process_group()
```

### 6.1 验证流程

```python
def _run_validation_epoch(self, val_dl) -> dict[str, float]:
    """单次 validation epoch：torch.no_grad + validate 模式 forward +
    DP all-reduce mean 聚合 val loss。

    返回 `{"loss": float, "num_tokens": int}`（num_tokens 用于加权聚合）。
    """
    # 切换 eval/validate 模式（关闭 dropout 等）
    for mp in self.model_parts:
        mp.eval()

    total_loss_sum = 0.0      # 跨 microbatch 累加 CE sum（已除以本 rank token 数）
    total_label_tokens = 0    # 本 rank 累计 label token 数
    n_micro = 0

    try:
        with torch.no_grad():
            for batch in val_dl:
                # 数据 → GPU
                batch = {
                    k: v.to(self.dist_env.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # CP batch 准备（与训练 §8 Step 2 一致；CP 维度非冗余）。
                # 已落地实现（hyper_models/components/distributed/cp_utils.py）：
                # Q 按 contiguous chunk 切分（shard_batch_for_cp，契约：
                # input_ids/labels/position_ids [B,S]、seq_lens/seq_lens_padded
                # 按 CP rank 重算并保留 -1000 哨兵、qkv_format="thd" 透传）；
                # K/V all-gather 由 apply_sharding_plan 编译期注入的 CP
                # inner-attention wrapper（_wrap_cp_inner_attention，cp_mesh
                # .size()>1 时生效）在 forward 内部完成——训练/验证循环
                # **无需任何 CP context manager 或额外 hook**。
                if self.mesh.cp_size > 1:
                    if hasattr(self.model_parts[0], "prepare_model_inputs_for_cp"):
                        batch = self.model_parts[0].prepare_model_inputs_for_cp(**batch)
                    else:
                        batch = shard_batch_for_cp(batch, self.mesh.cp_mesh)

                labels = batch.pop("labels", None)
                filtered_batch = filter_forward_kwargs(self.model_parts[0], batch)

                # validate 路径：模型走 forward 的 validate 分支。
                # PP 多 stage 时复用 §8 训练侧
                # PP forward 调度（send/recv），此处仅展示单 stage 路径。
                output = self.model_parts[0](**filtered_batch)
                logits = output.logits if hasattr(output, "logits") else output

                # 统计本 microbatch 的 label token 数
                num_tok = int((labels != -100).sum().item())
                # DP+CP 联合 all-reduce 得到全局 token 数（与训练一致）
                # _dp_cp_all_reduce_sum 返回 tensor，此处需标量
                num_tok_global = _dp_cp_all_reduce_sum(num_tok, self.dp_cp_mesh).item()

                # calculate_loss 返回 raw ce_sum（不除 N，覆盖本 rank 段 token）。
                # 累加 raw ce_sum，末尾再跨 DP+CP all-reduce SUM 后除以全局 token 数，
                # 还原正确的 token-mean（第六轮 P1 修复：旧实现 local_ce/global_tok
                # 每步相除再 DP-mean，少算 dp_size*cp_size 倍）。
                local_ce_sum = calculate_loss(
                    self.loss_fn,
                    logits=logits,
                    labels=labels,
                    model=self.model_parts[0],
                    num_label_tokens=num_tok_global,
                )

                total_loss_sum += local_ce_sum.detach().item()
                total_label_tokens += num_tok
                n_micro += 1
    finally:
        # 恢复 train 模式
        for mp in self.model_parts:
            mp.train()

    # token-mean = Σ_{dp,cp} ce_sum_local / Σ_{dp,cp} num_tok_local
    global_ce_sum = _dp_cp_all_reduce_sum(total_loss_sum, self.dp_cp_mesh).item()
    global_tokens = _dp_cp_all_reduce_sum(total_label_tokens, self.dp_cp_mesh).item()
    global_val_loss = global_ce_sum / max(global_tokens, 1)

    return {"loss": global_val_loss, "num_tokens": global_tokens}


# `build_validation_dataloader` 实现位于 02_data_pipeline.md，03 仅调用。
# 契约：复用 build_dataloader，但 drop_last=False、不维护 sampler state、
# 不做 sequence packing（保留 [B, S] 原形）；签名：
#   def build_validation_dataloader(cfg, ...) -> dict[str, "DataLoader"]:
#       ...  # 见 02_data_pipeline.md
```

---

## 7. 单步优化器步进

> **调用位置**: 时序树 ⑤.1 — `_run_train_optim_step()`

```python
def _run_train_optim_step(
    self, batches: list[dict], max_grad_norm: float
) -> dict:
    """执行一个完整的 optimizer step。

    三阶段：
    Phase 1: 统计全局 token 数
    Phase 2: 梯度累积（前向+反向 × grad_acc_steps）
    Phase 3: 梯度裁剪 + optimizer.step + lr_scheduler.step
    """
    num_batches = len(batches)
    model = self.model_parts[0]

    self._step_start_time = time.time()  # Track step timing for throughput calculation

    # ── Phase 1: 统计全局 token 数 ──
    num_label_tokens = 0
    for batch in batches:
        labels = batch.get("labels")
        if labels is not None:
            num_label_tokens += (labels != -100).sum().item()

    # DP+CP joint all-reduce (CP also shards the sequence, tokens need full count)
    # 返回 tensor → 取 .item() 转标量，供后续算术与 scale_grads 使用
    num_label_tokens = _dp_cp_all_reduce_sum(num_label_tokens, self.dp_cp_mesh).item()

    # ── Phase 2: 梯度累积 ──
    loss_buffer = []
    prepare_for_grad_accumulation(self.model_parts)

    for i, batch in enumerate(batches):
        is_last = (i == num_batches - 1)

        # FSDP2: 最后一个 microbatch 才同步梯度（遍历所有 PP stage）
        for mp in self.model_parts:
            if isinstance(mp, FSDPModule):
                mp.set_requires_gradient_sync(is_last)
        if is_last:
            prepare_for_final_backward(self.model_parts)

        self._forward_backward_step(
            i, batch,
            loss_buffer=loss_buffer,
            num_label_tokens=num_label_tokens,
            num_batches=num_batches,
        )

        if i == 0:
            prepare_after_first_microbatch(self.model_parts)

    # ── Phase 3: 梯度裁剪 + optimizer step ──
    # 注：签名 (model_parts, max_norm, num_label_tokens)，调用按名传参，
    #     避免 N-03 参数颠倒（旧调用 `scale_grads_and_clip_grad_norm(
    #     max_norm, self.model_parts, ...)` 前两参错位 → TypeError）。
    # num_label_tokens 仅在 token_weighted 且非 PP 时传入：rank_average
    # 等 mean 尺度 loss（§10）不能再除 N，PP 场景由 PP runtime 平衡（§10.1）
    _token_weighted = (
        getattr(self.cfg.loss_fn, "loss_aggregation", "token_weighted")
        == "token_weighted"
    )
    grad_norm = scale_grads_and_clip_grad_norm(
        self.model_parts, max_grad_norm,
        num_label_tokens=(
            num_label_tokens if (_token_weighted and not self.pp_enabled) else None
        ),
    )

    self.checkpointer.maybe_wait_for_staging()

    for opt in (self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]):
        opt.step()
        opt.zero_grad()

    # lr_scheduler 可能为 None（setup 中条件赋值）——加守卫避免 AttributeError。
    # None 时跳过 step，last_lr 退化为 optimizer param_groups 当前 lr。
    schedulers = (
        self.lr_scheduler
        if isinstance(self.lr_scheduler, list)
        else ([self.lr_scheduler] if self.lr_scheduler is not None else [])
    )
    for sch in schedulers:
        sch.step()

    # ── Loss 聚合（logged loss = token-mean） ──
    # local_loss 为 raw ce_sum（未除 N，见 §8 Step 5）。
    # 日志损失 = Σ_{microbatches, dp_ranks} ce_sum_local / N_global
    # （分子为所有 microbatch 的 raw CE sum 跨 DP all-reduce SUM 汇总，
    #  分母为全局 label token 数 N_global）。
    #
    # 注意：梯度路径和日志路径共享同一 N_global，但日志侧不必考虑
    # FSDP2 DP-mean 补偿——直接 SUM 各 rank 的 ce_sum_local 再除以 N_global
    # 即得正确的全局 token-mean 损失。
    total_ce_sum = sum(loss_buffer)
    global_ce_sum = _dp_cp_all_reduce_sum(total_ce_sum, self.dp_cp_mesh).item()
    global_loss = global_ce_sum / max(num_label_tokens, 1)

    # ── 计算吞吐 ──
    step_time = time.time() - self._step_start_time
    tps = num_label_tokens / max(step_time, 1e-8)
    mfu = calculate_mfu(
        tps, self.mfu_calc.flops_per_token, self.mfu_calc.peak_tflops,
        self.dist_env.world_size,
    )

    if schedulers:
        lr = schedulers[-1].get_last_lr()[0]
    else:
        first_opt = (
            self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
        )
        lr = first_opt.param_groups[0]["lr"]

    return {
        "loss": global_loss,
        "grad_norm": grad_norm,
        "lr": lr,
        "step_time": step_time,
        "tps": tps,
        "mfu": mfu,
        "num_tokens": num_label_tokens,
    }
```

### 7.1 梯度累积辅助函数

> 归属文件：`hyper_models/components/training/grad_accum.py`。FSDP2 梯度同步只在最后一个
> microbatch 触发（`set_requires_gradient_sync(is_last=True)`），其余 microbatch
> 走 `defer_fsdp_grad_sync=True` 路径，避免 grad accumulation 期间反复 all-reduce。

**FSDP 梯度同步的责任分工**（双层管理）：

| 层级 | 位置 | 职责 |
|------|------|------|
| 外层（`_run_train_optim_step`） | Phase 2 循环外 | `prepare_for_grad_accumulation` 关同步 / `prepare_for_final_backward` 开同步 + 挂 PP 钩子 |
| 内层（`_forward_backward_step`） | 每个 microbatch | `get_sync_ctx` 按 defer 开关切换各 FSDPModule 的 `set_requires_gradient_sync` |

两层不互相覆盖：外层在 microbatch 循环边界设置 `require_gradient_sync` 开关；
内层在每个 microbatch 的前向期间通过 `get_sync_ctx` 切换 sync 开关（deferred 微批关、
最后微批保持开）。FSDP2 的梯度 all-reduce 由 `set_requires_gradient_sync(True)` 在
本轮 backward 末尾自动触发，无需单独的上下文管理器。
最后 microbatch 前外层调 `prepare_for_final_backward` 置 `require_gradient_sync=True`，
内层 `get_sync_ctx(defer_fsdp_grad_sync=False)` 走 `is_optim_step` 路径但不再
关闭同步（因为 `require_gradient_sync` 已为 True，`set_requires_gradient_sync(False)`
不会被调用）。

```python
# hyper_models/components/training/grad_accum.py

def get_sync_ctx(
    model_parts: list[nn.Module],
    *,
    is_optim_step: bool,
    defer_fsdp_grad_sync: bool = False,
):
    """返回 forward 期间的上下文管理器。组件归属: hyper_models/components/training/grad_accum.py。

    FSDP2 的梯度 DP all-reduce 由 ``set_requires_gradient_sync(bool)`` 控制
    （而非一个单独的上下文管理器），因此本函数在所有分支都返回 ``nullcontext()``，
    仅副作用在于切换各 FSDPModule 的 sync 开关：
    - 非 optim step：返回 nullcontext()（不触碰 sync 开关）。
    - defer_fsdp_grad_sync=True（非最后 microbatch）：调
      ``set_requires_gradient_sync(False)`` 进入 deferred-sync，grad 不立即 all-reduce。
    - is_optim_step 且非 defer（最后 microbatch）：调用方已通过
      ``prepare_for_final_backward`` 将 ``require_gradient_sync`` 置为 True，
      本轮 backward 末尾自动触发梯度 all-reduce，故返回 nullcontext() 即可。
    """
    if not is_optim_step:
        return nullcontext()
    if defer_fsdp_grad_sync:
        for mp in model_parts:
            if isinstance(mp, FSDPModule):
                mp.set_requires_gradient_sync(False)
        return nullcontext()
    # 最后一个 microbatch：FSDP2 通过 set_requires_gradient_sync(True) 自动在
    # 本轮 backward 末尾触发梯度 DP all-reduce，无需额外上下文管理器。
    # 调用方（_run_train_optim_step）已在本 microbatch 前调 prepare_for_final_backward
    # 将 require_gradient_sync 置为 True，此处返回 nullcontext() 即可。
    return nullcontext()


def prepare_for_grad_accumulation(model_parts: list[nn.Module]) -> None:
    """梯度累积开始前的准备：组件归属: hyper_models/components/training/grad_accum.py。
    1. `opt.zero_grad(set_to_none=True)` 清空上一步梯度（由调用方 _run_train_optim_step
       在 Phase 3 统一执行，本函数不重复）；
    2. 对每个 FSDPModule 调 `set_requires_gradient_sync(False)`，
       进入 deferred-sync 模式；
    3. 记录 `_grad_accum_state`（用于 final backward 时还原）。
    """
    for mp in model_parts:
        if isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(False)
            mp._grad_accum_state = "deferred"


def prepare_for_final_backward(model_parts: list[nn.Module]) -> None:
    """最后一个 microbatch 反向前的准备：组件归属: hyper_models/components/training/grad_accum.py。
    1. 遍历所有 FSDPModule 调 `set_requires_gradient_sync(True)`，
       允许 backward 末尾触发 DP all-reduce；
    2. PP 多 stage 时还要在各 stage 间挂上 send/recv 钩子
       （PP 的 backward 由最后一 stage 触发）。
    """
    for mp in model_parts:
        if isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(True)
            mp._grad_accum_state = "final"
    if len(model_parts) > 1:
        # PP 多 stage：在各 stage 间挂 send/recv 钩子（实现见 PP runtime）。
        # _attach_pp_backward_hooks 注册 autograd hook 以在 backward 时
        # 跨 PP stage 传递梯度（send/recv），确保反向传播按 stage 顺序传播。
        # 【状态：待实现】PP 工具模块尚无属主代码，属主定为
        # hyper_models/components/parallel/pp_utils.py（路径保留，落地前 PP>1 不可用）。
        # 签名：def _attach_pp_backward_hooks(model_parts: list[nn.Module]) -> None
        _attach_pp_backward_hooks(model_parts)


def prepare_after_first_microbatch(model_parts: list[nn.Module]) -> None:
    """第一个 microbatch 前向后的准备：组件归属: hyper_models/components/training/grad_accum.py。
    1. 对 FSDPModule 调 `reset_lazy_init()` / 预热 unshard 缓存，
       避免后续 microbatch 重复 unshard；
    2. 标记 `_first_microbatch_done = True`。
    """
    for mp in model_parts:
        if isinstance(mp, FSDPModule):
            if hasattr(mp, "reset_lazy_init"):
                mp.reset_lazy_init()
            mp._first_microbatch_done = True


def set_requires_gradient_sync(
    model_parts: list[nn.Module], is_last: bool
) -> None:
    """逐 part 设置 FSDP2 梯度同步开关（中间 microbatch 关，最后一个开）。
    等价于 `for mp in model_parts: if FSDPModule: mp.set_requires_gradient_sync(is_last)`。
    """
    for mp in model_parts:
        if isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(is_last)


def scale_grads_and_clip_grad_norm(
    model_parts: list[nn.Module],
    max_norm: float,
    num_label_tokens: int | None = None,
) -> float:
    """梯度缩放 + 裁剪，返回 grad_norm。组件归属: hyper_models/components/training/grad_accum.py。

    步骤：
    1. 若 `num_label_tokens` 非 None（非 PP 场景）：对每个参数 grad 除以
       `num_label_tokens`，将 CE sum 还原为 token-mean（与 FSDP DP 平均
       配合后等价于全局 token-mean）。**这是 token-mean 归一化的唯一除法点**——
       `calculate_loss` 返回 raw `ce_sum`（不除 N），由本函数统一除 N，
       避免双除（见 §10.1）。
    2. 调 `torch.nn.utils.clip_grad_norm_(params, max_norm)` 得到
       裁剪前的总 grad_norm（跨 DP/TP all-reduce 后的总范数）。
    3. 返回 grad_norm（用于日志/metric）。
    """
    params = [p for mp in model_parts for p in mp.parameters() if p.grad is not None]
    if num_label_tokens is not None:
        for p in params:
            if p.grad is not None:
                # detach_() 在此处是安全措施：虽然 p.grad 本身已是 leaf tensor
                # 且 div_ 是 in-place 操作不会构建 autograd 图，但 FSDP2 的
                # _post_backward_hook 可能在 grad 上附加了 hook 句柄；
                # detach_() 确保这些 hook 不会在后续 clip_grad_norm_ 中被意外触发。
                p.grad.detach_().div_(num_label_tokens)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(grad_norm)


# 其他被调用但定义在他处的 helper：
def _dp_cp_all_reduce_sum(tensor, dp_cp_mesh) -> torch.Tensor:
    """在 DP+CP 联合 mesh 上做 all-reduce sum。
    用途：统计全局 label token 数、val token 数。
    返回 reduce 后的 tensor（形状与输入一致）。
    兼容 Python 标量入参（内部 wrap 为 tensor 再 reduce）。
    """
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, device=torch.cuda.current_device())
    if tensor.device.type != "cuda":
        tensor = tensor.cuda()
    # 确保 tensor 在 dp_cp_mesh 内所有 rank 上都是单元素标量
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_cp_mesh.get_group())
    return tensor


# shard_batch_for_cp 为 05 canonical，已落地于
# hyper_models/components/distributed/cp_utils.py，03 仅 import 调用：
#   from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
# 签名：def shard_batch_for_cp(batch: dict, cp_mesh) -> dict
# 契约：input_ids/labels/position_ids [B,S] int64；seq_lens/seq_lens_padded
# [B, max_num_packs] int64 且以 -1000 为哨兵填充；qkv_format="thd" 透传。
# 切分策略：pad 到 2*cp 倍数后按 contiguous chunk [cp_rank*chunk,
# (cp_rank+1)*chunk) 切片；seq_lens 系列由 _shard_seq_lens_for_cp 单独重算。
#
# CP 前向通信（K/V all-gather）不在训练循环内发生：apply_sharding_plan
# 编译期对标记 _needs_cp_attn 的边界模块调用
# sharding_applier._wrap_cp_inner_attention(
#     attn_module, cp_mesh, *, spec=None, mesh=None, mesh_dim_names=())
# （cp_mesh.size()>1 时生效），把 inner attention 的 forward 替换为
# CP-aware 版本——内部调 flex_cp_allgather(k, v, cp_dim=2, cp_mesh)
# （带 autograd 的 all-gather：前向 all-gather K/V，反向 reduce-scatter），
# is_causal 时替换为按本 rank Q 全局偏移的 offset-aware 显式 mask（D-04）。
# 因此训练循环只需 shard_batch_for_cp 切数据，**无需 make_cp_context /
# RingAttentionContext / attach_context_parallel_hooks 之类的运行时 hook**。
# （早期草案的 ring-attention 方案 D-01'' 已否决，本文统一为 all-gather K/V。）


def _dp_all_reduce_avg(tensor, dp_mesh=None) -> torch.Tensor:
    """纯 DP 维度 all-reduce mean（除以 dp_world_size）。
    用途：val loss 跨 DP rank 平均；CP 维度不参与（非冗余）。
    `dp_mesh`：DP 维 DeviceMesh（或其 SubMesh），用于取 DP process group。
    若为 None，退化为全局 all-reduce mean（需调用方确保仅在纯 DP 拓扑下使用）。
    兼容 Python 标量入参（内部 wrap 为 tensor 再 reduce）。
    """
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, device=torch.cuda.current_device())
    if tensor.device.type != "cuda":
        tensor = tensor.cuda()
    group = dp_mesh.get_group() if dp_mesh is not None else dist.group.WORLD
    world_size = dp_mesh.size() if dp_mesh is not None else dist.get_world_size(group)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    tensor.div_(world_size)
    return tensor


def calculate_mfu(
    tps: float,
    flops_per_token: float,
    peak_tflops: float,
    world_size: int,
) -> float:
    """Model FLOPs Utilization = (tps * flops_per_token) / (peak_tflops * world_size)。
    返回 [0, 1] 区间的标量。"""
    # peak_tflops 为单卡 TFLOPS（e.g. H100 bf16 ≈ 989），乘以 world_size 得总算力
    total_peak_tflops = peak_tflops * world_size
    if total_peak_tflops <= 0:
        return 0.0
    mfu = (tps * flops_per_token) / (total_peak_tflops * 1e12)
    return min(max(mfu, 0.0), 1.0)


def filter_forward_kwargs(model: nn.Module, batch: dict) -> dict:
    """过滤 batch 中 model.forward 不接受的 kwarg，避免 TypeError。
    依据 `inspect.signature(model.forward).parameters`。"""
    import inspect

    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        # 某些 wrapped module 的 forward 可能无法 inspect，退回全部 batch
        return dict(batch)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in batch.items() if k in accepted}


def calculate_mtp_loss(
    mtp_per_depth_logits: list[torch.Tensor],
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Multi-Token-Prediction 辅助 loss（Qwen3.5 等）。
    逐 depth 计算 CE 并求和（与主 loss 同尺度，token-mean）。
    """
    total_mtp_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    for depth_idx, (logits, h) in enumerate(
        zip(mtp_per_depth_logits, mtp_per_depth_h)
    ):
        # Shift: 预测下一个 token
        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()
        depth_loss = loss_fn(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
        )  # reduction="sum"
        total_mtp_loss = total_mtp_loss + depth_loss
    return total_mtp_loss


def setup_magi(cfg, device_mesh):
    """构建 MagiAttention 上下文（可选）；无配置时返回 None。"""
    magi_cfg = cfg.get("magi", None)
    if magi_cfg is None:
        return None
    try:
        from magi_attention import MagiAttentionContext
    except ImportError:
        logger.warning("magi_attention not installed; skipping MagiAttention setup")
        return None
    return MagiAttentionContext(
        device_mesh=device_mesh,
        **({} if isinstance(magi_cfg, bool) else dict(magi_cfg)),
    )


class AutoMFU:
    """MFU 计算器：缓存 flops_per_token / peak_tflops。"""

    def __init__(self, flops_per_token: float, peak_tflops: float):
        self.flops_per_token = flops_per_token
        self.peak_tflops = peak_tflops

    @classmethod
    def from_config(cls, model: nn.Module) -> "AutoMFU":
        """从 model config 推断 flops_per_token；peak_tflops 从设备读取。"""
        # 从 HuggingFace 模型 config 获取参数量和 hidden_size 等信息
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "num_hidden_layers"):
            # 标准 Transformer：每 token FLOPs ≈ 6N + 12 * L * H * d_ff
            # 简化估计：2 * 参数总量（前向）+ 4 * 参数总量（反向）= 6 * 参数总量
            num_params = sum(p.numel() for p in model.parameters())
            flops_per_token = 6.0 * num_params
        else:
            # fallback: 从 model 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            flops_per_token = 6.0 * num_params

        # peak_tflops：从 GPU 型号推断（保守取值）
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        peak_tflops = _infer_peak_tflops(device_name)

        return cls(flops_per_token=flops_per_token, peak_tflops=peak_tflops)


def _infer_peak_tflops(device_name: str) -> float:
    """根据 GPU 名称推断 bf16 峰值 TFLOPS（保守值）。"""
    name_lower = device_name.lower()
    if "h100" in name_lower or "h800" in name_lower:
        return 989.0   # H100/H800 bf16
    elif "a100" in name_lower or "a800" in name_lower:
        return 312.0   # A100/A800 bf16
    elif "h20" in name_lower:
        return 148.0   # H20 bf16
    elif "v100" in name_lower:
        return 125.0   # V100 (fp16, bf16 不支持)
    elif "4090" in name_lower:
        return 330.0   # RTX 4090 bf16
    else:
        # 默认值：假设为中等性能 GPU
        return 200.0


def _is_rank_0() -> bool:
    """当前进程是否为全局 rank 0。"""
    return dist.get_rank() == 0 if dist.is_initialized() else True


def _update_latest_symlink(checkpoint_dir: str, path: str) -> None:
    """原子更新 `{checkpoint_dir}/LATEST` 软链接，指向最新的 step 目录。
    用 `os.symlink` + rename 实现原子替换。

    软链接写**相对路径**（相对 checkpoint_dir）的行为保留；消费端 04 的
    `_resolve_latest_symlink` 需相对 checkpoint_dir 解析该链接（04 侧同步修）。
    """
    import tempfile

    latest = os.path.join(checkpoint_dir, "LATEST")
    # 计算相对路径（软链接存相对路径更健壮）
    rel_path = os.path.relpath(path, checkpoint_dir)

    # 原子替换：先写临时文件，再 os.rename（POSIX 保证原子性）
    tmp = os.path.join(checkpoint_dir, ".LATEST.tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(rel_path, tmp)
    os.rename(tmp, latest)
```

---

## 8. 前向+反向传播

> **调用位置**: 时序树 ⑤.1.2 — `_forward_backward_step()`

```python
def _forward_backward_step(
    self, idx: int, batch: dict, *,
    loss_buffer: list,
    num_label_tokens: int,
    num_batches: int,
) -> None:
    """单次 microbatch 的前向 + 反向传播。"""
    model = self.model_parts[0]

    # ── Step 1: 数据 → GPU ──
    batch = {
        k: v.to(self.dist_env.device, non_blocking=True)
        if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # ── Step 2: CP batch 准备 ──
    # CP（Context Parallel）沿序列维度切分：每个 cp rank 持有 Q 的 contiguous
    # chunk；K/V 在 attention 内部由 CP wrapper 做 all-gather（flex_cp_allgather，
    # 带 autograd），见 §7.1 末尾的 CP 机制说明。
    # shard_batch_for_cp(batch, cp_mesh) -> dict 返回切分后的 batch：
    # input_ids/labels/position_ids 按 [cp_rank*chunk, (cp_rank+1)*chunk) 切片
    # （labels 的 pad 用 -100，CE 的 ignore_index 天然屏蔽）；seq_lens /
    # seq_lens_padded 由 _shard_seq_lens_for_cp 按本 rank 区间重算并保留
    # -1000 哨兵，供模型侧 varlen attention 与 loss 还原使用；qkv_format 透传。
    # CP 包装由 apply_sharding_plan 编译期完成（_wrap_cp_inner_attention），
    # 训练循环无需任何 CP context manager / hook。
    if self.mesh.cp_size > 1:
        if hasattr(model, "prepare_model_inputs_for_cp"):
            batch = model.prepare_model_inputs_for_cp(**batch)
        else:
            batch = shard_batch_for_cp(batch, self.mesh.cp_mesh)

    # ── Step 3: 分离 labels ──
    labels = batch.pop("labels", None)

    # ── Step 4: 前向传播 ──
    # FSDP2 梯度同步策略：只有非最后一个 microbatch 才 defer 梯度 all-reduce；
    # 最后一个 microbatch 的 backward 必须触发 DP all-reduce，使梯度聚合完整。
    # 注意：_run_train_optim_step 在最后 microbatch 前已调
    # prepare_for_final_backward（set_requires_gradient_sync(True)），
    # 此处 defer_fsdp_grad_sync=False 确保 get_sync_ctx 不再次关闭同步。
    sync_ctx = get_sync_ctx(
        self.model_parts,
        is_optim_step=True,
        defer_fsdp_grad_sync=(idx != num_batches - 1),
    )

    with sync_ctx:
        # 过滤 forward 不接受的 kwargs
        filtered_batch = filter_forward_kwargs(model, batch)

        # CP 模式：K/V all-gather 在 CP wrapper 包裹的 inner attention forward
        # 内部发生（编译期注入，见 §7.1）；backward 沿 all-gather 的 autograd
        # Function 自动做 reduce-scatter，无需在此显式处理。
        output = model(**filtered_batch)

        # ── Step 5: Loss 计算 ──
        # local_loss = ce_sum_local（raw，不除 N）
        # 其中 ce_sum_local 是本 rank（含 DP/CP 切分）上的 CE sum；
        # token-mean 归一化（除以 N_global）推迟到 §7.1 scale_grads 统一完成，
        # 避免 calculate_loss 与 scale_grads 双除 num_label_tokens。
        # 见 §10.1 loss 归一化推导。
        logits = output.logits if hasattr(output, "logits") else output
        local_loss = calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=model,
            num_label_tokens=num_label_tokens,
            # loss_aggregation 从 LossConfig 透传（§10.0），缺省 token_weighted；
            # rank_average 路径 loss 为 mean 尺度，§7 Phase 3 相应跳过除 N
            loss_aggregation=getattr(
                self.cfg.loss_fn, "loss_aggregation", "token_weighted"
            ),
            hidden_states=getattr(output, "hidden_states", None),
            lm_weight=(
                model.lm_head.weight
                if hasattr(model, "lm_head") and model.lm_head is not None
                else None
            ),
        )

        # MTP loss（Qwen3.5 等）
        if hasattr(output, "mtp_per_depth_logits"):
            local_loss += calculate_mtp_loss(
                output.mtp_per_depth_logits,
                output.mtp_per_depth_h,
                labels,
                self.loss_fn,
            )

        loss_buffer.append(local_loss.detach())

        # ── Step 6: 反向传播 ──
        # 缩放: 取消 FSDP2 的 1/dp_size 除法
        # local_loss 为 raw ce_sum（未除 N）；backward 用 ce_sum，
        # FSDP2 DP-mean 后由 scale_grads 统一除 N 还原 token-mean。
        # 注：loss 乘以 dp_size 是为了在 FSDP2 all-reduce 均值后恢复总 loss；
        # cp 维度不需要额外乘法，因为每个 cp rank 处理不同的序列段（不是冗余计算）。
        dp_group_size = self.mesh.dp_size
        (local_loss * dp_group_size).backward()


# ── CP backward 机制说明（all-gather K/V 方案，D-01''） ──
#
# CP 维度的前向：每个 cp rank 持有 Q 的 contiguous chunk；inner attention
# 内部先由 flex_cp_allgather 把 K/V 沿序列维 all-gather 为全量，再用本 rank
# Q chunk 做 SDPA（is_causal 时替换为按本 rank Q 全局偏移 lo 的 offset-aware
# 显式 mask，D-04），得到本 rank 段的 attention 输出。loss 计算时
# ce_sum_local 只统计本 rank 持有段的 token（labels 的 CP pad 位为 -100，
# 被 ignore_index 屏蔽；packed 序列按切分后的 seq_lens 还原）。
#
# CP 维度的反向：flex_cp_allgather 是显式 autograd.Function
# （_AllGatherAlongDim），前向 all-gather、反向 reduce-scatter 语义
# （梯度跨 rank all-reduce 求和后取本 rank chunk），由 autograd 自动触发，
# 无需用户插入反向通信。因此 backward() 调用与无 CP 场景完全一致，
# 不需要额外的 cp_size 因子（CP 不是冗余计算，梯度无需除以 cp_size）。
#
# 相关签名（已落地：hyper_models/components/distributed/cp_utils.py、
# sharding_applier.py；05 §4.4.2 / §6.3.4 canonical）：
#   def shard_batch_for_cp(batch: dict, cp_mesh) -> dict:
#       """按 cp_rank 取 contiguous chunk 切分 batch（seq_lens/seq_lens_padded
#       按区间重算、保留 -1000 哨兵），只返回切分后的 batch dict。"""
#       ...
#
#   def flex_cp_allgather(k, v, cp_dim: int, cp_mesh):
#       """K/V 沿 cp_dim 在 CP 组内 all-gather（带 autograd；通信组取
#       cp_mesh.get_group()，禁 new_group）。cp_size<=1 时原样返回。"""
#       ...
#
#   def _wrap_cp_inner_attention(attn_module, cp_mesh, *, spec=None,
#                                mesh=None, mesh_dim_names=()):
#       """编译期（apply_sharding_plan Phase C）把 inner attention 替换为
#       CP-aware forward；cp_mesh.size()>1 且 spec._needs_cp_attn 时生效。"""
#       ...
#
#   def prepare_model_inputs_for_cp(self, **batch) -> dict:
#       """模型自带版 CP 输入准备（处理 seq_lens 等），返回切分后 batch。
#       若模型实现了该方法则优先使用，否则回退到 shard_batch_for_cp。"""
#       ...
#
# 注：01 §8.3 ⑩ 早期草案中的 attach_context_parallel_hooks 不存在——
# CP 包装由 apply_sharding_plan 内部完成，训练循环无需额外 hook。
```

---

## 9. Optimizer 与 LR Scheduler

> **调用位置**: 时序树 ④.8 / ④.12 — typed `.build()` 路径（`_target_` → typed config → `.build()`）

### 9.1 设计理念

AutoModel 为 optimizer/scheduler/loss 等**依赖运行时对象的组件**使用两层模式：

1. **Layer 1 — RecipeConfig.__init__**：`_callable_and_kwargs(node)` 提取 `_target_` factory + kwargs → **类型化 Config 实例**（类型校验完成）
2. **Layer 2 — Recipe.setup()**：`cfg.xxx.build(**runtime_deps)` → **真正的组件**

新增优化器类型只需修改 YAML 的 `_target_`，`build_optimizer_config()` 自动路由。

```yaml
# 使用 torch.optim.AdamW
optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1

# 切换到 Muon + AdamW（仅需改 _target_，Recipe 代码零改动）
optimizer:
  _target_: dion.Muon
  lr: 1.0e-3
  weight_decay: 0.01

# 调度器
lr_scheduler:
  _target_: hyper_models.components.optim.lr_scheduler.WarmupCosineScheduler
  warmup_steps_ratio: 0.1
  min_lr_ratio: 0.0
```

### 9.2 RecipeConfig.optimizer：`_target_` → OptimizerConfig

```python
# recipes/_typed_config.py

@cached_property
def optimizer(self) -> "OptimizerConfig | None":
    from hyper_models.components.optim.optimizer import build_optimizer_config

    node = self._raw.get("optimizer", None)
    if node is None:
        return None
    factory, kwargs = _callable_and_kwargs(node)
    # build_optimizer_config 将 factory(如 torch.optim.AdamW) + kwargs
    # 归一化为 OptimizerConfig 子类实例
    return build_optimizer_config(factory, kwargs)
```

### 9.3 OptimizerConfig.build(model) → 真正的优化器

```python
# hyper_models/components/optim/optimizer.py

class OptimizerConfig:
    """优化器配置基类——所有优化器类型的 typed config。

    子类实现 build()，负责参数分组 + 实例化。
    """
    # 梯度裁剪范数上限（由 train loop 在 scale_grads_and_clip_grad_norm 中使用，
    # 见 03 §7.1 / §10.1）。YAML optimizer 段可覆盖，缺省 1.0。
    max_grad_norm: float = 1.0

    def build(
        self,
        model: nn.Module,
        *,
        optimizer_init: "OptimizerInit | None" = None,
        device_mesh: DeviceMesh | None = None,
        is_peft: bool = False,
    ) -> list[torch.optim.Optimizer]:
        """构建优化器（支持 model.parts 返回多个）。

        `optimizer_init`：由 build_model 导出的 param 分组/mesh 描述（01 §6.2），
        可选——若传入则复用其 param_groups，避免重复推导。
        """
        raise NotImplementedError


class AdamWConfig(OptimizerConfig):
    """AdamW 的 typed config。"""
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.1
    foreach: bool = True

    def build(self, model, *, optimizer_init=None, device_mesh=None, is_peft=False):
        parts = getattr(model, "parts", [model])
        optimizers = []
        for part in parts:
            # 优先复用 optimizer_init.param_groups（已由 ShardingPlan 推导，
            # 01 §2.14 修正后 group 内为实际 weight_decay 值——此处原样复用，
            # 不再覆盖 weight_decay）；否则现场用 _is_no_decay +
            # _build_param_groups 推导
            if optimizer_init is not None and getattr(optimizer_init, "param_groups", None):
                param_groups = optimizer_init.param_groups
            else:
                param_groups = _build_param_groups(part, self.weight_decay)
            optimizers.append(torch.optim.AdamW(
                param_groups,
                lr=self.lr, betas=self.betas, eps=self.eps,
                foreach=self.foreach,
            ))
        return optimizers


class OptimizerFromFactoryConfig(OptimizerConfig):
    """外部优化器（如 dion.Muon）的 escape hatch——保留 factory 引用。"""
    def __init__(self, factory, kwargs):
        self.factory = factory
        self.kwargs = kwargs

    # 签名与基类 OptimizerConfig.build 一致（补 optimizer_init 形参），
    # 外部优化器（如 Muon 示例）路径可用。
    def build(self, model, *, device_mesh=None, optimizer_init=None,
              is_peft=False):
        # 与 AdamWConfig 同口径：优先复用 optimizer_init.param_groups
        # （01 §2.14 已在 group 内写入实际 weight_decay 值，此处不重复覆盖，
        # 也不再从 kwargs 取 weight_decay 传入 factory）
        if optimizer_init is not None and getattr(optimizer_init, "param_groups", None):
            param_groups = optimizer_init.param_groups
        else:
            param_groups = _build_param_groups(model, self.kwargs.get("weight_decay", 0.1))
        return [self.factory(param_groups, **{k: v for k, v in self.kwargs.items()
                                               if k != "weight_decay"})]
```

### 9.4 build_optimizer_config：factory + kwargs → OptimizerConfig 子类

```python
# hyper_models/components/optim/optimizer.py

def build_optimizer_config(
    target,  # OptimizerConfig | str | type | callable
    kwargs: dict | None = None,
) -> OptimizerConfig:
    """归一化入口：将 _target_ factory + kwargs 转为 OptimizerConfig 实例。

    支持的 target 类型：
    - OptimizerConfig 实例 → 直接返回
    - OptimizerConfig 子类 → target(**kwargs)
    - 字符串（"adamw" / "torch.optim.AdamW"）→ 查表 + 导入
    - 其他 callable（如 dion.Muon）→ OptimizerFromFactoryConfig(factory, kwargs)
    """
    if isinstance(target, OptimizerConfig):
        return target
    if isinstance(target, str):
        resolved = OPTIMIZER_CONFIG_REGISTRY.get(target.lower())
        if resolved is None:
            resolved = _import_from_path(target)
        target = resolved
    kwargs = dict(kwargs or {})
    if isinstance(target, type) and issubclass(target, OptimizerConfig):
        return target(**kwargs)
    if callable(target):
        return OptimizerFromFactoryConfig(factory=target, kwargs=kwargs)
    raise TypeError(f"Unsupported optimizer target: {target!r}")
```

### 9.5 参数分组（在 OptimizerConfig.build() 内部）

```python
def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """decay / no_decay 参数分组（与优化器类型无关）。"""
    decay_params, no_decay_params = [], []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_ids:
            continue
        seen_ids.add(param_id)

        if _is_no_decay(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _is_no_decay(name: str) -> bool:
    no_decay_patterns = ("bias", "norm", "rmsnorm", "layernorm", "ln_")
    return any(pattern in name.lower() for pattern in no_decay_patterns)
```

### 9.6 LR Scheduler：对齐 AutoModel 的 step-based 设计

**设计决策**：为兼容 AutoModel checkpoint（`OptimizerParamScheduler.state_dict()`），hyper_parallel 采用 AutoModel 的 **step-based** 配置（绝对步数，非比例），而非之前设计的 ratio-based `LambdaLR`。如果需要 ratio-based 便利性，可以额外提供一个轻量 wrapper 将比例转换为绝对步数。

> 注：`OptimizerParamScheduler` 来源于 AutoModel 的
> `nemo_automodel.components.optim.lr_scheduler`，但 hyper_parallel 不 import
> nemo_automodel（与 02 开头"不 import nemo_automodel"的约定一致）——需将
> 该类 **port 进 hyper_parallel**，存放于
> `hyper_models/components/optim/lr_scheduler.py`（文件头注明来源与出处
> commit），使用时直接：
> ```python
> from hyper_models.components.optim.lr_scheduler import OptimizerParamScheduler
> ```

```python
# RecipeConfig
@cached_property
def lr_scheduler(self) -> "LRSchedulerConfig | None":
    node = self._raw.get("lr_scheduler", None)
    return LRSchedulerConfig(**_section_kwargs(node)) if node else None


@dataclass
class LRSchedulerConfig:
    """LR 调度器 typed config —— 与 AutoModel 字段对齐。

    所有步数字段为绝对步数（非比例），未设置时从 step_scheduler 推断默认值。
    """
    # ── LR 衰减 ──
    lr_warmup_steps: int | None = None       # warmup 绝对步数
    lr_decay_steps: int | None = None        # 衰减总步数（None → max_steps - warmup_steps）
    lr_decay_style: str = "cosine"           # "cosine" | "linear" | "constant" | "inverse-square-root" | "WSD"
    init_lr: float | None = None             # 初始 LR（None → 从 optimizer 推断）
    max_lr: float | None = None              # 最大 LR（None → 从 optimizer 推断）
    min_lr: float | None = None              # 最小 LR

    # ── Weight Decay 调度 ──
    start_wd: float | None = None            # WD 起始值
    end_wd: float | None = None              # WD 终止值
    wd_incr_steps: int | None = None         # WD 增长步数
    wd_incr_style: str = "constant"

    # ── WSD 模式（Warmup-Stable-Decay） ──
    wsd_decay_steps: int | None = None       # WSD 衰减步数
    lr_wsd_decay_style: str | None = None    # WSD 衰减风格

    # ── 高级 ──
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False

    def build(self, optimizer, step_scheduler) -> list[OptimizerParamScheduler]:
        """构建 OptimizerParamScheduler（与 AutoModel checkpoint 兼容）。"""
        # 未设置的字段从 step_scheduler 推断默认值
        max_steps = step_scheduler.max_steps
        lr_warmup_steps = self.lr_warmup_steps if self.lr_warmup_steps is not None else 0
        lr_decay_steps = self.lr_decay_steps or (max_steps - lr_warmup_steps)

        # 从 optimizer param_groups 推断 init_lr / max_lr
        # optimizer 为 list[Optimizer]；取第一个优化器的 param_groups
        opt = optimizer if not isinstance(optimizer, list) else optimizer[0]
        # 注意：torch.optim.AdamW param_groups 无 "initial_lr" 键。
        # init_lr 优先取 YAML 显式配置（LRSchedulerConfig.init_lr），
        # 其次从 optimizer param_groups 的 "lr" 键推断（冷启动与初始 LR 相同；
        # resume 场景下已由 load_state_dict 恢复为断点 LR，此处作为 fallback 仍
        # 足够——因为 OptimizerParamScheduler 内部通过 self._last_lr 追踪当前 LR，
        # init_lr 仅用于 warmup 起始值计算）。
        init_lr = self.init_lr if self.init_lr is not None else opt.param_groups[0]["lr"]
        max_lr = self.max_lr if self.max_lr is not None else opt.param_groups[0]["lr"]

        return [OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=init_lr,
            max_lr=max_lr,
            min_lr=self.min_lr if self.min_lr is not None else 0.0,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=self.lr_decay_style,
            start_wd=self.start_wd,
            end_wd=self.end_wd,
            wd_incr_steps=self.wd_incr_steps,
            wd_incr_style=self.wd_incr_style,
            wsd_decay_steps=self.wsd_decay_steps,
            lr_wsd_decay_style=self.lr_wsd_decay_style,
        )]
```

**YAML 配置**（使用绝对步数，与 AutoModel 兼容）：

```yaml
lr_scheduler:
  lr_warmup_steps: 100        # 绝对步数，非比例
  lr_decay_style: cosine
  min_lr: 1.0e-6
  # WD 调度（可选）
  start_wd: 0.01
  end_wd: 0.1
  wd_incr_steps: 500
```

**Ratio-based 便利 wrapper**（可选，轻量包装）：

```yaml
# 如果偏好 ratio 方式，使用 wrapper _target_
lr_scheduler:
  _target_: hyper_models.components.optim.lr_scheduler.RatioBasedLRSchedulerConfig
  warmup_steps_ratio: 0.1
  min_lr_ratio: 0.0
  lr_decay_style: cosine
```
```python
@dataclass
class RatioBasedLRSchedulerConfig(LRSchedulerConfig):
    """接受 ratio 参数，在 build() 中转换为绝对步数。"""
    warmup_steps_ratio: float = 0.1
    min_lr_ratio: float = 0.0

    def build(self, optimizer, step_scheduler):
        self.lr_warmup_steps = int(step_scheduler.max_steps * self.warmup_steps_ratio)
        self.lr_decay_steps = step_scheduler.max_steps - self.lr_warmup_steps
        # max_lr fallback：未显式配置时取 optimizer 当前 lr
        # （#20 修复：旧值 `or 1.0` 会在用户漏配 lr 时得到错误的 1.0）
        # optimizer 为 list[Optimizer]，取第一个优化器的 param_groups
        opt = optimizer if not isinstance(optimizer, list) else optimizer[0]
        max_lr = self.max_lr or opt.param_groups[0]["lr"]
        self.min_lr = max_lr * self.min_lr_ratio
        return super().build(optimizer, step_scheduler)
```

---

## 10. Loss 计算

> **调用位置**: 时序树 ⑤.1.2 — `calculate_loss()` dispatcher（dispatcher 模式）

```python
# hyper_models/components/loss/utils.py

def calculate_loss(loss_fn: nn.Module, **kwargs) -> torch.Tensor:
    """统一的 loss 计算 —— 根据 loss_fn 类型分发。

    支持两种 loss 路径：
    - FusedLinearCrossEntropy：融合 lm_head + CE，直接接收 hidden_states
    - 标准 logit-based loss：CE / MaskedCrossEntropy 等

    注：num_label_tokens 由调用方传入 kwargs 但函数体内未直接使用——保留该参数
    是为自定义 loss_fn 提供归一化所需的全局 token 数（例如某些 loss 内部需要
    除以 N 做 token-mean，此时可通过 kwargs 取用）。
    """
    # ── 路径 A: FusedLinearCrossEntropy（融合 lm_head + CE） ──
    # 返回 raw ce_sum（不除 N）；token-mean 归一化由 §7.1 scale_grads 统一除以
    # num_label_tokens 完成，避免双除。FusedLinearCrossEntropy 内部以
    # reduction="sum" 计算，不传 num_label_tokens。
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        hidden_states = kwargs.get("hidden_states")
        lm_weight = kwargs.get("lm_weight")
        if hidden_states is not None and lm_weight is not None:
            return loss_fn(
                hidden_states=hidden_states,
                labels=kwargs["labels"],
                lm_weight=lm_weight,
            )
        # fallback：调用方未提供 hidden_states/lm_weight 时退回 logits 路径
        logits = kwargs.get("logits")
        labels = kwargs.get("labels")
        if logits is None:
            raise ValueError(
                "FusedLinearCrossEntropy requires hidden_states+lm_weight or logits; "
                "neither was provided."
            )
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        return loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )  # reduction="sum"

    # ── 路径 B: 标准 logit-based loss ──
    else:
        logits = kwargs["logits"]
        labels = kwargs["labels"]
        model = kwargs.get("model")

        # Shift: 标准自回归（预测下一个 token）—— causal LM 总是需要 shift
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        if kwargs.get("loss_aggregation", "token_weighted") == "token_weighted":
            # 返回 raw ce_sum（不除 N）；token-mean 归一化由 scale_grads 统一完成。
            # num_label_tokens 不在此处使用（避免与 scale_grads 双除）。
            return loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )  # reduction="sum"
        else:
            # rank_average: 等长 batch 场景（不参与 token-mean 归一化路径，
            # scale_grads 应传 num_label_tokens=None 跳过除法）
            return loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )  # reduction="mean"

# ── 支持的 Loss 类清单 ──
# 一期支持:
#   - MaskedCrossEntropy
#   - FusedLinearCrossEntropy（融合 lm_head + CE）
# 二期规划:
#   - KDLoss（知识蒸馏）
#   - ChunkedCrossEntropy（分块 CE，超长序列）
```

### 10.0 LossConfig —— typed config

```python
@dataclass
class LossConfig:
    """Loss typed config —— RecipeConfig.loss_fn 的返回类型。

    支持两种消费路径：
    - _target_ 未设置：默认构建 MaskedCrossEntropy()
    - _target_ 设置为 Loss 子类：.build() 实例化 _target_(**kwargs)
    """
    _target_: type | None = None
    loss_aggregation: str = "token_weighted"
    kwargs: dict = field(default_factory=dict)

    def build(self) -> nn.Module:
        if self._target_ is None:
            return MaskedCrossEntropy()
        return self._target_(**self.kwargs)


def build_loss_config(factory, **kwargs) -> LossConfig:
    """归一化入口：将 _target_ factory + kwargs 转为 LossConfig 实例。

    factory 无论是 nn.Module 子类还是普通 callable，最终都归一为同一个
    LossConfig（loss 的实例化由 RecipeConfig/ConfigNode.instantiate 统一处理）。
    此函数仅为调用点提供类型稳定的入口，不做分支特化。
    """
    return LossConfig(_target_=factory, kwargs=kwargs)
```

### 10.1 Loss 归一化数学推导

设全局 label token 数为 `N_global`（跨 DP+CP all-reduce 得到，见 §7 Phase 1）。
本 rank 持有的 CE sum 记为 `ce_sum_local`（含 DP 切分和 CP 切分，仅覆盖本
rank 实际负责的 token 子集）。归一化分两步：

1. **前向 / backward**：`calculate_loss` 返回 **raw `ce_sum_local`**（不除
   `N_global`）。backward 执行 `(ce_sum_local * dp_group_size).backward()`，
   乘 `dp_group_size` 抵消 FSDP2 的 DP-mean 除法。
2. **梯度归一化**：`scale_grads_and_clip_grad_norm` 对每个参数 `.grad` 除以
   `N_global`，将 CE sum 还原为 token-mean 梯度。

```
local_loss (backward 用) = ce_sum_local          # raw，不除 N
grad_after_dp_mean       = dp_mean(∂ce_sum_local/∂θ) * dp_group_size
                          = sum_dp(∂ce_sum_local/∂θ)   # 还原总 loss 梯度
grad_token_mean          = grad_after_dp_mean / N_global   # scale_grads 完成
```

> 该分工避免 `calculate_loss` 与 `scale_grads` 双除 `N_global`（旧实现两者
> 各除一次，梯度差 N 倍）。日志侧的 loss 数值在 §7 Phase 3 / §6.1 validation
> 中显式除以 `N_global` 还原 token-mean 用于上报。

**梯度累积天然正确**：在 grad accumulation 场景下，多个 microbatch 的
`ce_sum_local` 与对应 token 数都跨 microbatch 累加进同一份梯度，分子分母
同步累加，最终梯度等价于把所有 microbatch 拼成一个大 batch 计算的 token-mean
梯度。`scale_grads` 在所有 microbatch 反向完成后统一除以 `N_global`（累计
全局 token 数）即可。

**DP 因子**：FSDP2 在 backward 末尾会对梯度做 DP 维 all-reduce **mean**
（除以 `dp_size`）。为了让 `ce_sum_local * dp_group_size` 在 all-reduce 后
还原为正确的总 loss 梯度，前向 loss 需乘 `dp_size` 抵消（见 §8 Step 6）。

**CP 因子**：CP 不是冗余计算——每个 cp rank 处理序列的不同段，`ce_sum_local`
只覆盖本段 token，梯度无需跨 cp rank 平均。因此 **loss 不乘 cp_size**，
`N_global` 的统计也不除以 cp_size（全局 token 数 = 各 cp rank 持有段之和）。

**PP 因子**：PP 多 stage 时，`num_label_tokens` 只在最后一 stage 准确，
中间 stage 的 loss 缩放由 PP runtime 通过 send/recv 自动平衡；
`scale_grads_and_clip_grad_norm` 对 PP 场景传 `num_label_tokens=None`，
跳过本地除法。

### 10.2 混合精度 dtype 说明

模型在 bf16 下前向得到的 `logits` 也是 bf16。直接在 bf16 上做
`CrossEntropyLoss(reduction="sum")` 会因 `log_softmax` + `nll_loss` 的
大数值累加导致精度损失。`MaskedCrossEntropy` 内部对 logits 做 fp32 recast：

```python
# hyper_models/components/loss/masked_ce.py
def forward(self, logits, labels):
    # logits: bf16 [N, V] → recast 到 fp32 计算 CE，避免大词表累加溢出
    logits_fp32 = logits.float()
    # shift + CE（reduction="sum"），返回 fp32 标量
    return F.cross_entropy(
        logits_fp32.view(-1, logits.size(-1)), labels.view(-1),
        ignore_index=-100, reduction="sum",
    )
```

`calculate_loss` 拿到的 `ce_sum` 为 fp32 标量（raw，不除 N）；
`ce_sum * dp_group_size` 在 fp32 上反向，梯度由 autograd 自动
cast 回 bf16 注入参数 `.grad`；`scale_grads` 再在 bf16 `.grad` 上除以
`N_global`。MTP loss 同样走 fp32 recast 路径（返回 raw sum）。

---

## 11. DistributedSignalHandler

> **调用位置**: 时序树 ④ — `StepScheduler.__init__` 中创建，控制 SIGTERM 响应

```python
# hyper_models/components/training/signal_handler.py

class DistributedSignalHandler:
    """SIGTERM 分布式协调——任意 rank 收到 → 全体响应。"""

    def __init__(self):
        self._signal_received = False
        self._orig_handler = None

    def __enter__(self):
        self._orig_handler = signal.signal(signal.SIGTERM, self._handler)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGTERM, self._orig_handler)

    def _handler(self, signum, frame):
        logger.warning("Rank %d received SIGTERM", dist.get_rank())
        self._signal_received = True

    def signals_received(self) -> list[bool]:
        """all_gather：只要有一个 rank 收到 → 全体返回 True。"""
        # NCCL 不支持 CPU tensor 的集合通信——选型：把 tensor 搬到当前 CUDA
        # 设备再走默认（NCCL）group，而非新建 gloo group。理由：进程组初始
        # 化只有 nccl（initialize_distributed("nccl")），单后端少一个需要
        # 生命周期管理的专用 group，开销可忽略（每步 1 个 int32）。
        device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.tensor([int(self._signal_received)], dtype=torch.int32,
                              device=device)
        gathered = [torch.zeros(1, dtype=torch.int32, device=device)
                    for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return [bool(t.item()) for t in gathered]
```

---

## 12. Recipe 变体体系

> 参考 automodel 的继承体系，Hyper-Parallel 的 Recipe 变体全部继承自 `BaseRecipe`。
> 当前一期覆盖 LLM 和 VLM 两种任务，后续可按需扩展。

### 12.1 继承体系总览

```
BaseRecipe（__state_tracked + save/load_checkpoint + callback_manager）
  │
  ├─ FinetuneRecipe（LLM 标准训练，当前文档主体）
  │   └─ 覆盖: setup() / run_train_validation_loop() / _forward_backward_step()
  │            _run_train_optim_step() / _run_validation_epoch()
  │
  └─ FinetuneRecipeForVLM（VLM 多模态训练）
      └─ 覆盖: setup() / run_train_validation_loop() / _forward_backward_step()
               _run_train_optim_step() / _run_validation_epoch()
```

### 12.2 FinetuneRecipeForVLM —— 与 FinetuneRecipe 的差异

两个 Recipe 继承自同一 `BaseRecipe`，共享 `__state_tracked` 自动状态追踪、`save_checkpoint` / `load_checkpoint`、`CallbackManager` 和 `StepScheduler` 等基础设施。差异集中在以下 5 个方面：

| 差异点 | FinetuneRecipe（LLM） | FinetuneRecipeForVLM（VLM） | 原因 |
|--------|---------------------|----------------------------|------|
| **setup() 组件构建** | 标准模型 + 文本 DataLoader | 多模态模型 + VLM 专属 DataLoader（`vlm_dataloader`） | 多模态需要 vision tower、processor、chat_template |
| **forward_backward** | CE loss on logits | 多模态 loss（含 vision tower 梯度） | 多模态输出包含 logits + vision loss |
| **train_optim_step** | 同上 | 同上（gradient accumulation 逻辑相同） | 结构一致，可直接复用 |
| **validation** | `_run_validation_epoch` | 可能复用（取决于验证集格式） | 若验证集为纯文本，可直接复用 |
| **训练循环** | `run_train_validation_loop` | 可复用（混合方案下 Callback 处理 checkpoint/验证/日志） | 混合方案下训练循环骨架统一 |

**setup() 的核心差异**：
- VLM 需要构建 `processor`（处理图像/文本的 tokenizer）
- VLM 使用 `vlm_dataloader` 替代标准 `build_dataloader`，支持多模态 packed sequence（NEAT packing）
- VLM 需要处理 `vision tower` 的冻结（`freeze_vit` 参数）
- VLM 的 `model_parts` 可能返回 `model.parts`（PP 多 stage）或 `[model]`

**forward_backward_step 的核心差异**：
- 前向输入包含 `pixel_values`、`image_grid_thw` 等多模态字段
- Loss 计算可能涉及 vision tower 的梯度
- 需要处理 `mRoPE position_ids`（3D position_ids）

### 12.3 变体扩展原则

新增 Recipe 变体时，只需遵循以下原则：

1. **继承 BaseRecipe**，自动获得 `__state_tracked` + checkpoint + callback_manager
2. **按需覆盖** `setup()` / `run_train_validation_loop()` / `_forward_backward_step()` / `_run_train_optim_step()` / `_run_validation_epoch()` 中的部分方法
3. **不修改** `BaseRecipe` 的 `__state_tracked` 机制和 `CallbackManager` 的调用链
4. **Callback 不变**——所有内置 callback 在 `BaseRecipe` 层注册，子类只需确保 `run_train_validation_loop()` 中调用了 `callback_manager.on_step_end(state)`

---

## 13. 配置示例

> `recipe: FinetuneRecipe` 是字符串形式的 Recipe 类名，由 `main()` 中的
> Recipe 注册/导入机制解析为实际类。机制：维护 `RECIPE_REGISTRY` dict
> （`{"FinetuneRecipe": FinetuneRecipe, ...}`），`main()` 通过
> `cfg.get("recipe")` 取字符串 → `RECIPE_REGISTRY[name]` 查表；
> 未命中时尝试 `importlib.import_module` 动态导入。若 YAML 未设置 recipe，
> 默认使用 `FinetuneRecipe`。见 01_hf_compatibility_layer.md §4。

```yaml
recipe: FinetuneRecipe

model:
  _target_: hyper_models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16

distributed:
  tp: 4
  sequence_parallel: true

seed: 42

# ── typed: optimizer（_target_ → OptimizerConfig → .build(model)） ──
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.01

# ── typed: lr_scheduler（无 _target_，固定类型 → .build(optimizer, step_scheduler)） ──
lr_scheduler:
  lr_warmup_steps: 100              # 绝对步数，非比例
  lr_decay_style: cosine
  min_lr: 1.0e-6

# ── typed: loss_fn（_target_ → LossConfig → .build()） ──
loss_fn:
  _target_: hyper_models.components.loss.masked_ce.MaskedCrossEntropy

# WandB（可选，启用远程日志记录到 Weights & Biases）
# 注意：此键由 build_callback_manager() 中的 cfg.get("wandb") 读取
wandb:
  enabled: true
  project: my-training-project
  entity: my-team

# ── typed: step_scheduler（无 _target_，固定类型 → .build(dataloader, dp_size, local_bs)） ──
step_scheduler:
  ckpt_every_steps: 500
  val_every_steps: 500
  max_steps: 1000
  global_batch_size: 32

# ── typed: checkpoint（无 _target_，固定类型 → .build(dp_rank, tp_rank, ...)） ──
checkpoint:
  checkpoint_dir: outputs/qwen35_08b
  model_save_format: safetensors
  save_consolidated: final
  is_async: true
  restore_from: LATEST

# ── untyped: dataset（.instantiate() 直接调用） ──
dataset:
  _target_: datasets.load_dataset
  path: HuggingFaceFW/fineweb
  name: sample-10BT
  split: train
  streaming: true
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B

# ── untyped: dataloader（.instantiate() 直接调用） ──
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 2
  pin_memory: true

packed_sequence:
  packed_sequence_size: 8192
  packing_strategy: thd
```

---

## 14. 组件清单与分期规划

### 14.1 Optimizer 配置类

| 类 | 一期 | 说明 |
|------|:--:|------|
| `AdamWConfig` | ✓ | torch.optim.AdamW |
| `OptimizerFromFactoryConfig` | ✓ | 外部优化器 escape hatch（YAML `_target_` 直接路由） |
| `FusedAdamConfig` | 二期 | NVIDIA apex FusedAdam |
| `FlashAdamWConfig` | 二期 | torchao FlashAdamW |
| `MuonConfig` | 二期 | dion.Muon |
| `NorMuonConfig` | 二期 | dion.NorMuon |
| `Dion2Config` | 二期 | dion.Dion2 |
| `DionConfig` | 二期 | dion.Dion |

### 14.2 Loss 类

| 类 | 一期 | 说明 |
|------|:--:|------|
| `MaskedCrossEntropy` | ✓ | 标准 CE loss |
| `FusedLinearCrossEntropy` | ✓ | 融合 lm_head + CE |
| `KDLoss` | 二期 | 知识蒸馏 loss |
| `ChunkedCrossEntropy` | 二期 | 分块 CE（超长序列） |

### 14.3 HyperAutoModel 变体

| 类 | 一期 | 说明 |
|------|:--:|------|
| `HyperAutoModelForCausalLM` | ✓ | LLM 自回归 |
| `HyperAutoModelForImageTextToText` | ✓ | VLM 多模态 |
| `HyperAutoModelForSequenceClassification` | ✓ | 序列分类 |
| `HyperAutoModelForMultimodalLM` | 二期 | 多模态语言模型 |
| `HyperAutoModelForTokenClassification` | 二期 | Token 分类 |
| `HyperAutoModelForTextToWaveform` | 二期 | 文本转波形 |
| `HyperAutoModelBiEncoder` | 二期 | 双编码器 |
| `HyperAutoModelCrossEncoder` | 二期 | 交叉编码器 |
