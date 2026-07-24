# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""BaseRecipe — 自动状态追踪 + checkpoint save/load。

Following design doc 03_training_loop.md §3（状态追踪 / save_checkpoint）
与 04_checkpoint.md §8（load_checkpoint canonical）/ §7.2（辅助函数）。
"""

import json
import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyper_models.components.distributed.infrastructure import _is_rank_0
from hyper_models.components.training.callback import CallbackManager
from hyper_models.components.training.grad_accum import _update_latest_symlink

logger = logging.getLogger(__name__)


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
        # （实际存储为 _BaseRecipe__state_tracked）。子类应始终使用
        # register_state() 方法，不要直接操作 self.__state_tracked。
        self.__state_tracked: list[tuple[str, str]] = []
        # 反查表，避免重复注册同名状态
        self.__state_names: set[str] = set()
        # Callback 管理器（延迟初始化，在 setup() 中由 build_callback_manager 赋值）
        self.callback_manager: Optional[CallbackManager] = None

    def register_state(self, name: str, kind: str) -> None:
        """显式注册一个需要 checkpoint 追踪的组件。

        name: Recipe 上的属性名（如 "model" / "optimizer"）。
        kind: 04 `_state_path` 所用的 state kind，取值：
              "model" / "optimizer" / "lr_scheduler" / "rng" / "dataloader"
              / "train_state"。同名重复注册将被忽略。
        """
        if name in self.__state_names:
            return
        self.__state_tracked.append((name, kind))
        self.__state_names.add(name)

    @property
    def _state_tracked(self) -> list[tuple[str, str]]:
        """只读视图（测试/子类检查用；请勿直接修改）。"""
        return list(self.__state_tracked)

    # ── rank / group size 查询 ──
    # 委托给 self.mesh（MeshContext，见 06 §2）；mesh 在 setup 中由
    # self.distributed_setup.mesh_context 赋值。

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

    # ── 外围关注点辅助方法（Callback 调用） ──

    def log_val_metrics(self, val_losses: dict) -> None:
        """记录验证指标（仅 rank 0 输出，EvaluateCallback 在 is_val_step 调用）。

        val_losses 形如 {"validation": {"loss": float, "num_tokens": int}}
        （_run_validation_epoch 的返回结构，见 03 §6.1）。
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
    # save 侧整体委托 04 Checkpointer 走 `_state_path` per-rank 子目录规则，
    # 保证 save/load 同源。本方法仅负责遍历 `__state_tracked` 分发。

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        step: int,
        train_loss: float,
        val_losses: Optional[dict] = None,
        is_final_checkpoint: bool = False,
    ) -> None:
        """遍历 __state_tracked，按 kind 委托 Checkpointer 保存。

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
                # 【canonical】self.optimizer 为 list[Optimizer]，原样（不拆包）
                # 传给 checkpointer；04 OptimizerState 接受 list[Optimizer]。
                self.checkpointer.save_optimizer(model_ref, obj, f"{path}/optimizer")
            elif kind == "lr_scheduler":
                # lr_scheduler build 返回 list（03 §9.6），逐个保存 state_dict
                # 到 scheduler.pt（聚合 dict）。
                schedulers = obj if isinstance(obj, list) else [obj]
                torch.save(
                    {f"sch_{i}": s.state_dict() for i, s in enumerate(schedulers)},
                    f"{path}/scheduler.pt",
                )
            elif kind == "rng":
                rng_path = self._state_path(path, name, kind)
                os.makedirs(os.path.dirname(rng_path), exist_ok=True)
                torch.save(obj.state_dict(), rng_path)
            elif kind == "dataloader":
                dl_path = self._state_path(path, name, kind)
                os.makedirs(os.path.dirname(dl_path), exist_ok=True)
                torch.save(obj.state_dict(), dl_path)
            elif kind == "train_state":
                # 训练元信息（epoch/step/loss）——与 04 load `kind=="train_state"`
                # 分支对称。state_dict 先展开，显式键居后，确保 epoch/global_step
                # 不被 state_dict 覆盖；train_loss/val_losses 一并落盘。
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

        # 更新 LATEST symlink（_update_latest_symlink 为模块级函数，见 03 §7.1）
        _update_latest_symlink(checkpoint_dir, path)

    # ── Checkpoint load（canonical：04 §8） ──

    def load_checkpoint(self, restore_from: Optional[str]) -> None:
        """从 checkpoint 恢复所有组件（04 §8 canonical 流程）。

        restore_from 解析：
        - None → 跳过恢复
        - "LATEST" → 读取 LATEST symlink
        - "epoch_0_step_100" → 直接使用路径
        """
        if restore_from is None:
            return

        if restore_from == "LATEST":
            restore_from = _resolve_latest_symlink(self.checkpoint_config.checkpoint_dir)
            if restore_from is None:
                logger.info("No LATEST checkpoint found, starting from scratch.")
                return

        if not os.path.exists(restore_from):
            logger.warning("Checkpoint %s not found, starting from scratch.", restore_from)
            return

        logger.info("Loading checkpoint from %s", restore_from)

        # ① 兼容性检查
        self._validate_checkpoint_compatibility(restore_from)

        # ② 遍历 __state_tracked 注册表，按 (name, kind) 对称解包加载
        for name, kind in sorted(self.__state_tracked):
            path = self._state_path(restore_from, name, kind)
            if not os.path.exists(path):
                continue
            self._load_state_by_kind(name, kind, path)

    def _load_state_by_kind(self, name: str, kind: str, path: str) -> None:
        """按 state 种类分发加载（04 §8）。"""
        obj = getattr(self, name)
        if kind == "model":
            # PP 多 stage：传 model_parts 列表
            self.checkpointer.load_model(
                getattr(self, "model_parts", obj), model_path=path
            )
        elif kind == "optimizer":
            self.checkpointer.load_optimizer(
                getattr(self, "model_parts", None), obj, path
            )
        elif kind == "lr_scheduler":
            # 与 save 侧 {f"sch_{i}": state_dict} 聚合对称
            state = torch.load(path, weights_only=False)
            schedulers = obj if isinstance(obj, list) else [obj]
            for i, sch in enumerate(schedulers):
                sch.load_state_dict(state[f"sch_{i}"])
        elif kind == "train_state":
            with open(path) as f:
                extra = json.load(f)
            obj.load_state_dict(extra)
        elif kind in ("rng", "dataloader"):
            state = torch.load(path, weights_only=False)
            obj.load_state_dict(state)

    def _state_path(self, root: str, name: str, kind: str) -> str:
        """根据 state 种类和 name 计算 checkpoint 子路径（04 §8）。"""
        if kind in ("model", "optimizer"):
            return f"{root}/{kind}"
        if kind in ("rng", "dataloader"):
            return f"{root}/{kind}/{kind}_dp_rank_{self._get_dp_rank()}.pt"
        if kind == "train_state":
            return f"{root}/extra_state.json"
        if kind == "lr_scheduler":
            return f"{root}/scheduler.pt"
        return f"{root}/{name}.pt"

    def _validate_checkpoint_compatibility(self, restore_from: str) -> None:
        """校验 checkpoint 与当前运行环境兼容性（04 §7.2）。

        Stub — 完整实现需读取 extra_state.json / .dtensor_metadata.json 对比
        DP/TP/PP size 与 DTensor placements。
        """
        mesh = getattr(self, "mesh", None)
        if mesh is not None:
            logger.debug(
                "checkpoint compatibility check (stub): dp=%d tp=%d pp=%d, path=%s",
                mesh.dp_size, mesh.tp_size, mesh.pp_size, restore_from,
            )


# ── 模块级辅助（04 §7.2） ──

def _resolve_latest_symlink(checkpoint_dir: str) -> Optional[str]:
    """读取 LATEST symlink 指向的最新 checkpoint 目录，不存在则返回 None。

    03 §7.1 `_update_latest_symlink` 写入的是相对路径，消费端必须拼回
    checkpoint_dir 再判 exists——直接对 readlink 结果调 os.path.exists 会
    依赖 CWD。
    """
    symlink = os.path.join(checkpoint_dir, "LATEST")
    if os.path.islink(symlink):
        target = os.path.join(checkpoint_dir, os.readlink(symlink))
        if os.path.exists(target):
            return target
        return None
    return _maybe_load_latest_marker(checkpoint_dir)


def _maybe_load_latest_marker(checkpoint_dir: str) -> Optional[str]:
    """无 symlink 时尝试读取 LATEST marker 文件（兼容无符号链接的 FS）。"""
    marker = os.path.join(checkpoint_dir, "LATEST")
    if os.path.isfile(marker) and not os.path.islink(marker):
        with open(marker) as f:
            lines = f.read().strip().splitlines()
            if lines:
                return os.path.join(checkpoint_dir, lines[-1])
    return None


def _is_stateful(obj: Any) -> bool:
    """判断对象是否需要 checkpoint 追踪（仅用于 setup 期辅助判断）。"""
    return isinstance(obj, (
        nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler,
        DataLoader,
    )) or hasattr(obj, "state_dict")
