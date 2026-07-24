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
"""StepScheduler — 训练节奏控制：梯度累积、checkpoint/validation 步判断、SIGTERM 响应。

替代旧的 _make_micro_batch_iterator + callback 判断逻辑。
"""

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from hyper_models.components.training.signal_handler import DistributedSignalHandler


@dataclass
class StepSchedulerConfig:
    """StepScheduler typed config —— TrainerConfig.step_scheduler 的返回类型。

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
    # local_batch_size 不参与 StepScheduler 构造（由 build() 的调用方注入），
    # 但作为 YAML 配置键保留于此：Recipe.setup() ⑬⑭⑮ 读取
    # cfg.step_scheduler.local_batch_size 传给 build_dataloader 与 build()，
    # 保证 dataloader 批大小与 grad_acc_steps 计算同源（03 §5.3）。
    local_batch_size: int = 1
    # global_batch_size 为正式字段。
    # None 时退化为 local_batch_size * dp_world_size（即 grad_acc_steps=1）。
    # 注意：本字段与 build_dataloader 的 global_batch_size 读取的是同一 YAML 键（同源），
    # grad_acc_steps 计算与 dataloader 的 global_batch_size 不会出现口径分叉。
    global_batch_size: int | None = None

    def build(
        self,
        dataloader: DataLoader,
        dp_world_size: int,
        local_batch_size: int,
        start_step: int = 0,
        start_epoch: int = 0,
    ) -> "StepScheduler":
        """构建 StepScheduler 实例。

        Args:
            dataloader: 训练数据加载器。
            dp_world_size: 数据并行 world size。
            local_batch_size: 每设备每 micro-batch 的样本数。
            start_step: 断点续训起始步（默认 0）。
            start_epoch: 断点续训起始 epoch（默认 0）。

        Returns:
            配置好的 StepScheduler 实例。
        """
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
        max_steps: int = -1,
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
                f"local_batch_size * dp_world_size "
                f"({local_batch_size * dp_world_size})"
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
        is_final_step 跳过，避免同一步重复保存。
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