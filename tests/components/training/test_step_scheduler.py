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
"""Tests for StepScheduler — pure CPU logic (design doc 03 §4/§4.1)."""

from unittest.mock import MagicMock

import pytest

from hyper_models.components.training.step_scheduler import (
    StepScheduler,
    StepSchedulerConfig,
)


def _make_scheduler(dataloader=None, **kwargs):
    defaults = dict(
        dataloader=dataloader if dataloader is not None else [],
        global_batch_size=32,
        local_batch_size=2,
        dp_world_size=4,
        max_steps=-1,
    )
    defaults.update(kwargs)
    return StepScheduler(**defaults)


# ── grad_acc_steps ──

def test_grad_acc_steps_basic():
    sched = _make_scheduler(global_batch_size=32, local_batch_size=2, dp_world_size=4)
    assert sched.grad_acc_steps == 4  # 32 / (2*4)


def test_grad_acc_steps_unit():
    sched = _make_scheduler(global_batch_size=8, local_batch_size=8, dp_world_size=1)
    assert sched.grad_acc_steps == 1


def test_grad_acc_steps_divisible_check():
    with pytest.raises(ValueError, match="divisible"):
        _make_scheduler(global_batch_size=32, local_batch_size=3, dp_world_size=4)


def test_grad_acc_steps_max_1():
    # global=0 可整除任意 local*dp → floor division 得 0 → max(1, 0) 下限为 1
    # （非零场景下不整除会被 divisibility check 拒绝，见 test_grad_acc_steps_divisible_check）
    sched = _make_scheduler(global_batch_size=0, local_batch_size=4, dp_world_size=1)
    assert sched.grad_acc_steps == 1


# ── 迭代行为 ──

def test_iter_yields_groups():
    sched = _make_scheduler(
        dataloader=list(range(10)),
        global_batch_size=3, local_batch_size=1, dp_world_size=1,
    )
    groups = list(sched)
    assert [len(g) for g in groups] == [3, 3, 3, 1]


def test_iter_step_increments():
    sched = _make_scheduler(
        dataloader=list(range(10)),
        global_batch_size=3, local_batch_size=1, dp_world_size=1,
    )
    list(sched)
    # 4 组（含余量 1），step 在每次 yield 前自增（设计文档 §4：余量也自增）
    assert sched.step == 4


def test_iter_step_start():
    sched = _make_scheduler(
        dataloader=list(range(4)),
        global_batch_size=2, local_batch_size=1, dp_world_size=1,
        start_step=5,
    )
    assert sched.step == 5
    groups = list(sched)
    assert len(groups) == 2
    assert sched.step == 7


def test_max_steps_reached():
    sched = _make_scheduler(
        dataloader=list(range(6)),
        global_batch_size=3, local_batch_size=1, dp_world_size=1,
        max_steps=2,
    )
    groups = list(sched)
    assert len(groups) == 2
    assert sched._max_steps_reached


def test_max_steps_negative():
    sched = _make_scheduler(
        dataloader=list(range(4)),
        global_batch_size=4, local_batch_size=1, dp_world_size=1,
        max_steps=-1,
    )
    assert not sched._max_steps_reached
    groups = list(sched)
    assert len(groups) == 1  # epoch 驱动，不按步数限制


# ── is_ckpt_step ──

def test_is_ckpt_step_periodic():
    sched = _make_scheduler(ckpt_every_steps=500)
    sched.step = 500
    assert sched.is_ckpt_step
    sched.step = 499
    assert not sched.is_ckpt_step


def test_is_ckpt_step_max_steps():
    sched = _make_scheduler(max_steps=1000)
    sched.step = 1000
    assert sched.is_ckpt_step  # 最终步标记


def test_is_ckpt_step_epoch():
    sched = _make_scheduler(save_checkpoint_every_epoch=True, ckpt_every_steps=500)
    sched.step = 1
    assert sched.is_ckpt_step  # epoch 边界（未标记）
    sched.mark_epoch_ckpt_saved()
    assert not sched.is_ckpt_step  # 标记后不再触发


# ── is_val_step / is_log_step / is_gc_step ──

def test_is_val_step_default():
    sched = _make_scheduler(val_every_steps=None, ckpt_every_steps=500)
    sched.step = 500
    assert sched.is_val_step == sched.is_ckpt_step
    sched.step = 499
    assert sched.is_val_step == sched.is_ckpt_step


def test_is_val_step_custom():
    sched = _make_scheduler(val_every_steps=200)
    sched.step = 200
    assert sched.is_val_step
    sched.step = 199
    assert not sched.is_val_step


def test_is_log_step():
    sched = _make_scheduler(log_remote_every_steps=10)
    sched.step = 10
    assert sched.is_log_step
    sched.step = 9
    assert not sched.is_log_step


def test_is_gc_step():
    sched = _make_scheduler(gc_every_steps=100)
    sched.step = 100
    assert sched.is_gc_step
    sched.step = 99
    assert not sched.is_gc_step
    sched_no_gc = _make_scheduler(gc_every_steps=None)
    sched_no_gc.step = 100
    assert not sched_no_gc.is_gc_step


# ── 状态序列化 ──

def test_state_dict():
    sched = _make_scheduler()
    sched.step = 7
    sched.epoch = 2
    assert sched.state_dict() == {"step": 7, "epoch": 2}


def test_load_state_dict():
    sched = _make_scheduler()
    sched.load_state_dict({"step": 5, "epoch": 1})
    assert sched.step == 5
    assert sched.epoch == 1
    assert sched.start_epoch == 1


def test_load_state_dict_legacy_keys():
    sched = _make_scheduler()
    sched.load_state_dict({"global_step": 5, "current_epoch": 1})
    assert sched.step == 5
    assert sched.epoch == 1
    assert sched.start_epoch == 1


# ── epochs ──

def test_epochs_property():
    sched = _make_scheduler(num_train_epochs=3, start_epoch=1)
    assert list(sched.epochs) == [1, 2]


def test_epochs_stop_on_max_steps():
    sched = _make_scheduler(num_train_epochs=3, max_steps=1)
    sched.step = 1
    assert list(sched.epochs) == [0]  # 第 1 个 epoch 内达到 max_steps → 停止


def test_set_epoch():
    dataloader = MagicMock()
    sched = _make_scheduler(dataloader=dataloader)
    sched.mark_epoch_ckpt_saved()
    sched.set_epoch(3)
    dataloader.sampler.set_epoch.assert_called_once_with(3)
    assert not sched._epoch_ckpt_saved  # 重置标记


def test_mark_epoch_ckpt_saved():
    sched = _make_scheduler(save_checkpoint_every_epoch=True, ckpt_every_steps=500)
    sched.step = 1
    assert sched.is_ckpt_step
    sched.mark_epoch_ckpt_saved()
    assert not sched.is_ckpt_step


def test_iter_stop_on_sigterm():
    sched = _make_scheduler(
        dataloader=list(range(10)),
        global_batch_size=3, local_batch_size=1, dp_world_size=1,
    )
    sched._sigterm_flag = True  # 模拟已收到 SIGTERM（跳过 all_gather）
    groups = list(sched)
    assert len(groups) == 1  # 首组 yield 后即停止，余量也不再 yield


# ── StepSchedulerConfig.build ──

def test_config_build_injects_runtime_deps():
    cfg = StepSchedulerConfig(max_steps=100, global_batch_size=None)
    dataloader = list(range(8))
    sched = cfg.build(dataloader, dp_world_size=2, local_batch_size=4)
    # global_batch_size=None → 退化为 local*dp = 8 → grad_acc=1
    assert sched.grad_acc_steps == 1
    assert sched.max_steps == 100
    assert sched.dataloader is dataloader


def test_config_build_explicit_global_batch_size():
    cfg = StepSchedulerConfig(global_batch_size=16)
    sched = cfg.build(list(range(8)), dp_world_size=2, local_batch_size=2)
    assert sched.grad_acc_steps == 4  # 16 / (2*2)
