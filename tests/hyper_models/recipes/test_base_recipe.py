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
"""Tests for BaseRecipe — state tracking + checkpoint save/load (03 §3 / 04 §8)."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from hyper_models.recipes.base_recipe import (
    BaseRecipe,
    _maybe_load_latest_marker,
    _resolve_latest_symlink,
)


def _make_recipe():
    recipe = BaseRecipe()
    recipe.checkpointer = MagicMock()
    return recipe


# ── 状态注册 ──

def test_init_state_tracked_empty():
    assert BaseRecipe()._state_tracked == []


def test_register_state():
    recipe = BaseRecipe()
    recipe.register_state("model", "model")
    assert recipe._state_tracked == [("model", "model")]


def test_register_state_duplicate():
    recipe = BaseRecipe()
    recipe.register_state("model", "model")
    recipe.register_state("model", "model")
    assert recipe._state_tracked == [("model", "model")]  # 防重复


# ── rank / group size 查询（无 mesh 兜底） ──

def test_get_dp_rank_no_mesh():
    assert BaseRecipe()._get_dp_rank() == 0


def test_get_dp_group_size_no_mesh():
    assert BaseRecipe()._get_dp_group_size() == 1


def test_get_tp_rank_no_mesh():
    assert BaseRecipe()._get_tp_rank() == 0


def test_get_pp_rank_no_mesh():
    assert BaseRecipe()._get_pp_rank() == 0


def test_rank_helpers_delegate_to_mesh():
    recipe = BaseRecipe()
    recipe.mesh = SimpleNamespace(dp_rank=2, tp_rank=1, pp_rank=0, dp_size=4)
    assert recipe._get_dp_rank() == 2
    assert recipe._get_tp_rank() == 1
    assert recipe._get_pp_rank() == 0
    assert recipe._get_dp_group_size() == 4


# ── save_checkpoint ──

def test_save_checkpoint_model(tmp_path):
    recipe = _make_recipe()
    recipe.model = nn.Linear(2, 2)
    recipe.register_state("model", "model")
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    recipe.checkpointer.save_model.assert_called_once()
    args = recipe.checkpointer.save_model.call_args[0]
    assert args[0] is recipe.model
    assert args[1].endswith("/model")


def test_save_checkpoint_optimizer(tmp_path):
    recipe = _make_recipe()
    recipe.model = nn.Linear(2, 2)
    recipe.optimizer = [MagicMock()]
    recipe.register_state("model", "model")
    recipe.register_state("optimizer", "optimizer")
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    recipe.checkpointer.save_optimizer.assert_called_once()
    args = recipe.checkpointer.save_optimizer.call_args[0]
    assert args[0] is recipe.model  # model_ref
    assert args[1] is recipe.optimizer  # list 原样传递（不拆包）


def test_save_checkpoint_lr_scheduler(tmp_path):
    recipe = _make_recipe()
    sch = MagicMock()
    sch.state_dict.return_value = {"last_epoch": 3}
    recipe.lr_scheduler = [sch]
    recipe.register_state("lr_scheduler", "lr_scheduler")
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    sch_path = tmp_path / "epoch_0_step_1" / "scheduler.pt"
    assert sch_path.exists()
    state = torch.load(sch_path, weights_only=False)
    assert state == {"sch_0": {"last_epoch": 3}}


def test_save_checkpoint_train_state(tmp_path):
    recipe = _make_recipe()
    recipe.step_scheduler = MagicMock()
    recipe.step_scheduler.state_dict.return_value = {"step": 1, "epoch": 0}
    recipe.register_state("step_scheduler", "train_state")
    recipe.save_checkpoint(
        str(tmp_path), epoch=0, step=1, train_loss=0.5,
        val_losses={"validation": {"loss": 0.2}},
    )
    extra_path = tmp_path / "epoch_0_step_1" / "extra_state.json"
    assert extra_path.exists()
    extra = json.loads(extra_path.read_text())
    # 显式键居后：epoch/global_step 覆盖 state_dict 展开值
    assert extra["epoch"] == 0
    assert extra["global_step"] == 1
    assert extra["train_loss"] == 0.5
    assert extra["val_losses"] == {"validation": {"loss": 0.2}}


def test_save_checkpoint_rng_and_dataloader_paths(tmp_path):
    recipe = _make_recipe()
    recipe.rng = MagicMock()
    recipe.rng.state_dict.return_value = {"seed": 42}
    recipe.dataloader = MagicMock()
    recipe.dataloader.state_dict.return_value = {"epoch": 0}
    recipe.register_state("rng", "rng")
    recipe.register_state("dataloader", "dataloader")
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    step_dir = tmp_path / "epoch_0_step_1"
    assert (step_dir / "rng" / "rng_dp_rank_0.pt").exists()
    assert (step_dir / "dataloader" / "dataloader_dp_rank_0.pt").exists()


def test_save_checkpoint_missing_attr(tmp_path):
    recipe = _make_recipe()
    recipe.register_state("ghost", "model")  # 注册但未赋值
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    recipe.checkpointer.save_model.assert_not_called()  # 跳过不报错


def test_save_checkpoint_updates_latest_symlink(tmp_path):
    recipe = _make_recipe()
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)
    latest = tmp_path / "LATEST"
    assert latest.is_symlink()
    assert os.readlink(latest) == "epoch_0_step_1"


# ── load_checkpoint ──

def test_load_checkpoint_none():
    recipe = _make_recipe()
    recipe.load_checkpoint(None)  # 跳过恢复，不抛异常


def test_load_checkpoint_latest_not_found(tmp_path):
    recipe = _make_recipe()
    recipe.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    recipe.load_checkpoint("LATEST")  # 无 LATEST → from scratch，不抛异常


def test_load_checkpoint_missing_path(tmp_path):
    recipe = _make_recipe()
    recipe.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    recipe.load_checkpoint(str(tmp_path / "nonexistent"))  # warning + 返回


def test_load_checkpoint_restores_train_state(tmp_path):
    # 先 save 再 load，验证 save/load 对称
    recipe = _make_recipe()
    recipe.step_scheduler = MagicMock()
    recipe.step_scheduler.state_dict.return_value = {"step": 1, "epoch": 0}
    recipe.register_state("step_scheduler", "train_state")
    recipe.save_checkpoint(str(tmp_path), epoch=0, step=1, train_loss=0.5)

    recipe2 = _make_recipe()
    recipe2.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    recipe2.step_scheduler = MagicMock()
    recipe2.register_state("step_scheduler", "train_state")
    recipe2.load_checkpoint("LATEST")
    loaded = recipe2.step_scheduler.load_state_dict.call_args[0][0]
    assert loaded["step"] == 1
    assert loaded["global_step"] == 1


# ── _state_path ──

def test_state_path_kinds():
    recipe = BaseRecipe()
    assert recipe._state_path("/root", "model", "model") == "/root/model"
    assert recipe._state_path("/root", "optimizer", "optimizer") == "/root/optimizer"
    assert recipe._state_path("/root", "rng", "rng") == "/root/rng/rng_dp_rank_0.pt"
    assert recipe._state_path("/root", "dataloader", "dataloader") == \
        "/root/dataloader/dataloader_dp_rank_0.pt"
    assert recipe._state_path("/root", "step_scheduler", "train_state") == \
        "/root/extra_state.json"
    assert recipe._state_path("/root", "lr_scheduler", "lr_scheduler") == \
        "/root/scheduler.pt"
    assert recipe._state_path("/root", "custom", "other") == "/root/custom.pt"


# ── LATEST 解析辅助 ──

def test_resolve_latest_symlink(tmp_path):
    step_dir = tmp_path / "epoch_0_step_1"
    step_dir.mkdir()
    os.symlink("epoch_0_step_1", tmp_path / "LATEST")
    assert _resolve_latest_symlink(str(tmp_path)) == str(step_dir)


def test_resolve_latest_symlink_missing(tmp_path):
    assert _resolve_latest_symlink(str(tmp_path)) is None


def test_maybe_load_latest_marker(tmp_path):
    step_dir = tmp_path / "epoch_0_step_1"
    step_dir.mkdir()
    (tmp_path / "LATEST").write_text("epoch_0_step_1\n")
    assert _maybe_load_latest_marker(str(tmp_path)) == str(step_dir)


# ── 外围关注点辅助 ──

def test_log_val_metrics():
    recipe = BaseRecipe()
    with patch("hyper_models.recipes.base_recipe._is_rank_0", return_value=True), \
         patch("hyper_models.recipes.base_recipe.logger") as mock_logger:
        recipe.log_val_metrics({"validation": {"loss": 0.1234, "num_tokens": 8}})
        mock_logger.info.assert_called_once()


def test_log_val_metrics_non_rank0():
    recipe = BaseRecipe()
    with patch("hyper_models.recipes.base_recipe._is_rank_0", return_value=False), \
         patch("hyper_models.recipes.base_recipe.logger") as mock_logger:
        recipe.log_val_metrics({"validation": {"loss": 0.1}})
        mock_logger.info.assert_not_called()


def test_maybe_collect_garbage():
    recipe = BaseRecipe()
    with patch("gc.collect") as mock_gc, \
         patch("torch.cuda.is_available", return_value=False):
        recipe._maybe_collect_garbage()
        mock_gc.assert_called_once()


def test_is_rank_0_helper():
    from hyper_models.components.distributed.infrastructure import _is_rank_0
    assert not torch.distributed.is_initialized()
    assert _is_rank_0() is True
