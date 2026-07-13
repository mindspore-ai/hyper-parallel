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
"""Distributed driver for BaseTrainer checkpoint round trip."""
import os
import shutil
import types
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel import (
    destroy_process_group,
    get_platform,
    init_process_group,
)
from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, TrainerState

platform = get_platform()

_SAVE_ROOT = "/tmp/hp_pr649_ckpt_st"


def _local_rank() -> int:
    """Return local device index for this torchrun worker."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return platform.get_rank() % int(os.environ.get("LOCAL_WORLD_SIZE", "8"))


def _set_local_device() -> None:
    """Bind the current worker to its local NPU before any collective."""
    platform.get_device_handle(platform.device_type()).set_device(_local_rank())


class _RecordingDataloader:
    """Stateful dataloader stand-in (``state_dict`` / ``load_state_dict``)."""

    def __init__(self, position: int = 0) -> None:
        """Seed the dataloader at ``position`` (0 by default)."""
        self.position = position

    def state_dict(self) -> dict:
        """Return a serialisable view of the current position."""
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        """Restore ``position`` from the dict produced by ``state_dict``."""
        self.position = state["position"]


def _build_trainer(save_dir: str, *, load_path=None):
    """Wire the BaseTrainer fields the checkpoint lifecycle reads."""
    ckpt_cfg = types.SimpleNamespace(
        output_dir=save_dir,
        save_steps=1,
        save_async=False,
        load_path=load_path,
        save_hf_weights=False,
    )
    _set_local_device()
    device = platform.device(_local_rank())
    model = nn.Linear(4, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    dataloader = _RecordingDataloader(position=42)
    args = types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        checkpoint=ckpt_cfg,
        train=types.SimpleNamespace(
            max_steps=20,
            num_train_epochs=1,
            checkpoint=ckpt_cfg,
        ),
    )
    with patch.object(trainer_base, "get_spec", return_value=types.SimpleNamespace(state_dict_adapter=None)):
        trainer = BaseTrainer(args, setup=False)
    trainer.global_rank = int(platform.get_rank())
    trainer.local_rank = int(_local_rank())
    trainer.world_size = int(platform.get_world_size())
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.train_dataloader = dataloader
    trainer.on_save = lambda *_a, **_kw: None
    trainer.on_resume = lambda *_a, **_kw: None
    return trainer, model, dataloader


def _prepare_root():
    """Rank-0 cleans + recreates the shared save root, then barrier across ranks."""
    if platform.get_rank() == 0:
        shutil.rmtree(_SAVE_ROOT, ignore_errors=True)
        os.makedirs(_SAVE_ROOT, exist_ok=True)
    platform.barrier()


def test_checkpoint_lifecycle_round_trip_4card():
    """
    Feature: BaseTrainer checkpoint save → load round-trip on real DCP + torch.save.
    Description: Build a plain ``nn.Linear`` on every rank, mutate the
        weight to a distinctive value, invoke ``BaseTrainer._save_checkpoint``
        at ``global_step=5 / epoch=1``. Then mutate the weight to zero, build
        a fresh checkpoint load path and call ``_resume_from_checkpoint``.
        Exercises ``dcp_save`` /
        ``dcp_load`` for the model state-dict plus the five other artifact
        buckets via real ``torch.save`` / ``torch.load`` on a 4-card PG.
    Expectation: After load, every rank sees ``model.weight ≈ 0.42``,
        ``state.global_step == 5``, ``state.epoch == 1``, and the
        dataloader position is restored from the saved snapshot.
    """
    init_process_group()
    _set_local_device()
    try:
        _prepare_root()
        rank = platform.get_rank()
        trainer, model, dataloader = _build_trainer(_SAVE_ROOT)

        with torch.no_grad():
            model.weight.fill_(0.42)
        saved_weight = model.weight.detach().clone()  # pylint: disable=not-callable
        dataloader.position = 7 * (rank + 1)
        saved_position = dataloader.position
        state = TrainerState(max_steps=20)
        state.global_step = 5
        state.epoch = 1
        trainer._save_checkpoint(state)  # pylint: disable=protected-access

        # Cross-rank barrier so every rank finishes its write before any
        # rank tries to read (otherwise rank N can race ahead and find an
        # incomplete dir).
        platform.barrier()

        with torch.no_grad():
            model.weight.fill_(0.0)
        dataloader.position = 0

        save_dir = os.path.join(_SAVE_ROOT, "step_5")
        assert os.path.isdir(save_dir), f"save_dir missing: {save_dir}"
        trainer.args.checkpoint.load_path = save_dir
        trainer.state = TrainerState(max_steps=20)
        trainer._resume_from_checkpoint(save_dir)  # pylint: disable=protected-access

        assert trainer.state.global_step == 5, (
            f"global_step not restored: expected 5, got {trainer.state.global_step}"
        )
        assert trainer.state.epoch == 1, (
            f"epoch not restored: expected 1, got {trainer.state.epoch}"
        )
        assert torch.allclose(model.weight.cpu(), saved_weight.cpu(), atol=1e-6), (
            f"rank={rank}: model weight not restored, "
            f"diff_sum={(model.weight - saved_weight).abs().sum().item()}"
        )
        assert dataloader.position == saved_position, (
            f"rank={rank}: dataloader position not restored, "
            f"expected {saved_position}, got {dataloader.position}"
        )
    finally:
        # Rank 0 cleans up the shared dir so re-runs start fresh.
        try:
            platform.barrier()
        except RuntimeError:
            pass
        if platform.get_rank() == 0:
            shutil.rmtree(_SAVE_ROOT, ignore_errors=True)
        destroy_process_group()
