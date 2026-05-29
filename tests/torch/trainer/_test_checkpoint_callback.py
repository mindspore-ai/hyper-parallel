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
"""Distributed driver for ``CheckpointCallback`` real-DCP round trip."""
import os
import shutil
import types

import torch
from torch import nn

from hyper_parallel import (
    destroy_process_group,
    get_platform,
    init_process_group,
)
from hyper_parallel.trainer.base import TrainerState
from hyper_parallel.trainer.callbacks.base import CheckpointCallback

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
    """Wire the minimal trainer-shaped namespace the callback reads from."""
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
    trainer = types.SimpleNamespace(
        args=types.SimpleNamespace(checkpoint=ckpt_cfg),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=dataloader,
        dispatch_save_event=lambda *_a, **_kw: None,
        dispatch_load_event=lambda *_a, **_kw: None,
        spec=None,
    )
    return trainer, model, dataloader


def _prepare_root():
    """Rank-0 cleans + recreates the shared save root, then barrier across ranks."""
    if platform.get_rank() == 0:
        shutil.rmtree(_SAVE_ROOT, ignore_errors=True)
        os.makedirs(_SAVE_ROOT, exist_ok=True)
    platform.barrier()


def test_checkpoint_callback_round_trip_4card():
    """
    Feature: ``CheckpointCallback`` save → load round-trip on real DCP + torch.save.
    Description: Build a plain ``nn.Linear`` on every rank, mutate the
        weight to a distinctive value, invoke ``CheckpointCallback._save``
        at ``global_step=5 / epoch=1``. Then mutate the weight to zero, build
        a fresh callback against the same checkpoint dir as ``load_path``,
        and let ``on_train_begin`` restore. Exercises ``dcp_save`` /
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
        cb = CheckpointCallback(trainer)

        with torch.no_grad():
            model.weight.fill_(0.42)
        saved_weight = model.weight.detach().clone()  # pylint: disable=not-callable
        dataloader.position = 7 * (rank + 1)
        saved_position = dataloader.position
        state = TrainerState(max_steps=20)
        state.global_step = 5
        state.epoch = 1
        cb._save(state)  # pylint: disable=protected-access

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
        cb2 = CheckpointCallback(trainer)
        restart_state = TrainerState(max_steps=20)
        cb2.on_train_begin(restart_state)

        assert restart_state.global_step == 5, (
            f"global_step not restored: expected 5, got {restart_state.global_step}"
        )
        assert restart_state.epoch == 1, (
            f"epoch not restored: expected 1, got {restart_state.epoch}"
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
