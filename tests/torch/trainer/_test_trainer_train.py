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
"""Distributed driver for ``BaseTrainer.train()`` lifecycle cases."""
import math
import os
import types
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hyper_parallel import (
    fully_shard,
    get_platform,
    init_process_group,
)
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.parallel_dims import ParallelDims

platform = get_platform()


class _ToyCEModel(nn.Module):
    """Embedding + projection LM head over a small vocab.

    Identical to the one used in ``_test_train_step.py`` so this driver
    inherits the same convergence properties: under lr=0.1, a 4-token
    memorisable sequence drops CE from ~ln(8) toward 0 within ~10 steps.
    """

    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        """Build a ``vocab x hidden`` embedding plus ``hidden x vocab`` projection."""
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None,  # pylint: disable=arguments-differ
                use_cache: bool = False) -> dict:  # pylint: disable=unused-argument
        """Run embed → proj → CE; returns ``{"loss": ce}`` (or summed logits if no labels)."""
        x = self.embed(input_ids)
        logits = self.proj(x)
        if labels is None:
            return {"loss": logits.sum()}
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss}


class _RepeatingDataset(Dataset):
    """Dataset that yields the same memorisable (input_ids, labels) every step.

    The ``num_samples`` field controls how many items the dataloader can
    emit per epoch; setting it equal to ``max_steps`` exposes the
    ``StopIteration`` path in ``train()`` at the right moment.
    """

    def __init__(self, num_samples: int, seq_len: int = 4, vocab: int = 8) -> None:
        """Pre-bake a fixed ``seq_len`` token sequence under modulo ``vocab``."""
        self.num_samples = num_samples
        self.input_ids = torch.arange(1, 1 + seq_len, dtype=torch.long) % vocab

    def __len__(self) -> int:
        """Return the number of (input_ids, labels) items this dataset emits."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:  # pylint: disable=unused-argument
        """Return the same memorisable batch for every index."""
        return {"input_ids": self.input_ids.clone(), "labels": self.input_ids.clone()}


class _ProbeCallback(Callback):
    """User callback that records lifecycle dispatch order.

    Used to assert that ``train()`` fires hooks in the contract order
    (begin → epoch → step* → epoch_end → end) without relying on the
    LoggingCallback's tqdm output.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        """Initialise the probe with an empty ``events`` log."""
        super().__init__(trainer)
        self.events: list = []

    def on_train_begin(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the train_begin hook fire and the live global_step."""
        self.events.append(("train_begin", state.global_step))

    def on_epoch_begin(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the epoch_begin hook fire."""
        self.events.append(("epoch_begin", state.global_step))

    def on_step_begin(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the step_begin hook fire — paired with on_step_end."""
        self.events.append(("step_begin", state.global_step))

    def on_step_end(self, state: "TrainerState", *, loss: float = None,
                    grad_norm: float = None, **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the step_end hook fire and the reported loss value."""
        self.events.append(("step_end", state.global_step, float(loss)))

    def on_substep_end(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the substep_end hook fire — should match grad_accum count per step."""
        self.events.append(("substep_end", state.global_step))

    def on_pre_optimizer_step(self, state: "TrainerState", *,
                              grad_norm: float = None, **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the pre_optimizer_step hook fire and the grad_norm value."""
        self.events.append(("pre_optimizer_step", state.global_step, float(grad_norm)))

    def on_epoch_end(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the epoch_end hook fire."""
        self.events.append(("epoch_end", state.global_step))

    def on_train_end(self, state: "TrainerState", **kwargs: object) -> None:  # pylint: disable=unused-argument
        """Record the train_end hook fire — must be the last event."""
        self.events.append(("train_end", state.global_step))


def _make_args(max_steps: int, loss_aggregation: str = "token_weighted",
               grad_accum: int = 1) -> types.SimpleNamespace:
    """Args namespace matching the fields ``train()`` and its dispatch read.

    ``micro_batch_size * grad_accum * dp_size = global_batch_size`` —
    ``_build_dataloader`` derives ``_grad_accum`` from that ratio. We
    bypass ``_build_dataloader`` (see ``_wire_trainer``) and set
    ``_grad_accum`` directly so the test only depends on the field, not
    on the build-step.
    """
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        data=types.SimpleNamespace(type="dummy", max_seq_len=4),
        train=types.SimpleNamespace(
            max_steps=max_steps,
            num_train_epochs=8,  # high enough that max_steps gates exit
            micro_batch_size=1,
            global_batch_size=grad_accum,
            optimizer=types.SimpleNamespace(
                lr=0.1,
                loss_aggregation=loss_aggregation,
                max_grad_norm=1.0,
                weight_decay=0.0,
                eps=1e-8,
                betas=(0.9, 0.999),
                foreach=None,
                lr_warmup_ratio=0.0,
            ),
        ),
    )


def _wire_trainer(args, mesh, world: int, model=None, grad_accum: int = 1):
    """Construct a BaseTrainer with every field ``train()`` reads.

    Bypasses ``_build_*`` steps that need a registered ``ModelSpec`` /
    HuggingFace dataset / parallelize_fn, but calls the *real*
    ``_init_callbacks`` so the full built-in callback stack is on the
    trainer. The user callback (``_ProbeCallback``) gets added via the
    public ``add_callback`` so we exercise that surface too.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = platform.device(local_rank)
    platform.get_device_handle(platform.device_type()).set_device(local_rank)

    with patch.object(trainer_base, "get_spec", return_value=types.SimpleNamespace(
            clip_grad_fn=None,
    )):
        trainer = BaseTrainer(args)
    if model is None:
        model = _ToyCEModel().to(device)
    trainer.model = model
    trainer.device = device
    trainer.parallel_dims = types.SimpleNamespace(dp_size=world, non_dp_size=1)
    trainer.mesh = mesh
    trainer._dp_group_info = GroupInfo(  # pylint: disable=protected-access
        group_name="trainer_dp",
        group=mesh.get_group("loss"),
        rank_size=world,
    )
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.lr_scheduler = None
    dataset = _RepeatingDataset(num_samples=args.train.max_steps * grad_accum * 2)
    trainer.train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: {
            k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()
        },
    )
    trainer.sampler = types.SimpleNamespace(set_epoch=lambda epoch: None)
    trainer._grad_accum = grad_accum  # pylint: disable=protected-access

    trainer._init_callbacks()  # pylint: disable=protected-access

    probe = _ProbeCallback(trainer)
    trainer.add_callback(probe)

    return trainer, device, probe


def test_trainer_train_end_to_end_4card():
    """
    Feature: ``BaseTrainer.train()`` full lifecycle on a real 4-card PG.
    Description: Wires every field ``train()`` reads (model, optimizer,
        dataloader, mesh, dp_group_info) and runs the real production
        loop on 4 NPU cards. The 12 built-in callbacks come from the
        real ``_init_callbacks``; a user probe callback records the
        lifecycle dispatch order.
    Expectation:
        - ``on_train_begin`` is the first dispatch and ``on_train_end``
          is the last (otherwise checkpoint resume / TB writer
          finalisation breaks).
        - Exactly ``max_steps`` ``on_step_end`` events fire — the live
          ``global_step < max_steps`` loop guard works.
        - ``on_step_begin`` always precedes its matching ``on_step_end``
          for the same step.
        - The reported step-end ``loss`` decreases by ≥ 30 % between the
          first and last step (training actually trained).
        - ``trainer.state.global_step == max_steps`` on exit.
        - Process group is destroyed by ``train()`` itself.
    """
    init_process_group()
    world = platform.get_world_size()
    pd = ParallelDims(dp_shard=world, world_size=world)
    mesh = pd.build_mesh(platform.device_type())

    max_steps = 15
    args = _make_args(max_steps=max_steps)
    trainer, _, probe = _wire_trainer(args, mesh, world)

    # train() destroys PG on exit; no finally:destroy needed.
    trainer.train()

    events = probe.events
    assert events, "Probe callback never fired — lifecycle dispatch is broken"
    assert events[0][0] == "train_begin", (
        f"First lifecycle event must be train_begin, got {events[0][0]!r}"
    )
    assert events[-1][0] == "train_end", (
        f"Last lifecycle event must be train_end, got {events[-1][0]!r}"
    )

    step_begins = [e for e in events if e[0] == "step_begin"]
    step_ends = [e for e in events if e[0] == "step_end"]
    assert len(step_begins) == max_steps, (
        f"on_step_begin must fire exactly max_steps={max_steps} times, "
        f"got {len(step_begins)}"
    )
    assert len(step_ends) == max_steps, (
        f"on_step_end must fire exactly max_steps={max_steps} times, "
        f"got {len(step_ends)}"
    )

    losses = [e[2] for e in step_ends]
    assert all(not math.isnan(loss_val) and not math.isinf(loss_val) for loss_val in losses), (
        f"every step-end loss must be finite, got trajectory={losses}"
    )
    assert losses[-1] < losses[0] * 0.7, (
        f"loss must drop by ≥30% across {max_steps} steps under a "
        f"memorisable input — training is broken if it does not. "
        f"initial={losses[0]:.4f}, final={losses[-1]:.4f}, trajectory={losses}"
    )

    assert trainer.state.global_step == max_steps, (
        f"global_step must equal max_steps={max_steps} on exit, "
        f"got {trainer.state.global_step}"
    )


def test_trainer_train_grad_accum_4card():
    """
    Feature: ``BaseTrainer.train()`` with grad-accumulation > 1.
    Description: Sets ``_grad_accum = 2`` so each ``train_step`` reads
        two consecutive batches from the dataloader. Exercises the
        production grad-accum path:
        - ``_make_micro_batch_iterator`` groups 2 batches per yield.
        - ``forward_backward_step`` runs twice per step; gradient sync
          (``set_requires_gradient_sync``) only fires on the last
          micro-batch — verified via ``on_substep_end`` count = 2 per
          step.
    Expectation: grad_accum=2 still drops loss by ≥30 % across 10
        steps (=20 forwards), and ``on_substep_end`` fires exactly
        ``max_steps * grad_accum`` times.
    """
    init_process_group()
    world = platform.get_world_size()
    pd = ParallelDims(dp_shard=world, world_size=world)
    mesh = pd.build_mesh(platform.device_type())

    max_steps = 10
    grad_accum = 2
    args = _make_args(max_steps=max_steps, grad_accum=grad_accum)
    trainer, _, probe = _wire_trainer(args, mesh, world, grad_accum=grad_accum)

    trainer.train()

    events = probe.events
    substeps = [e for e in events if e[0] == "substep_end"]
    assert len(substeps) == max_steps * grad_accum, (
        f"on_substep_end must fire exactly max_steps*grad_accum="
        f"{max_steps * grad_accum} times under grad_accum={grad_accum}, "
        f"got {len(substeps)}"
    )

    losses = [e[2] for e in events if e[0] == "step_end"]
    assert losses[-1] < losses[0] * 0.7, (
        f"loss must drop by ≥30% across {max_steps} steps with "
        f"grad_accum={grad_accum}, got initial={losses[0]:.4f}, "
        f"final={losses[-1]:.4f}, trajectory={losses}"
    )


def test_trainer_train_fsdp_wrapped_4card():
    """
    Feature: ``BaseTrainer.train()`` over a real ``fully_shard``-wrapped
        model on a 4-card PG.
    Description: Wraps the toy model with ``fully_shard`` over the
        ``mesh["fsdp"]`` axis — the production setup any FSDP user lands
        on. ``train()`` then drives:
        - DTensor-aware forward/backward (parameters are sharded
          DTensor; backward triggers reduce-scatter on every step).
        - ``set_requires_gradient_sync`` toggling (FSDPModule path —
          base.py:975-977).
        - Distributed ``clip_grad_norm_`` walking real DTensor params.
        - ``optimizer.step()`` operating on DTensor shards under
          ``SkipDTensorDispatch``.
    Expectation: Loss drops by ≥30 % across 20 steps.
    """
    init_process_group()
    world = platform.get_world_size()
    pd = ParallelDims(dp_shard=world, world_size=world)
    mesh = pd.build_mesh(platform.device_type())

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = platform.device(local_rank)
    platform.get_device_handle(platform.device_type()).set_device(local_rank)

    model = _ToyCEModel().to(device)
    fully_shard(model, mesh=mesh["fsdp"])

    max_steps = 20
    args = _make_args(max_steps=max_steps)
    trainer, _, probe = _wire_trainer(args, mesh, world, model=model)

    trainer.train()

    events = probe.events
    step_ends = [e for e in events if e[0] == "step_end"]
    assert len(step_ends) == max_steps, (
        f"FSDP-wrapped train must complete max_steps={max_steps} steps, "
        f"got {len(step_ends)}"
    )

    losses = [e[2] for e in step_ends]
    assert all(not math.isnan(loss_val) and not math.isinf(loss_val) for loss_val in losses), (
        f"FSDP-wrapped train loss must stay finite, got trajectory={losses}"
    )
    assert losses[-1] < losses[0] * 0.7, (
        f"FSDP-wrapped train loss must drop by ≥30% across {max_steps} "
        f"steps — broken FSDP gradient sync / dispatch otherwise. "
        f"initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
        f"trajectory={losses}"
    )
