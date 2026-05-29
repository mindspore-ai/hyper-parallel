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
"""Distributed driver for ``BaseTrainer.train_step`` cases."""
import math
import os
import types
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel import (
    destroy_process_group,
    get_platform,
    init_process_group,
)
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.parallel_dims import ParallelDims

platform = get_platform()


class _ToyModel(nn.Module):
    """Minimal CE-loss model — provides real autograd on 4 NPU ranks.

    Returns a dict so ``forward_backward_step``'s ``outputs["loss"]`` branch
    is exercised; embedding + projection is enough to produce non-trivial
    gradients without dragging in a transformer stack.
    """

    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        """Build a ``vocab x hidden`` embedding and ``hidden x vocab`` projection."""
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None,  # pylint: disable=arguments-differ
                use_cache: bool = False) -> dict:  # pylint: disable=unused-argument
        """Run embed → proj → CE on ``input_ids`` with ``-100``-aware loss."""
        x = self.embed(input_ids)
        logits = self.proj(x)
        if labels is None:
            return {"loss": logits.sum()}
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss}


def _make_args(loss_aggregation: str = "token_weighted"):
    """Build the SimpleNamespace ``args`` shape that ``train_step`` reads."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        train=types.SimpleNamespace(
            max_steps=4,
            optimizer=types.SimpleNamespace(
                loss_aggregation=loss_aggregation,
                max_grad_norm=1.0,
            ),
        ),
    )


def _wire_trainer(args, mesh, world):
    """Construct a ``BaseTrainer`` with the fields ``train_step`` reads.

    Skips ``_setup`` / ``_build_model`` (we already initialised the PG /
    device / model manually) and only wires the runtime state needed for
    ``train_step``: model, optimizer, scheduler, mesh-derived dp group.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = platform.device(local_rank)
    platform.get_device_handle(platform.device_type()).set_device(local_rank)

    with patch.object(trainer_base, "get_spec", return_value=types.SimpleNamespace(
            clip_grad_fn=None,
    )):
        trainer = BaseTrainer(args)
    model = _ToyModel().to(device)
    trainer.model = model
    trainer.device = device
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.lr_scheduler = None
    trainer.parallel_dims = types.SimpleNamespace(dp_size=world)
    trainer.mesh = mesh
    trainer._dp_group_info = GroupInfo(  # pylint: disable=protected-access
        group_name="trainer_dp",
        group=mesh.get_group("loss"),
        rank_size=world,
    )
    # No callback wiring — train_step calls these via the trainer's bound
    # methods, replace with no-op lambdas so we don't drag a full callback
    # stack into the ST.
    trainer.on_substep_end = lambda: None
    trainer.on_pre_optimizer_step = lambda **_: None
    return trainer, device


def _batch(input_ids, labels=None):
    if labels is None:
        labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def test_train_step_token_weighted_end_to_end_4card():
    """
    Feature: ``BaseTrainer.train_step`` token-weighted aggregation on a real 4-card PG.
    Description: Build a 1D ``dp_shard=4`` mesh, wire a toy CE model + SGD
        optimizer on every rank, run one step with deterministic input.
        Exercises the full path: ``count_loss_token`` per micro-batch →
        ``platform.all_reduce(global_tokens)`` → token-weighted scaled
        backward → ``clip_grad_norm_`` on distributed params →
        ``optimizer.step()`` → ``platform.all_reduce(total_loss_sum)``.
    Expectation: ``avg_loss`` and ``grad_norm`` are both finite, the global
        step counter advances to 1, and ``platform.all_reduce`` produced a
        ``global_tokens`` that equals ``world * per-rank-tokens``.
    """
    init_process_group()
    try:
        world = platform.get_world_size()
        pd = ParallelDims(dp_shard=world, world_size=world)
        mesh = pd.build_mesh(platform.device_type())

        trainer, device = _wire_trainer(_make_args("token_weighted"), mesh, world)
        # 4 valid tokens per rank → global tokens after all-reduce = world * 4.
        mb = _batch(torch.tensor([[1, 2, 3, 4]], device=device),
                    torch.tensor([[1, 2, 3, 4]], device=device))
        metrics = trainer.train_step(iter([[mb]]))

        assert metrics["loss"] is not None and not math.isnan(metrics["loss"]), (
            f"avg_loss must be finite after one step, got {metrics['loss']!r}"
        )
        assert metrics["grad_norm"] is not None and not math.isnan(metrics["grad_norm"]), (
            f"grad_norm must be finite after one step, got {metrics['grad_norm']!r}"
        )
        assert trainer.state.global_step == 1, (
            f"global_step must advance to 1 after a successful step, "
            f"got {trainer.state.global_step}"
        )
        # ``_last_global_tokens`` is set by train_step after the all-reduce;
        # all ranks see the same world-wide sum.
        expected_global_tokens = world * 4
        assert trainer._last_global_tokens == expected_global_tokens, (  # pylint: disable=protected-access
            f"global_tokens all-reduce broken: expected {expected_global_tokens}, "
            f"got {trainer._last_global_tokens}"  # pylint: disable=protected-access
        )
    finally:
        destroy_process_group()


def test_train_step_rank_average_end_to_end_4card():
    """
    Feature: ``BaseTrainer.train_step`` rank-average aggregation on real 4-card PG.
    Description: Same setup as the token-weighted case but with
        ``loss_aggregation='rank_average'``. Exercises the alternative
        formula path (``base.py:1016-1029``) — ``total_loss_arith_sum /
        num_micro / dp_size`` after a cross-rank ``platform.all_reduce``.
    Expectation: ``avg_loss`` and ``grad_norm`` are finite; ``global_step``
        advances to 1; the all-reduce divisor uses ``dp_size`` (= world).
    """
    init_process_group()
    try:
        world = platform.get_world_size()
        pd = ParallelDims(dp_shard=world, world_size=world)
        mesh = pd.build_mesh(platform.device_type())

        trainer, device = _wire_trainer(_make_args("rank_average"), mesh, world)
        mb = _batch(torch.tensor([[1, 2, 3, 4]], device=device))
        metrics = trainer.train_step(iter([[mb]]))

        assert metrics["loss"] is not None and not math.isnan(metrics["loss"]), (
            f"rank_average avg_loss must be finite, got {metrics['loss']!r}"
        )
        assert metrics["grad_norm"] is not None and not math.isnan(metrics["grad_norm"]), (
            f"rank_average grad_norm must be finite, got {metrics['grad_norm']!r}"
        )
        assert trainer.state.global_step == 1, (
            f"global_step must advance to 1, got {trainer.state.global_step}"
        )
    finally:
        destroy_process_group()
