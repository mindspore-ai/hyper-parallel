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
"""Tests for BaseTrainer FSDP controls during gradient accumulation."""

# The tests intentionally exercise BaseTrainer's micro-step control helper.
# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import Mock, call

from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.trainer.base import BaseTrainer


def _trainer(
        *,
        dp_shard_size: int,
        dp_replicate_size: int,
        requires_grad_sync: bool = True,
) -> BaseTrainer:
    """Build a minimal trainer instance for FSDP control tests."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = SimpleNamespace(
        fsdp_config=FSDP2Config(
            dp_shard_size=dp_shard_size,
            reshard_after_backward=False,
            requires_grad_sync=requires_grad_sync,
        )
    )
    trainer.mesh = SimpleNamespace(dp_replicate_size=dp_replicate_size)
    trainer.hsdp_model_parts = [Mock()]
    return trainer


def test_model_reshard_defers_until_last_micro_batch() -> None:
    trainer = _trainer(dp_shard_size=2, dp_replicate_size=1)
    model_part = trainer.hsdp_model_parts[0]

    trainer.model_reshard(micro_step=0, num_micro_steps=2)
    trainer.model_reshard(micro_step=1, num_micro_steps=2)

    assert model_part.set_reshard_after_backward.call_args_list == [
        call(False),
        call(True),
    ]


def test_fsdp_gradient_sync_runs_only_on_last_micro_batch_when_disabled() -> None:
    """Verify disabled FSDP synchronization resumes on the final micro batch."""
    trainer = _trainer(
        dp_shard_size=2,
        dp_replicate_size=1,
        requires_grad_sync=False,
    )
    model_part = trainer.hsdp_model_parts[0]

    trainer._configure_fsdp_gradient_sync(0, 2)
    trainer._configure_fsdp_gradient_sync(1, 2)

    assert model_part.set_requires_gradient_sync.call_args_list == [
        call(False),
        call(True),
    ]
    assert model_part.set_is_last_backward.call_args_list == [
        call(False),
        call(True),
    ]
    model_part.set_requires_all_reduce.assert_not_called()


def test_fsdp_gradient_sync_runs_on_every_micro_batch_when_enabled() -> None:
    """Verify enabled FSDP synchronization runs on every micro batch."""
    trainer = _trainer(
        dp_shard_size=2,
        dp_replicate_size=1,
        requires_grad_sync=True,
    )
    model_part = trainer.hsdp_model_parts[0]

    trainer._configure_fsdp_gradient_sync(0, 2)
    trainer._configure_fsdp_gradient_sync(1, 2)

    assert model_part.set_requires_gradient_sync.call_args_list == [
        call(True),
        call(True),
    ]
    assert model_part.set_is_last_backward.call_args_list == [
        call(False),
        call(True),
    ]
    model_part.set_requires_all_reduce.assert_not_called()


def test_hsdp_all_reduce_runs_only_on_last_micro_batch_when_grad_sync_disabled() -> None:
    """Verify HSDP all-reduce runs on the final micro batch when grad sync is disabled."""
    trainer = _trainer(
        dp_shard_size=2,
        dp_replicate_size=2,
        requires_grad_sync=False,
    )
    model_part = trainer.hsdp_model_parts[0]

    trainer._configure_fsdp_gradient_sync(0, 2)
    trainer._configure_fsdp_gradient_sync(1, 2)

    assert model_part.set_requires_all_reduce.call_args_list == [
        call(False),
        call(True),
    ]


def test_hsdp_all_reduce_runs_only_on_last_micro_batch_when_grad_sync_enabled() -> None:
    """Verify HSDP all-reduce remains limited to the final micro batch."""
    trainer = _trainer(
        dp_shard_size=2,
        dp_replicate_size=2,
        requires_grad_sync=True,
    )
    model_part = trainer.hsdp_model_parts[0]

    trainer._configure_fsdp_gradient_sync(0, 2)
    trainer._configure_fsdp_gradient_sync(1, 2)

    assert model_part.set_requires_all_reduce.call_args_list == [
        call(False),
        call(True),
    ]
