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
"""Checkpoint lifecycle configuration tests."""
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer


def _trainer(args):
    """Build BaseTrainer with get_spec patched."""
    with patch.object(trainer_base, "get_spec", return_value=object()):
        return BaseTrainer(args, setup=False)


def test_checkpoint_lifecycle_reads_nested_train_checkpoint():
    """
    Feature: nested checkpoint config
    Description: BaseTrainer checkpoint lifecycle reads train.checkpoint.
    Expectation: save/load fields match the nested config values.
    """
    ckpt = SimpleNamespace(
        save_steps=7,
        output_dir="/tmp/glm5_ckpt",
        load_path="/tmp/glm5_ckpt/step_7",
        save_async=True,
        save_hf_weights=False,
    )
    args = SimpleNamespace(
        model=SimpleNamespace(name="toy"),
        train=SimpleNamespace(max_steps=10, num_train_epochs=1, checkpoint=ckpt),
    )

    trainer = _trainer(args)

    assert trainer._checkpoint_save_steps() == 7
    assert trainer._checkpoint_output_dir() == "/tmp/glm5_ckpt"
    assert trainer._checkpoint_load_path() == "/tmp/glm5_ckpt/step_7"
    assert trainer._checkpoint_save_async() is True


def test_checkpoint_lifecycle_keeps_top_level_fallback():
    """
    Feature: checkpoint config fallback
    Description: Legacy top-level checkpoint config remains supported.
    Expectation: BaseTrainer uses args.checkpoint when train.checkpoint is absent.
    """
    ckpt = SimpleNamespace(
        save_steps=3,
        output_dir="/tmp/legacy_ckpt",
        load_path=None,
        save_async=False,
        save_hf_weights=False,
    )
    args = SimpleNamespace(
        model=SimpleNamespace(name="toy"),
        train=SimpleNamespace(max_steps=10, num_train_epochs=1),
        checkpoint=ckpt,
    )

    trainer = _trainer(args)

    assert trainer._checkpoint_save_steps() == 3
    assert trainer._checkpoint_output_dir() == "/tmp/legacy_ckpt"
    assert trainer._checkpoint_load_path() is None
    assert trainer._checkpoint_save_async() is False


def test_maybe_export_hf_checkpoint_reads_nested_train_checkpoint():
    """
    Feature: nested HF export config
    Description: _maybe_export_hf_checkpoint reads train.checkpoint.save_hf_weights.
    Expectation: disabled branch exits before gathering model state.
    """
    ckpt = SimpleNamespace(
        save_hf_weights=False,
        save_steps=11,
        output_dir="/tmp/glm5_hf",
    )
    args = SimpleNamespace(
        model=SimpleNamespace(name="toy"),
        train=SimpleNamespace(max_steps=10, num_train_epochs=1, checkpoint=ckpt),
    )

    trainer = _trainer(args)
    trainer.model = object()

    with patch.object(trainer_base, "get_model_state_dict") as mock_get:
        trainer._maybe_export_hf_checkpoint(trainer.state)
    mock_get.assert_not_called()
