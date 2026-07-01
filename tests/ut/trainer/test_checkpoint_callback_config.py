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
"""Checkpoint callback configuration tests."""
from types import SimpleNamespace

from hyper_parallel.trainer.callbacks.base import (
    CheckpointCallback,
    SafetensorsExportCallback,
)


def _trainer(args):
    return SimpleNamespace(args=args)


def test_checkpoint_callback_reads_nested_train_checkpoint():
    """
    Feature: nested checkpoint config
    Description: CheckpointCallback reads checkpoint settings from train.checkpoint.
    Expectation: save/load fields match the nested config values.
    """
    ckpt = SimpleNamespace(
        save_steps=7,
        output_dir="/tmp/glm5_ckpt",
        load_path="/tmp/glm5_ckpt/step_7",
        save_async=True,
    )
    args = SimpleNamespace(train=SimpleNamespace(checkpoint=ckpt))

    callback = CheckpointCallback(_trainer(args))

    assert callback.save_steps == 7
    assert callback.output_dir == "/tmp/glm5_ckpt"
    assert callback.load_path == "/tmp/glm5_ckpt/step_7"
    assert callback.save_async is True


def test_checkpoint_callback_keeps_top_level_fallback():
    """
    Feature: checkpoint config fallback
    Description: Legacy top-level checkpoint config remains supported.
    Expectation: callback uses args.checkpoint when train.checkpoint is absent.
    """
    ckpt = SimpleNamespace(
        save_steps=3,
        output_dir="/tmp/legacy_ckpt",
        load_path=None,
        save_async=False,
    )
    args = SimpleNamespace(checkpoint=ckpt)

    callback = CheckpointCallback(_trainer(args))

    assert callback.save_steps == 3
    assert callback.output_dir == "/tmp/legacy_ckpt"
    assert callback.load_path is None
    assert callback.save_async is False


def test_hf_export_callback_reads_nested_train_checkpoint():
    """
    Feature: nested HF export config
    Description: SafetensorsExportCallback reads settings from train.checkpoint.
    Expectation: export controls match the nested checkpoint config.
    """
    ckpt = SimpleNamespace(
        save_hf_weights=True,
        save_steps=11,
        output_dir="/tmp/glm5_hf",
    )
    args = SimpleNamespace(train=SimpleNamespace(checkpoint=ckpt))

    callback = SafetensorsExportCallback(_trainer(args))

    assert callback.enabled is True
    assert callback.save_steps == 11
    assert callback.output_dir == "/tmp/glm5_hf"
