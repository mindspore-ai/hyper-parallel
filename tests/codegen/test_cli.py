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
"""Codegen CLI contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from hyper_parallel.codegen import cli
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_generate_passes_typed_config_overrides(monkeypatch):
    """Unknown argparse tokens are trainer config overrides."""
    captured = {}

    def fake_load_training_config(yaml_path, overrides=()):
        captured["yaml_path"] = yaml_path
        captured["overrides"] = list(overrides)
        return SimpleNamespace(codegen=True)

    def fake_ensure_codegen_artifact(_config, _yaml_path):
        return None

    monkeypatch.setattr("hyper_parallel.trainer.config.manager.load_training_config", fake_load_training_config)
    monkeypatch.setattr("hyper_parallel.codegen.manager.ensure_codegen_artifact", fake_ensure_codegen_artifact)

    ret = cli.main([
        "generate",
        "--config",
        "train.yaml",
        "--accelerator.tp_size=1",
        "--accelerator.cp_size=2",
    ])

    assert ret == 0
    assert captured == {
        "yaml_path": "train.yaml",
        "overrides": ["--accelerator.tp_size=1", "--accelerator.cp_size=2"],
    }
