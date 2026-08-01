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
"""Regression checks for the MXFP8 training skeleton configuration."""

from pathlib import Path

import yaml


def test_low_precision_skeleton_uses_aligned_tiny_checkpoint():
    """Keep the dedicated skeleton checkpoint separate and 32-aligned."""
    root = Path(__file__).resolve().parents[3]
    main_source = (root / "examples/training_skeleton/main.py").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(
        (root / "examples/training_skeleton/train_low_precision.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "TINY_VOCAB_SIZE = 1024" in main_source
    assert 1024 % 32 == 0
    assert config["model"]["weights_path"].endswith("tiny_model_mxfp8")
    assert config["model"]["tokenizer_path"].endswith("tiny_model_mxfp8")
    assert config["training"]["train_url"].endswith("training_skeleton_low_precision")
    assert "output" not in config["low_precision"]["precision_debug"]
    assert "OBSERVER_SOURCE" in main_source
