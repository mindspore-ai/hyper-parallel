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
"""Prediction-local configuration stubs shared by task-host unit tests."""

from types import SimpleNamespace
from typing import Any

from hyper_parallel.trainer.config.tasks import PredictionConfig


def build_model_stub(name: str = "stub") -> str:
    """Provide an inspectable model target without constructing a framework model.

    Args:
        name: Sentinel value returned when the configuration target is built.

    Returns:
        The supplied name, unchanged.
    """
    return name


def prediction_config(**overrides: Any) -> SimpleNamespace:
    """Build a minimal TrainerConfig-like object for pure validation checks."""
    defaults = {
        "run_mode": "predict",
        "prediction": PredictionConfig(input="prompts.jsonl", output_file="results.jsonl"),
        "optimizer": None,
        "dataset": None,
        "dataloader": None,
        "compile": SimpleNamespace(enabled=False),
        "activation_checkpoint": None,
        "hyper_offload": SimpleNamespace(enable=False),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
