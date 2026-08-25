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
"""Focused tests for Transformers model construction dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.auto_models._transformers.model_init import _init_model


def test_hf_random_initialization_uses_from_config() -> None:
    """Build a checkpoint-free HF model through its native from_config API."""
    model = object()
    model_class = SimpleNamespace(
        _from_config_parent_class=Mock(return_value=model),
        _from_pretrained_parent_class=Mock(),
    )
    config = SimpleNamespace(architectures=["ExampleModel"])

    is_custom_model, result = _init_model(
        model_class,
        None,
        config,
        "eager",
        torch.bfloat16,
        True,
    )

    assert not is_custom_model
    assert result is model
    model_class._from_config_parent_class.assert_called_once_with(
        config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model_class._from_pretrained_parent_class.assert_not_called()
