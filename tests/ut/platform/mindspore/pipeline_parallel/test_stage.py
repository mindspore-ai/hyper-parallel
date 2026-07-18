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
"""Unit tests for MindSpore pipeline stage gradient target selection."""
# pylint: disable=protected-access,wrong-import-position

import os
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import pytest

pytest.importorskip("mindspore")

import mindspore as ms
from tests.common.mark_utils import arg_mark

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore.pipeline_parallel import backward as pipeline_backward
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStageBase


class _FakeTensor:
    """Device-free stand-in for Tensor gradient-selection tests."""

    def __init__(self, requires_grad: bool = False) -> None:
        """Initialize the gradient requirement flag."""
        self._requires_grad = requires_grad


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_stage_grad_position_does_not_mark_keyword_tensor() -> None:
    """
    Feature: MindSpore pipeline-stage input gradients.
    Description: Select positional activation gradients without including integer keyword Tensors.
    Expectation: The stage uses explicit positional indices and leaves the keyword Tensor non-differentiable.
    """
    activation = _FakeTensor(requires_grad=True)
    input_ids = _FakeTensor()

    with patch.object(ms, "Tensor", _FakeTensor), patch.object(pipeline_backward, "Tensor", _FakeTensor):
        grad_position = PipelineStageBase._grad_position_from_requires_grad((activation,))
        pipeline_backward._set_requires_grad((activation,), {"input_ids": input_ids}, grad_position)

    assert grad_position == (0,), f"Expected only positional activation index 0, got={grad_position}"
    assert activation._requires_grad is True, (
        f"Expected activation._requires_grad=True, got={activation._requires_grad}"
    )
    assert input_ids._requires_grad is False, (
        f"Expected input_ids._requires_grad=False, got={input_ids._requires_grad}"
    )
