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
"""Unit tests for explicit backward-target transport."""
from unittest.mock import patch

import pytest

from hyper_parallel.core.backward_target import (
    AuxiliaryOutput,
    BackwardTarget,
    attach_backward_targets,
    backward_targets,
    split_backward_targets,
)


def test_attach_and_split_preserve_multiple_targets_and_scales() -> None:
    """Target collection must preserve declaration order and distinct sensitivities."""
    first = BackwardTarget("loss_0", "scale_0")
    second = BackwardTarget("loss_1", "scale_1")

    wrapped = attach_backward_targets("hidden", (first,), (second,))
    value, targets = split_backward_targets(wrapped)

    assert isinstance(wrapped, AuxiliaryOutput)
    assert value == "hidden"
    assert targets == (first, second)


def test_attach_without_targets_keeps_regular_tuple_unchanged() -> None:
    """Ordinary tuple outputs must never be interpreted as backward metadata."""
    regular_output = ("hidden", "context")

    assert attach_backward_targets(regular_output, ()) is regular_output
    assert split_backward_targets(regular_output) == (regular_output, ())


def test_backward_targets_uses_one_platform_multi_root_call() -> None:
    """Every root and its matching sensitivity are forwarded in one call."""
    targets = (
        BackwardTarget("loss_0", "scale_0"),
        BackwardTarget("loss_1", "scale_1"),
    )

    with patch("hyper_parallel.platform.get_platform") as get_platform:
        backward_targets(targets)

    get_platform.return_value.backward.assert_called_once_with(
        ("loss_0", "loss_1"), ("scale_0", "scale_1")
    )


def test_backward_targets_rejects_empty_input() -> None:
    """An empty target set is a caller error rather than a silent no-op."""
    with pytest.raises(ValueError, match="at least one"):
        backward_targets(())
