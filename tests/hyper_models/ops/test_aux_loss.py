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
"""NPU tests for auxiliary-loss gradient scaling."""

# Importing the public interface after importorskip keeps collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch

pytest.importorskip("torch_npu")

from hyper_parallel.auto_models.ops import aux_loss_auto_scale, set_aux_loss_scale
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_aux_loss_auto_scale_preserves_output_and_injects_gradient() -> None:
    """Auxiliary scaling must preserve output values and pass both gradients."""
    main_input = torch.tensor([1.0, -2.0, 3.0], device="npu", requires_grad=True)
    aux_input = torch.tensor([2.0, -1.0], device="npu", requires_grad=True)
    main_output = main_input.square()
    aux_loss = aux_input.square().sum()
    scale = torch.tensor(0.25, device="npu")

    try:
        set_aux_loss_scale(scale)
        output = aux_loss_auto_scale(main_output, aux_loss)
        torch.testing.assert_close(output, main_output, rtol=0.0, atol=0.0)
        output.sum().backward()

        torch.testing.assert_close(
            main_input.grad,
            2 * main_input.detach(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            aux_input.grad,
            scale * 2 * aux_input.detach(),
            rtol=0.0,
            atol=0.0,
        )
    finally:
        set_aux_loss_scale(torch.tensor(1.0))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_set_aux_loss_scale_updates_subsequent_autograd_calls() -> None:
    """The latest configured scale must control each subsequent aux gradient."""
    try:
        for scale_value in (0.5, 2.0):
            aux_input = torch.tensor(3.0, device="npu", requires_grad=True)
            aux_loss = aux_input.square()
            output = torch.ones((), device="npu", requires_grad=True)
            scale = torch.tensor(scale_value, device="npu")
            set_aux_loss_scale(scale)

            aux_loss_auto_scale(output, aux_loss).backward()

            torch.testing.assert_close(
                aux_input.grad,
                scale * 2 * aux_input.detach(),
                rtol=0.0,
                atol=0.0,
            )
    finally:
        set_aux_loss_scale(torch.tensor(1.0))
