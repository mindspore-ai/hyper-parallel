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
"""Tests for MindSpore pipeline stage integration with forward_and_gradfn."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn, ops

from hyper_parallel.core.pipeline_parallel.stage import PipelineStage
from tests.common.mark_utils import arg_mark


class CountNet(nn.Cell):
    """Minimal network that counts how many forwards actually happen."""

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.array([[2.0]], np.float32)), name="w")
        self.forward_calls = 0

    def construct(self, x):
        self.forward_calls += 1
        return ops.matmul(x, self.w)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pipeline_stage_backward_reuses_forward_and_gradfn_result():
    """
    Feature: pipeline stage backward integration.
    Description: backward_one_chunk should reuse the grad_fn created during forward_one_chunk.
    Expectation: The stage runs forward only once and parameter grad is materialized correctly.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)

    net = CountNet()
    stage = PipelineStage(net, stage_index=0, stage_num=1, has_backward=True)
    x = Tensor(np.array([[3.0]], np.float32))

    out = stage.forward_one_chunk(0, args=(x,), kwargs={})
    np.testing.assert_allclose(out.asnumpy(), np.array([[6.0]], dtype=np.float32))
    assert net.forward_calls == 1

    stage.backward_one_chunk(0)

    assert net.forward_calls == 1
    assert net.w.grad is not None
    np.testing.assert_allclose(net.w.grad.asnumpy(), np.array([[3.0]], dtype=np.float32))
