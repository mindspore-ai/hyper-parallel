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
"""ST cases for MindSpore autograd compatibility."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
from mindspore import Tensor, ops

from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.common.mark_utils import arg_mark

enable_mindspore_backward_compat()


def _make_tensor(data):
    tensor = Tensor(np.array(data, dtype=np.float32))
    tensor.requires_grad = True
    return tensor


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scalar_backward_grad_correctness():
    """
    Feature: MindSpore backward compatibility on scalar loss.
    Description: Compute scalar loss with torch-like `loss.backward()` and check leaf gradients.
    Expectation: Gradients match analytical results.
    """
    x = _make_tensor([1.0, 2.0, 3.0])
    w = _make_tensor([0.5, -1.0, 4.0])

    loss = (x * w).sum()
    loss.backward()

    np.testing.assert_allclose(x.grad.asnumpy(), np.array([0.5, -1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(w.grad.asnumpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_backward_with_sens_grad_correctness():
    """
    Feature: MindSpore backward compatibility with explicit sens.
    Description: Compute vector output and call `loss.backward(sens)` to verify Jacobian-vector product.
    Expectation: Gradients match analytical results.
    """
    x = _make_tensor([1.0, 2.0, 3.0])
    w = _make_tensor([4.0, 5.0, 6.0])
    sens = Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    out = x * w
    out.backward(sens)

    np.testing.assert_allclose(x.grad.asnumpy(), np.array([0.4, 1.0, 1.8], dtype=np.float32))
    np.testing.assert_allclose(w.grad.asnumpy(), np.array([0.1, 0.4, 0.9], dtype=np.float32))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_retain_grad_keeps_non_leaf_grad_correctness():
    """
    Feature: MindSpore backward compatibility for retain_grad.
    Description: Retain a non-leaf tensor gradient and verify it is populated after backward.
    Expectation: Non-leaf gradient matches the expected upstream gradient.
    """
    x = _make_tensor([2.0, 4.0, 6.0])
    w = _make_tensor([1.0, 3.0, 5.0])

    out = x * w
    out.retain_grad()
    loss = ops.sum(out)
    loss.backward()

    np.testing.assert_allclose(out.grad.asnumpy(), np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(x.grad.asnumpy(), np.array([1.0, 3.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(w.grad.asnumpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))
