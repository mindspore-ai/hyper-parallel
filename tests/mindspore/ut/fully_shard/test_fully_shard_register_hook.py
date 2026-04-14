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
"""Unit tests for _register_post_backward_hook in MindSporeHSDPSchedulerV2."""
import os
import unittest

import pytest

# Skip entire module if mindspore is not installed
pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import mindspore as ms
import numpy as np

from mindspore.common.api import _no_grad
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2


def _make_scheduler_stub() -> MindSporeHSDPSchedulerV2:
    """Create a minimal MindSporeHSDPSchedulerV2 stub that can call _register_post_backward_hook."""
    scheduler = object.__new__(MindSporeHSDPSchedulerV2)
    return scheduler


def _call_register_post_backward_hook(scheduler, args, kwargs):
    """Call the tested protected hook helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return scheduler._register_post_backward_hook(args, kwargs)


class TestRegisterPostBackwardHook(unittest.TestCase):
    """Unit tests for MindSporeHSDPSchedulerV2._register_post_backward_hook."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
        self.scheduler = _make_scheduler_stub()

    def test_all_no_grad_returns_early(self):
        """When no tensor requires grad, args/kwargs are returned unchanged.

        description: Pass only requires_grad=False tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook early return.
        """
        t1 = ms.Tensor(np.random.randn(2).astype(np.float32))
        t2 = ms.Tensor(np.random.randn(3).astype(np.float32))

        args = (t1, t2)
        kwargs = {}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object (early return), "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object (early return), "
            "got different object"
        )

    def test_no_grad_context_returns_early(self):
        """Under _no_grad context, args/kwargs are returned unchanged.

        description: Wrap the call in _no_grad() context with a requires_grad=True tensor.
        expectation: Returns early with original args and kwargs since grad is disabled.
        feature: _register_post_backward_hook _no_grad context path.
        """
        t1 = ms.Tensor(np.random.randn(2).astype(np.float32))
        t1.requires_grad = True
        args = (t1,)
        kwargs = {}

        with _no_grad():
            out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object under _no_grad, "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object under _no_grad, "
            "got different object"
        )

if __name__ == "__main__":
    unittest.main()
