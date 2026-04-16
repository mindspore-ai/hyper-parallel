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
"""Unit tests for MindSpore Async Context Parallel platform helpers."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import mindspore as ms

from hyper_parallel.core.context_parallel import async_context_parallel as async_cp_module
from hyper_parallel.platform.mindspore import platform as ms_platform_module
from hyper_parallel.platform.mindspore.platform import (
    MindSporePlatform,
    _a2a_reconstruct_ms,
)


class _FakeWork:
    """Minimal async work stub exposing wait()."""

    def __init__(self):
        self.wait_called = 0

    def wait(self):
        self.wait_called += 1


class _FakeProj:
    """Minimal projection stub for hook-registration tests."""

    def __init__(self):
        self.forward_hooks = []

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)


def _call_proj_post_hook(async_cp, *args, **kwargs):
    """Call the protected projection forward hook in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return async_cp._proj_post_hook(*args, **kwargs)


def _call_proj_bwd_pre_hook(async_cp, *args, **kwargs):
    """Call the protected projection backward pre-hook in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return async_cp._proj_bwd_pre_hook(*args, **kwargs)


def _call_register_proj_hooks(async_cp, *args, **kwargs):
    """Call the protected hook registration helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return async_cp._register_proj_hooks(*args, **kwargs)


def _reconstruct_a2a(async_cp_module_ref, out_perm, head_dim):
    """Call the module-private reconstruction helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return async_cp_module_ref._a2a_reconstruct(out_perm, head_dim)


class TestMindSporeAsyncContextParallel(unittest.TestCase):
    """Unit tests for MindSpore Async Context Parallel helpers."""

    @classmethod
    def setUpClass(cls):
        """Run tests in PyNative mode so custom autograd is active."""
        cls._original_mode = ms.get_context("mode")
        ms.set_context(mode=ms.PYNATIVE_MODE)

    @classmethod
    def tearDownClass(cls):
        """Restore the original execution mode after tests finish."""
        ms.set_context(mode=cls._original_mode)

    @staticmethod
    def _tensor(data):
        return ms.Tensor(np.array(data, dtype=np.float32))

    def test_a2a_reconstruct_ms_matches_expected_layout(self):
        """The helper should reconstruct the gathered sequence layout correctly."""
        out_perm = self._tensor(np.arange(8).reshape(2, 1, 2, 2))

        result = _a2a_reconstruct_ms(out_perm, concat_dim=1)

        expected = np.array([[[0, 1], [2, 3], [4, 5], [6, 7]]], dtype=np.float32)
        np.testing.assert_allclose(result.asnumpy(), expected)

    def test_all_to_all_single_normalizes_handle_result(self):
        """The platform helper should return the preallocated output with the async handle."""
        input_tensor = self._tensor([[1.0, 2.0], [3.0, 4.0]])
        fake_handle = object()

        def fake_all_to_all_single(output, input_data, group=None, async_op=False):
            self.assertEqual(group, "cp_group")
            self.assertTrue(async_op)
            output.copy_(input_data)
            return fake_handle

        with patch.object(ms_platform_module.ops_comm, "all_to_all_single", side_effect=fake_all_to_all_single):
            output, handle = MindSporePlatform.all_to_all_single(
                input_tensor,
                [2, 2],
                "cp_group",
                async_op=True,
            )

        self.assertIs(handle, fake_handle)
        np.testing.assert_allclose(output.asnumpy(), input_tensor.asnumpy())

    def test_all_to_all_single_normalizes_tuple_result(self):
        """The platform helper should accept tuple-style returns as well."""
        input_tensor = self._tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_output = self._tensor([[4.0, 3.0], [2.0, 1.0]])
        fake_handle = object()

        with patch.object(
            ms_platform_module.ops_comm,
            "all_to_all_single",
            return_value=(expected_output, fake_handle),
        ):
            output, handle = MindSporePlatform.all_to_all_single(
                input_tensor,
                [2, 2],
                "cp_group",
                async_op=True,
            )

        self.assertIs(output, expected_output)
        self.assertIs(handle, fake_handle)

    def test_differentiable_async_a2a_wait_waits_and_reconstructs_forward(self):
        """Forward should wait on the async handle and reconstruct the gathered tensor."""
        x = self._tensor(np.ones((1, 2, 2, 1)))
        work = _FakeWork()
        out_perm = self._tensor(np.arange(4).reshape(2, 1, 2, 1))

        output = MindSporePlatform.differentiable_async_a2a_wait(
            x,
            work,
            out_perm,
            "cp_group",
            2,
            1,
            2,
            [],
        )

        self.assertEqual(work.wait_called, 1)
        expected = np.array([[[0.0], [1.0], [2.0], [3.0]]], dtype=np.float32)
        np.testing.assert_allclose(output.asnumpy(), expected)

    def test_differentiable_async_a2a_wait_backward_launches_reverse_a2a(self):
        """Backward should launch reverse A2A, append the handle, and return a zero grad."""
        x = self._tensor(np.ones((1, 2, 2, 1)))
        x.requires_grad = True
        work = _FakeWork()
        out_perm = self._tensor(np.arange(4).reshape(2, 1, 2, 1))
        handle_box = []
        fake_handle = object()

        def fake_all_to_all_single(output, input_data, group=None, async_op=False):
            self.assertEqual(group, "cp_group")
            self.assertTrue(async_op)
            output.copy_(input_data)
            return fake_handle

        def loss_fn(inp):
            output = MindSporePlatform.differentiable_async_a2a_wait(
                inp,
                work,
                out_perm,
                "cp_group",
                2,
                1,
                2,
                handle_box,
            )
            return output.sum()

        with patch.object(ms_platform_module.ops_comm, "all_to_all_single", side_effect=fake_all_to_all_single):
            grad = ms.grad(loss_fn)(x)

        self.assertEqual(work.wait_called, 1)
        self.assertEqual(len(handle_box), 1)
        handle, backward_out_perm = handle_box[0]
        self.assertIs(handle, fake_handle)
        np.testing.assert_allclose(grad.asnumpy(), np.zeros((1, 2, 2, 1), dtype=np.float32))
        np.testing.assert_allclose(backward_out_perm.asnumpy(), np.ones((2, 1, 2, 1), dtype=np.float32))

    def test_proj_post_hook_launches_async_a2a_without_waiting(self):
        """Projection post-hook should only launch async A2A and must not wait immediately."""
        async_cp = async_cp_module.AsyncContextParallel(seq_dim=1, head_dim=2)
        fake_work = _FakeWork()
        fake_out_perm = self._tensor(np.arange(4).reshape(2, 1, 2, 1))
        fwd_slots = {"q": None, "k": None, "v": None}
        output = self._tensor(np.ones((1, 2, 2, 1)))

        with patch.object(
            async_cp_module,
            "_launch_async_a2a_seq_to_head",
            return_value=(fake_work, fake_out_perm),
        ) as mock_launch:
            returned = _call_proj_post_hook(
                async_cp,
                module=None,
                inputs=(),
                output=output,
                key="q",
                group="cp_group",
                world_size=2,
                fwd_slots=fwd_slots,
            )

        self.assertIs(returned, output)
        self.assertEqual(fake_work.wait_called, 0)
        self.assertEqual(fwd_slots["q"], (fake_work, fake_out_perm))
        mock_launch.assert_called_once()

    def test_reverse_a2a_wait_happens_in_backward_pre_hook(self):
        """Backward should launch reverse A2A first, and the projection pre-hook should perform the wait later."""
        x = self._tensor(np.ones((1, 2, 2, 1)))
        x.requires_grad = True
        forward_work = _FakeWork()
        out_perm = self._tensor(np.arange(4).reshape(2, 1, 2, 1))
        handle_box = []
        reverse_work = _FakeWork()

        def fake_all_to_all_single(output, input_data, group=None, async_op=False):
            self.assertEqual(group, "cp_group")
            self.assertTrue(async_op)
            output.copy_(input_data)
            return reverse_work

        def loss_fn(inp):
            output = MindSporePlatform.differentiable_async_a2a_wait(
                inp,
                forward_work,
                out_perm,
                "cp_group",
                2,
                1,
                2,
                handle_box,
            )
            return output.sum()

        with patch.object(ms_platform_module.ops_comm, "all_to_all_single", side_effect=fake_all_to_all_single):
            _ = ms.grad(loss_fn)(x)

        self.assertEqual(forward_work.wait_called, 1)
        self.assertEqual(reverse_work.wait_called, 0)
        self.assertEqual(len(handle_box), 1)
        backward_out_perm = handle_box[0][1]

        async_cp = async_cp_module.AsyncContextParallel(seq_dim=1, head_dim=2)
        grad_output = (self._tensor(np.ones((1, 4, 1))),)
        hooked_grad = _call_proj_bwd_pre_hook(
            async_cp,
            module=None,
            grad_output=grad_output,
            bwd_slot=handle_box,
        )

        self.assertEqual(reverse_work.wait_called, 1)
        self.assertEqual(len(handle_box), 0)
        expected_grad = _reconstruct_a2a(async_cp_module, backward_out_perm, async_cp.head_dim)
        np.testing.assert_allclose(hooked_grad[0].asnumpy(), expected_grad.asnumpy())

    def test_async_context_parallel_registers_backward_hooks_via_platform(self):
        """MindSpore Async CP should use the platform hook API instead of a torch-only method name."""
        async_cp = async_cp_module.AsyncContextParallel(seq_dim=1, head_dim=2)
        q_proj = _FakeProj()
        k_proj = _FakeProj()
        v_proj = _FakeProj()

        with patch.object(async_cp_module.platform, "register_full_backward_pre_hook") as mock_register:
            _call_register_proj_hooks(
                async_cp,
                q_proj,
                k_proj,
                v_proj,
                group="cp_group",
                world_size=2,
                fwd_slots={"q": None, "k": None, "v": None},
                bwd_slots={"q": [], "k": [], "v": []},
            )

        self.assertEqual(len(q_proj.forward_hooks), 1)
        self.assertEqual(len(k_proj.forward_hooks), 1)
        self.assertEqual(len(v_proj.forward_hooks), 1)
        self.assertEqual(mock_register.call_count, 3)


if __name__ == "__main__":
    unittest.main()
