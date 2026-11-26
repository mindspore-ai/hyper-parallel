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
"""CPU lifecycle tests for optional DeepEP transport; no communication kernels are emulated numerically."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.deep_ep import (
    DeepEPBuffer, deep_ep_combine, deep_ep_dispatch,
)


class TestDeepEPBuffer(unittest.TestCase):
    """Verify native argument preservation, event ordering, failure recovery and destruction."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.order = []
        self.event = SimpleNamespace(current_stream_wait=lambda: self.order.append("wait"))
        self.handle = object()
        self.native = SimpleNamespace(dispatch=Mock(return_value=(torch.ones(1), None, None, self.handle, self.event)),
                                      combine=Mock(return_value=(torch.ones(1), None, self.event)),
                                      destroy=Mock(side_effect=lambda: self.order.append("destroy")))
        self.constructor = Mock(return_value=self.native)
        dependency = patch("hyper_parallel.distributed.expert_parallel.deep_ep.import_module",
                           return_value=SimpleNamespace(EPBuffer=self.constructor))
        dependency.start()
        self.addCleanup(dependency.stop)

    def test_expanded_layout_preserves_handles_and_options(self) -> None:
        with DeepEPBuffer(object(), deterministic=True) as owner:
            x, indices, weights = torch.ones(2, 8), torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1)
            result = owner.dispatch(x, indices, weights, async_with_compute_stream=True)
            self.assertIs(result[3], self.handle)
            self.assertIs(result[4], self.event)
            self.assertEqual(self.native.dispatch.call_args.kwargs,
                             {"do_expand": True, "do_handle_copy": False, "async_with_compute_stream": True})
            owner.combine(x, result[3], topk_weights=weights, defer_epilogue=True,
                          async_with_compute_stream=True)
            self.assertIs(self.native.combine.call_args.args[1], self.handle)
        self.assertEqual(self.order, ["wait", "wait", "destroy"])
        self.assertTrue(self.constructor.call_args.kwargs["explicitly_destroy"])
        owner.destroy()
        self.native.destroy.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            owner.dispatch(x)

    def test_exception_scope_cleanup_and_released_events(self) -> None:
        self.event.event = None
        with self.assertRaisesRegex(ValueError, "training failure"):
            with DeepEPBuffer(object()) as owner:
                owner.dispatch(torch.ones(1, 8))
                raise ValueError("training failure")
        self.assertEqual(self.order, ["destroy"])

    def test_failed_wait_keeps_buffer_alive_for_retry(self) -> None:
        event = Mock()
        event.current_stream_wait.side_effect = [RuntimeError("pending"), None]
        self.native.dispatch.return_value = event
        owner = DeepEPBuffer(object())
        owner.dispatch(torch.ones(1, 8))
        with self.assertRaisesRegex(RuntimeError, "pending"):
            owner.close()
        self.native.destroy.assert_not_called()
        owner.close()
        self.native.destroy.assert_called_once()
        self.assertEqual(event.current_stream_wait.call_count, 2)

    def test_direct_calls_cannot_silently_drop_autograd(self) -> None:
        x = torch.ones(2, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "explicit autograd Function"):
            deep_ep_dispatch(self.native, x)
        with self.assertRaisesRegex(RuntimeError, "explicit autograd Function"):
            deep_ep_combine(self.native, x, self.handle)
        with torch.no_grad():
            result = deep_ep_dispatch(self.native, x)
        self.assertIs(result[3], self.handle)
        with self.assertRaisesRegex(ValueError, "do_handle_copy=False"):
            deep_ep_dispatch(self.native, x.detach(), do_handle_copy=True)
