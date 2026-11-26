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
"""CPU resource/ABI checks; these tests do not validate collective MegaMoE numerics."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.deep_gemm import (
    MegaMoEBuffer, transform_mega_moe_weights, transform_gemm_scales, transform_k_grouped_gemm_scales,
)


class TestMegaMoEBuffer(unittest.TestCase):
    """Check optional import, borrowed-view lifetime, option forwarding and cleanup order."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.order = []
        self.native = SimpleNamespace(x=torch.ones(2, 64), x_sf=torch.ones(2, 2), topk_weights=torch.ones(2, 1),
                                      destroy=Mock(side_effect=lambda: self.order.append("destroy")))
        self.extension = SimpleNamespace(SymmBuffer=Mock(return_value=self.native), fp8_fp4_mega_moe=Mock(),
                                         transform_weights_for_mega_moe=Mock(return_value=object()),
                                         transform_sf_into_required_layout=Mock(return_value=object()),
                                         transform_k_grouped_sf_into_required_layout=Mock(return_value=object()))
        dependency = patch("hyper_parallel.distributed.expert_parallel.deep_gemm.import_module",
                           return_value=self.extension)
        dependency.start()
        self.addCleanup(dependency.stop)
        synchronize = patch("hyper_parallel.distributed.expert_parallel.deep_gemm.torch.npu.synchronize",
                            side_effect=lambda device: self.order.append("synchronize"))
        synchronize.start()
        self.addCleanup(synchronize.stop)
        self.weights = (torch.ones(2, 32, dtype=torch.int8), torch.ones(2, 1, dtype=torch.int16))

    def test_native_views_and_collective_options(self):
        group = object()
        output = torch.empty(2, 64, dtype=torch.bfloat16)
        with MegaMoEBuffer(group, num_experts=8, num_topk=1, hidden=64) as owner:
            self.assertIs(owner.native, self.native)
            self.assertIs(owner.forward(output, self.weights, self.weights, low_latency=True,
                                         activation_clamp=7.0), output)
            args, kwargs = self.extension.fp8_fp4_mega_moe.call_args
            self.assertIs(args[3], self.native)
            self.assertTrue(kwargs["low_latency"])
            self.assertEqual(kwargs["activation_clamp"], 7.0)
        self.assertEqual(self.order, ["synchronize", "destroy"])
        owner.close()
        self.native.destroy.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = owner.native

    def test_cleanup_after_exception_and_sync_failure_retry(self):
        with self.assertRaisesRegex(ValueError, "failed inference"):
            with MegaMoEBuffer(object()):
                raise ValueError("failed inference")
        self.assertEqual(self.order, ["synchronize", "destroy"])
        owner = MegaMoEBuffer(object())
        with patch("hyper_parallel.distributed.expert_parallel.deep_gemm.torch.npu.synchronize",
                   side_effect=RuntimeError("pending work")):
            with self.assertRaisesRegex(RuntimeError, "pending work"):
                owner.close()
        self.assertIs(owner.native, self.native)
        owner.close()
        self.assertEqual(self.native.destroy.call_count, 2)

    def test_layout_helpers_preserve_native_encodings(self):
        self.assertIs(transform_mega_moe_weights(self.weights, self.weights),
                      self.extension.transform_weights_for_mega_moe.return_value)
        scale = torch.ones(2, 2, dtype=torch.int16)
        ends = torch.tensor([2], dtype=torch.int32)
        expected = transform_gemm_scales(scale, 2, 64, (1, 1, 32), is_sfa=True, psum_layout=ends)
        self.assertIs(expected, self.extension.transform_sf_into_required_layout.return_value)
        self.assertIs(self.extension.transform_sf_into_required_layout.call_args.args[-1], ends)
        expected = transform_k_grouped_gemm_scales(scale, 2, 64, (1, 1, 32), is_sfa=False, psum_layout=ends)
        self.assertIs(expected, self.extension.transform_k_grouped_sf_into_required_layout.return_value)

    def test_forward_cannot_drop_input_or_parameter_gradients(self):
        owner = MegaMoEBuffer(object())
        self.addCleanup(owner.close)
        self.native.topk_weights.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            owner.forward(torch.empty(2, 64), self.weights, self.weights)
        self.extension.fp8_fp4_mega_moe.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            transform_gemm_scales(torch.ones(2, 2, requires_grad=True), 2, 64, (1, 1, 32), is_sfa=True)
