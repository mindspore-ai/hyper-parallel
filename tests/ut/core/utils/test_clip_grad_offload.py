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
"""CPU-offloaded norm communication without accelerator hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from hyper_parallel.core.utils.clip_grad import _all_reduce_offloaded, _get_total_norm


class TestOffloadedReduction(unittest.TestCase):
    """Transfer offloaded inputs only when the process group requires it."""

    def test_gloo_reduces_cpu_tensor_without_copy(self):
        """CPU-capable groups keep the original tensor."""
        value = torch.tensor(3.0)
        group = object()
        with patch("hyper_parallel.core.utils.clip_grad.dist.get_backend", return_value="gloo"), \
                patch("hyper_parallel.core.utils.clip_grad.dist.all_reduce") as reduce:
            result = _all_reduce_offloaded(value, dist.ReduceOp.SUM, group)
        self.assertIs(result, value)
        reduce.assert_called_once_with(value, op=dist.ReduceOp.SUM, group=group)

    def test_accelerator_backends_stage_cpu_tensor_and_restore_device(self):
        """NCCL/HCCL staging uses the rank's current accelerator."""
        for backend, device_type in (("nccl", "cuda"), ("hccl", "npu")):
            with self.subTest(backend=backend):
                value = MagicMock(spec=torch.Tensor)
                value.device = torch.device("cpu")
                staged = value.to.return_value
                group = object()
                module = SimpleNamespace(current_device=lambda: 3)
                with patch("hyper_parallel.core.utils.clip_grad.dist.get_backend", return_value=backend), \
                        patch("hyper_parallel.core.utils.clip_grad.torch.get_device_module", return_value=module), \
                        patch("hyper_parallel.core.utils.clip_grad.dist.all_reduce") as reduce:
                    result = _all_reduce_offloaded(value, dist.ReduceOp.MAX, group)
                value.to.assert_called_once_with(torch.device(device_type, 3))
                reduce.assert_called_once_with(staged, op=dist.ReduceOp.MAX, group=group)
                staged.to.assert_called_once_with(torch.device("cpu"))
                self.assertIs(result, staged.to.return_value)

    def test_device_resident_tensor_keeps_fast_path(self):
        """Accelerator inputs need neither backend lookup nor staging."""
        value = MagicMock(spec=torch.Tensor)
        value.device = torch.device("cuda", 1)
        value.to.return_value = value
        with patch("hyper_parallel.core.utils.clip_grad.dist.get_backend") as backend, \
                patch("hyper_parallel.core.utils.clip_grad.dist.all_reduce") as reduce:
            self.assertIs(_all_reduce_offloaded(value, dist.ReduceOp.SUM, None), value)
        backend.assert_not_called()
        reduce.assert_called_once_with(value, op=dist.ReduceOp.SUM, group=None)

    def test_norm_variants_preserve_cpu_result(self):
        """Finite, L0 and infinite norms all use the offload-aware reduction."""
        group = object()
        mesh = SimpleNamespace(get_group=lambda dim: group)
        key = (id(mesh), (0,))
        grads = [torch.tensor([3.0, 4.0])]
        for order, expected in ((2.0, 5.0), (1.0, 7.0), (0.0, 2.0), (float("inf"), 4.0),
                                (-float("inf"), 3.0)):
            with self.subTest(order=order), \
                    patch("hyper_parallel.core.utils.clip_grad._all_reduce_offloaded",
                          side_effect=lambda tensor, *args: tensor) as reduce, \
                    patch("hyper_parallel.core.utils.clip_grad.dist.get_process_group_ranks", return_value=[0, 1]):
                result = _get_total_norm({key: grads}, order, {id(mesh): mesh}, torch.device("cpu"), grads, [key])
                self.assertEqual(result.device.type, "cpu")
                self.assertAlmostEqual(float(result), expected)
                reduce.assert_called_once()

    def test_empty_local_shard_still_enters_reduction(self):
        """A rank without gradients contributes an identity instead of skipping."""
        group = object()
        mesh = SimpleNamespace(get_group=lambda dim: group)
        key = (id(mesh), (0,))
        with patch("hyper_parallel.core.utils.clip_grad._all_reduce_offloaded",
                   side_effect=lambda tensor, *args: tensor) as reduce, \
                patch("hyper_parallel.core.utils.clip_grad.dist.get_process_group_ranks", return_value=[0, 1]):
            result = _get_total_norm({key: []}, 2.0, {id(mesh): mesh}, torch.device("cpu"), [], [])
        self.assertEqual(float(result), 0.0)
        reduce.assert_called_once()
