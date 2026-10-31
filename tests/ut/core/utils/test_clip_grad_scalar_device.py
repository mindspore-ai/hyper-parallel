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
"""Norm-scalar reductions pick their device from the group's backend.

FSDP CPU offload keeps parameters, their sharded gradients and the per-rank norm
scalars they produce on the CPU.  clip_grad_norm_ therefore builds each rank's norm
scalar on the CPU, but the accelerator backends (hccl/nccl) have no CPU backend
registered, so the first optimizer step died with ``No backend type associated with
device type cpu``.  The reduction helper moves such a scalar to the accelerator only
when the group cannot reduce CPU tensors itself; a CPU-capable backend (gloo/mpi) must
keep it in place, because moving it there fails the other way round.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.utils import clip_grad as clip_grad_mod
from tests.common.mark_utils import arg_mark


class TestClipGradScalarReductionDevice(unittest.TestCase):
    """Backend-aware device choice for norm-scalar reductions."""

    def _reduce_scalar(self, backend: str, tensor: torch.Tensor | None = None):
        """Reduce *tensor* over a group reporting *backend*.

        Returns the tensor handed to ``all_reduce`` plus the accelerator-device mock.
        """
        scalar = torch.tensor(3.0) if tensor is None else tensor
        group = MagicMock(name="group")
        accelerator = MagicMock(return_value=torch.device("cpu"))
        with patch.object(clip_grad_mod.dist, "get_backend", return_value=backend), \
                patch.object(clip_grad_mod, "_accelerator_device", accelerator), \
                patch.object(clip_grad_mod.dist, "all_reduce") as all_reduce:
            result = clip_grad_mod._reduce_scalar_norm(  # pylint: disable=W0212
                scalar, clip_grad_mod.dist.ReduceOp.SUM, group,
            )
        return result, all_reduce.call_args[0][0], accelerator

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cpu_capable_backend_keeps_scalar_on_cpu(self):
        """
        Feature: FSDP CPU offload under a Gloo group.

        Description: Gloo reduces CPU tensors natively, so moving the scalar to an
        accelerator would be both pointless and wrong on a CPU-only rank.

        Expectation: the scalar is reduced in place.
        """
        result, reduced, accelerator = self._reduce_scalar("gloo")
        self.assertTrue(result.is_cpu)
        self.assertTrue(reduced.is_cpu)
        accelerator.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_accelerator_backend_moves_the_scalar_and_copies_back(self):
        """
        Feature: FSDP CPU offload under an HCCL group.

        Description: hccl has no CPU backend, so the scalar has to travel to the
        accelerator for the collective while the caller keeps its CPU tensor.

        Expectation: the accelerator device is used and the result stays on the CPU.
        """
        result, reduced, accelerator = self._reduce_scalar("hccl")
        accelerator.assert_called_once()
        # The caller keeps a CPU tensor: the result is copied back after the move.
        self.assertTrue(result.is_cpu)
        self.assertEqual(float(result), float(reduced))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_composite_backend_with_a_cpu_capable_member_stays_in_place(self):
        """
        Feature: multi-backend process groups.

        Description: ``dist.get_backend`` reports composite strings such as
        ``cpu:gloo,cuda:nccl``; the CPU-capable member must win.

        Expectation: no move for the composite group.
        """
        _, reduced, accelerator = self._reduce_scalar("cpu:gloo,cuda:nccl")
        self.assertTrue(reduced.is_cpu)
        accelerator.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_backend_leaves_the_scalar_alone(self):
        """
        Feature: backends without a known device requirement.

        Description: An unrecognized backend (for example a third-party or legacy
        name) must not trigger a device move the framework cannot justify.

        Expectation: the scalar is reduced in place.
        """
        _, reduced, accelerator = self._reduce_scalar("unknown_backend")
        self.assertTrue(reduced.is_cpu)
        accelerator.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unqueryable_backend_keeps_the_historical_path(self):
        """
        Feature: groups whose backend cannot be queried.

        Description: Test doubles and custom process-group wrappers make
        ``dist.get_backend`` raise; that must not turn a working reduction into an
        exception.

        Expectation: the tensor is reduced in place and no device is touched.
        """
        scalar = torch.tensor(3.0)
        with patch.object(clip_grad_mod.dist, "get_backend", side_effect=ValueError("bad group")), \
                patch.object(clip_grad_mod, "_accelerator_device") as accelerator, \
                patch.object(clip_grad_mod.dist, "all_reduce") as all_reduce:
            result = clip_grad_mod._reduce_scalar_norm(  # pylint: disable=W0212
                scalar, clip_grad_mod.dist.ReduceOp.SUM, MagicMock(name="group"),
            )
        self.assertIs(result, scalar)
        self.assertTrue(result.is_cpu)
        accelerator.assert_not_called()
        all_reduce.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backend_device_type_mapping(self):
        """
        Feature: backend to device-type mapping.

        Description: The mapping drives the move decision, so the accelerator names
        and the CPU-capable set are pinned.

        Expectation: hccl -> npu, nccl -> cuda, gloo/mpi -> None.
        """
        group = MagicMock(name="group")
        for backend, expected in (("hccl", "npu"), ("nccl", "cuda"), ("gloo", None), ("mpi", None)):
            with patch.object(clip_grad_mod.dist, "get_backend", return_value=backend):
                self.assertEqual(
                    clip_grad_mod._backend_device_type(group),  # pylint: disable=W0212
                    expected,
                    backend,
                )


if __name__ == "__main__":
    unittest.main()
