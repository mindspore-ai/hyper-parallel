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
"""Hardware-independent tests of Hyper-only DTensor compatibility."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.distributed import _tensor, tensor
from torch.distributed.device_mesh import DeviceMesh as TorchMesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Partial as TorchPartial
from torch.distributed.tensor import Replicate as TorchReplicate
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
    StridedShard,
)
from hyper_parallel.core.utils import clip_grad_norm_ as hyper_clip_grad_norm
from hyper_parallel.integration import torch as integration
from hyper_parallel.integration.torch import DTensorAdapter, clip_grad_norm_, is_dtensor
from hyper_parallel.integration.torch import dtensor as compatibility


def _reference(local: torch.Tensor) -> MagicMock:
    reference = MagicMock(spec=HyperDTensor)
    reference.shape = (4, 3)
    reference.placements = (Shard(0),)
    reference.to_local.return_value = local
    reference.layout = MagicMock()
    reference.layout.tensor_stride = None
    reference.grad = None
    reference.main_grad = None
    return reference


class TestDTensorAdapter(unittest.TestCase):
    """Verify explicit Hyper boundaries without setting up process groups."""

    def test_only_hyper_tensors_enter_adapter(self):
        """Torch DTensors and plain tensors keep their original entry points."""
        self.assertTrue(is_dtensor(_reference(torch.ones(2, 3))))
        for value in (torch.ones(2, 3), MagicMock(spec=TorchDTensor)):
            self.assertFalse(is_dtensor(value))
            with self.assertRaises(TypeError):
                DTensorAdapter(value)

    def test_local_copy_preserves_storage_and_identity(self):
        """Loading writes original local storage without replacing the parameter."""
        local = torch.zeros(2, 3, requires_grad=True)
        reference = _reference(local)
        pointer = local.data_ptr()
        adapter = DTensorAdapter(reference)
        adapter.copy_local_(torch.ones_like(local))
        self.assertIs(adapter.tensor, reference)
        self.assertEqual(local.data_ptr(), pointer)
        torch.testing.assert_close(local, torch.ones_like(local))

    def test_copy_rejects_broadcast_cast_and_distributed_payloads(self):
        """Boundary checks do not silently broadcast, cast or combine DTensor types."""
        adapter = DTensorAdapter(_reference(torch.ones(2, 3)))
        for value in (torch.ones(1, 3), torch.ones(2, 3, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                adapter.copy_local_(value)
        for value in (adapter.tensor, MagicMock(spec=TorchDTensor)):
            with self.assertRaises(TypeError):
                adapter.copy_local_(value)

    def test_from_local_delegates_to_hyper(self):
        """Hyper reconstructs tensors with original shape, mesh and placements."""
        local = torch.ones(2, 3)
        reference = _reference(local)
        for stride in (None, (3, 1)):
            with self.subTest(stride=stride):
                reference.layout.tensor_stride = stride
                with patch.object(HyperDTensor, "from_local") as factory:
                    DTensorAdapter(reference).from_local(local)
                options = {"shape": (4, 3), "run_check": False}
                if stride is not None:
                    options["stride"] = stride
                factory.assert_called_once_with(local, reference.device_mesh, reference.placements, **options)

    def test_local_gradient_prefers_main_grad(self):
        """Hyper's FP32 main gradient takes precedence over ordinary gradients."""
        reference = _reference(torch.ones(2, 3))
        adapter = DTensorAdapter(reference)
        self.assertIsNone(adapter.local_grad())
        grad = torch.ones(2, 3)
        reference.grad = grad
        self.assertIs(adapter.local_grad(), grad)
        reference.grad = _reference(grad)
        self.assertIs(adapter.local_grad(), grad)
        main_grad = torch.full_like(grad, 2)
        reference.main_grad = main_grad
        self.assertIs(adapter.local_grad(), main_grad)
        reference.main_grad = _reference(main_grad)
        self.assertIs(adapter.local_grad(), main_grad)

    def test_local_access_does_not_restrict_hyper_layouts(self):
        """Reading local storage does not impose extra layout restrictions."""
        for case in ("partial", "uneven", "padding"):
            with self.subTest(case=case):
                reference = _reference(torch.ones(2, 3))
                if case == "partial":
                    reference.placements = (Partial(),)
                elif case == "uneven":
                    reference.shape = (5, 3)
                else:
                    reference.to_local.return_value = torch.ones(3, 3)
                self.assertIs(DTensorAdapter(reference).to_local(), reference.to_local())

    def test_import_leaves_torch_api_unchanged(self):
        """The integration neither aliases Torch classes nor wraps native optimizers."""
        self.assertIs(tensor.DTensor, TorchDTensor)
        self.assertIs(_tensor.DTensor, TorchDTensor)
        self.assertFalse(hasattr(integration, "local_update_context"))

    def test_full_load_delegates_to_hyper(self):
        """Hyper owns local slicing and source-rank communication for every call."""
        full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        local = torch.zeros(2, 3)
        reference = _reference(local)
        adapter = DTensorAdapter(reference)
        pointer = local.data_ptr()
        for source_rank in (None, 0):
            with self.subTest(source_rank=source_rank):
                with patch("hyper_parallel.integration.torch.dtensor.hyper_distribute_tensor") as factory:
                    factory.return_value.to_local.return_value = full[2:]
                    adapter.copy_from_full_(full, src_data_rank=source_rank)
                factory.assert_called_once_with(
                    full, reference.device_mesh, reference.placements, src_data_rank=source_rank
                )
                torch.testing.assert_close(local, full[2:])
                self.assertEqual(local.data_ptr(), pointer)

    def test_full_load_rejects_wrong_payload(self):
        """A local shard or distributed input is not a plain full checkpoint payload."""
        adapter = DTensorAdapter(_reference(torch.zeros(2, 3)))
        with self.assertRaises(ValueError):
            adapter.copy_from_full_(torch.ones(2, 3))
        with self.assertRaises(TypeError):
            adapter.copy_from_full_(MagicMock(spec=TorchDTensor))

    def test_clipping_is_original_hyper_function(self):
        """No wrapper changes Hyper's signature, options, return or norm algorithm."""
        self.assertIs(clip_grad_norm_, hyper_clip_grad_norm)


class TestTorchLayoutCompatibility(unittest.TestCase):
    """Check descriptor conversion separately from any external framework."""

    def test_mesh_reuses_existing_groups(self):
        """Translate layout descriptions without creating a new process group."""
        mesh = MagicMock(spec=TorchMesh)
        mesh.device_type = "cpu"
        mesh.mesh = torch.tensor([0, 1])
        mesh.mesh_dim_names = ("dp",)
        groups = [object()]
        mesh.get_all_groups.return_value = groups
        with patch.object(DeviceMesh, "from_group") as convert:
            self.assertIs(compatibility.to_hyper_mesh(mesh), convert.return_value)
        convert.assert_called_once_with(groups, "cpu", mesh=mesh.mesh, mesh_dim_names=("dp",))
        native = MagicMock(spec=DeviceMesh)
        self.assertIs(compatibility.to_hyper_mesh(native), native)
        with self.assertRaises(TypeError):
            compatibility.to_hyper_mesh(object())

    def test_placement_conversion_rejects_partial_and_special_shards(self):
        """Do not erase pending reduction semantics or custom shard ordering."""
        self.assertIsInstance(compatibility.to_hyper_placement(TorchShard(1)), Shard)
        self.assertEqual(compatibility.to_hyper_placement(TorchShard(1)).dim, 1)
        self.assertIsInstance(compatibility.to_hyper_placement(TorchReplicate()), Replicate)

        class SpecialShard(TorchShard):
            """Represent a layout with additional semantics."""

        for placement in (TorchPartial(), SpecialShard(0)):
            with self.subTest(placement=placement), self.assertRaises(NotImplementedError):
                compatibility.to_hyper_placement(placement)

    def test_checkpoint_view_passes_plain_values_through(self):
        """Ordinary tensors and metadata are not reconstructed or copied."""
        for value in (torch.ones(2), 4, "state"):
            self.assertIs(compatibility.to_torch_view(value), value)
            self.assertIs(compatibility.from_torch_view(value), value)

    def test_checkpoint_view_rejects_partial_before_mesh_conversion(self):
        """A pending reduction cannot be represented as an already reduced shard."""
        value = _reference(torch.ones(2, 3))
        value.placements = (Partial(),)
        with patch.object(TorchMesh, "from_group") as convert, self.assertRaises(NotImplementedError):
            compatibility.to_torch_view(value)
        convert.assert_not_called()

    def test_checkpoint_view_rejects_special_placements_before_conversion(self):
        """Unknown ordering must fail rather than become a plain Torch shard."""

        class CustomShard(Shard):
            """Represent shard semantics not supported by checkpoint views."""

        class CustomReplicate(Replicate):
            """Represent replication semantics not supported by checkpoint views."""

        for placement in (StridedShard(0, 2), CustomShard(0), CustomReplicate()):
            with self.subTest(placement=placement):
                value = _reference(torch.ones(2, 3))
                value.placements = (placement,)
                with (
                    patch.object(TorchMesh, "from_group") as mesh,
                    patch.object(TorchDTensor, "from_local") as factory,
                    self.assertRaisesRegex(NotImplementedError, type(placement).__name__),
                ):
                    compatibility.to_torch_view(value)
                mesh.assert_not_called()
                factory.assert_not_called()

    def test_checkpoint_view_shares_local_storage_without_autograd_bridge(self):
        """The Torch factory receives detached storage with the same data pointer."""
        local = torch.ones(2, 3, requires_grad=True)
        value = _reference(local)
        value.device_mesh.device_type = "cpu"
        value.device_mesh.mesh = torch.tensor([0, 1])
        value.device_mesh.mesh_dim_names = ("dp",)
        value.layout.tensor_stride = (3, 1)
        with patch.object(TorchMesh, "from_group") as mesh, patch.object(TorchDTensor, "from_local") as factory:
            compatibility.to_torch_view(value)
        actual = factory.call_args
        self.assertEqual(actual.args[0].data_ptr(), local.data_ptr())
        self.assertFalse(actual.args[0].requires_grad)
        self.assertIs(actual.args[1], mesh.return_value)
        self.assertEqual(actual.kwargs["shape"], (4, 3))
        self.assertEqual(actual.kwargs["stride"], (3, 1))
        self.assertFalse(actual.kwargs["run_check"])

    def test_checkpoint_view_supplies_stride_when_source_omits_it(self):
        """A logical contiguous stride completes shape-only Hyper metadata."""
        value = _reference(torch.ones(2, 3))
        value.device_mesh.device_type = "cpu"
        value.device_mesh.mesh = torch.tensor([0, 1])
        value.device_mesh.mesh_dim_names = ("dp",)
        with patch.object(TorchMesh, "from_group"), patch.object(TorchDTensor, "from_local") as factory:
            compatibility.to_torch_view(value)
        self.assertEqual(factory.call_args.kwargs["shape"], (4, 3))
        self.assertEqual(factory.call_args.kwargs["stride"], (3, 1))
