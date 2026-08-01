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
"""Unit tests for the fully_shard local-tensor extension protocol."""

import os
import unittest
from types import SimpleNamespace

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.platform.torch.fully_shard.extension import (
    FSDPGatherContext,
    fsdp_post_all_gather,
    fsdp_pre_all_gather,
    fsdp_shard_tensor,
    fsdp_to_dtensor,
)


class _ExtensionDTensor(torch.Tensor):
    """DTensor-compatible wrapper used by the extension protocol double."""

    def __new__(cls, local_tensor, mesh, placements):
        result = torch.Tensor._make_subclass(cls, local_tensor, local_tensor.requires_grad)
        result._layout = SimpleNamespace(mesh=mesh, alias_placements=placements)
        return result

    @property
    def layout(self):
        return self._layout

    def to_local(self):
        return self.as_subclass(torch.Tensor)


class _ExtensionTensor(torch.Tensor):
    """Small protocol double that owns two physical communication tensors."""

    _hp_fsdp_extension = True

    def __new__(cls, post_result=None):
        del post_result
        return torch.Tensor._make_subclass(cls, torch.ones(2), False)

    def __init__(self, post_result=None):
        self.post_result = post_result or self

    def hp_fsdp_to_dtensor(self, mesh, placements):
        return _ExtensionDTensor(self, mesh, placements)

    def fsdp_pre_all_gather(self, context):
        return (torch.ones(2), torch.ones(3)), {"phase": context.phase}

    def fsdp_post_all_gather(self, tensors, metadata, *, out=None):
        self.post_args = (tensors, metadata, out)
        return out if out is not None else self.post_result, {
            "physical_tensor_count": len(tensors)
        }


class TestFullyShardExtension(unittest.TestCase):
    """Exercise protocol validation without a process group or device mesh."""

    def test_pre_all_gather_returns_all_physical_tensors(self):
        extension = _ExtensionTensor()
        context = FSDPGatherContext("forward", True, "model.weight")

        tensors, metadata = fsdp_pre_all_gather(extension, context)

        self.assertEqual([tensor.numel() for tensor in tensors], [2, 3])
        self.assertEqual(metadata, {"phase": "forward"})

    def test_post_all_gather_preserves_extension_contract(self):
        extension = _ExtensionTensor()
        outputs = (torch.ones(4), torch.ones(6))
        out = _ExtensionTensor()

        result, inner_tensor = fsdp_post_all_gather(
            extension,
            outputs,
            {"x": 1},
            out=out,
        )

        self.assertIs(result, out)
        self.assertEqual(extension.post_args, (outputs, {"x": 1}, out))
        self.assertEqual(inner_tensor, {"physical_tensor_count": 2})

    def test_post_all_gather_requires_unsharded_and_inner_tensor_tuple(self):
        extension = _ExtensionTensor()
        extension.fsdp_post_all_gather = lambda *args, **kwargs: extension

        with self.assertRaisesRegex(ValueError, "unsharded_tensor, inner_tensor"):
            fsdp_post_all_gather(extension, (torch.ones(1),), None)

    def test_to_dtensor_requires_layout_and_to_local(self):
        extension = _ExtensionTensor()
        extension.hp_fsdp_to_dtensor = lambda mesh, placements: object()

        with self.assertRaisesRegex(ValueError, "torch.Tensor subclass"):
            fsdp_to_dtensor(extension, "mesh", ())

    def test_to_dtensor_rejects_non_tensor_wrapper(self):
        extension = _ExtensionTensor()
        extension.hp_fsdp_to_dtensor = lambda mesh, placements: SimpleNamespace(
            layout=object(),
            to_local=lambda: extension,
        )

        with self.assertRaisesRegex(ValueError, "torch.Tensor subclass"):
            fsdp_to_dtensor(extension, "mesh", ())

    def test_hp_owned_logical_sharding_does_not_require_gradient_wrapper(self):
        tensor = _ExtensionTensor()

        shard = fsdp_shard_tensor(
            tensor,
            shard_dim=0,
            shard_rank=0,
            shard_world_size=2,
        )
        wrapper = fsdp_to_dtensor(shard, "mesh", ("shard",))

        self.assertIsInstance(shard, _ExtensionTensor)
        self.assertTrue(callable(wrapper.to_local))

    def test_post_all_gather_rejects_out_replacement(self):
        extension = _ExtensionTensor()
        extension.fsdp_post_all_gather = lambda *args, **kwargs: (
            _ExtensionTensor(),
            None,
        )

        with self.assertRaisesRegex(ValueError, "return that exact out"):
            fsdp_post_all_gather(
                extension,
                (torch.ones(1),),
                None,
                out=_ExtensionTensor(),
            )


if __name__ == "__main__":
    unittest.main()
