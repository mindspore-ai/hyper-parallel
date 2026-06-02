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
"""Unit tests for ``Platform.is_collective_op``.

Note:
    The ``keep_collectives_policy`` mapping test is intentionally disabled below.
    Importing ``hyper_parallel.core.activation_checkpoint`` from this module — which is
    collected before the MindSpore platform UTs and under PyTorch — makes it the first
    importer of that module, which bakes ``CheckpointWrapper``'s base class to the
    PyTorch backend at import time and breaks
    ``tests/ut/platform/mindspore/test_activation_wrapper.py`` (its MindSpore model then
    receives a PyTorch-based wrapper, so ``cells_and_names()`` no longer registers it).
    Re-enable once the wrapper resolves its activation-wrapper base lazily per backend.
    The ``is_collective_op`` tests below do not import activation checkpointing and stay
    enabled.
"""
import types
import unittest

from hyper_parallel.platform import get_platform

platform = get_platform()
_IS_MINDSPORE = "MindSpore" in type(platform).__name__
_IS_TORCH = "Torch" in type(platform).__name__


def _op(name):
    """Stand-in dispatched op carrying a ``.name`` (mirrors a primitive/OpOverload)."""
    return types.SimpleNamespace(name=name)


# Disabled until ``CheckpointWrapper`` resolves its base lazily (see module docstring):
#
# import hyper_parallel.core.activation_checkpoint.activation_checkpoint as ackpt
# from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, keep_collectives_policy
#
# class TestKeepCollectivesPolicy(unittest.TestCase):
#     """``keep_collectives_policy`` maps collective→MUST_SAVE, compute→PREFER_RECOMPUTE."""
#
#     def test_collective_maps_to_must_save(self):
#         with mock.patch.object(ackpt.plat, "is_collective_op", return_value=True):
#             policy = keep_collectives_policy(None, object())
#         assert policy == CheckpointPolicy.MUST_SAVE
#
#     def test_compute_maps_to_prefer_recompute(self):
#         with mock.patch.object(ackpt.plat, "is_collective_op", return_value=False):
#             policy = keep_collectives_policy(None, object())
#         assert policy == CheckpointPolicy.PREFER_RECOMPUTE


@unittest.skipUnless(_IS_MINDSPORE, "MindSpore-specific is_collective_op test")
class TestMindSporeIsCollectiveOp(unittest.TestCase):
    """MindSpore primitive-name recognition for ``is_collective_op``."""

    def test_collectives_recognised(self):
        for name in ("InnerCommAllToAllV", "InnerCommAllReduce", "InnerCommAllGather",
                     "AllReduce", "AllGather", "ReduceScatter", "AllToAll", "AllToAllV",
                     "Broadcast", "NeighborExchange"):
            assert platform.is_collective_op(_op(name)) is True, \
                f"{name} should be recognised as a collective"

    def test_compute_ops_rejected(self):
        # Names sharing a token with a collective must NOT match (no false positives).
        for name in ("MatMul", "MatMulExt", "Add", "Gather", "GatherV2", "ReduceSum",
                     "ScatterUpdate", "BroadcastTo", "Reshape", "Softmax"):
            assert platform.is_collective_op(_op(name)) is False, \
                f"{name} must NOT be recognised as a collective"


@unittest.skipUnless(_IS_TORCH, "PyTorch-specific is_collective_op test")
class TestTorchIsCollectiveOp(unittest.TestCase):
    """PyTorch functional-collective recognition for ``is_collective_op``."""

    def test_functional_collectives_recognised(self):
        for name in ("_c10d_functional::all_to_all_single.default",
                     "_c10d_functional::all_reduce.default",
                     "_c10d_functional::all_gather_into_tensor.default",
                     "_c10d_functional::reduce_scatter_tensor.default",
                     "c10d::allreduce_.default"):
            assert platform.is_collective_op(_op(name)) is True, \
                f"{name} should be recognised as a collective"

    def test_aten_ops_rejected(self):
        for name in ("aten::mm.default", "aten::add.Tensor", "aten::gelu.default",
                     "aten::gather.default", "aten::sum.default"):
            assert platform.is_collective_op(_op(name)) is False, \
                f"{name} must NOT be recognised as a collective"


if __name__ == "__main__":
    unittest.main()
