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
"""Layout and autograd coverage for consecutive pipelined mHC post calls."""

from contextlib import ExitStack
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.modules.mhc import pipelined_mhc_post
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.common.mark_utils import arg_mark


def _post_formula(output: torch.Tensor, residual: torch.Tensor,
                  post: torch.Tensor, combine: torch.Tensor) -> torch.Tensor:
    """Model the native binding's local tensor result with an independent formula.

    Args:
        output: Sublayer output.
        residual: Residual streams.
        post: Output mixing coefficients.
        combine: Residual mixing coefficients.

    Returns:
        Local mixed streams, without a Python DTensor wrapper.
    """
    output, residual, post, combine = (
        value.to_local() if isinstance(value, DTensor) else value
        for value in (output, residual, post, combine)
    )
    return (post.unsqueeze(-1) * output.unsqueeze(-2)
            + torch.einsum("bsij,bsic->bsjc", combine, residual))


class TestPipelinedMhcPost(unittest.TestCase):
    """Keep local fused execution inside the decoder's DTensor boundaries."""

    def tearDown(self) -> None:
        """Remove backend-less meshes before other distributed tests run."""
        clear_device_mesh_cache()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_consecutive_sequence_shards_keep_layout_and_gradients(self) -> None:
        """Feature: CP and TP/SP layouts across consecutive mHC post calls.

        Description: Compare eager and mocked CANN calls on DP/CP/TP local shards.
        Expectation: Both calls keep global geometry, local values, and all input gradients.
        """
        generator = torch.Generator().manual_seed(1560)
        # Keep this CPU boundary test independent of FP32 reduction-order
        # differences between full-sequence and sliced contractions.
        values = [torch.randn(shape, generator=generator, dtype=torch.float64) for shape in (
            (4, 16, 4), (4, 16, 4, 4), (4, 16, 4), (4, 16, 4, 4),
        )]
        for dp_size, cp_size, tp_size in ((1, 2, 1), (1, 1, 2), (2, 2, 2), (1, 1, 1)):
            world_size = dp_size * cp_size * tp_size
            for rank in range(world_size):
                for fused in (True, False):
                    with self.subTest(dp=dp_size, cp=cp_size, tp=tp_size, rank=rank, fused=fused):
                        ranks = tuple((index - rank) % world_size for index in range(world_size))
                        mesh = init_device_mesh(
                            "cpu", (dp_size, cp_size, tp_size), mesh_dim_names=("dp", "cp", "tp"),
                            rank_list=ranks, init_backend=False,
                        )
                        batch_length = values[0].shape[0] // dp_size
                        sequence_length = values[0].shape[1] // (cp_size * tp_size)
                        batch_start = (rank // (cp_size * tp_size)) * batch_length
                        sequence_start = (rank % (cp_size * tp_size)) * sequence_length
                        interval = (slice(batch_start, batch_start + batch_length),
                                    slice(sequence_start, sequence_start + sequence_length))
                        local = [value[interval].clone().requires_grad_() for value in values]
                        placements = (Shard(0), Shard(1), Shard(1))
                        inputs = [DTensor.from_local(value, mesh, placements) for value in local]
                        kernel = Mock(side_effect=lambda residual, combine, output, post:
                                      _post_formula(output, residual, post, combine))
                        ops = ModuleType("cann_ops_transformer.ops")
                        ops.mhc_post = kernel
                        package = ModuleType("cann_ops_transformer")
                        package.ops = ops

                        with ExitStack() as stack:
                            if fused:
                                stack.enter_context(patch.dict(sys.modules, {
                                    "cann_ops_transformer": package, "cann_ops_transformer.ops": ops,
                                }))
                                stack.enter_context(patch.object(
                                    torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu")),
                                ))
                            first = pipelined_mhc_post(*inputs)
                            second = pipelined_mhc_post(inputs[0], first, inputs[2], inputs[3])

                        reference_inputs = [value.clone().requires_grad_() for value in values]
                        reference_first = _post_formula(*reference_inputs)
                        reference_second = _post_formula(
                            reference_inputs[0], reference_first, reference_inputs[2], reference_inputs[3],
                        )
                        for actual, expected in ((first, reference_first), (second, reference_second)):
                            self.assertIsInstance(actual, DTensor)
                            self.assertIs(actual.device_mesh, mesh)
                            self.assertEqual(actual.placements, placements)
                            self.assertEqual(actual.shape, expected.shape)
                            torch.testing.assert_close(actual.to_local(), expected[interval])
                        second.to_local().square().sum().backward()
                        reference_second[interval].square().sum().backward()
                        for value, reference in zip(local, reference_inputs):
                            torch.testing.assert_close(value.grad, reference.grad[interval])
                        self.assertEqual(kernel.call_count, 2 if fused else 0)
                        for call in kernel.call_args_list:
                            for argument in call.args:
                                self.assertNotIsInstance(argument, DTensor)
