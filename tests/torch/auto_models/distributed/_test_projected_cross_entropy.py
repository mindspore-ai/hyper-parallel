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
"""Check rank-order chunk gathering against a full-vocabulary reference."""

from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather
from torch.utils.checkpoint import DefaultDeviceType
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel import ColwiseParallel, parallelize_module
from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.loss_parallel import loss_parallel
from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel


@pytest.fixture(autouse=True, scope="module")
def cpu_checkpoint_device() -> Iterator[None]:
    """CPU-only ranks must not initialize an installed accelerator during recomputation."""
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        yield
    finally:
        DefaultDeviceType.set_device_type(previous)


def setup_module() -> None:
    """Initialize only a CPU group; this worker never acquires NPU devices."""
    dist.init_process_group("gloo")


def teardown_module() -> None:
    """Release the process group after comparisons."""
    dist.destroy_process_group()


class SequenceGatherHead(nn.Module):
    """Gather SP states before applying one vocabulary shard of the output head."""

    def __init__(self, weight: torch.Tensor) -> None:
        """Own one vocabulary shard of the reference projection weight."""
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Use a differentiable gather, including its matching backward reduction."""
        gathered = torch.cat(all_gather(hidden.contiguous()), dim=1)
        return gathered @ self.weight.t()


class TestParallelCrossEntropy:
    """Run the essential CE and projection contracts in one two-rank launch."""

    @pytest.mark.parametrize("tp_size,sequence_parts", ((1, 1), (2, 1), (2, 2)))
    def test_sequence_and_vocab_shards(self, tp_size: int, sequence_parts: int) -> None:
        """Uneven vocab shards, tail/empty chunks and SP preserve loss and gradients."""
        rank = dist.get_rank() % tp_size
        device_mesh = init_device_mesh("cpu", (2 // tp_size, tp_size), mesh_dim_names=("dp", "tp"))
        torch.manual_seed(18)
        hidden = torch.randn(2, 10, 5)
        weight = torch.randn(13, 5)
        labels = torch.tensor([[-100, -100, 2, 4, -100, -100, -100, 8, 11, 12]])
        labels = torch.cat((labels, labels.flip(1)), dim=0)
        for targets in (labels, torch.full_like(labels, -100)):
            start, stop = (rank * 5, (rank + 1) * 5) if sequence_parts == 2 else (0, 10)
            local = hidden[:, start:stop].clone().requires_grad_()
            local_weight = weight.chunk(tp_size, dim=0)[rank]
            if sequence_parts == 2:
                head = SequenceGatherHead(local_weight)
            else:
                head = nn.Linear(5, local_weight.shape[0], bias=False)
                head.load_state_dict({"weight": local_weight})
            tp_mesh = device_mesh["tp"]
            actual = projected_cross_entropy(local, targets, head=head, loss_fn=causal_lm_loss_parallel,
                                              vocab_size=13, chunk_size=4,
                                              sequence_parallel_size=sequence_parts, tp_mesh=tp_mesh)
            reference_hidden = hidden.clone().requires_grad_()
            reference_weight = weight.clone().requires_grad_()
            expected = ForCausalLMLoss(logits=reference_hidden @ reference_weight.t(), labels=None,
                                       vocab_size=13, shift_labels=targets,
                                       num_items_in_batch=(targets != -100).sum().clamp_min(1))
            torch.testing.assert_close(actual, expected)
            (actual * 0.37).backward()
            (expected * 0.37).backward()
            if tp_size > 1 and sequence_parts == 1:
                # Without SP's gather backward, the caller's replicated-input boundary sums vocabulary contributions.
                dist.all_reduce(local.grad, group=tp_mesh.get_group())
            torch.testing.assert_close(local.grad, reference_hidden.grad[:, start:stop],
                                       atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(head.weight.grad, reference_weight.grad.chunk(tp_size, dim=0)[rank],
                                       atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("sequence_parts,shard_logits", ((1, True), (2, True), (2, False)))
    def test_colwise_head_matches_unchunked_parallel_path(self, sequence_parts: int, shard_logits: bool) -> None:
        """Preserve real ColwiseParallel hooks during forward and recomputation."""
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(31)
        hidden = torch.randn(2, 10, 5)
        targets = torch.randint(0, 16, (2, 10))
        targets[:, :2] = -100
        weight = torch.randn(16, 5)
        losses, input_gradients, weight_gradients = [], [], []
        for chunked in (False, True):
            head = nn.Linear(5, 16, bias=False)
            head.load_state_dict({"weight": weight})
            head = parallelize_module(head, mesh, ColwiseParallel(
                input_layouts=Shard(1) if sequence_parts > 1 else Replicate(),
                output_layouts=Shard(-1) if shard_logits else Replicate(),
                use_local_output=True,
            ))
            local = hidden.chunk(2, dim=1)[dist.get_rank()] if sequence_parts > 1 else hidden
            local = local.clone().requires_grad_()
            loss_fn = causal_lm_loss_parallel if shard_logits else ForCausalLMLoss
            if chunked:
                loss = projected_cross_entropy(local, targets, head=head, loss_fn=loss_fn,
                                               vocab_size=16, chunk_size=4,
                                               sequence_parallel_size=sequence_parts,
                                               tp_mesh=mesh if shard_logits else None)
            else:
                with loss_parallel(mesh=mesh):
                    loss = loss_fn(logits=head(local), labels=None, vocab_size=16,
                                   shift_labels=targets, num_items_in_batch=(targets != -100).sum())
            loss.backward()
            losses.append(loss.detach().reshape(()))
            input_gradients.append(local.grad)
            weight_gradients.append(head.weight.grad)
        torch.testing.assert_close(losses[0], losses[1])
        torch.testing.assert_close(input_gradients[0], input_gradients[1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(weight_gradients[0], weight_gradients[1], atol=1e-6, rtol=1e-5)
