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
"""Four-rank Gloo oracle for Xing40 CP sequencewise auxiliary loss."""
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.models.xing40.modeling_xing40 import Xing40MoE


EXPERTS, TOP_K, ALPHA = 4, 2, 0.0006
RTOL, ATOL = 1e-11, 1e-12


def _oracle(scores, selected):
    """Independently compute per-sequence counts, load, affinity and objective."""
    counts = functional.one_hot(selected, num_classes=EXPERTS).sum(dim=(1, 2)).to(scores.dtype)
    load = counts / (scores.shape[1] * TOP_K)
    affinity = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).mean(dim=1)
    loss = EXPERTS * (load * affinity).sum(dim=-1).mean()
    return counts, load, affinity, loss


def _scenario(rank, group, forced):
    """Compare one routing scenario with two independent DP references."""
    dp, cp = divmod(rank, 2)
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randn(2, 2, 5, 5, generator=generator, dtype=torch.float64)
    initial = torch.randn(5, EXPERTS, generator=generator, dtype=torch.float64)
    weight = initial.clone().requires_grad_()
    all_forced = torch.tensor([[[[0, 1], [0, 1], [0, 2], [0, 3], [1, 3]],
                                [[3, 2], [3, 2], [3, 1], [3, 0], [2, 0]]],
                               [[[1, 0], [1, 0], [1, 2], [1, 3], [0, 3]],
                                [[2, 3], [2, 3], [2, 1], [2, 0], [3, 0]]]])
    start, end = ((0, 2), (2, 5))[cp]
    local_scores = (inputs[dp, :, start:end] @ weight).sigmoid()
    local_forced = all_forced[dp, :, start:end].reshape(-1, TOP_K) if forced else None
    obj = SimpleNamespace(top_k=TOP_K, num_experts=EXPERTS, aux_loss_coeff=ALPHA,
                          aux_loss_cp_group=group, last_aux_loss=None)
    weights = torch.zeros(2 * (end - start), TOP_K, dtype=torch.float64, requires_grad=True)
    with aux_loss_scale_context():
        output = Xing40MoE._attach_balance_loss(obj, weights, local_scores.reshape(-1, EXPERTS), 2, local_forced)
        task_loss = output.sum() * 0
        bind_aux_loss_scale(task_loss)
        task_loss.backward()
    averaged_grad = weight.grad.clone()
    dist.all_reduce(averaged_grad)
    averaged_grad /= 4  # Simulate the framework's DP x CP parameter gradient average.

    reference_weight = initial.clone().requires_grad_()
    references = []
    for data_parallel in range(2):
        scores = (inputs[data_parallel] @ reference_weight).sigmoid()
        selected = all_forced[data_parallel] if forced else scores.detach().topk(TOP_K, dim=-1, sorted=False).indices
        references.append(_oracle(scores, selected))
    (ALPHA * torch.stack([v[3] for v in references]).mean()).backward()
    torch.testing.assert_close(averaged_grad, reference_weight.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(obj.last_aux_loss, references[dp][3].detach(), rtol=RTOL, atol=ATOL)
    assert not torch.equal(references[dp][0][0], references[dp][0][1]), (
        f"expected counts_equal=False, got=True for dp={dp}"
    )



def test_xing40_cp_sequence_aux_worker() -> None:
    """Exercise real differentiable CP SUM and world gradient averaging."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 4, (
            f"expected world_size=4, got={dist.get_world_size()}"
        )
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        rank = dist.get_rank()
        for forced in (False, True):
            _scenario(rank, groups[rank // 2], forced)
    finally:
        dist.destroy_process_group()
