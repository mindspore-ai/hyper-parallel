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
"""Independent CPU oracle for Hybrid rank order and convolution halos."""
import copy
from datetime import timedelta

import torch
import torch.distributed as dist

import hyper_parallel as hp
from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_hybrid import KimiDeltaAttentionLayerHybridCP


def test_kda_hybrid_layout() -> None:
    """Match the serial layer, including halo and replicated parameter gradients."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = hp.init_device_mesh("cpu", (size,), mesh_dim_names=("cp",))
    torch.manual_seed(412)
    reference = KimiDeltaAttention(hidden_size=8, num_heads=4, num_v_heads=4,
                                   head_k_dim=4, head_v_dim=4, conv_kernel_size=3,
                                   chunk_size=4).float().train()
    model = copy.deepcopy(reference)
    executor = KimiDeltaAttentionLayerHybridCP(model, mesh, ulysses_degree=2,
                                               chunk_size=4, backend="eager")
    torch.manual_seed(523)
    full = torch.randn(1, 32, 8, requires_grad=True)
    dout = torch.randn_like(full)
    local = full.detach().chunk(size, dim=1)[rank].contiguous().requires_grad_()
    expected = reference(full)
    actual = executor(local)
    actual.backward(dout.chunk(size, dim=1)[rank].contiguous())
    expected.backward(dout)
    torch.testing.assert_close(actual, expected.detach().chunk(size, dim=1)[rank], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(local.grad, full.grad.chunk(size, dim=1)[rank], atol=2e-6, rtol=2e-5)
    for (_, parameter), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-6, rtol=2e-5)
    dist.destroy_process_group()
