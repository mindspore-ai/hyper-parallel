# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test vpp"""
import numpy as np
import torch
import torch.distributed as dist
from hyper_parallel import DTensor, PipelineStage, ScheduleInterleaved1F1B, init_device_mesh, manual_seed
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.dtensor.random import is_rng_supported_mesh
from tests.torch.utils import _DEVICE_TYPE, to_device
from .simple_mlp import (
    SimpleMLP,
    model_split_manual,
    run_standalone,
    init_hccl,
    get_stage_index,
    get_rank_list,
)


def _pp_domain_dtensor_rng_smoke() -> None:
    """Per PP-domain ``DeviceMesh``: ``manual_seed`` then one sharded ``randn_like`` on ``DTensor``.

    Uses two domains of two ranks each (ranks ``{0,1}`` and ``{2,3}``) when ``world_size==4``,
    mirroring different parallel seeds per domain (PyTorch DTensor PP guidance).
    """
    world_size = dist.get_world_size()
    if world_size != 4:
        return
    num_pp_domains = 2
    stage_ranks = tuple(get_rank_list(num_pp_domains))
    domain_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(len(stage_ranks),),
        mesh_dim_names=("pp_domain",),
        rank_list=stage_ranks,
        init_backend=False,
    )
    if not is_rng_supported_mesh(domain_mesh):
        return
    pp_domain_id = get_stage_index(num_pp_domains)
    parallel_seed = 31_415 + pp_domain_id * 271_828
    manual_seed(parallel_seed, domain_mesh)

    local = torch.zeros((1, 8), dtype=torch.float32, device=_DEVICE_TYPE)
    dt = DTensor.from_local(local, domain_mesh, [Shard(0)])
    out = torch.randn_like(dt)
    assert isinstance(out, DTensor), f"expected DTensor, got {type(out)}"
    assert tuple(out.shape) == (1 * len(stage_ranks), 8)
    assert torch.isfinite(out.to_local()).all()


def run_parallel(micro_batch_num):
    """
    Feature: PipelineParallel.
    Description: Test simple mlp net; before PP run, smoke-test per-domain ``manual_seed`` and
        ``torch.randn_like`` on a sharded ``DTensor`` (two PP domains when world size is 4).
    Expectation: Run success.
    """
    init_hccl()
    _pp_domain_dtensor_rng_smoke()

    # pp config
    num_stages = 4
    stage_index = get_stage_index(num_stages)
    local_batch_size = micro_batch_num

    model0 = SimpleMLP(8, 16, 16)
    model1 = SimpleMLP(8, 16, 16)
    model_split_manual(model0, stage_index, 8)
    model_split_manual(model1, stage_index + 4, 8)

    # pp stage
    device = torch.device(_DEVICE_TYPE)
    pipeline_stage0 = PipelineStage(model0, stage_index , num_stages + 4, device)
    pipeline_stage1 = PipelineStage(model1, stage_index + 4, num_stages + 4, device)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    # input
    local_hidden_size = 16
    x = to_device(torch.ones(local_batch_size, local_hidden_size, dtype=torch.float32), _DEVICE_TYPE)

    # train config
    epochs = 1
    for _ in range(epochs):
        if stage_index == 0:
            loss = schedule.run(x)
        else:
            loss = schedule.run()
    return loss, (model0, model1)


def test_vpp():
    """
    Feature: VPP.
    Description: Test simple mlp net + vpp.
    Expectation: Run success.
    """
    micro = 8
    standalone_model = SimpleMLP(8, 16, 16)
    standalone_loss = run_standalone(micro, standalone_model)
    pp_loss, pp_model = run_parallel(micro)
    stage_index = get_stage_index(4)
    if stage_index == 3:
        assert np.allclose(standalone_loss.cpu().detach().numpy(), pp_loss[0].cpu().detach().numpy())
    assert np.allclose(standalone_model.mlp_layers[str(stage_index)].weight.cpu().detach().numpy(),
                       pp_model[0].mlp_layers[str(stage_index)].weight.cpu().detach().numpy())
    assert np.allclose(standalone_model.mlp_layers[str(stage_index+4)].weight.cpu().detach().numpy(),
                       pp_model[1].mlp_layers[str(stage_index+4)].weight.cpu().detach().numpy())


if __name__ == "__main__":
    test_vpp()
