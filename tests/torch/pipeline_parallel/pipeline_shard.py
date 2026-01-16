# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel import DTensor, Layout, shard
from hyper_parallel import PipelineStage
from hyper_parallel import ScheduleInterleaved1F1B
from .simple_mlp import SimpleMLP, model_split_manual, run_standalone, init_hccl, get_stage_index, get_rank_list


def run_parallel(micro_batch_num):
    """
    Feature: PipelineParallel + Tensor parallel.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    init_hccl()

    # pp config
    num_stages = 4
    local_batch_size = micro_batch_num
    rank_list = get_rank_list(num_stages)
    stage_index = get_stage_index(num_stages)
    layout = Layout((1, 2), ("dp", "tp"), rank_list=rank_list)

    sharding_plan = {"parameter": {"mlp_layers.0.weight": layout("None", "tp"),
                                    "mlp_layers.1.weight": layout("tp", "None"),
                                    "mlp_layers.2.weight": layout("None", "tp"),
                                    "mlp_layers.3.weight": layout("tp", "None"),
                                    "mlp_layers.4.weight": layout("None", "tp"),
                                    "mlp_layers.5.weight": layout("tp", "None"),
                                    "mlp_layers.6.weight": layout("None", "tp"),
                                    "mlp_layers.7.weight": layout("tp", "None")}}
    local_hidden_size = 16

    model0 = SimpleMLP(8, local_hidden_size, local_hidden_size)
    model0 = shard(model0, sharding_plan)
    model1 = SimpleMLP(8, local_hidden_size, local_hidden_size)
    model1 = shard(model1, sharding_plan)
    model_split_manual(model0, stage_index, 8)
    model_split_manual(model1, stage_index + 4, 8)

    # pp stage
    device = torch.device("npu")
    pipeline_stage0 = PipelineStage(model0, stage_index, num_stages + 4, device)
    pipeline_stage1 = PipelineStage(model1, stage_index + 4, num_stages + 4, device)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    # input
    local_batch_size = 8
    local_hidden_size = 16
    x = torch.ones(local_batch_size, local_hidden_size, dtype=torch.float32).npu()
    d_x = DTensor.from_local(x, layout("dp", "None"))

    # train config
    epochs = 1
    for _ in range(epochs):
        if stage_index == 0:
            loss = schedule.run(d_x)
        else:
            loss = schedule.run()
    return loss, (model0, model1)


def test_vpp_shard():
    """
    Feature: VPP.
    Description: Test simple mlp net + vpp.
    Expectation: Run success.
    """
    micro = 8
    standalone_model = SimpleMLP(8, 16, 16).npu()
    standalone_loss = run_standalone(micro, standalone_model)
    pp_loss, pp_model = run_parallel(micro)
    stage_index = get_stage_index(4)
    if stage_index == 3:
        assert np.allclose(standalone_loss.cpu().detach().numpy(), pp_loss[0].to_local().cpu().detach().numpy())
    assert np.allclose(standalone_model.mlp_layers[str(stage_index)].weight.grad.cpu().detach().numpy()[:8, :8],
                       pp_model[0].mlp_layers[str(stage_index)].weight.grad.cpu().detach().numpy()[:8, :8])
    assert np.allclose(standalone_model.mlp_layers[str(stage_index+4)].weight.grad.cpu().detach().numpy()[:8, :8],
                       pp_model[1].mlp_layers[str(stage_index+4)].weight.grad.cpu().detach().numpy()[:8, :8])


if __name__ == "__main__":
    test_vpp_shard()
