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
import gc
import os

import numpy as np
import torch
import torch.distributed as dist
from hyper_parallel import (
    DTensor,
    PipelineStage,
    ScheduleInterleaved1F1B,
    init_device_mesh,
    manual_seed,
)
from hyper_parallel.core.activation_checkpoint import swap_wrapper
from hyper_parallel.core.activation_checkpoint.swap import SwapManager
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

_SWAP_REAL_STAGES = 4
_SWAP_LOCAL_CHUNKS = 2
_SWAP_VIRTUAL_STAGES = _SWAP_REAL_STAGES * _SWAP_LOCAL_CHUNKS
_SWAP_LAYERS_PER_CHUNK = 3
_SWAP_TOTAL_LAYERS = _SWAP_VIRTUAL_STAGES * _SWAP_LAYERS_PER_CHUNK
_SWAP_MICROBATCHES = 8
_SWAP_HIDDEN_SIZE = 16
_SWAP_ROWS_PER_MICROBATCH = int(os.environ.get("TORCH_PP_SWAP_ROWS_PER_MICROBATCH", "131072"))


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


def _build_swap_chunk(virtual_stage_index: int, enable_swap: bool) -> torch.nn.Module:
    """Build one VPP chunk with several layers and deterministic identity weights."""
    model = SimpleMLP(_SWAP_TOTAL_LAYERS, _SWAP_HIDDEN_SIZE, _SWAP_HIDDEN_SIZE)
    first_layer = virtual_stage_index * _SWAP_LAYERS_PER_CHUNK
    owned_layers = set(range(first_layer, first_layer + _SWAP_LAYERS_PER_CHUNK))
    for layer_name in list(model.mlp_layers.keys()):
        if int(layer_name) not in owned_layers:
            del model.mlp_layers[layer_name]
    identity = torch.eye(_SWAP_HIDDEN_SIZE, dtype=torch.float32, device=_DEVICE_TYPE)
    with torch.no_grad():
        for layer in model.mlp_layers.values():
            layer.weight.copy_(identity)
    if enable_swap:
        return swap_wrapper(model, group_swap=True)
    return model


def _run_pipeline_swap_phase(
        stage_index: int, enable_swap: bool,
) -> tuple[float, list[torch.Tensor], dict[str, torch.Tensor], int]:
    """Run one no-swap or swap VPP phase and snapshot outputs, gradients, and peak memory."""
    device = torch.device(_DEVICE_TYPE)
    virtual_stage_indices = [
        stage_index + chunk_index * _SWAP_REAL_STAGES
        for chunk_index in range(_SWAP_LOCAL_CHUNKS)
    ]
    chunks = [
        _build_swap_chunk(virtual_stage_index, enable_swap)
        for virtual_stage_index in virtual_stage_indices
    ]
    stages = [
        PipelineStage(
            chunk,
            virtual_stage_index,
            _SWAP_VIRTUAL_STAGES,
            device,
        )
        for chunk, virtual_stage_index in zip(chunks, virtual_stage_indices)
    ]
    schedule = ScheduleInterleaved1F1B(
        stages,
        micro_batch_num=_SWAP_MICROBATCHES,
        swap=enable_swap,
    )
    swap_step_count = sum(
        step is not None and step.type.name.startswith("SWAP_")
        for step in schedule.exec_order[stage_index]
    )
    if enable_swap:
        assert swap_step_count > 0, f"rank {dist.get_rank()} has no pipeline-swap control steps"

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    if stage_index == 0:
        input_tensor = torch.full(
            (_SWAP_MICROBATCHES, _SWAP_ROWS_PER_MICROBATCH, _SWAP_HIDDEN_SIZE),
            0.125,
            dtype=torch.float32,
            device=device,
        )
        losses = schedule.run(input_tensor)
    else:
        losses = schedule.run()
    torch.npu.synchronize()
    peak_memory_mb = torch.npu.max_memory_allocated() / (1024 ** 2)

    output_sums = [
        output.detach().float().sum().cpu()
        for output in losses
    ]
    gradients = {
        f"stage{stage.stage_index}.{name}": parameter.grad.detach().float().cpu().clone()
        for stage in stages
        for name, parameter in stage.submodule.named_parameters()
    }
    assert SwapManager().active_group_count() == 0, (
        f"rank {dist.get_rank()} leaked pipeline swap groups"
    )
    return peak_memory_mb, output_sums, gradients, swap_step_count


def test_vpp_pipeline_swap() -> None:
    """
    Feature: PyTorch VPP activation swap.
    Description: Run paired no-swap/swap Interleaved 1F1B phases using the existing
        two-local-chunk VPP layout, compare final outputs and every local gradient,
        and verify that activation swap reduces peak NPU memory.
    Expectation: Outputs and gradients match while swap peak memory is lower on every rank.
    """
    init_hccl()
    stage_index = get_stage_index(_SWAP_REAL_STAGES)

    baseline_peak_mb, baseline_outputs, baseline_gradients, _ = _run_pipeline_swap_phase(
        stage_index, enable_swap=False,
    )
    dist.barrier()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    dist.barrier()

    swap_peak_mb, swap_outputs, swap_gradients, swap_step_count = _run_pipeline_swap_phase(
        stage_index, enable_swap=True,
    )
    assert len(swap_outputs) == len(baseline_outputs)
    for output, baseline_output in zip(swap_outputs, baseline_outputs):
        torch.testing.assert_close(output, baseline_output, rtol=1e-5, atol=1e-6)
    assert swap_gradients.keys() == baseline_gradients.keys()
    for name, gradient in swap_gradients.items():
        torch.testing.assert_close(
            gradient,
            baseline_gradients[name],
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, param_name=name: f"{param_name}: {message}",
        )
    assert swap_peak_mb < baseline_peak_mb, (
        f"rank {dist.get_rank()} swap peak ({swap_peak_mb:.1f} MB) should be less than "
        f"no-swap peak ({baseline_peak_mb:.1f} MB)"
    )
    reduction = (baseline_peak_mb - swap_peak_mb) / baseline_peak_mb * 100
    print(
        f"[rank {dist.get_rank()}] torch VPP pipeline swap passed: "
        f"no_swap={baseline_peak_mb:.1f} MB, swap={swap_peak_mb:.1f} MB, "
        f"reduction={reduction:.1f}%, swap_steps={swap_step_count}, accuracy=matched",
        flush=True,
    )


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


def test_vpp_dynamic_batch_p2p_cold_start():
    """
    Feature: Dynamic-shape VPP with batched P2P on cold PP subgroups.
    Description: Run two virtual stages per rank on a ``pp=4, dp=2`` mesh with
        ``micro_batch_num=1``, ``dyn_shape=True``, and ``overlap_b_f=True``
        before any tensor collective.
    Expectation: The schedule completes and the final-stage output is finite.
    """
    init_hccl()
    pp_size = 4
    world_size = dist.get_world_size()
    if world_size % pp_size != 0:
        raise ValueError(f"world_size must be divisible by pp_size, but got {world_size} and {pp_size}.")
    dp_size = world_size // pp_size
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(pp_size, dp_size),
        mesh_dim_names=("pp", "dp"),
    )
    pp_mesh = mesh["pp"]
    stage_index = pp_mesh.get_local_rank()

    chunks = [SimpleMLP(8, 16, 16), SimpleMLP(8, 16, 16)]
    for chunk_index, chunk in enumerate(chunks):
        model_split_manual(
            chunk,
            stage_index + chunk_index * pp_size,
            pp_size * len(chunks),
        )
    stages = [
        PipelineStage(
            chunk,
            stage_index + chunk_index * pp_size,
            pp_size * len(chunks),
            torch.device(_DEVICE_TYPE),
            group=pp_mesh.get_group(),
            mesh=pp_mesh,
            dyn_shape=True,
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
    schedule = ScheduleInterleaved1F1B(
        stages,
        micro_batch_num=1,
        overlap_b_f=True,
    )
    assert getattr(schedule, "_p2p_mode") == "batch"
    input_tensor = to_device(torch.ones((1, 16), dtype=torch.float32), _DEVICE_TYPE)
    losses = schedule.run(input_tensor) if stage_index == 0 else schedule.run()

    if stage_index == pp_size - 1:
        assert len(losses) == 1
        assert torch.isfinite(losses[0]).all()


if __name__ == "__main__":
    test_vpp()
