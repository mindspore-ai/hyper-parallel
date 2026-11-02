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
"""Real-device HSDP state roundtrip and next-update equality acceptance."""

import copy
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import StorageWeakRef

from rl.roles.phase_offload import TrainingStateOffload
from rl.trainer import SyncTrainer

from hyper_parallel import fully_shard, hsdp_sync_stream
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


def local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a device-local tensor without collective communication."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def tensors(model: torch.nn.Module, optimizer: Float16OptimizerWithFloat16Params) -> dict[str, torch.Tensor]:
    """Name model shards, fp32 masters, moments, and CPU step counters."""
    result = {f"model/{name}": p for name, p in model.named_parameters()}
    result.update({f"buffer/{name}": p for name, p in model.named_buffers()})
    for index, component in enumerate(optimizer.chained_optimizers):
        for group_index, group in enumerate(component.param_groups):
            for param_index, param in enumerate(group["params"]):
                key = f"optim/{index}/{group_index}/{param_index}"
                result[key] = param
                result.update({f"{key}/{name}": value for name, value in component.state[param].items()
                               if isinstance(value, torch.Tensor)})
    return result


def update(model: torch.nn.Module, optimizer: Float16OptimizerWithFloat16Params, inputs: torch.Tensor) -> None:
    """Run shared HSDP and shared mixed-precision AdamW without RL sampling."""
    with SkipDTensorDispatch():
        loss = model(inputs).float().square().mean()
    loss.backward()
    hsdp_sync_stream()
    optimizer.prepare_grads()
    with SkipDTensorDispatch():
        optimizer.step_with_ready_grads()
    optimizer.zero_grad(set_to_none=True)
    # Exercise the same residency boundary as the production trainer.
    SyncTrainer._reshard_model(model)  # pylint: disable=protected-access
    torch.npu.synchronize()


def run(training_offload: bool) -> dict:
    """Check CPU policy preservation and exact numerical roundtrip on every rank."""
    torch.manual_seed(83)
    base = torch.nn.Sequential(torch.nn.Linear(2048, 2048, bias=False), torch.nn.ReLU(),
                               torch.nn.Linear(2048, 2048, bias=False)).to(dtype=torch.bfloat16)
    base.register_buffer("device_buffer", torch.ones(3))
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    models = []
    for _ in range(2):
        model = copy.deepcopy(base).npu()
        policy = CPUOffloadPolicy() if training_offload else OffloadPolicy()
        fully_shard(model, mesh=mesh, mp_policy=precision, offload_policy=policy, reshard_after_forward=True,
                    comm_fusion=True, comm_fusion_zero_copy=True)
        SyncTrainer._reshard_model(model)  # pylint: disable=protected-access
        leaf = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
        models.append((model, optimizer))
    inputs = torch.randn(4, 2048, device="npu", dtype=torch.bfloat16)
    for model, optimizer in models:
        update(model, optimizer, inputs)
    model, optimizer = models[0]
    manager = TrainingStateOffload([(model, optimizer)])
    before = {name: local(t).detach().cpu().clone() for name, t in tensors(model, optimizer).items()}
    devices = {name: t.device for name, t in tensors(model, optimizer).items()}
    identities = [id(p) for p in model.parameters()]
    storage_refs = [StorageWeakRef(local(t).untyped_storage()) for t in tensors(model, optimizer).values()
                    if t.device.type != "cpu"]
    allocated_before = torch.npu.memory_allocated()
    moved = manager.offload()
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    allocated_offloaded = torch.npu.memory_allocated()
    if not all(ref.expired() for ref in storage_refs):
        raise RuntimeError("Original device storage retained after phase offload")
    for name, tensor in tensors(model, optimizer).items():
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Device state survived offload: {name}")
        torch.testing.assert_close(local(tensor), before[name], rtol=0, atol=0)
    manager.restore()
    torch.npu.synchronize()
    if identities != [id(p) for p in model.parameters()]:
        raise RuntimeError("Parameter identity changed")
    for name, tensor in tensors(model, optimizer).items():
        if tensor.device != devices[name]:
            raise RuntimeError(f"Original device not restored: {name}")
        torch.testing.assert_close(local(tensor).cpu(), before[name], rtol=0, atol=0)
    for current_model, current_optimizer in models:
        update(current_model, current_optimizer, inputs)
    actual = tensors(*models[0])
    expected = tensors(*models[1])
    for name in actual:
        torch.testing.assert_close(local(actual[name]), local(expected[name]), rtol=0, atol=0)
    return {"training_cpu_offload": training_offload, "copied_bytes": moved,
            "allocated_before": allocated_before, "allocated_offloaded": allocated_offloaded,
            "old_storage_expired": True, "state_roundtrip_exact": True, "next_update_exact": True}


def main() -> None:
    """Run both independent CPU policies on the allocated distributed devices."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="cpu:gloo,npu:hccl")
    results = [run(False), run(True)]
    root = Path(os.environ.get("HYPER_COMPARISON_OUTPUT", "/results"))
    (root / f"roundtrip-rank-{dist.get_rank()}.json").write_text(json.dumps(results, indent=2) + "\n")
    dist.barrier()
    if dist.get_rank() == 0:
        (root / "measurement.json").write_text(json.dumps({"status": "passed", "ranks": dist.get_world_size()}) + "\n")
        (root / "work_done").touch()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
