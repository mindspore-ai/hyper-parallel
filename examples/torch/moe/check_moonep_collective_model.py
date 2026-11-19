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

"""Strict FP32 full-model all-to-all/collective gradient and SGD parity check."""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from benchmark_moonep_collective import build_model

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


def main():
    """Check all local parameter gradients and three SGD updates on each rank."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("npu", (size,), mesh_dim_names=("ep",))
    tokens = torch.randint(
        0, 256, (1, 128), generator=torch.Generator().manual_seed(123 + rank)
    ).npu()
    reference = []
    failures = []
    maximum = {"gradient": 0.0, "parameter": 0.0}
    for backend in ("all_to_all", "moonep_collective"):
        model = build_model(backend, mesh, SimpleNamespace(grouped=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                (
                    model(input_ids=tokens, labels=tokens, use_cache=False).loss / 2
                ).backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if ".experts." not in name:
                    dist.all_reduce(parameter.grad)
                parameter.grad.div_(size)
            gradients = {
                name: p.grad.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            optimizer.step()
            parameters = {
                name: p.detach().cpu().clone() for name, p in model.named_parameters()
            }
            if backend == "all_to_all":
                reference.append((gradients, parameters))
            else:
                for kind, actual, expected in zip(
                    ("gradient", "parameter"), (gradients, parameters), reference[step]
                ):
                    if actual.keys() != expected.keys():
                        failures.append((step, kind, "keys"))
                    for name in actual:
                        maximum[kind] = max(
                            maximum[kind],
                            (actual[name] - expected[name]).abs().max().item(),
                        )
                        try:
                            torch.testing.assert_close(
                                actual[name], expected[name], atol=3e-6, rtol=3e-5
                            )
                        except AssertionError:
                            failures.append((step, kind, name))
        del optimizer, model
        torch.npu.empty_cache()
    summary = torch.tensor(
        [len(failures), maximum["gradient"], maximum["parameter"]], device="npu"
    )
    dist.all_reduce(summary, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "max_failures_per_rank": summary[0].item(),
                    "max_gradient_abs": summary[1].item(),
                    "max_parameter_abs": summary[2].item(),
                    "steps": 3,
                    "accumulation": 2,
                    "atol": 3e-6,
                    "rtol": 3e-5,
                }
            ),
            flush=True,
        )
    if failures:
        print(rank, failures, flush=True)
    passed = summary[0].item() == 0
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("Full-model strict parity failed")


if __name__ == "__main__":
    main()
