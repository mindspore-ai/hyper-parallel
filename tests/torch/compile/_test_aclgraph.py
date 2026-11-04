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
"""Native NPU recording/replay and HCCL numerical acceptance workers."""

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.passes.aclgraph_runtime import ACLGraphWrapper


def _loss(model, value, target):
    return (model(value) - target).square().mean()


def test_aclgraph_mutated_state():
    """Native capture must preserve live buffer identity and retained outputs."""
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu")
    state = torch.zeros((), device=device)

    def _step(value, accumulator):
        accumulator.add_(value.sum())
        return accumulator, value * 2

    wrapper = ACLGraphWrapper(_step, static_input_indices=(1,))
    retained = []
    try:
        for index in range(1, 5):
            output_state, gradient = wrapper(torch.full((4,), float(index), device=device), state)
            assert output_state is state, f"Captured state lost its identity on call {index}"
            torch.testing.assert_close(state, torch.tensor(2 * index * (index + 1), device=device).float())
            retained.append((gradient, torch.full((4,), float(2 * index), device=device)))
            for actual, expected in retained:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch_npu.npu.synchronize()
        wrapper.teardown()


def _compare_gradients(model, reference, rank, world_size):
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        expected = expected_parameter.grad
        if world_size > 1:
            dist.all_reduce(expected)
            expected.div_(world_size)
            expected = expected.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-6)


def _compare_parameters(model, reference, rank, world_size):
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        expected = expected_parameter.detach()
        if world_size > 1:
            expected = expected.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)


def _check_captures(wrappers, previous, mode, rank, step):
    current = [wrapper.graph for wrapper in wrappers]
    assert current and all(graph is not None for graph in current), (
        f"Expected captured regions on rank {rank}, got {current}"
    )
    if previous is not None:
        same = [graph is prior for graph, prior in zip(current, previous, strict=True)]
        if mode == "static":
            assert all(same), f"Static capture unexpectedly re-recorded on rank {rank}: {same}"
        elif step >= 2:
            assert not all(same), f"Rank {rank} did not coordinate changed inputs: {same}"
    return current


def _check_inductor_capture(compiler):
    gm = compiler._joint_graph.graph_module
    captured = {entry["name"] for entry in gm.meta["aclgraph_partitions"] if entry["capture"]}
    compiled_calls = [node for name, region in gm.named_children() if name in captured
                      for node in region.graph.nodes if node.meta.get("hyper_parallel_compiled_region")]
    assert compiled_calls, f"Expected compiled functions inside captured regions, got {gm.graph}"


def _refresh_parameter_storage(model, mode, inductor_mode, step, rank):
    if mode == "dynamic" and inductor_mode == "regional" and step == 2 and rank == 0:
        for parameter in model.parameters():
            parameter.data = parameter.detach().clone()


@pytest.mark.parametrize("mode,inductor_mode", [
    ("static", "off"), ("dynamic", "off"), ("static", "regional"), ("dynamic", "regional"),
])
def test_aclgraph_training(mode, inductor_mode):
    """Capture joint forward/backward, replay with live weights, then release."""
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device(f"npu:{rank}")
    torch.manual_seed(2026)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    reference = copy.deepcopy(model)
    symbolic_shapes = mode == "dynamic" and inductor_mode == "off"
    compiler = GraphCompiler(
        model, _loss, device=device,
        pass_config=PassConfig(fsdp_enabled=world_size > 1, aclgraph_mode=mode, inductor_mode=inductor_mode),
        dynamic_arg_dims={"value": [0], "target": [0]} if symbolic_shapes else None,
    )
    optimizer = None
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    retained = []
    graphs = None
    try:
        # Change one rank's capture signature: batch shape for plain FX, or
        # parameter storage for compiled regions. Symbolic shape support also
        # depends on the selected Inductor backend, independently of capture.
        for step in range(4):
            _refresh_parameter_storage(model, mode, inductor_mode, step, rank)
            reference_optimizer.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for microbatch in range(2):
                size = 6 if symbolic_shapes and step >= 2 and rank == 0 else 4
                generator = torch.Generator().manual_seed(100 + rank + 10 * step + microbatch)
                value = torch.randn(size, 16, generator=generator).to(device)
                target = torch.randn(size, 8, generator=generator).to(device)
                expected = _loss(reference, value, target)
                expected.backward()
                actual, _ = compiler.forward_backward(value=value, target=target)
                if optimizer is None:
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
                torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
                for previous, snapshot in retained:
                    torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)
                retained.append((actual, actual.clone()))
            _compare_gradients(model, reference, rank, world_size)
            optimizer.step()
            reference_optimizer.step()
            _compare_parameters(model, reference, rank, world_size)
            wrappers = compiler._joint_graph.graph_module._aclgraph_wrappers
            current_graphs = _check_captures(wrappers, graphs, mode, rank, step)
            if graphs is None:
                graphs = current_graphs
        if inductor_mode == "regional":
            _check_inductor_capture(compiler)
        print(f"ACLGraph acceptance: rank={rank}, mode={mode}, inductor={inductor_mode}, "
              f"world_size={world_size}, regions={len(wrappers)}")
    finally:
        # Drop test-only graph references before releasing HCCL resources.
        graphs = None
        current_graphs = None
        compiler.close()
        dist.destroy_process_group()
