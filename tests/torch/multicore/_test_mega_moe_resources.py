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
"""Real-device acceptance of serial shared workspaces and mixed SHMEM owners."""

from dataclasses import asdict

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.shmem import lifecycle
from hyper_parallel.core.multicore.shmem.lifecycle import acquire_symmetric_memory
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    assert_stable_memory,
    collect_memory,
    start_shmem_lifetime,
    write_evidence,
)


def _new_stack(shape: baseline.MoeShape, layers: int = 3) -> list[MegaMoeExperts]:
    """Build distinct expert layers without initializing any native resource."""
    return [
        MegaMoeExperts(
            local_num_tokens=shape.local_num_tokens,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            ep_size=shape.ep_size,
            ep_group=dist.group.WORLD,
        ).to(device=baseline.DEVICE, dtype=torch.bfloat16)
        for _ in range(layers)
    ]


def _stack_step(
    layers: list[MegaMoeExperts],
    shape: baseline.MoeShape,
    step: int,
    streams: list,
) -> list[torch.Tensor]:
    """Keep two whole-stack forwards outstanding, then run backward in reverse."""
    hidden, upstream = baseline.make_data(shape)
    hidden = hidden * (1.0 + step / 32.0)
    # Keep repeated SGD on the unnormalized three-layer stack numerically stable.
    upstream = upstream / shape.local_num_tokens**0.5
    topk_ids, weights, counts = baseline.make_balanced_route(shape)
    inputs = [(hidden * (1.0 + microbatch / 8.0)).detach().requires_grad_(True) for microbatch in range(2)]
    route_weights = [weights.clone().requires_grad_(True) for _ in range(6)]
    default_stream = torch.npu.current_stream()
    outputs = []
    for microbatch, input_tensor in enumerate(inputs):
        output = input_tensor
        previous_stream = default_stream
        for index, layer in enumerate(layers):
            stream = streams[(microbatch + index) % len(streams)]
            with torch.npu.stream(stream):
                stream.wait_stream(previous_stream)
                output = layer(
                    output, topk_ids, route_weights[microbatch * 3 + index],
                    tokens_per_expert=counts,
                )
            previous_stream = stream
        outputs.append(output)
    for output in reversed(outputs):
        output.backward(upstream)
    torch.npu.synchronize()
    compared = outputs + [tensor.grad for tensor in inputs + route_weights]
    compared += [parameter.grad for layer in layers for parameter in layer.parameters()]
    for index, tensor in enumerate(compared):
        assert tensor is not None, f"rank={baseline.RANK}: missing gradient at {index}."
        baseline.assert_finite(f"stack step={step} tensor={index}", tensor)
        assert torch.count_nonzero(tensor).item() > 0, (
            f"rank={baseline.RANK}: stack tensor {index} is entirely zero."
        )
    snapshot = [tensor.detach().cpu() for tensor in compared]
    with torch.no_grad():
        for layer in layers:
            for parameter in layer.parameters():
                parameter.add_(parameter.grad, alpha=-1e-3)
            layer.zero_grad(set_to_none=True)
    return snapshot


def _shared_stack_comparison(alternate_streams: bool) -> dict:
    """Compare independent and shared groups over warmup and stable steps."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=1024)
    torch.manual_seed(61_000 + baseline.RANK)
    independent = _new_stack(shape)
    shared = _new_stack(shape)
    for source, target in zip(independent, shared):
        target.load_state_dict(source.state_dict())
    MegaMoeExperts.share_execution_resources(shared)
    streams = [torch.npu.Stream() for _ in range(2 if alternate_streams else 1)]
    samples = []
    try:
        for step in range(8):
            print(
                f"rank={baseline.RANK}: Level1 sharing alternate_streams={alternate_streams} step={step}", flush=True,
            )
            reference = _stack_step(independent, shape, step, streams)
            actual = _stack_step(shared, shape, step, streams)
            for index, (expected, observed) in enumerate(zip(reference, actual)):
                torch.testing.assert_close(
                    observed, expected, rtol=2e-2, atol=1e-5,
                    msg=f"rank={baseline.RANK}: shared step={step} tensor={index}",
                )
            for source, target in zip(independent, shared):
                for expected, observed in zip(source.parameters(), target.parameters()):
                    baseline.assert_close("updated expert parameter", observed, expected)
            if step >= 3:
                sample = collect_memory()
                # Count tracked allocations to detect leaks inside the native heap.
                sample["shmem_allocations"] = sum(
                    len(
                        layer._resource_group.resources.symmetric_memory._allocations  # pylint: disable=W0212
                    )
                    for layer in independent + shared[:1]
                )
                samples.append(sample)
        assert_stable_memory(samples)
        assert len({sample["shmem_allocations"] for sample in samples}) == 1, (
            f"rank={baseline.RANK}: SHMEM allocations grew across steps: {samples}."
        )
        # Resource identity is internal; verify that sharing uses the same actual allocation group.
        resources = shared[0]._resource_group.resources  # pylint: disable=protected-access
        assert all(
            layer._resource_group.resources is resources for layer in shared  # pylint: disable=protected-access
        ), (
            f"rank={baseline.RANK}: shared layers acquired different resources."
        )
        assert len({
            id(layer._resource_group.resources) for layer in independent  # pylint: disable=protected-access
        }) == 3, (
            f"rank={baseline.RANK}: independent layers unexpectedly share resources."
        )
        return {
            "shape": asdict(shape), "layers": 3, "outstanding_forwards": 6,
            "steps": 8, "warmup_steps": 3, "alternate_streams": alternate_streams,
            "gradient_scale": 1.0 / shape.local_num_tokens**0.5, "rtol": 2e-2, "atol": 1e-5,
            "outputs_and_all_gradients_match": True, "parameter_updates_match": True,
            "stable_memory": samples,
        }
    finally:
        for layer in independent + shared:
            layer.close()


def _assert_invalidated(tensors: list[torch.Tensor]) -> None:
    """Inspect storage metadata without touching freed device data."""
    for tensor in tensors:
        size = tensor.untyped_storage().nbytes()
        assert size == 0, f"rank={baseline.RANK}: closed SHMEM allocation retained {size} bytes."


def _mixed_owner_close_order(internal_first: bool) -> dict:
    """Exercise real private allocations and a live managed backward together."""
    start_shmem_lifetime()
    shape = baseline.MoeShape()
    layer = _new_stack(shape, layers=1)[0]
    hidden, upstream = baseline.make_data(shape)
    hidden.requires_grad_(True)
    topk_ids, weights, counts = baseline.make_balanced_route(shape)
    output = layer(hidden, topk_ids, weights, tokens_per_expert=counts)
    # Retain the actual workspace tensors to verify their storage is invalidated on close.
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    workspace_tensors = [resources.workspace.expert_buffer, resources.workspace.routed_buffer]
    internal_owner = acquire_symmetric_memory()
    internal_tensor = internal_owner.empty((256,), torch.int32)
    internal_tensor.fill_(baseline.RANK + 1)
    internal_owner.barrier()
    # Reference counts and manager identity verify that only the last owner finalizes.
    state = lifecycle._PROCESS_STATE  # pylint: disable=protected-access
    manager = state.manager
    assert state.clients == 2, f"rank={baseline.RANK}: expected two owners, got {state.clients}."
    try:
        if internal_first:
            internal_owner.close()
            _assert_invalidated([internal_tensor])
            assert state.manager is manager and state.clients == 1, (
                f"rank={baseline.RANK}: internal close finalized the managed owner."
            )
        output.backward(upstream)
        torch.npu.synchronize()
        baseline.assert_finite("mixed-owner input gradient", hidden.grad)
        layer.close()
        _assert_invalidated(workspace_tensors)
        if not internal_first:
            assert state.manager is manager and state.clients == 1, (
                f"rank={baseline.RANK}: managed close finalized the internal owner."
            )
            internal_tensor.add_(1)
            internal_owner.barrier()
            expected = torch.full_like(internal_tensor, baseline.RANK + 2)
            baseline.assert_close("surviving internal allocation", internal_tensor, expected)
            internal_owner.close()
            _assert_invalidated([internal_tensor])
        assert not state.initialized and state.clients == 0, (
            f"rank={baseline.RANK}: last owner did not finalize: clients={state.clients}."
        )
        return {"internal_first": internal_first, "survivor_usable": True, "last_close_invalidated": True}
    finally:
        layer.close()
        internal_owner.close()


def test_mega_moe_shared_resource_acceptance() -> None:
    """Accept serial sharing and private/managed ownership with real NPU kernels."""
    results = [_shared_stack_comparison(alternate) for alternate in (False, True)]
    close_orders = [_mixed_owner_close_order(order) for order in (True, False)]
    write_evidence({"sharing": results, "close_orders": close_orders, "checkpoint_tested": False})
