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
"""MoonEP regressions reusing HP's reference data, routes and numerical assertions."""
from __future__ import annotations

import gc
from contextlib import ExitStack
import time
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from hyper_parallel.core.expert_parallel.moonep import placement as placement_module
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler, schedule as diagnostic_schedule
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import _records_for_task
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.expert_parallel.moonep.mega_moe import MoonEPMegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import function as native
from hyper_parallel.core.expert_parallel.moonep.shmem_transfer import load_bridge
from hyper_parallel.core.expert_parallel.moonep import mega_moe as moon_module
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from tests.torch.multicore import _test_mega_moe as reference
from tests.torch.multicore import _test_mega_moe_lifecycle as lifecycle
from tests.torch.multicore import _test_mega_moe_zeroing as zeroing
from tests.torch.multicore import _test_swiglu_clamp_st as clamp


_FORCED_TRANSFER_COST = False


@pytest.fixture
def _force_transfer_cost(monkeypatch: pytest.MonkeyPatch):
    """Inject only cost selection to stress real transfers on inexpensive tiny cases."""
    monkeypatch.setitem(globals(), "_FORCED_TRANSFER_COST", True)
    with patch.object(MoonEPMegaMoeExperts, "_placement_is_profitable", return_value=True):
        yield



@pytest.fixture
def _routed_snapshot_contract():
    """Keep the target stream schedule while validating cold expert gradients exactly."""

    def _snapshot_and_update(layers, outputs, leaves, step):
        tensors = outputs + [tensor.grad for tensor in leaves]
        for index, tensor in enumerate(tensors):
            assert tensor is not None, f"Missing output/input/route gradient {index}"
            reference.assert_finite(f"stream step={step} tensor={index}", tensor)
            assert torch.count_nonzero(tensor).item() > 0
        first = layers[0]
        shape = reference.MoeShape(
            local_num_tokens=leaves[0].shape[0], hidden_size=first.hidden_size,
            intermediate_size=first.intermediate_size, num_experts=first.num_experts,
            top_k=first.top_k, ep_size=first.ep_size)
        _, _, counts = reference.make_balanced_route(shape)
        counts = counts.clone()
        dist.all_reduce(counts)
        start = dist.get_rank() * first.local_experts
        active = (counts[start:start + first.local_experts] > 0).cpu()
        for layer in layers:
            for parameter in layer.parameters():
                gradient = parameter.grad
                assert gradient is not None
                reference.assert_finite("stream expert gradient", gradient)
                local = gradient.detach().cpu()
                assert torch.count_nonzero(local[~active]).item() == 0, "Cold experts must return exact zero dW"
                for expert in torch.where(active)[0].tolist():
                    assert torch.count_nonzero(local[expert]).item() > 0, "A routed expert lost its gradient"
                tensors.append(gradient)
        snapshots = [tensor.detach().cpu() for tensor in tensors]
        with torch.no_grad():
            for layer in layers:
                for parameter in layer.parameters():
                    parameter.add_(parameter.grad, alpha=-1e-3)
                layer.zero_grad(set_to_none=True)
        return snapshots

    with patch.object(lifecycle, "_snapshot_and_update", _snapshot_and_update):
        yield

def _new_pair(shape, mode, layout="EIO"):
    template, common = reference.new_layers(shape, expert_weight_layout=layout)
    # Use per-projection fan-in initialization so a three-layer test has useful
    # output/gradient magnitudes, rather than compounding flattened-expert scales.
    with torch.no_grad():
        for parameter in common.parameters():
            local = reference._local_tensor(parameter)
            local.normal_(std=local.shape[-1] ** -0.5)
    reference._copy_common_weights_to_mega(common, template)
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD,
                               dispatch_mode=mode, expert_weight_layout=layout).to(
                                   device=reference.DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        moon.gate_up_weight.copy_(template.gate_up_weight)
        moon.down_weight.copy_(template.down_weight)
    template.close()
    return moon, common


def _forward(layers, hidden, probabilities, routes, checkpoint_mode):
    output = hidden
    for layer, probs, (ids, _, counts) in zip(layers, probabilities, routes):
        def _call(value, weights, current=layer, current_ids=ids, current_counts=counts):
            return reference.forward_layer(current, value, current_ids, weights, tokens_per_expert=current_counts)
        if checkpoint_mode == "none":
            output = _call(output, probs)
        else:
            output = checkpoint(_call, output, probs, use_reentrant=checkpoint_mode == "true")
    return output


def _capture(layers, output, hidden, probabilities):
    result = {"output": output.detach().clone(), "input_grad": hidden.grad.detach().clone()}
    for index, (layer, probs) in enumerate(zip(layers, probabilities)):
        result[f"probs_{index}"] = probs.grad.detach().clone()
        weight1, weight2 = reference.expert_weight_gradients(layer)
        result[f"weight1_{index}"] = weight1.detach().clone()
        result[f"weight2_{index}"] = weight2.detach().clone()
    return result


def _compare(expected, actual):
    maxima = {}
    for name in expected:
        reference.assert_close(name, actual[name], expected[name])
        maxima[name] = float((actual[name].float() - expected[name].float()).abs().max().item())
    return maxima


def _write(records):
    evidence = [None] * dist.get_world_size()
    dist.all_gather_object(
        evidence, {"rank": dist.get_rank(), "transfer_cost_forced": _FORCED_TRANSFER_COST, **records})
    if dist.get_rank() == 0:
        path = Path(os.environ["HP_MOONEP_LIFECYCLE_RESULT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_original_precision():
    """Run HP's original level0 route/SGD case with the MoonEP class factory."""
    original = reference.MegaMoeExperts
    reference.MegaMoeExperts = MoonEPMegaMoeExperts
    try:
        reference.test_mega_moe_level0_balanced()
        _write({"original_case": "test_mega_moe_level0_balanced", "passed": True})
    finally:
        reference.MegaMoeExperts = original


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_checkpoint_retained_layers():
    """Match every gradient through three layers, checkpoint modes and retained backward."""
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    checkpoint_mode = os.environ["HP_MOONEP_CHECKPOINT_MODE"]
    shape = reference.MoeShape()
    pairs = [_new_pair(shape, mode) for _ in range(3)]
    moons, commons = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    MoonEPMegaMoeExperts.share_execution_resources(moons)
    routes = [reference._make_fixed_route(shape, pattern)
              for pattern in ("balanced", "single_destination", "skew")]
    hidden, grad = reference.make_data(shape)
    actual_hidden = hidden.detach().clone().requires_grad_(True)
    actual_probs = [route[1].detach().clone().requires_grad_(True) for route in routes]
    output = None
    try:
        with set_checkpoint_early_stop(checkpoint_mode != "false_full"):
            output = _forward(moons, actual_hidden, actual_probs, routes, checkpoint_mode)
        iterations = []
        for retained in (True, False):
            for layer in moons + commons:
                layer.zero_grad(set_to_none=True)
            actual_hidden.grad = None
            for probs in actual_probs:
                probs.grad = None
            expected_hidden = hidden.detach().clone().requires_grad_(True)
            expected_probs = [route[1].detach().clone().requires_grad_(True) for route in routes]
            expected_output = _forward(commons, expected_hidden, expected_probs, routes, "none")
            expected_output.backward(grad)
            expected = _capture(commons, expected_output, expected_hidden, expected_probs)
            output.backward(grad, retain_graph=retained)
            actual = _capture(moons, output, actual_hidden, actual_probs)
            assert expected["output"].float().abs().max().item() > 1e-3
            assert expected["input_grad"].float().abs().max().item() > 1e-3
            iterations.append({"retain_graph": retained, "max_abs": _compare(expected, actual)})
        active_pools = [layer._transfer_pool for layer in moons if layer._transfer_pool is not None]
        pools = {id(pool) for pool in active_pools}
        engines = [group.resources for layer in moons
                   for executor in (layer, layer._virtual_executor) if executor is not None
                   for group in (executor._resource_group,) if group.resources is not None]
        by_geometry = {}
        for engine in engines:
            by_geometry.setdefault(engine.spec.num_experts, set()).add(id(engine.workspace))
        assert len(pools) == 1, "Borrowing layers did not share their expert bank"
        assert set(by_geometry) == {shape.num_experts, 2 * shape.num_experts}
        assert all(len(workspaces) == 1 for workspaces in by_geometry.values()), (
            "Compatible layers did not share the original HP workspace for each plan")
        stats = dict(active_pools[0].statistics)
        assert stats["remote_prefetches"] > 0 and stats["remote_gradient_returns"] > 0
        assert moons[0].placement_statistics["native_balanced_calls"] > 0
        _write({"dispatch_mode": mode, "checkpoint_mode": checkpoint_mode,
                "iterations": iterations, "statistics": stats,
                "expert_banks": len(pools),
                "native_workspace_per_geometry": {str(key): len(value) for key, value in by_geometry.items()}})
    finally:
        output = None
        gc.collect()
        for layer in moons:
            layer.close()

@pytest.mark.usefixtures("_force_transfer_cost", "_routed_snapshot_contract")
def test_moonep_stream_exception_recovery():
    """Reuse HP's staggered two-stream schedule with genuine borrowed experts."""

    shape = reference.MoeShape()
    original = lifecycle._new_stack(shape)
    with patch.object(lifecycle, "MegaMoeExperts", MoonEPMegaMoeExperts):
        moons = lifecycle._new_stack(shape)
    for source, target in zip(original, moons):
        target.load_state_dict(source.state_dict())
    MoonEPMegaMoeExperts.share_execution_resources(moons)
    streams = [torch.npu.Stream() for _ in range(2)]
    records = []
    try:
        # Keep the target schedule; snapshot checks distinguish routed and cold experts.
        with patch.object(reference, "make_balanced_route",
                          side_effect=lambda current: reference._make_fixed_route(current, "single_destination")):
            for step, replay in enumerate((False, True)):
                expected = lifecycle._stack_step(original, shape, streams, False, step)
                actual = lifecycle._stack_step(moons, shape, streams, replay, step)
                for index, (left, right) in enumerate(zip(expected, actual)):
                    torch.testing.assert_close(right, left, rtol=2e-2, atol=1e-5,
                                               msg=f"MoonEP streams replay={replay} tensor={index}")
                records.append({"checkpoint": replay, "tensors": len(actual)})

            source, _ = reference.make_data(shape)
            ids, probs, counts = reference.make_balanced_route(shape)
            with patch.object(native.multicore_ops, "mega_moe",
                              side_effect=RuntimeError("injected MoonEP launch error")):
                try:
                    moons[0](source, ids, probs, tokens_per_expert=counts)
                except RuntimeError as error:
                    assert str(error) == "injected MoonEP launch error"
                else:
                    raise AssertionError("The injected native failure did not propagate")
            pool = moons[0]._transfer_pool
            workspace = moons[0]._virtual_executor._resource_group.resources.workspace
            logical_workspace = moons[0]._resource_group.resources.workspace
            assert not pool.in_use and not workspace.in_use and not logical_workspace.in_use, (
                "Failed launch kept a resource lease")
            expected = lifecycle._stack_step(original, shape, streams, False, 2)
            actual = lifecycle._stack_step(moons, shape, streams, True, 2)
            for index, (left, right) in enumerate(zip(expected, actual)):
                torch.testing.assert_close(right, left, rtol=2e-2, atol=1e-5,
                                           msg=f"MoonEP exception recovery tensor={index}")
        assert pool.statistics["remote_prefetches"] > 0
        assert pool.statistics["remote_gradient_returns"] > 0
        _write({"dispatch_mode": os.environ["HP_MEGA_MOE_DISPATCH_MODE"],
                "alternating_streams": 2, "rank_staggered": True, "records": records,
                "exception_recovered": True, "statistics": dict(pool.statistics)})
    finally:
        gc.collect()
        for layer in original + moons:
            layer.close()


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_inference_then_training():
    """Ensure first-call inference does not poison persistent bank or HP buffers."""
    shape = reference.MoeShape()
    moon, common = _new_pair(shape, os.environ["HP_MEGA_MOE_DISPATCH_MODE"])
    hidden, grad = reference.make_data(shape)
    ids, probs, counts = reference._make_fixed_route(shape, "single_destination")
    try:
        with torch.inference_mode():
            moon(hidden, ids, probs, tokens_per_expert=counts)
        expected = reference.run_layer(common, hidden, ids, probs, counts, grad)
        actual = reference.run_layer(moon, hidden, ids, probs, counts, grad)
        reference.assert_results_close(expected, actual)
        assert not any(tensor.is_inference() for tensor in moon._transfer_pool.weights)
        _write({"inference_first": True, "training_afterwards": True,
                "statistics": dict(moon._transfer_pool.statistics)})
    finally:
        moon.close()


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_original_poisoned_buffers():
    """Reuse every original zero-token/tail/NaN test with MoonEP enabled."""

    with patch.object(reference, "MegaMoeExperts", MoonEPMegaMoeExperts):
        zeroing.test_mega_moe_poisoned_buffers()

@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_heap_rebuild_with_live_graph():
    """Preserve the first layer's graph while HP replaces its managed SHMEM heap."""
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    checkpoint_mode = os.environ["HP_MOONEP_CHECKPOINT_MODE"]
    shape = reference.MoeShape()
    moon, common = _new_pair(shape, mode)
    late = None
    output = None
    hidden, grad = reference.make_data(shape)
    route = reference._make_fixed_route(shape, "single_destination")
    leaf = hidden.detach().clone().requires_grad_(True)
    probs = route[1].detach().clone().requires_grad_(True)
    try:
        output = _forward([moon], leaf, [probs], [route], checkpoint_mode)
        manager = moon._transfer_pool.manager
        before = manager.epoch
        late_shape = reference.MoeShape(intermediate_size=512)
        late, _ = _new_pair(late_shape, mode)
        with torch.no_grad():
            late(hidden, route[0], route[1], tokens_per_expert=route[2])
        assert manager.epoch > before, "The regression did not force a physical heap replacement"
        expected_leaf = hidden.detach().clone().requires_grad_(True)
        expected_probs = route[1].detach().clone().requires_grad_(True)
        expected = _forward([common], expected_leaf, [expected_probs], [route], "none")
        expected.backward(grad)
        output.backward(grad)
        maxima = _compare(_capture([common], expected, expected_leaf, [expected_probs]),
                          _capture([moon], output, leaf, [probs]))
        _write({"dispatch_mode": mode, "checkpoint_mode": checkpoint_mode,
                "before_epoch": before, "after_epoch": manager.epoch, "max_abs": maxima,
                "statistics": dict(moon._transfer_pool.statistics)})
    finally:
        output = None
        gc.collect()
        for layer in (moon, late):
            if layer is not None:
                layer.close()


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_clipping_with_borrowed_experts():
    """Match HP's clamp boundary factory with real EP2 copies and mixed limits."""

    shape = reference.MoeShape()
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    hidden, grad = clamp._boundary_inputs(shape)
    ids, probs, counts = reference._make_fixed_route(shape, "single_destination")
    pairs = []
    try:
        for limit in (None, clamp._CLAMP_LIMIT):
            baseline, common = reference.new_layers(shape, dispatch_mode=mode)
            # Constructor owns HP's native plan for each activation configuration.
            baseline.close()
            baseline = reference.MegaMoeExperts(
                **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
                ep_group=dist.group.WORLD,
                                               dispatch_mode=mode, swiglu_limit=limit).to(
                                                   device=reference.DEVICE, dtype=torch.bfloat16)
            clamp._initialize_weights(common, baseline, shape)
            moon = MoonEPMegaMoeExperts(
                **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
                ep_group=dist.group.WORLD,
                                       dispatch_mode=mode, swiglu_limit=limit).to(
                                           device=reference.DEVICE, dtype=torch.bfloat16)
            moon.load_state_dict(baseline.state_dict())
            pairs.append((baseline, moon))
        records = []
        # Alternate activation plans over the same geometric expert bank.
        for index in (0, 1, 0, 1):
            baseline, moon = pairs[index]
            expected = reference.run_layer(baseline, hidden, ids, probs, counts, grad)
            actual = reference.run_layer(moon, hidden, ids, probs, counts, grad)
            reference.assert_results_close(expected, actual)
            records.append({"limit": (None, clamp._CLAMP_LIMIT)[index], "passed": True})
        assert pairs[0][1]._transfer_pool is pairs[1][1]._transfer_pool
        assert pairs[0][1]._transfer_pool.statistics["remote_gradient_returns"] > 0
        _write({"dispatch_mode": mode, "mixed_clipping_plans": records,
                "statistics": dict(pairs[0][1]._transfer_pool.statistics)})
    finally:
        for pair in pairs:
            for layer in pair:
                layer.close()

def test_moonep_prefetch_ub_reject():
    """Reject a too-wide row and a copy table consuming all UB before any launch."""

    shape = reference.MoeShape(local_num_tokens=128, hidden_size=512, intermediate_size=128,
                               num_experts=16, top_k=4, ep_size=2)
    layer, _ = _new_pair(shape, "push")
    try:
        ids, probabilities, counts = reference.make_balanced_route(shape)
        hidden, _ = reference.make_data(shape)
        with torch.no_grad():
            layer(hidden, ids, probabilities, tokens_per_expert=counts)
        torch.npu.synchronize()
        for slots_count, width in ((1, 49152), (24056, 16)):
            home = torch.empty((2 * slots_count, 1, width), dtype=torch.bfloat16, device=reference.DEVICE)
            slots = torch.empty((slots_count, 1, width), dtype=torch.bfloat16, device=reference.DEVICE)
            copies = torch.full((slots_count,), -1, dtype=torch.int32, device=reference.DEVICE)
            with pytest.raises(RuntimeError, match="MoonEP prefetch UB budget"):
                load_bridge().prefetch(home, slots, copies, dist.get_rank(), dist.get_world_size(),
                                       torch.npu.get_device_limit(torch.npu.current_device())["cube_core_num"])
        dist.barrier()
    finally:
        layer.close()


def _assert_complete_moon_forward_trace(trace: dict, spec) -> None:
    """Reject incomplete or ambiguously named device records."""
    device_limits = torch.npu.get_device_limit(torch.npu.current_device())
    hardware_cube_cores = device_limits["cube_core_num"]
    assert hardware_cube_cores == spec.num_cube_cores, (
        f"device cube count {hardware_cube_cores} differs from HP plan {spec.num_cube_cores}"
    )
    # split values are token tile sizes, not task counts. Rebuild the same CPU
    # graph as the active plan so every direction's exact task count is checked.
    graph, runtime, _, _ = _build_runtime_artifacts(spec)
    tasks = {operator.name: operator.task_num for operator in graph.topological_sort()}
    assert tasks["up_proj"] == spec.local_experts * hardware_cube_cores
    assert tasks["down_proj"] == spec.local_experts * hardware_cube_cores
    # PUT emits no scheduler trigger record, whereas GET does. Count the
    # actual queued task descriptors instead of assuming one transport mode.
    expected_core_records = {}
    for core, queue, index in (
        ("AIC", runtime.cube_task_indices, 0),
        ("AIV", runtime.vector_task_indices, 1),
    ):
        expected_core_records[core] = sum(
            _records_for_task(runtime.all_tasks[int(queue[position])])
            for position in range(int(runtime.task_index_num[index]))
        )
    expected_records = sum(expected_core_records.values())
    metadata = trace["megaKernelCycleTrace"]
    internal_events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"]
    core_record_counts = Counter(event["args"]["core_type"] for event in internal_events)

    assert metadata["invocationCount"] == 1, f"rank={reference.RANK}: expected one forward invocation, got {metadata}."
    assert (
        metadata["recordCount"] == expected_records
    ), f"rank={reference.RANK}: incomplete record count: {metadata}."
    assert metadata["droppedRecordCount"] == 0, f"rank={reference.RANK}: profiling dropped records: {metadata}."
    assert (
        core_record_counts == expected_core_records
    ), f"rank={reference.RANK}: unexpected AIC/AIV records: {core_record_counts}."
    assert all(
        0 < event["args"]["start_cycle"] <= event["args"]["end_cycle"]
        for event in internal_events
    ), f"rank={reference.RANK}: profiling exported an invalid cycle record."
    assert {event["args"]["direction"] for event in internal_events} == {"forward"}
    event_names = {event["name"] for event in internal_events}
    assert any("GMM1_WaitDependency" in name for name in event_names)
    assert any("GMM1_TriggerEvent" in name for name in event_names)
    assert any("GMM2_WaitDependency" in name for name in event_names)
    assert any("GMM2_TriggerEvent" in name for name in event_names)
    metadata["testExpectedRecordCount"] = expected_records
    metadata["testExpectedCoreRecordCounts"] = expected_core_records
    metadata["testHardwareCubeCores"] = hardware_cube_cores
    metadata["testPlanCubeCores"] = spec.num_cube_cores

@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_hotspot_profiling():
    """Profile real borrowing under a forced-profitable test policy, then return gradients."""
    shape = reference.MoeShape(local_num_tokens=128, hidden_size=512, intermediate_size=128,
                               num_experts=16, top_k=4, ep_size=2)
    layer = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD, dispatch_mode=os.environ["HP_MEGA_MOE_DISPATCH_MODE"],
    ).to(device=reference.DEVICE, dtype=torch.bfloat16)
    ids, probabilities, counts = reference._make_fixed_route(shape, "single_destination")
    hidden, upstream = reference.make_data(shape)
    hidden = hidden.detach().requires_grad_()
    probabilities = probabilities.detach().requires_grad_()
    try:
        layer(hidden, ids, probabilities, tokens_per_expert=counts).backward(upstream)
        assert layer._virtual_executor is not None
        pool = layer._transfer_pool
        before = dict(pool.statistics)
        layer.zero_grad(set_to_none=True)
        hidden.grad = probabilities.grad = None
        torch.npu.synchronize()
        dist.barrier()
        with multicore_profiler.mega_kernel_profile(
                schedule=diagnostic_schedule(wait=0, warmup=0, active=1, repeat=1),
                detailed_task_names=True) as profiler:
            output = layer(hidden, ids, probabilities, tokens_per_expert=counts)
            profiler.step()
        path = Path(os.environ["HP_MEGA_MOE_PROFILE_RESULT_DIR"]) / f"rank{reference.RANK}_moon_trace.json"
        trace = profiler.export_chrome_trace(path)
        resources = layer._virtual_executor._resource_group.resources
        plan = resources.plan_for_tokens(shape.local_num_tokens, reference.DEVICE)
        assert plan.spec.num_experts == 2 * shape.num_experts
        _assert_complete_moon_forward_trace(trace, plan.spec)
        output.backward(upstream)
        torch.npu.synchronize()
        transfers = torch.tensor([
            pool.statistics["remote_prefetches"] - before["remote_prefetches"],
            pool.statistics["remote_gradient_returns"] - before["remote_gradient_returns"],
        ], dtype=torch.int64, device=reference.DEVICE)
        dist.all_reduce(transfers)
        prefetches, gradient_returns = transfers.cpu().tolist()
        assert prefetches > 0 and gradient_returns > 0
        assert layer.placement_statistics["moonep_calls"] >= 2
        trace["megaKernelCycleTrace"].update(
            testMoonForcedProfitable=True, testMoonRemotePrefetches=prefetches,
            testMoonRemoteGradientReturns=gradient_returns, testMoonVirtualExperts=plan.spec.num_experts)
        path.write_text(json.dumps(trace), encoding="utf-8")
        print(f"MOONEP_HOTSPOT_PROFILE_TRACE={path}", flush=True)
    finally:
        layer.close()
    dist.barrier()

@pytest.mark.usefixtures("_force_transfer_cost", "_routed_snapshot_contract")
def test_moonep_distinct_geometries_and_streams():
    """Keep independent HP plans and borrowed banks correct across two streams."""

    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    shapes = [reference.MoeShape(intermediate_size=size) for size in (128, 512, 256)]
    originals = [lifecycle._new_stack(shape, layers=1)[0] for shape in shapes]
    moons = [MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD,
                                  dispatch_mode=mode).to(device=reference.DEVICE, dtype=torch.bfloat16)
             for shape in shapes]
    for source, target in zip(originals, moons):
        target.load_state_dict(source.state_dict())
    streams = [torch.npu.Stream() for _ in range(2)]
    try:
        with patch.object(reference, "make_balanced_route",
                          side_effect=lambda current: reference._make_fixed_route(current, "single_destination")):
            for step in range(2):
                expected = lifecycle._stack_step(originals, shapes[0], streams, False, step)
                actual = lifecycle._stack_step(moons, shapes[0], streams, bool(step), step)
                for index, (left, right) in enumerate(zip(expected, actual)):
                    torch.testing.assert_close(right, left, rtol=2e-2, atol=1e-5,
                                               msg=f"distinct geometry step={step} tensor={index}")
        assert len({id(layer._transfer_pool) for layer in moons}) == len(shapes)
        _write({"dispatch_mode": mode, "intermediate_sizes": [shape.intermediate_size for shape in shapes],
                "streams": 2, "microbatches": 2, "checkpoint_modes": [False, True]})
    finally:
        gc.collect()
        for layer in originals + moons:
            layer.close()

@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_external_weights():
    """Keep HP external-weight autograd for contiguous and strided local shards."""
    shape = reference.MoeShape()
    records = []
    home = (torch.randn(shape.local_experts, shape.hidden_size, 2 * shape.intermediate_size,
                        device=reference.DEVICE, dtype=torch.bfloat16) * shape.hidden_size ** -0.5,
            torch.randn(shape.local_experts, shape.intermediate_size, shape.hidden_size,
                        device=reference.DEVICE, dtype=torch.bfloat16) * shape.intermediate_size ** -0.5)
    hidden, upstream = reference.make_data(shape)
    for mode in ("push", "pull"):
        baseline = reference.MegaMoeExperts(
            **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
            ep_group=dist.group.WORLD,
                                            dispatch_mode=mode, create_parameters=False)
        moon = MoonEPMegaMoeExperts(
            **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
            ep_group=dist.group.WORLD,
                                    dispatch_mode=mode, create_parameters=False)
        try:
            assert list(moon.state_dict()) == list(baseline.state_dict()) == []
            for strided in (False, True):
                for pattern in ("balanced", "single_destination"):
                    ids, probabilities, counts = reference._make_fixed_route(shape, pattern)
                    snapshots = []
                    for layer in (baseline, moon):
                        leaf = hidden.detach().clone().requires_grad_(True)
                        probs = probabilities.detach().clone().requires_grad_(True)
                        weights = tuple(weight.detach().clone() for weight in home)
                        if strided:
                            weights = tuple(weight.transpose(1, 2).contiguous().transpose(1, 2) for weight in weights)
                        weights = tuple(weight.requires_grad_(True) for weight in weights)
                        output = layer(leaf, ids, probs, tokens_per_expert=counts, expert_weights=weights)
                        gradients = torch.autograd.grad(output, (leaf, probs, *weights), upstream)
                        snapshots.append((output.detach().clone(), *(value.detach().clone() for value in gradients)))
                    for index, (expected, actual) in enumerate(zip(*snapshots)):
                        reference.assert_close(f"external mode={mode} strided={strided} tensor={index}",
                                               actual, expected)
                    records.append({"mode": mode, "strided": strided, "route": pattern, "passed": True})
        finally:
            moon.close()
            baseline.close()
    _write({"external_weights": records})

def test_moonep_profitability_policy():
    """Small/rank-balanced expert hotspots use ordinary HP with no placement resources."""

    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    shape = reference.MoeShape()
    baseline, common = reference.new_layers(shape, dispatch_mode=mode)
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD,
                                dispatch_mode=mode).to(device=reference.DEVICE, dtype=torch.bfloat16)
    moon.load_state_dict(baseline.state_dict())
    hidden, upstream = reference.make_data(shape)
    records = []
    try:
        with patch.object(moon_module, "build_placement", side_effect=AssertionError("unprofitable allocator")):
            for pattern in ("balanced", "zero_token_experts", "single_destination"):
                ids, probs, counts = reference._make_fixed_route(shape, pattern)
                expected = reference.run_layer(common, hidden, ids, probs, counts, upstream)
                original = reference.run_layer(baseline, hidden, ids, probs, counts, upstream)
                actual = reference.run_layer(moon, hidden, ids, probs, counts, upstream)
                reference.assert_results_close(expected, original)
                reference.assert_results_close(expected, actual)
                assert moon._virtual_executor is None, "Skipped route created a virtual workspace"
                assert moon._transfer_pool is None, "Skipped route created an expert bank"
                records.append({"pattern": pattern, "passed": True})
        assert moon.placement_statistics["native_balanced_calls"] == 2
        assert moon.placement_statistics["native_cost_skip_calls"] == 1
        _write({"dispatch_mode": mode, "production_policy": True,
                "scenarios": records, "placement": dict(moon.placement_statistics)})
    finally:
        moon.close()
        baseline.close()

def test_moonep_alternating_production_routes():
    """Real profitability selection remains correct across outstanding ordinary/borrowed graphs."""
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    checkpoint_mode = os.environ["HP_MOONEP_CHECKPOINT_MODE"]
    shape = reference.MoeShape(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                               num_experts=16, top_k=8)
    baseline, common = reference.new_layers(shape, dispatch_mode=mode)
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD,
                                dispatch_mode=mode).to(device=reference.DEVICE, dtype=torch.bfloat16)
    moon.load_state_dict(baseline.state_dict())
    del common
    hidden, upstream = reference.make_data(shape)
    routes = [reference._make_fixed_route(shape, pattern)
              for pattern in ("balanced", "single_destination", "balanced", "single_destination")]

    def _run(layer):
        layer.zero_grad(set_to_none=True)
        inputs = [hidden.detach().clone().requires_grad_(True) for _ in routes]
        probabilities = [route[1].detach().clone().requires_grad_(True) for route in routes]
        outputs = []
        for leaf, probs, (ids, _, counts) in zip(inputs, probabilities, routes):
            def _call(value, weights, current=layer, current_ids=ids, current_counts=counts):
                return current(value, current_ids, weights, tokens_per_expert=current_counts)
            outputs.append(_call(leaf, probs) if checkpoint_mode == "none" else
                           checkpoint(_call, leaf, probs, use_reentrant=False))
        for output in reversed(outputs):
            output.backward(upstream)
        result = {f"output_{index}": output.detach().clone() for index, output in enumerate(outputs)}
        result.update({f"input_{index}": leaf.grad.detach().clone() for index, leaf in enumerate(inputs)})
        result.update({f"probs_{index}": probs.grad.detach().clone()
                       for index, probs in enumerate(probabilities)})
        weights = reference.expert_weight_gradients(layer)
        result["weight1"], result["weight2"] = (weight.detach().clone() for weight in weights)
        return result

    try:
        expected = _run(baseline)
        actual = _run(moon)
        maxima = _compare(expected, actual)
        assert moon.placement_statistics["native_balanced_calls"] >= 2
        assert moon.placement_statistics["moonep_calls"] >= 2
        assert moon._transfer_pool.statistics["remote_gradient_returns"] >= 2
        _write({"production_policy": True, "dispatch_mode": mode, "checkpoint_mode": checkpoint_mode,
                "routes": ["balanced", "single_destination", "balanced", "single_destination"],
                "max_abs": maxima, "placement": dict(moon.placement_statistics),
                "statistics": dict(moon._transfer_pool.statistics)})
    finally:
        gc.collect()
        moon.close()
        baseline.close()

def test_moonep_fastpath_work_counts():
    """Count actual warm-path collectives, host copies and workspace operations."""

    shape = reference.MoeShape()
    baseline, _ = reference.new_layers(shape, dispatch_mode="push")
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD).to(
        device=reference.DEVICE, dtype=torch.bfloat16)
    moon.load_state_dict(baseline.state_dict())
    hidden, upstream = reference.make_data(shape)
    ids, probabilities, counts = reference.make_balanced_route(shape)
    records = {}
    try:
        for layer in (baseline, moon):
            for _ in range(5):
                reference._timed_fwd_bwd(layer, hidden, ids, probabilities, counts, upstream)
        for tag, layer in (("baseline", baseline), ("moonep", moon)):
            counter = {}
            def _watch(owner, name, label):
                original = getattr(owner, name)
                def _wrapped(*args, **kwargs):
                    started = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        record = counter.setdefault(label, {"calls": 0, "cpu_us": 0.0})
                        record["calls"] += 1
                        record["cpu_us"] += (time.perf_counter() - started) * 1e6
                return patch.object(owner, name, _wrapped)
            for _ in range(10):
                layer.zero_grad(set_to_none=True)
                leaf = hidden.detach().clone().requires_grad_(True)
                probs = probabilities.detach().clone().requires_grad_(True)
                with ExitStack() as stack:
                    for owner, name, label in (
                        (dist, "all_gather_into_tensor", "count_collective"),
                        (torch.Tensor, "tolist", "host_tolist"),
                        (torch.Tensor, "cpu", "host_cpu"),
                        (torch.npu, "synchronize", "global_synchronize"),
                        (torch_npu, "npu_moe_token_permute", "permute"),
                        (torch_npu, "npu_moe_token_unpermute", "unpermute"),
                        (MegaMoeWorkspace, "ensure", "workspace_ensure"),
                        (MegaMoeWorkspace, "claim", "workspace_claim"),
                        (MegaMoeWorkspace, "release", "workspace_release"),
                        (MegaMoeWorkspace, "wait_for_reuse", "workspace_wait"),
                    ):
                        stack.enter_context(_watch(owner, name, label))
                    output = layer(leaf, ids, probs, tokens_per_expert=counts)
                    output.backward(upstream)
                torch.npu.synchronize()
            assert counter["count_collective"]["calls"] == 10
            assert counter["host_tolist"]["calls"] == 10
            assert counter.get("global_synchronize", {"calls": 0})["calls"] == 0
            records[tag] = counter
        assert moon._virtual_executor is None and moon._transfer_pool is None
        start = time.perf_counter()
        for _ in range(1000):
            with torch.inference_mode(False), torch.no_grad():
                pass
        _write({"warm_iterations": 10, "work_counts": records,
                "persistent_guard_mean_cpu_us": (time.perf_counter() - start) * 1e3})
    finally:
        moon.close()
        baseline.close()

def test_moonep_native_inference_growth():
    """A cost-skipped inference hotspot may grow ordinary storage before training."""
    shape = reference.MoeShape()
    baseline, common = reference.new_layers(shape, dispatch_mode="push")
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD).to(
        device=reference.DEVICE, dtype=torch.bfloat16)
    moon.load_state_dict(baseline.state_dict())
    hidden, upstream = reference.make_data(shape)
    ids, probabilities, counts = reference._make_fixed_route(shape, "single_destination")
    try:
        with torch.inference_mode():
            moon(hidden, ids, probabilities, tokens_per_expert=counts)
        workspace = moon._resource_group.resources.workspace
        for value in vars(workspace).values():
            if isinstance(value, torch.Tensor):
                assert not value.is_inference(), "Persistent growth created an inference-only buffer"
        expected = reference.run_layer(common, hidden, ids, probabilities, counts, upstream)
        actual = reference.run_layer(moon, hidden, ids, probabilities, counts, upstream)
        reference.assert_results_close(expected, actual)
        assert moon._virtual_executor is None
        _write({"native_inference_growth": True, "placement": dict(moon.placement_statistics)})
    finally:
        moon.close()
        baseline.close()


@pytest.mark.usefixtures("_force_transfer_cost")
def test_moonep_dynamic_shared_tokens():
    """Keep ragged and empty source graphs over growth, shared layers and EIO/EOI banks."""
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    checkpoint_mode = os.environ["HP_MOONEP_CHECKPOINT_MODE"]
    records = []
    world_size = dist.get_world_size()
    base_shape = reference.MoeShape(num_experts=2 * world_size, ep_size=world_size)
    for layout in ("EIO", "EOI"):
        pairs = [_new_pair(base_shape, mode, layout) for _ in range(2)]
        moons = [pair[0] for pair in pairs]
        commons = [pair[1] for pair in pairs]
        MoonEPMegaMoeExperts.share_execution_resources(moons)
        retained = []
        try:
            warm_shape = base_shape
            warm_hidden, _ = reference.make_data(warm_shape)
            warm_route = reference._make_fixed_route(warm_shape, "single_destination")
            with torch.inference_mode():
                _forward(moons, warm_hidden, [warm_route[1]] * 2, [warm_route] * 2, "none")
            manager = moons[0]._transfer_pool.manager
            initial_epoch = manager.epoch
            ordinary = moons[0]._resource_group.resources
            assert ordinary.workspace.forward_event_counters is None, (
                "A borrowed-only call materialized the unused ordinary workspace")
            epochs = []
            capacities = []
            for lengths in ((0, 129), (257, 64), (64, 513), (128, 128), (64, 513)):
                shape = reference.MoeShape(
                    local_num_tokens=lengths[dist.get_rank() % len(lengths)],
                    num_experts=base_shape.num_experts, ep_size=world_size,
                )
                source, upstream = reference.make_data(shape)
                ids, probabilities, counts = reference._make_fixed_route(shape, "single_destination")
                source_common = source.detach().clone().requires_grad_(True)
                source_moon = source.detach().clone().requires_grad_(True)
                common_probs = [probabilities.detach().clone().requires_grad_(True) for _ in range(2)]
                moon_probs = [probabilities.detach().clone().requires_grad_(True) for _ in range(2)]
                routes = [(ids, probabilities, counts)] * 2
                expected = _forward(commons, source_common, common_probs, routes, "none")
                actual = _forward(moons, source_moon, moon_probs, routes, checkpoint_mode)
                reference.assert_close("dynamic output", actual, expected)
                resources = moons[0]._virtual_executor._resource_group.resources
                epochs.append(manager.epoch)
                capacities.append({"input_rows": [lengths[rank % len(lengths)] for rank in range(world_size)],
                                   "epoch": manager.epoch,
                                   "source_capacity": resources._token_capacity,
                                   "receive_capacity": resources.workspace.capacity_floor,
                                   "heap_bytes": manager.heap_bytes})
                retained.append((expected, actual, upstream, source_common, source_moon, common_probs, moon_probs))
            assert epochs[-1] == epochs[-2] == epochs[-3], (
                "Previously observed peak rows caused another SHMEM restart")
            assert ordinary.workspace.forward_event_counters is None, (
                "Borrowed routes grew an unused ordinary workspace")
            for expected, actual, upstream, left, right, left_probs, right_probs in reversed(retained):
                expected.backward(upstream)
                actual.backward(upstream)
                reference.assert_close("dynamic dX", right.grad, left.grad)
                for expected_prob, actual_prob in zip(left_probs, right_probs):
                    reference.assert_close("dynamic dProbs", actual_prob.grad, expected_prob.grad)
            for moon, common in pairs:
                for expected_grad, actual_grad in zip(reference.expert_weight_gradients(common),
                                                       reference.expert_weight_gradients(moon)):
                    reference.assert_close("dynamic dW", actual_grad, expected_grad)
                for parameter in moon.parameters():
                    assert parameter.grad.stride() == parameter.stride()
            pool = moons[0]._transfer_pool
            assert pool.statistics["remote_gradient_returns"] > 0
            assert pool.manager.epoch > initial_epoch
            assert pool.manager.epoch == epochs[-1], "Backward replay grew a previously committed route"
            assert (moons[0]._virtual_executor._resource_group.resources
                    is moons[1]._virtual_executor._resource_group.resources)
            assert all(not tensor.is_inference() for tensor in pool.weights)
            records.append({"layout": layout, "source_lengths": [0, 129, 257, 64, 513, 128],
                            "initial_epoch": initial_epoch, "final_epoch": pool.manager.epoch,
                            "statistics": dict(pool.statistics), "pending_forwards": 5,
                            "capacity_trace": capacities})
        finally:
            retained.clear()
            gc.collect()
            for moon in moons:
                moon.close()
    _write({"dynamic_shared_tokens": records, "checkpoint_mode": checkpoint_mode, "dispatch_mode": mode})


def test_moonep_hotspot() -> None:
    """Match common MoE and HP baseline on output, dX, probabilities and dW."""
    shape = reference.MoeShape()
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    baseline, common = reference.new_layers(shape)
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD,
                               dispatch_mode=mode).to(device=reference.DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        moon.gate_up_weight.copy_(baseline.gate_up_weight)
        moon.down_weight.copy_(baseline.down_weight)
    hidden, grad = reference.make_data(shape)
    ids, probs, counts = reference._make_fixed_route(shape, "single_destination")
    try:
        expected = reference.run_layer(common, hidden, ids, probs, counts, grad)
        original = reference.run_layer(baseline, hidden, ids, probs, counts, grad)
        allocate = placement_module.allocate_experts
        placements = []

        def recorded_allocate(counts: torch.Tensor, capacity: int) -> list[torch.Tensor]:
            """Record the real native plan without modifying the native module."""
            result = allocate(counts, capacity)
            placements.append(result[-1].cpu().tolist())
            return result

        with patch.object(placement_module, "allocate_experts", side_effect=recorded_allocate), \
             patch.object(MoonEPMegaMoeExperts, "_placement_is_profitable", return_value=True):
            actual = reference.run_layer(moon, hidden, ids, probs, counts, grad)
        assert placements and any(expert >= 0 for plan in placements for row in plan for expert in row), (
            "Hotspot allocator never assigned a borrowed expert")
        assert any(expert >= 0 and expert // shape.local_experts != rank
                   for plan in placements for rank, row in enumerate(plan) for expert in row), (
            "Hotspot placement did not assign an expert to a remote rank")
        reference.assert_results_close(expected, original)
        reference.assert_results_close(expected, actual)
        pool = moon._transfer_pool
        assert pool is not None, "Hotspot did not create a shared expert bank"
        assert pool.statistics["remote_prefetches"] > 0, "Hotspot never fetched borrowed experts"
        assert pool.statistics["remote_gradient_returns"] > 0, "Hotspot never returned borrowed gradients"
        record = {"rank": dist.get_rank(), "dispatch_mode": mode, "statistics": dict(pool.statistics),
                  "allocator_copy_plans": placements, "transfer_cost_forced": True}
        evidence = [None] * dist.get_world_size()
        dist.all_gather_object(evidence, record)
        if dist.get_rank() == 0 and os.getenv("HP_MOONEP_HOTSPOT_RESULT"):
            Path(os.environ["HP_MOONEP_HOTSPOT_RESULT"]).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"MOONEP_HOTSPOT_PASS rank={dist.get_rank()} dispatch={mode}", flush=True)
    finally:
        moon.close()
        baseline.close()


# Pair transport and checkpoint variants instead of repeating their Cartesian product.
_CONTRACT_CASES = {
    "push": (
        (test_moonep_hotspot, "none"),
        (test_moonep_original_precision, "none"),
        (test_moonep_checkpoint_retained_layers, "true"),
        (test_moonep_stream_exception_recovery, "none"),
        (test_moonep_original_poisoned_buffers, "none"),
        (test_moonep_heap_rebuild_with_live_graph, "none"),
        (test_moonep_clipping_with_borrowed_experts, "none"),
        (test_moonep_external_weights, "none"),
        (test_moonep_profitability_policy, "none"),
        (test_moonep_alternating_production_routes, "none"),
        (test_moonep_fastpath_work_counts, "none"),
        (test_moonep_native_inference_growth, "none"),
        (test_moonep_dynamic_shared_tokens, "false"),
        (test_moonep_prefetch_ub_reject, "none"),
    ),
    "pull": (
        (test_moonep_hotspot, "none"),
        (test_moonep_checkpoint_retained_layers, "false_full"),
        (test_moonep_heap_rebuild_with_live_graph, "false"),
        (test_moonep_inference_then_training, "none"),
        (test_moonep_distinct_geometries_and_streams, "none"),
        (test_moonep_alternating_production_routes, "false"),
        (test_moonep_dynamic_shared_tokens, "true"),
    ),
}


@pytest.mark.parametrize(
    "case,checkpoint_mode",
    [pytest.param(case, mode, id=f"{case.__name__}-{mode}")
     for case, mode in _CONTRACT_CASES[os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")]],
)
def test_moonep_contracts(
    case: Callable[[], None], checkpoint_mode: str, request: pytest.FixtureRequest, tmp_path: Path,
) -> None:
    """Reuse the worker process while keeping each scenario's fixtures and environment isolated."""
    with patch.dict(os.environ):
        os.environ.pop("HYPER_PARALLEL_SHMEM_HEAP_SIZE", None)
        os.environ["HP_MOONEP_CHECKPOINT_MODE"] = checkpoint_mode
        evidence = str(tmp_path / "result.json")
        os.environ["HP_MOONEP_LIFECYCLE_RESULT"] = evidence
        os.environ["HP_MOONEP_HOTSPOT_RESULT"] = evidence
        os.environ["HP_MEGA_MOE_LEVEL1_RESULT"] = evidence
        for marker in getattr(case, "pytestmark", ()):
            if marker.name == "usefixtures":
                for fixture in marker.args:
                    request.getfixturevalue(fixture)
        case()
        dist.barrier()
