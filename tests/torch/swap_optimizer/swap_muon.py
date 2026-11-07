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
"""Torch Muon momentum-only swap optimizer ST cases.

These cases run the plain single-card Muon path and the momentum-swapped path
over the very same inputs and compare them parameter by parameter, plus the
device/CPU lifecycle of the swapped momentum buffers.  HSDP collective ordering
and multi-rank checkpoint behaviour are covered separately.
"""

import gc
from typing import Dict, List, Optional, Tuple
from unittest import mock

import torch
import torch.distributed as dist

from hyper_parallel import DTensor, Replicate, Shard, SkipDTensorDispatch, distribute_tensor, init_device_mesh
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer import Muon, SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.torch.utils import init_dist

_TRAIN_STEPS = 8
_RTOL = 0.0
_ATOL = 0.0
_INPUT_DIM = 48
_HIDDEN_DIM = 64
_OUTPUT_DIM = 32
_MODEL_SEED = 7
_BATCH_SEED = 99


def _release_device_memory() -> None:
    """Release cached device memory after dropping Python references."""
    gc.collect()
    torch.npu.empty_cache()


def _fixed_batches() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build inputs once so every run sees identical data regardless of ordering."""
    generator = torch.Generator().manual_seed(_BATCH_SEED)
    batches = []
    for step in range(_TRAIN_STEPS):
        x = torch.randn(_INPUT_DIM, 4, generator=generator) * 0.1 + step * 0.001
        target = torch.randn(_OUTPUT_DIM, 4, generator=generator) * 0.1
        batches.append((x.npu(), target.npu()))
    return batches


class _MuonNet(torch.nn.Module):
    """Small deterministic matrix-only network matching Muon's 2D constraint."""

    def __init__(self) -> None:
        """Initialize the network weights."""
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.randn(_HIDDEN_DIM, _INPUT_DIM) * 0.02)
        self.weight1 = torch.nn.Parameter(torch.randn(_OUTPUT_DIM, _HIDDEN_DIM) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward network."""
        return self.weight1 @ torch.relu(self.weight0 @ x)


def _initial_state() -> Dict[str, torch.Tensor]:
    """Return the canonical initial weights every run starts from."""
    torch.manual_seed(_MODEL_SEED)
    return {name: param.detach().clone() for name, param in _MuonNet().named_parameters()}


def _build(
        use_swap: bool,
        swap_times: int,
        initial: Dict[str, torch.Tensor],
        packed_swap: bool = False,
):
    """Build a network/optimizer pair starting from ``initial`` weights."""
    torch.manual_seed(_MODEL_SEED)
    model = _MuonNet().npu()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(initial[name])
    optimizer = Muon(model.parameters(), lr=0.013, weight_decay=0.02, momentum=0.93)
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=swap_times, min_numel=1, packed_swap=packed_swap),
        )
    return model, optimizer


def _train(model, optimizer, batches, steps=None) -> List[float]:
    """Train ``model`` for ``steps`` steps and return the per-step losses."""
    losses = []
    for x, target in batches if steps is None else batches[:steps]:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _assert_losses_align(base: List[float], swap: List[float]) -> None:
    """Assert two loss trajectories are identical."""
    assert len(base) == len(swap), f"loss count mismatch: {len(base)} vs {len(swap)}"
    for step, (base_loss, swap_loss) in enumerate(zip(base, swap)):
        assert abs(base_loss - swap_loss) <= _ATOL + _RTOL * abs(base_loss), (
            f"step {step} loss mismatch: base={base_loss!r} swap={swap_loss!r}"
        )


def _assert_swap_state_offloaded(optimizer, expected_params: int) -> None:
    """Assert every momentum slot is host-resident with no device storage."""
    assert bool(getattr(optimizer, "_is_swap_optimizer", False))
    slots = tuple(optimizer.adapter.all_slots())
    assert len(slots) == expected_params, f"expected {expected_params} slots, got {len(slots)}"
    assert all(slot.name == "momentum_buffer" for slot in slots)
    assert all(slot.swappable for slot in slots), "momentum slots must be swap-eligible"
    assert all(slot.state == "host" for slot in slots), "momentum slots must be offloaded"
    assert all(slot.cpu_tensor is not None for slot in slots), "host mirrors must exist"
    for slot in slots:
        local_tensor = to_local_if_dtensor(slot.tensor)
        if slot.packed:
            assert local_tensor.device.type == "cpu", "packed momentum must point at its persistent host view"
            assert slot.tensor is slot.cpu_tensor, "packed state must not retain a staging-arena view"
        else:
            assert local_tensor.untyped_storage().size() == 0, "device storage must be released after the step"


def _run_align_case(case_name: str, swap_times: int, packed_swap: bool = False) -> None:
    """Compare bare and swapped Muon over identical inputs for ``swap_times``."""
    print(case_name)
    batches = _fixed_batches()
    initial = _initial_state()

    base_model, base_optimizer = _build(False, swap_times, initial)
    base_losses = _train(base_model, base_optimizer, batches)
    base_params = {name: param.detach().clone() for name, param in base_model.named_parameters()}
    del base_optimizer
    _release_device_memory()

    swap_model, swap_optimizer_inst = _build(True, swap_times, initial, packed_swap=packed_swap)
    swap_losses = _train(swap_model, swap_optimizer_inst, batches)

    _assert_losses_align(base_losses, swap_losses)
    for name, base_param in base_params.items():
        swap_param = dict(swap_model.named_parameters())[name].detach()
        diff = (base_param - swap_param).abs().max().item()
        assert diff <= 1e-6, f"parameter {name} mismatch: {diff}"
    _assert_swap_state_offloaded(swap_optimizer_inst, expected_params=len(base_params))
    _release_device_memory()


def test_swap_muon_parameter_align() -> None:
    """
    Feature: Muon momentum-only swap optimizer numerical parity.
    Description: Train the same network with plain Muon and with Muon wrapped by the swap optimizer over
        identical inputs, then compare per-step losses and final parameters.
    Expectation: Losses and parameters align exactly, and every momentum slot ends host-resident.
    """
    _run_align_case("test_swap_muon_parameter_align", swap_times=2)


def test_swap_muon_swap_times_parameter_align() -> None:
    """
    Feature: Muon swap optimizer with different transfer batch counts.
    Description: Repeat the parity case with swap_times=1 and swap_times=16, which put every update unit in
        one batch and spread them across many batches respectively.
    Expectation: Both schedules match the unswapped trajectory exactly.
    """
    print("test_swap_muon_swap_times_parameter_align")
    _run_align_case("test_swap_muon_swap_times_parameter_align[swap_times=1]", swap_times=1)
    _run_align_case("test_swap_muon_swap_times_parameter_align[swap_times=16]", swap_times=16)


def test_swap_muon_post_update_fn_align() -> None:
    """
    Feature: Muon swap optimizer with a post-update callback.
    Description: Run both paths with post_update_fn recording parameter context, so the callback still fires once
        per updated parameter and before the momentum slot is offloaded.
    Expectation: The recorded callback sequence is identical and losses align.
    """
    print("test_swap_muon_post_update_fn_align")
    batches = _fixed_batches()
    initial = _initial_state()

    def _run(use_swap: bool):
        torch.manual_seed(_MODEL_SEED)
        model = _MuonNet().npu()
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(initial[name])
        seen = []
        optimizer = Muon(
            model.parameters(), lr=0.013, momentum=0.93,
            post_update_fn=lambda param, tensor, context: seen.append(
                (context.step, getattr(param, "model_name", None), float(tensor.abs().sum()))
            ),
        )
        if use_swap:
            optimizer = swap_optimizer(
                optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False)
            )
        return model, optimizer, seen

    base_model, base_optimizer, base_seen = _run(False)
    base_losses = _train(base_model, base_optimizer, batches)
    del base_optimizer
    _release_device_memory()

    swap_model, swap_optimizer_inst, swap_seen = _run(True)
    swap_losses = _train(swap_model, swap_optimizer_inst, batches)

    _assert_losses_align(base_losses, swap_losses)
    assert base_seen == swap_seen, "post_update_fn must observe the same updates in the same order"
    _release_device_memory()


def test_swap_muon_checkpoint_host_state() -> None:
    """
    Feature: Muon swap optimizer checkpoint export while all momentum sits on host.
    Description: Export the state dict after training, then reload it into a fresh swap optimizer and assert the
        checkpoint carries real momentum values on CPU rather than released device placeholders.
    Expectation: Exported momentum tensors are dense CPU tensors, and the reloaded optimizer keeps them
        host-resident until the next step prefetches them.
    """
    print("test_swap_muon_checkpoint_host_state")
    batches = _fixed_batches()
    initial = _initial_state()

    model, optimizer = _build(True, 2, initial)
    _train(model, optimizer, batches)
    state_dict = optimizer.state_dict()

    momentum = [
        saved["momentum_buffer"]
        for saved in state_dict["state"].values()
        if "momentum_buffer" in saved
    ]
    assert len(momentum) == len(initial), f"expected {len(initial)} momentum entries, got {len(momentum)}"
    for tensor in momentum:
        assert tensor.device.type == "cpu", f"checkpoint momentum must be CPU, got {tensor.device}"
        assert tensor.untyped_storage().size() == tensor.numel() * tensor.element_size(), (
            "checkpoint momentum must not expose released storage"
        )
        assert torch.count_nonzero(tensor) > 0, "checkpoint momentum must carry real values"

    reloaded_model, reloaded = _build(True, 2, initial)
    reloaded.load_state_dict(state_dict)

    slots = tuple(reloaded.adapter.all_slots())
    assert slots, "reloading must register momentum slots"
    assert all(slot.state == "host" for slot in slots)
    assert all(
        to_local_if_dtensor(slot.tensor).untyped_storage().size() == 0 for slot in slots
    )

    # The next step must prefetch, update and offload again.
    _train(reloaded_model, reloaded, batches, steps=1)
    _assert_swap_state_offloaded(reloaded, expected_params=len(initial))
    _release_device_memory()


def test_swap_muon_hsdp_checkpoint_8p() -> None:
    """
    Feature: HSDP checkpoint lifecycle for the swap Muon optimizer.
    Description: Run fully-sharded Muon on a (replicate=2, shard=4) mesh, save
        and reload a checkpoint, continue training, and save again.
    Expectation: Every rank exports dense CPU momentum values, reload restores
        owner state only, and local slots return to host after every checkpoint.
    """
    print("test_swap_muon_hsdp_checkpoint_8p")
    init_dist()
    torch.manual_seed(_MODEL_SEED)
    model = fully_shard(
        _MuonNet().npu(),
        mesh=init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("replicate", "shard")
        ),
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(),
    )
    optimizer = Muon(model.parameters(), lr=0.013, weight_decay=0.02, momentum=0.93,
                     hsdp_replica_count=2)
    optimizer = swap_optimizer(
        optimizer,
        SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False),
    )

    batches = _fixed_batches()
    _train(model, optimizer, batches, steps=2)
    state_dict = optimizer.state_dict()

    momentum = [
        saved["momentum_buffer"]
        for saved in state_dict["state"].values()
        if "momentum_buffer" in saved
    ]
    assert momentum, "HSDP checkpoint must contain replicated Muon momentum"
    assert all(tensor.device.type == "cpu" for tensor in momentum)
    assert all(torch.count_nonzero(to_local_if_dtensor(tensor)) > 0 for tensor in momentum)

    slots = tuple(optimizer.adapter.all_slots())
    assert slots, "owner ranks must retain local momentum slots"
    assert all(slot.state == "host" for slot in slots)
    assert all(slot.cpu_tensor is not None for slot in slots)
    assert all(
        to_local_if_dtensor(slot.tensor).untyped_storage().size() == 0 for slot in slots
    )

    optimizer.load_state_dict(state_dict)
    local_param_ids = {
        id(param)
        for params_by_group in (optimizer.unshard_params_by_group, optimizer.shard_params_by_group)
        for params in params_by_group.values()
        for param in params
    }
    assert local_param_ids, "HSDP must expose rank-local ownership"
    assert all(id(param) in local_param_ids for param in optimizer.state), (
        "checkpoint reload must not retain non-owner momentum state"
    )
    assert {id(slot.tensor) for slot in optimizer.adapter.all_slots()} == {
        id(state["momentum_buffer"]) for state in optimizer.state.values()
    }

    _train(model, optimizer, batches, steps=1)
    resumed_state_dict = optimizer.state_dict()
    resumed_momentum = [
        saved["momentum_buffer"]
        for saved in resumed_state_dict["state"].values()
        if "momentum_buffer" in saved
    ]
    assert resumed_momentum
    assert all(tensor.device.type == "cpu" for tensor in resumed_momentum)
    assert all(slot.state == "host" for slot in optimizer.adapter.all_slots())
    assert all(
        to_local_if_dtensor(slot.tensor).untyped_storage().size() == 0
        for slot in optimizer.adapter.all_slots()
    )
    _release_device_memory()


def _schedule_params(mesh):
    """Build uneven shard and unshard assignments plus a no-comm parameter."""
    shapes = ((6, 8), (17, 8), (4, 8), (1, 8), (8, 8, 8), (8, 4, 12), (8, 3, 6), (4, 2, 8, 8))
    generator = torch.Generator().manual_seed(_MODEL_SEED)
    params = []
    for index, shape in enumerate(shapes):
        value = (torch.randn(shape, generator=generator) * 0.02).npu()
        if index:
            value = distribute_tensor(value, mesh, [Replicate(), Shard(0)])
        param = torch.nn.Parameter(value)
        param.model_name = f"weight{index}"
        params.append(param)
    return params


def _assign_schedule_grads(params, step):
    """Keep logical gradients fixed across ranks, including empty local shards."""
    generator = torch.Generator().manual_seed(_BATCH_SEED + step)
    for index, param in enumerate(params):
        if index == 0 and step == 1:
            param.grad = None
            continue
        grad = (torch.randn(tuple(param.shape), generator=generator) * 0.05).npu()
        if isinstance(param, DTensor):
            grad = distribute_tensor(grad, param.device_mesh, param.placements)
        param.grad = grad


def _step_collective_trace(optimizer):
    """Run real collectives while recording membership, order and payload metadata."""
    trace = []
    real_broadcast = dist.broadcast
    real_gather = dist.all_gather_into_tensor

    def broadcast(
            tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None, async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Record the real broadcast and preserve its asynchronous work handle."""
        trace.append(("broadcast", tuple(dist.get_process_group_ranks(group)), src,
                      tuple(tensor.shape), str(tensor.dtype), async_op))
        return real_broadcast(tensor, src=src, group=group, async_op=async_op)

    def gather(
            output: torch.Tensor, tensor: torch.Tensor,
            group: Optional[dist.ProcessGroup] = None, async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Record the real input gather without changing its completion semantics."""
        trace.append(("all_gather", tuple(dist.get_process_group_ranks(group)),
                      tuple(output.shape), tuple(tensor.shape), str(tensor.dtype), async_op))
        return real_gather(output, tensor, group=group, async_op=async_op)

    with mock.patch.object(dist, "broadcast", side_effect=broadcast), \
            mock.patch.object(dist, "all_gather_into_tensor", side_effect=gather), SkipDTensorDispatch():
        optimizer.step()
    return trace


def _assert_global_schedule(pipeline):
    """Compare the executed batch plan across ranks after optimizer collectives finish."""
    batches = pipeline.call_args.args[0]
    signature = [[(unit.schedule_key, unit.logical_nbytes, unit.flushes_broadcast) for unit in batch]
                 for batch in batches]
    local_bytes = sum(slot.storage_nbytes for batch in batches for unit in batch
                      for slot in unit.slots if slot.swappable)
    empty_owner = any(unit.kind == "hsdp" and not unit.hsdp_assign.owned_params
                      for batch in batches for unit in batch)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (signature, local_bytes, empty_owner))
    assert all(item[0] == signature for item in gathered), (
        f"Expected identical logical batches: local={signature}, gathered={gathered}"
    )
    assert len({item[1] for item in gathered}) > 1, f"Expected unequal transfer bytes, got {gathered}"
    assert any(item[2] for item in gathered), f"Expected an empty owner assignment, got {gathered}"


def _run_hsdp_logical_schedule(shard_size, packed_swap=False):
    """Compare global plans and real collective/numerical behavior on an HSDP mesh."""
    init_dist()
    mesh = init_device_mesh("npu", (2, shard_size), mesh_dim_names=("replicate", "shard"))
    for swap_times in (1, 2, 16):
        bare_params = _schedule_params(mesh)
        swapped_params = _schedule_params(mesh)
        bare = Muon(bare_params, lr=0.013, momentum=0.93, hsdp_replica_count=2)
        swapped = swap_optimizer(
            Muon(swapped_params, lr=0.013, momentum=0.93, hsdp_replica_count=2),
            SwapOptimizerConfig(swap_times=swap_times, min_numel=1, packed_swap=packed_swap),
        )
        for step in range(3):
            _assign_schedule_grads(bare_params, step)
            _assign_schedule_grads(swapped_params, step)
            bare_trace = _step_collective_trace(bare)
            with mock.patch.object(swapped.runtime, "run_pipeline", wraps=swapped.runtime.run_pipeline) as pipeline:
                swapped_trace = _step_collective_trace(swapped)
            assert bare_trace == swapped_trace, f"Collective mismatch: bare={bare_trace}, swap={swapped_trace}"
            _assert_global_schedule(pipeline)
            for param, swapped_param in zip(bare_params, swapped_params):
                torch.testing.assert_close(
                    to_local_if_dtensor(param), to_local_if_dtensor(swapped_param), rtol=1e-5, atol=1e-6,
                )
            for slot in swapped.adapter.all_slots():
                if slot.swappable:
                    assert slot.state == "host", f"Expected host slot, got {slot.state}"
                    if slot.packed:
                        assert slot.tensor is slot.cpu_tensor, "Packed HSDP state retained a staging view"
                    else:
                        assert to_local_if_dtensor(slot.tensor).untyped_storage().size() == 0, (
                            f"Expected released storage for {slot.name}, got {slot.state}"
                        )
            with SkipDTensorDispatch():
                bare_state = bare.state_dict()
                swapped_state = swapped.state_dict()
                for param_id, state in bare_state["state"].items():
                    torch.testing.assert_close(
                        to_local_if_dtensor(state["momentum_buffer"]).cpu(),
                        swapped_state["state"][param_id]["momentum_buffer"].cpu(), rtol=0, atol=0,
                    )
                if step == 1:
                    swapped.load_state_dict(swapped_state)
        del bare, swapped, bare_params, swapped_params
        _release_device_memory()


def test_swap_muon_hsdp_logical_schedule_4p() -> None:
    """Check uneven ownership and empty shards on a (replicate=2, shard=2) mesh."""
    _run_hsdp_logical_schedule(2)


def test_swap_muon_hsdp_logical_schedule_8p() -> None:
    """Verify global batch signatures, collective parity and checkpoint continuation.

    Covers swap_times=1/2/16, no-comm, shard/unshard assignments, uneven and empty
    local shards, unequal ownership and an assignment with an empty owner rank.
    """
    _run_hsdp_logical_schedule(4)


def test_swap_muon_hsdp_packed_logical_schedule_4p() -> None:
    """Verify packed local layouts preserve the global schedule on four devices."""
    _run_hsdp_logical_schedule(2, packed_swap=True)


def test_swap_muon_hsdp_packed_logical_schedule_8p() -> None:
    """Verify packed HSDP/DTensor state and checkpoint continuation on eight devices."""
    _run_hsdp_logical_schedule(4, packed_swap=True)


def test_swap_muon_first_step_is_lazy_and_offloaded() -> None:
    """
    Feature: Muon swap optimizer first-step lazy momentum creation.
    Description: Wrap Muon before any state exists, run a single step, and check that the lazily created momentum
        never stays device-resident.
    Expectation: After the first step every momentum slot is host-resident with released device storage.
    """
    print("test_swap_muon_first_step_is_lazy_and_offloaded")
    batches = _fixed_batches()
    initial = _initial_state()

    model, optimizer = _build(True, 2, initial)
    assert not optimizer.state, "the optimizer must start with no materialized state"

    _train(model, optimizer, batches, steps=1)

    _assert_swap_state_offloaded(optimizer, expected_params=len(initial))
    _release_device_memory()


def _mixed_reset_optimizer(
        use_swap: bool, warm: bool, packed_adam: bool,
        mesh: Optional[object] = None, packed_muon: bool = False,
):
    """Replace BF16 parameters after optionally offloading pre-existing state."""
    model = torch.nn.Linear(8, 8, dtype=torch.bfloat16).npu()
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(1)
    if mesh is not None:
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Replicate(), Shard(0)]))
    muon = Muon([model.weight], lr=0.013)
    adam = torch.optim.AdamW([model.bias], lr=0.013, foreach=False)
    if warm and use_swap:
        muon.state[model.weight]["momentum_buffer"] = torch.full_like(model.weight, 0.25)
        adam.state[model.bias] = {
            "step": torch.tensor(2.), "exp_avg": torch.full_like(model.bias, 0.125),
            "exp_avg_sq": torch.full_like(model.bias, 0.5),
        }
    if use_swap:
        muon = swap_optimizer(muon, SwapOptimizerConfig(min_numel=1, packed_swap=packed_muon))
        adam = swap_optimizer(adam, SwapOptimizerConfig(min_numel=1, packed_swap=packed_adam))
    chain = ChainedOptimizer(model, {"muon": muon, "adamw": adam}, flatten=True)
    optimizer = Float16OptimizerWithFloat16Params(chain, model)
    if warm and not use_swap:
        owned = muon.unshard_params_by_group[0] + muon.shard_params_by_group[0]
        for param in owned:
            muon.state[param]["momentum_buffer"] = torch.full_like(param, 0.25)
        adam.state[model.bias.main_param] = {
            "step": torch.tensor(2.), "exp_avg": torch.full_like(model.bias.main_param, 0.125),
            "exp_avg_sq": torch.full_like(model.bias.main_param, 0.5),
        }
    return model, optimizer


def _step_mixed_precision_optimizer(model, optimizer, step: int) -> None:
    """Assign deterministic gradients and advance one mixed-precision optimizer."""
    for param in model.parameters():
        if isinstance(param, DTensor):
            gradient = torch.zeros_like(param)
            gradient.fill_(step + 1)
            param.main_grad = gradient
        else:
            param.grad = torch.full_like(param, step + 1)
    with SkipDTensorDispatch(no_skip={torch.zeros_like}):
        optimizer.step()
        optimizer.zero_grad()


def _run_mixed_precision_reset(shard_size: Optional[int] = None, packed_muon: bool = False) -> None:
    """Compare reset and checkpoint continuation on local or sharded main params."""
    init_dist()
    mesh = None if shard_size is None else init_device_mesh(
        "npu", (2, shard_size), mesh_dim_names=("replicate", "shard"),
    )
    for warm in (False, True):
        for packed_adam in (False, True):
            with SkipDTensorDispatch():
                reference_model, reference = _mixed_reset_optimizer(False, warm, packed_adam, mesh)
                model, optimizer = _mixed_reset_optimizer(True, warm, packed_adam, mesh, packed_muon)
            muon = optimizer.optimizers_dict["muon"]
            adam = optimizer.optimizers_dict["adamw"]
            owned = len(muon.unshard_params_by_group[0]) + len(muon.shard_params_by_group[0])
            if warm and owned:
                _assert_swap_state_offloaded(muon, expected_params=owned)
                slot = next(iter(muon.adapter.all_slots()))
                torch.testing.assert_close(slot.cpu_tensor, torch.full(slot.shape, 0.25), rtol=0, atol=0)
            for step in range(3):
                for current_model, current_optimizer in ((reference_model, reference), (model, optimizer)):
                    _step_mixed_precision_optimizer(current_model, current_optimizer, step)
                for expected, actual in zip(reference_model.parameters(), model.parameters()):
                    torch.testing.assert_close(
                        to_local_if_dtensor(actual.main_param), to_local_if_dtensor(expected.main_param),
                        rtol=0, atol=0,
                        msg=lambda message, warm=warm, packed_adam=packed_adam, step=step: (
                            f"warm={warm}, packed_adam={packed_adam}, step={step}: {message}"
                        ),
                    )
                    torch.testing.assert_close(
                        to_local_if_dtensor(actual), to_local_if_dtensor(expected), rtol=0, atol=0,
                    )
                _assert_swap_state_offloaded(muon, expected_params=owned)
                assert muon.runtime.copy_stream is not adam.runtime.copy_stream, (
                    f"Expected independent streams, got muon={muon.runtime.copy_stream}, "
                    f"adam={adam.runtime.copy_stream}"
                )
                if step == 1:
                    with SkipDTensorDispatch():
                        checkpoint = optimizer.state_dict()
                        reference_state = reference.state_dict()
                    for key in ("state.weight.momentum_buffer", "state.bias.exp_avg", "state.bias.exp_avg_sq"):
                        actual_state = to_local_if_dtensor(checkpoint[key]).cpu()
                        expected_state = to_local_if_dtensor(reference_state[key]).cpu()
                        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
                    with SkipDTensorDispatch():
                        optimizer.load_state_dict(checkpoint)
            del reference_model, reference, model, optimizer, muon, adam
            _release_device_memory()


def test_swap_muon_mixed_precision_reset() -> None:
    """Verify real offloaded-state migration, two-leaf isolation and checkpoint resume."""
    _run_mixed_precision_reset()


def test_swap_muon_hsdp_mixed_precision_reset_4p() -> None:
    """Verify main-parameter identity, owner state and shard collectives after reset."""
    _run_mixed_precision_reset(2)


def test_swap_muon_packed_mixed_precision_reset() -> None:
    """Verify packed-state reset, main-parameter copies and checkpoint continuation."""
    _run_mixed_precision_reset(packed_muon=True)


def test_swap_muon_hsdp_packed_mixed_precision_reset_4p() -> None:
    """Verify logical DTensor shells are rebound to packed FP32 main state."""
    _run_mixed_precision_reset(2, packed_muon=True)


def test_swap_muon_packed_swap() -> None:
    """
    Feature: Muon packed momentum staging.
    Description: Compare packed Muon with bare Muon for several transfer counts, then verify lazy state and
        checkpoint continuation keep persistent host views outside the update window.
    Expectation: Numerical results align and no optimizer state retains an A/B staging-arena view.
    """
    print("test_swap_muon_packed_swap")
    for swap_times in (1, 2, 16):
        _run_align_case(
            f"test_swap_muon_packed_swap[swap_times={swap_times}]",
            swap_times,
            packed_swap=True,
        )

    batches = _fixed_batches()
    initial = _initial_state()
    model, optimizer = _build(True, 2, initial, packed_swap=True)
    assert not optimizer.state, "packed momentum must remain lazy until a gradient arrives"
    live_views = []
    original_step_batch = optimizer.adapter.step_batch

    def _step_batch(batch, step_context):
        """Check that Muon sees the active staging views during its update."""
        for unit in batch:
            for param, slot in zip(unit.params, unit.slots):
                state_tensor = optimizer.state[param]["momentum_buffer"]
                assert state_tensor is slot.tensor
                assert to_local_if_dtensor(state_tensor).device.type == "npu"
                live_views.append(id(state_tensor))
        return original_step_batch(batch, step_context)

    with mock.patch.object(optimizer.adapter, "step_batch", side_effect=_step_batch):
        _train(model, optimizer, batches, steps=1)
    assert live_views, "the packed update must activate at least one device staging view"
    _assert_swap_state_offloaded(optimizer, expected_params=len(initial))

    checkpoint = optimizer.state_dict()
    reloaded_model, reloaded = _build(True, 2, initial, packed_swap=True)
    reloaded_model.load_state_dict(model.state_dict())
    reloaded.load_state_dict(checkpoint)
    _train(model, optimizer, batches, steps=1)
    _train(reloaded_model, reloaded, batches, steps=1)
    for reference, actual in zip(model.parameters(), reloaded_model.parameters()):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            reloaded.state[actual]["momentum_buffer"], optimizer.state[reference]["momentum_buffer"], rtol=0, atol=0,
        )
    _assert_swap_state_offloaded(reloaded, expected_params=len(initial))
    _release_device_memory()


def test_swap_muon_packed_mixed_dtype() -> None:
    """Compare packed and bare Muon updates across separate dtype regions."""
    print("test_swap_muon_packed_mixed_dtype")
    initial = [
        torch.arange(24, dtype=torch.float32).reshape(4, 6) * 0.01,
        torch.arange(40, dtype=torch.bfloat16).reshape(5, 8) * 0.01,
    ]
    bare_params = [torch.nn.Parameter(value.npu()) for value in initial]
    per_tensor_params = [torch.nn.Parameter(value.npu()) for value in initial]
    packed_params = [torch.nn.Parameter(value.npu()) for value in initial]
    # Match the existing per-unit foreach boundaries for exact bare parity:
    # NPU mixed-dtype foreach has different BF16 rounding from separate calls.
    bare = Muon([{"params": [param]} for param in bare_params], lr=0.013, momentum=0.93)
    per_tensor = swap_optimizer(
        Muon(per_tensor_params, lr=0.013, momentum=0.93),
        SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False),
    )
    packed = swap_optimizer(
        Muon(packed_params, lr=0.013, momentum=0.93),
        SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=True),
    )

    for step in range(3):
        for bare_param, per_tensor_param, packed_param in zip(bare_params, per_tensor_params, packed_params):
            grad = torch.arange(bare_param.numel(), dtype=torch.float32).reshape(bare_param.shape)
            grad = (grad * 0.001 + step + 1).to(dtype=bare_param.dtype, device="npu")
            bare_param.grad = grad
            per_tensor_param.grad = grad.clone()
            packed_param.grad = grad.clone()
        bare.step()
        per_tensor.step()
        packed.step()
        for bare_param, per_tensor_param, packed_param in zip(bare_params, per_tensor_params, packed_params):
            torch.testing.assert_close(packed_param, bare_param, rtol=0, atol=0)
            torch.testing.assert_close(packed_param, per_tensor_param, rtol=0, atol=0)
            torch.testing.assert_close(
                packed.state[packed_param]["momentum_buffer"],
                bare.state[bare_param]["momentum_buffer"].cpu(),
                rtol=0,
                atol=0,
            )

    assert {slot.cpu_tensor.dtype for slot in packed.adapter.all_slots()} == {torch.float32, torch.bfloat16}
    _assert_swap_state_offloaded(packed, expected_params=2)
    _release_device_memory()
