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
from typing import Dict, List, Tuple

import torch

from hyper_parallel.core.optimizer import Muon, SwapOptimizerConfig, swap_optimizer
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


def _build(use_swap: bool, swap_times: int, initial: Dict[str, torch.Tensor]):
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
            SwapOptimizerConfig(swap_times=swap_times, min_numel=1, packed_swap=False),
        )
    return model, optimizer


def _train(model, optimizer, batches, steps=None) -> List[float]:
    """Train ``model`` for ``steps`` steps and return the per-step losses."""
    losses = []
    for step, (x, target) in enumerate(batches if steps is None else batches[:steps]):
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
    assert all(
        slot.tensor.untyped_storage().size() == 0 for slot in slots
    ), "device storage must be released after the step"


def _run_align_case(case_name: str, swap_times: int) -> None:
    """Compare bare and swapped Muon over identical inputs for ``swap_times``."""
    print(case_name)
    batches = _fixed_batches()
    initial = _initial_state()

    base_model, base_optimizer = _build(False, swap_times, initial)
    base_losses = _train(base_model, base_optimizer, batches)
    base_params = {name: param.detach().clone() for name, param in base_model.named_parameters()}
    del base_optimizer
    _release_device_memory()

    swap_model, swap_optimizer_inst = _build(True, swap_times, initial)
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
    assert all(slot.tensor.untyped_storage().size() == 0 for slot in slots)

    # The next step must prefetch, update and offload again.
    _train(reloaded_model, reloaded, batches, steps=1)
    _assert_swap_state_offloaded(reloaded, expected_params=len(initial))
    _release_device_memory()


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


def test_swap_muon_rejects_packed_swap() -> None:
    """
    Feature: Muon swap optimizer packed staging rejection.
    Description: Muon swaps momentum tensor by tensor, so requesting packed staging must fail at wrap time.
    Expectation: A ValueError names packed_swap.
    """
    print("test_swap_muon_rejects_packed_swap")
    init_dist()
    model = _MuonNet().npu()
    optimizer = Muon(model.parameters(), lr=0.013)

    try:
        swap_optimizer(optimizer, SwapOptimizerConfig(packed_swap=True))
    except ValueError as error:
        assert "packed_swap" in str(error), f"unexpected error message: {error}"
        return
    raise AssertionError("packed_swap=True must be rejected for Muon")
