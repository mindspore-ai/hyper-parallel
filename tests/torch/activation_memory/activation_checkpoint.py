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
"""Activation checkpoint memory comparison: None vs Recompute vs Save vs Swap"""
import multiprocessing as mp
import queue
import traceback
from typing import Any

import torch
from torch.utils.checkpoint import DefaultDeviceType
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel.core.activation_memory import (
    CheckpointPolicy,
    SwapManager,
    clear_recompute_session,
    checkpoint,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)
from tests.torch.common_net import SimpleTransformer
from tests.torch.activation_memory.utils import prepare_data, seed_memory_time_context, set_seed, train_one_mode


MEMORY_COMPARISON_MODES = ("none", "recompute", "save", "swap", "group_swap")
MEMORY_COMPARISON_VOCAB_SIZE = 8192


def _prefire_recompute(handles: list, session_id: Any) -> None:
    """Run all collected checkpoint frames in one retained session."""
    with recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
        for handle in handles:
            recompute_handle(handle, session_id)


def _scheduled_dx_dw(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    weights: tuple,
    session_id: Any,
) -> tuple:
    """Compute separate input and weight gradients from one prefired session."""
    with recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
        input_grad = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
    with recompute_session_ctx(session_id=session_id, retain_on_unpack=False):
        weight_grads = torch.autograd.grad(output.sum(), weights)
    return input_grad, weight_grads


def _run_closure_only_npu_rng_case(checkpoint_fn):
    """Return checkpoint output and gradient when the NPU tensor is captured only by closure."""
    captured_tensor = torch.ones(4096, device="npu", requires_grad=True)
    cpu_argument = torch.zeros(1)
    torch.npu.manual_seed(2026)

    def function(unused_argument: torch.Tensor) -> torch.Tensor:
        """Run NPU dropout while the device tensor is captured only by closure."""
        del unused_argument
        return torch.nn.functional.dropout(captured_tensor, p=0.5, training=True)

    output = checkpoint_fn(function, cpu_argument)
    output.sum().backward()
    return output.detach(), captured_tensor.grad.detach()


class _SaveRngNetwork(torch.nn.Module):
    """Exercise RNG both inside and after a checkpoint-excluded SAVE region."""

    def __init__(self) -> None:
        """Initialize the excluded random region and a trainable scale."""
        super().__init__()

        class _RandomSave(torch.nn.Module):
            """Consume device RNG while producing a SAVE-region activation."""

            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                """Multiply by a device RNG mask that must match during replay."""
                return tensor * torch.rand_like(tensor)

        self.excluded = checkpoint_exclude_wrapper(_RandomSave())
        self.scale = torch.nn.Parameter(torch.tensor([1.3, 0.7], device="npu"))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Use RNG after the skipped SAVE body during checkpoint replay."""
        saved = self.excluded(tensor * self.scale).sum()
        replay_consumer = (torch.rand_like(tensor) * tensor).sum()
        return saved + replay_consumer


def test_checkpoint_exclude_save_preserves_rng_state():
    """A skipped SAVE body must still advance RNG consistently during replay."""
    set_seed(2032)
    eager_net = _SaveRngNetwork()
    set_seed(2033)
    checkpoint_net = _SaveRngNetwork()
    checkpoint_net.load_state_dict(eager_net.state_dict())
    eager_input = torch.tensor([1.0, 2.0], device="npu", requires_grad=True)
    checkpoint_input = eager_input.detach().clone().requires_grad_()

    set_seed(2034)
    eager_output = eager_net(eager_input)
    eager_grads = torch.autograd.grad(eager_output, (eager_input, eager_net.scale))
    eager_rng_state = torch.npu.get_rng_state()
    set_seed(2034)
    checkpoint_output = checkpoint_wrapper(checkpoint_net)(checkpoint_input)
    checkpoint_grads = torch.autograd.grad(
        checkpoint_output,
        (checkpoint_input, checkpoint_net.scale),
    )
    checkpoint_rng_state = torch.npu.get_rng_state()

    torch.testing.assert_close(checkpoint_output, eager_output)
    torch.testing.assert_close(checkpoint_grads[0], eager_grads[0])
    torch.testing.assert_close(checkpoint_grads[1], eager_grads[1])
    assert torch.equal(checkpoint_rng_state, eager_rng_state)


def test_hyper_npu_rng_for_closure_only_tensor_matches_native():
    """Hyper should match native RNG behavior when the NPU tensor exists only in a closure."""
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("npu")
    try:
        native_output, native_grad = _run_closure_only_npu_rng_case(
            lambda function, argument: torch_checkpoint(function, argument, use_reentrant=False)
        )
        hyper_output, hyper_grad = _run_closure_only_npu_rng_case(checkpoint)
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)

    assert not torch.equal(native_grad, native_output)
    assert torch.equal(hyper_output, native_output)
    assert torch.equal(hyper_grad, native_grad)


def test_scheduled_recompute_supports_dx_dw_split() -> None:
    """A prefired recomputation should be shared by separate dx and dw autograd calls."""
    set_seed(2026)
    input_tensor = torch.randn(8, 16, device="npu", requires_grad=True)
    weight = torch.randn(16, 16, device="npu", requires_grad=True)
    reference_input = input_tensor.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()

    reference_output = torch.nn.functional.gelu(reference_input @ reference_weight)
    expected_dx, expected_dw = torch.autograd.grad(
        reference_output.sum(),
        (reference_input, reference_weight),
    )

    checkpoint_calls = 0

    def checkpointed_function(current_input: torch.Tensor, current_weight: torch.Tensor) -> torch.Tensor:
        """Count checkpoint executions while applying a weight-dependent function."""
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return torch.nn.functional.gelu(current_input @ current_weight)

    with recompute_handle_collector_ctx() as handles:
        output = checkpoint(checkpointed_function, input_tensor, weight)

    assert len(handles) == 1
    session_id = ("dxdw_split_e2e", id(output))
    try:
        _prefire_recompute(handles, session_id)
        actual_dx, actual_dws = _scheduled_dx_dw(output, input_tensor, (weight,), session_id)
    finally:
        clear_recompute_session(session_id)

    assert checkpoint_calls == 2
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(actual_dx, expected_dx)
    torch.testing.assert_close(actual_dws[0], expected_dw)


def test_scheduled_recompute_npu_preserves_rng_state() -> None:
    """Share the dropout replay across dx and dw while preserving the caller RNG state."""
    set_seed(2030)
    input_tensor = torch.randn(16, 32, device="npu", requires_grad=True)
    weight = torch.randn(32, 32, device="npu", requires_grad=True)
    reference_input = input_tensor.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()

    set_seed(88)
    reference_output = torch.nn.functional.dropout(
        torch.nn.functional.gelu(reference_input @ reference_weight),
        p=0.4,
        training=True,
    )
    expected_dx, expected_dw = torch.autograd.grad(
        reference_output.sum(),
        (reference_input, reference_weight),
    )

    checkpoint_calls = 0

    def checkpointed_function(current_input: torch.Tensor, current_weight: torch.Tensor) -> torch.Tensor:
        """Count executions of the checkpointed dropout function."""
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return torch.nn.functional.dropout(
            torch.nn.functional.gelu(current_input @ current_weight),
            p=0.4,
            training=True,
        )

    set_seed(88)
    with recompute_handle_collector_ctx() as handles:
        output = checkpoint(checkpointed_function, input_tensor, weight, preserve_rng_state=True)

    assert len(handles) == 1
    session_id = ("npu_rng", id(output))
    rng_state_before_prefire = torch.npu.get_rng_state()
    try:
        _prefire_recompute(handles, session_id)
        rng_state_after_prefire = torch.npu.get_rng_state()
        actual_dx, actual_dws = _scheduled_dx_dw(output, input_tensor, (weight,), session_id)
    finally:
        clear_recompute_session(session_id)

    assert checkpoint_calls == 2
    assert torch.equal(rng_state_after_prefire, rng_state_before_prefire)
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(actual_dx, expected_dx)
    torch.testing.assert_close(actual_dws[0], expected_dw)


def test_scheduled_recompute_npu_restores_autocast() -> None:
    """A prefire outside autocast should restore the NPU forward autocast settings."""
    set_seed(2031)
    input_tensor = torch.randn(16, 32, device="npu", requires_grad=True)
    weight = torch.randn(32, 32, device="npu", requires_grad=True)
    reference_input = input_tensor.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()

    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        reference_output = torch.nn.functional.gelu(reference_input @ reference_weight)
    expected_dx, expected_dw = torch.autograd.grad(
        reference_output.float().sum(),
        (reference_input, reference_weight),
    )
    execution_dtypes = []

    def checkpointed_function(current_input: torch.Tensor, current_weight: torch.Tensor) -> torch.Tensor:
        """Record the matmul dtype used in forward and scheduled replay."""
        result = current_input @ current_weight
        execution_dtypes.append(result.dtype)
        return torch.nn.functional.gelu(result)

    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(checkpointed_function, input_tensor, weight)

    session_id = ("npu_autocast", id(output))
    try:
        _prefire_recompute(handles, session_id)
        actual_dx, actual_dws = _scheduled_dx_dw(output.float(), input_tensor, (weight,), session_id)
    finally:
        clear_recompute_session(session_id)

    assert execution_dtypes == [torch.bfloat16, torch.bfloat16]
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(actual_dx, expected_dx, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_dws[0], expected_dw, atol=5e-3, rtol=5e-3)


def apply_recompute(model, mode):
    """Apply activation checkpointing based on the specified mode."""
    op_non_recompute = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default
    }
    if mode == "none":
        return model

    if mode == "recompute":
        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer)
    elif mode == "swap":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in op_non_recompute:
                return CheckpointPolicy.MUST_SWAP
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn, swap_inputs=True)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == "group_swap":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in op_non_recompute:
                return CheckpointPolicy.MUST_SWAP
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn, swap_inputs=True, group_swap=True)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == 'save':
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in op_non_recompute:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def _run_memory_comparison_mode(mode, train_steps, result_queue):
    """Run one memory-comparison mode in an isolated NPU process."""
    try:
        set_seed()
        dataloader = prepare_data(num_samples=train_steps * 8, vocab_size=MEMORY_COMPARISON_VOCAB_SIZE)
        with seed_memory_time_context() as stats:
            base_model = SimpleTransformer(vocab_size=MEMORY_COMPARISON_VOCAB_SIZE, dim=2048, depth=16).npu()
            base_model = base_model.bfloat16()
            model = apply_recompute(base_model, mode)
            losses = train_one_mode(model, dataloader, train_steps, optimizer_cls=torch.optim.SGD)
        result_queue.put({
            "mode": mode,
            "losses": losses,
            "peak_mem_gb": stats["peak_mem"],
            "time_sec": stats["exec_time"],
        })
        del model
        del base_model
        torch.npu.empty_cache()
    except Exception:  # pylint: disable=W0718
        result_queue.put({"mode": mode, "error": traceback.format_exc()})


def _run_memory_comparison_modes(modes, train_steps):
    """Run memory-comparison modes concurrently and collect their measurements."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_run_memory_comparison_mode, args=(mode, train_steps, result_queue))
        for mode in modes
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise AssertionError(f"Memory comparison process exited with code {process.exitcode}")

    results = {}
    for _ in modes:
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError("Memory comparison process did not report its result") from exc
        if "error" in result:
            raise AssertionError(f"Memory comparison mode '{result['mode']}' failed:\n{result['error']}")
        results[result["mode"]] = result
    result_queue.close()
    return results


def test_ac_memory_comparison():
    """
    Feature: Activation Checkpointing and Swapping Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline), 'recompute' (full activation checkpointing),
                 'save' (partial activation saving), and 'swap' (fine-grained tensor swapping).
                 Validates that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy:
                 NONE > SAVE > RECOMPUTE ≈ SWAP.
    Expectation: All modes produce consistent losses (within 1e-5 tolerance),
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("🚀 Starting memory and time comparison: none vs recompute vs save vs swap")
    train_steps = 3
    modes = MEMORY_COMPARISON_MODES
    results = _run_memory_comparison_modes(modes, train_steps)

    print("\n" + "="*70)
    print("📊 FINAL COMPARISON")
    print("="*70)
    print(f"{'Mode':<12} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-"*70)
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():<12} | {r['peak_mem_gb']:<15.5f} | {r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}")

    # loss assert
    base_losses = results["none"]["losses"]
    tol = 1e-5
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["recompute", "save", "swap", "group_swap"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\n✅ All {train_steps} steps: losses are consistent across modes (tol={tol}).")

    # mem assert
    mem_none = results["none"]["peak_mem_gb"]
    mem_save = results["save"]["peak_mem_gb"]
    mem_recompute = results["recompute"]["peak_mem_gb"]
    mem_group_swap = results["group_swap"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]

    # none > save > recompute ≈ swap
    # none > save
    assert mem_none > mem_save, f"Expected NONE ({mem_none:.5f}) > SAVE ({mem_save:.5f})"
    print(f"✅ Verified: NONE ({mem_none:.5f}) > SAVE ({mem_save:.5f})")
    # none > swap_group
    assert mem_none > mem_group_swap, f"Expected NONE ({mem_none:.5f}) > SWAP_GROUP ({mem_group_swap:.5f})"
    print(f"✅ Verified: NONE ({mem_none:.5f}) > SWAP_GROUP ({mem_group_swap:.5f})")
    # save > recompute
    assert mem_save > mem_recompute, f"Expected SAVE ({mem_save:.5f}) > RECOMPUTE ({mem_recompute:.5f})"
    print(f"✅ Verified: SAVE ({mem_save:.5f}) > RECOMPUTE ({mem_recompute:.5f})")
    # recompute ≈ swap
    tol_mem = 0.15
    assert abs(mem_recompute - mem_swap) < tol_mem, \
        f"Expected RECOMPUTE ({mem_recompute:.5f}) ≈ SWAP ({mem_swap:.5f})"
    print(f"✅ Verified: RECOMPUTE ({mem_recompute:.5f}) ≈ SWAP ({mem_swap:.5f}) within tolerance ({tol_mem:.5f} GB)")
