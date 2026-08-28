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
import copy
import multiprocessing as mp
import queue
import traceback
from typing import Any

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel.core.activation_checkpoint import (
    CheckpointPolicy,
    SwapManager,
    checkpoint,
    checkpoint_wrapper,
    swap_wrapper,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.torch.activation_checkpoint import checkpoint as hyper_checkpoint
from tests.torch.common_net import SimpleTransformer
from tests.torch.activation_checkpoint.utils import prepare_data, seed_memory_time_context, set_seed, train_one_mode


MEMORY_COMPARISON_MODES = ("none", "recompute", "save", "swap", "group_swap")
MEMORY_COMPARISON_VOCAB_SIZE = 8192
platform = get_platform()


def _prefire_recompute(handles: list, session_id: Any) -> None:
    """Run all collected checkpoint frames in one retained session."""
    with platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
        for handle in handles:
            platform.recompute_handle(handle, session_id)


def _scheduled_dx_dw(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    weights: tuple,
    session_id: Any,
) -> tuple:
    """Compute separate input and weight gradients from one prefired session."""
    with platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
        input_grad = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
    with platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=False):
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


def test_native_npu_rng_for_closure_only_tensor_is_not_restored():
    """Native checkpoint does not discover an NPU tensor that exists only in a closure."""
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("npu")
    try:
        native_output, native_grad = _run_closure_only_npu_rng_case(
            lambda function, argument: torch_checkpoint(function, argument, use_reentrant=False)
        )
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)

    assert not torch.equal(native_grad, native_output)


def test_hyper_npu_rng_for_closure_only_tensor_matches_native():
    """Hyper should match native RNG behavior when the NPU tensor exists only in a closure."""
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("npu")
    try:
        native_output, native_grad = _run_closure_only_npu_rng_case(
            lambda function, argument: torch_checkpoint(function, argument, use_reentrant=False)
        )
        hyper_output, hyper_grad = _run_closure_only_npu_rng_case(hyper_checkpoint)
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

    with platform.recompute_handle_collector_ctx() as handles:
        output = checkpoint(checkpointed_function, input_tensor, weight)

    assert len(handles) == 1
    session_id = ("dxdw_split_e2e", id(output))
    try:
        _prefire_recompute(handles, session_id)
        actual_dx, actual_dws = _scheduled_dx_dw(output, input_tensor, (weight,), session_id)
    finally:
        platform.clear_recompute_session(session_id)

    assert checkpoint_calls == 2
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(actual_dx, expected_dx)
    torch.testing.assert_close(actual_dws[0], expected_dw)


def test_scheduled_recompute_npu_preserves_rng_state() -> None:
    """NPU dropout should replay its forward mask without advancing caller RNG state."""
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

    set_seed(88)
    with platform.recompute_handle_collector_ctx() as handles:
        output = checkpoint(
            lambda current_input, current_weight: torch.nn.functional.dropout(
                torch.nn.functional.gelu(current_input @ current_weight),
                p=0.4,
                training=True,
            ),
            input_tensor,
            weight,
            preserve_rng_state=True,
        )

    session_id = ("npu_rng", id(output))
    rng_state_before_prefire = torch.npu.get_rng_state()
    try:
        _prefire_recompute(handles, session_id)
        rng_state_after_prefire = torch.npu.get_rng_state()
        actual_dx, actual_dws = _scheduled_dx_dw(output, input_tensor, (weight,), session_id)
    finally:
        platform.clear_recompute_session(session_id)

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
        with platform.recompute_handle_collector_ctx() as handles:
            output = checkpoint(checkpointed_function, input_tensor, weight)

    session_id = ("npu_autocast", id(output))
    try:
        _prefire_recompute(handles, session_id)
        actual_dx, actual_dws = _scheduled_dx_dw(output.float(), input_tensor, (weight,), session_id)
    finally:
        platform.clear_recompute_session(session_id)

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


class _SmallNet(torch.nn.Module):
    """Two-layer MLP where the hidden activation is exposed as a plain function."""

    def __init__(self, dim: int, use_ckpt: bool):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

        def _hidden(x):
            return torch.relu(self.fc1(x))

        self.hidden = checkpoint_wrapper(_hidden) if use_ckpt else _hidden

    def forward(self, x):
        return self.fc2(self.hidden(x))


def test_checkpoint_wrapper_accepts_func():
    """
    Feature: checkpoint_wrapper accepts plain callable (func) as module argument
    Description: Build a small two-layer MLP whose hidden activation is a plain
                 Python function.  Run several training steps on two identical
                 copies of the network — one with the hidden function wrapped by
                 checkpoint_wrapper, one without — and verify that the per-step
                 losses are numerically identical.
    Expectation: Losses across all training steps match within 1e-6 tolerance,
                 confirming that wrapping a func does not alter forward/backward
                 semantics.
    """

    torch.manual_seed(42)
    dim = 32
    batch, train_steps = 8, 5
    tol = 1e-6

    # reference model (no checkpoint)
    ref_model = _SmallNet(dim, use_ckpt=False)
    # wrapped model starts from identical weights
    ckpt_model = _SmallNet(dim, use_ckpt=True)
    ckpt_model.fc1.load_state_dict(copy.deepcopy(ref_model.fc1.state_dict()))
    ckpt_model.fc2.load_state_dict(copy.deepcopy(ref_model.fc2.state_dict()))

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    ckpt_opt = torch.optim.SGD(ckpt_model.parameters(), lr=0.01)

    torch.manual_seed(42)
    for step in range(train_steps):
        x = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        # reference forward/backward
        ref_opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref_model(x), target)
        ref_loss.backward()
        ref_opt.step()

        # checkpoint-wrapped forward/backward (same input & target)
        ckpt_opt.zero_grad()
        ckpt_loss = torch.nn.functional.mse_loss(ckpt_model(x), target)
        ckpt_loss.backward()
        ckpt_opt.step()

        diff = abs(ref_loss.item() - ckpt_loss.item())
        assert diff < tol, (
            f"Loss mismatch at step {step}: "
            f"ref={ref_loss.item():.8f}, ckpt={ckpt_loss.item():.8f}, diff={diff:.2e}"
        )


class _OverlapTransformerBlock(torch.nn.Module):
    """Small transformer-style block used for wrapper overlap detection."""

    def __init__(self, dim=4, hidden_dim=8):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.out_proj = torch.nn.Linear(dim * 3, dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ffn1 = torch.nn.Linear(dim, hidden_dim)
        self.ffn2 = torch.nn.Linear(hidden_dim, dim)

        def local_activation(x):
            return torch.relu(x)

        self.local_activation = local_activation
        self.framework_mul = torch.mul

    def scale(self, x):
        return x * 2

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.out_proj(self.qkv(x))
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.local_activation(self.ffn1(x))
        x = self.framework_mul(x, torch.tensor(1.0, device=x.device, dtype=x.dtype))
        return residual + self.ffn2(x)


class _OverlapTransformerNet(torch.nn.Module):
    """Three-block net used to validate nested wrapper overlap detection."""

    def __init__(self, dim=4, depth=3):
        super().__init__()
        self.embed = torch.nn.Linear(dim, dim)
        self.blocks = torch.nn.ModuleList([_OverlapTransformerBlock(dim=dim) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, dim)

        def root_activation(x):
            return torch.relu(x)

        self.root_activation = root_activation
        self.framework_mul = torch.mul

    def scale(self, x):
        return x * 2

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.root_activation(x)
        x = self.norm(x)
        return self.head(x)


def _overlap_same_module_twice():
    net = _OverlapTransformerNet()
    checkpoint_wrapper(net.blocks[0].ffn1)
    checkpoint_wrapper(net.blocks[0].ffn1)


def _overlap_checkpoint_then_swap_same_module():
    net = _OverlapTransformerNet()
    checkpoint_wrapper(net.blocks[0].ffn1)
    swap_wrapper(net.blocks[0].ffn1)


def _overlap_leaf_then_parent():
    net = _OverlapTransformerNet()
    net.blocks[0].ffn1 = checkpoint_wrapper(net.blocks[0].ffn1)
    checkpoint_wrapper(net)


def _overlap_parent_then_leaf():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net)
    checkpoint_wrapper(wrapped.blocks[0].ffn1)


def _overlap_block_then_parent():
    net = _OverlapTransformerNet()
    net.blocks[1] = checkpoint_wrapper(net.blocks[1])
    checkpoint_wrapper(net)


def _overlap_parent_then_block():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net)
    checkpoint_wrapper(wrapped.blocks[1])


def _overlap_leaf_then_block():
    net = _OverlapTransformerNet()
    net.blocks[1].ffn1 = checkpoint_wrapper(net.blocks[1].ffn1)
    checkpoint_wrapper(net.blocks[1])


def _overlap_block_then_leaf():
    net = _OverlapTransformerNet()
    wrapped_block = checkpoint_wrapper(net.blocks[1])
    checkpoint_wrapper(wrapped_block.ffn1)


@pytest.mark.parametrize(
    "case",
    [
        _overlap_same_module_twice,
        _overlap_checkpoint_then_swap_same_module,
        _overlap_leaf_then_parent,
        _overlap_parent_then_leaf,
        _overlap_block_then_parent,
        _overlap_parent_then_block,
        _overlap_leaf_then_block,
        _overlap_block_then_leaf,
    ],
)
def test_wrapper_overlap_detection_cases(case):
    """Overlapping wrapper regions should warn consistently."""
    with pytest.warns(UserWarning, match="Wrapping overlapping module regions is not allowed"):
        case()


def _allowed_distinct_sibling_modules():
    net = _OverlapTransformerNet()
    net.blocks[0].ffn1 = checkpoint_wrapper(net.blocks[0].ffn1)
    net.blocks[0].ffn2 = swap_wrapper(net.blocks[0].ffn2)


def _allowed_distinct_blocks():
    net = _OverlapTransformerNet()
    net.blocks[0] = checkpoint_wrapper(net.blocks[0])
    net.blocks[1] = swap_wrapper(net.blocks[1])


def _allowed_distinct_block_callables():
    net = _OverlapTransformerNet()
    net.blocks[0].local_activation = checkpoint_wrapper(net.blocks[0].local_activation)
    net.blocks[1].local_activation = swap_wrapper(net.blocks[1].local_activation)


def _allowed_framework_function_reuse():
    checkpoint_wrapper(torch.mul)
    swap_wrapper(torch.mul)


def _allowed_framework_function_attr_after_parent_wrap():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net)
    checkpoint_wrapper(wrapped.blocks[0].framework_mul)
    checkpoint_wrapper(wrapped.blocks[1].framework_mul)


def _allowed_root_framework_function_attr_after_parent_wrap():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net)
    checkpoint_wrapper(wrapped.framework_mul)


def _allowed_bound_method_reuse():
    net = _OverlapTransformerNet()
    checkpoint_wrapper(net.blocks[0].scale)
    swap_wrapper(net.blocks[0].scale)


def _allowed_root_bound_method_reuse():
    net = _OverlapTransformerNet()
    checkpoint_wrapper(net.scale)
    swap_wrapper(net.scale)


def _allowed_callable_attr_then_parent():
    net = _OverlapTransformerNet()
    net.blocks[2].local_activation = checkpoint_wrapper(net.blocks[2].local_activation)
    checkpoint_wrapper(net.blocks[2])


def _allowed_parent_then_callable_attr():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net.blocks[2])
    checkpoint_wrapper(wrapped.local_activation)


def _allowed_root_callable_attr_then_parent():
    net = _OverlapTransformerNet()
    net.root_activation = checkpoint_wrapper(net.root_activation)
    checkpoint_wrapper(net)


def _allowed_parent_then_root_callable_attr():
    net = _OverlapTransformerNet()
    wrapped = checkpoint_wrapper(net)
    checkpoint_wrapper(wrapped.root_activation)


def _allowed_diff_callables():
    net = _OverlapTransformerNet()
    net.blocks[1].local_activation = checkpoint_wrapper(net.blocks[1].local_activation)
    net.blocks[2].local_activation = checkpoint_wrapper(net.blocks[2].local_activation)


@pytest.mark.parametrize(
    "case",
    [
        _allowed_distinct_sibling_modules,
        _allowed_distinct_blocks,
        _allowed_distinct_block_callables,
        _allowed_framework_function_reuse,
        _allowed_framework_function_attr_after_parent_wrap,
        _allowed_root_framework_function_attr_after_parent_wrap,
        _allowed_bound_method_reuse,
        _allowed_root_bound_method_reuse,
        _allowed_callable_attr_then_parent,
        _allowed_parent_then_callable_attr,
        _allowed_root_callable_attr_then_parent,
        _allowed_parent_then_root_callable_attr,
        _allowed_diff_callables,
    ],
)
def test_wrapper_non_overlapping_allowed_cases(case):
    """Non-overlapping or explicitly exempt callable configurations should be allowed."""
    case()
