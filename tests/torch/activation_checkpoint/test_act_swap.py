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
"""Test Activation Swap memory comparison: None vs Swap"""
from tests.common.mark_utils import arg_mark
from tests.torch.common_net import SimpleTransformer
from hyper_parallel.core.activation_checkpoint import SwapManager
from hyper_parallel.platform.torch.activation_checkpoint import ActivationPolicy, swap_wrapper
from .utils import prepare_data, train_one_mode, seed_memory_time_context


def apply_swap(model, mode):
    """Apply activation swap to the model based on the specified mode."""
    if mode == "none":
        return model

    if mode == "swap":
        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == "swap_with_policy":
        policy_threshold = 32 * 512 * 512
        def policy_fn(x):
            if x.storage().size() <= policy_threshold:
                return ActivationPolicy.SAVE
            return ActivationPolicy.SWAP

        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn, policy_fn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_act_swap_memory_comparison():
    """
    Feature: Activation Swap Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline),
                 'swap' (swap all tensors),
                 'swap_with_policy' (swap part of tensors according to the policy function).
                 Validate that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy.
    Expectation: The peak memory usage should follow the hierarchy: NONE > SWAP_WITH_POLICY > SWAP,
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("🚀 Starting memory and time comparison: none vs swap vs swap_with_policy")
    dataloader = prepare_data()
    train_steps=3

    modes = ["none", "swap", "swap_with_policy"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        with seed_memory_time_context() as stats:
            base_model = SimpleTransformer(vocab_size=32000, dim=2048, depth=16).npu()
            model = apply_swap(base_model, mode)
            losses = train_one_mode(model, dataloader, train_steps)
        peak_mem, duration = stats.get("peak_mem"), stats.get("exec_time")
        results[mode] = {
            "losses": losses,
            "peak_mem_gb": peak_mem,
            "time_sec": duration
        }
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "="*70)
    print("📊 FINAL COMPARISON")
    print("="*70)
    print(f"{'Mode':<15} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-"*70)
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():<20} | {r['peak_mem_gb']:<15.5f} | {r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}")

    # loss assert
    base_losses = results["none"]["losses"]
    tol = 1e-5
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["swap", "swap_with_policy"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\n✅ All {train_steps} steps: losses are consistent across modes (tol={tol}).")

    # memory assert
    mem_none = results["none"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]
    mem_swap_with_policy = results["swap_with_policy"]["peak_mem_gb"]

    # mem_none > mem_swap_with_policy
    assert mem_none > mem_swap_with_policy, \
        f"Expected NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})"
    print(f"✅ Verified: NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})")
    # mem_swap_with_policy > mem_swap
    assert mem_swap_with_policy > mem_swap, \
        f"Expected SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})"
    print(f"✅ Verified: SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})")
