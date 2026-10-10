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
"""NPU parity for V4.1 MegaGate replacement, FSDP and expert dispatch."""

from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer

from hyper_parallel import SkipDTensorDispatch, fully_shard, init_device_mesh
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate import (
    DeepseekV41MegaGate,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _initialize_v41_owned_module,
    _v41_sparse_moe_forward,
)
from tests.torch.utils import init_dist


def _config(vision=True):
    return SimpleNamespace(
        hidden_size=128, num_local_experts=16, num_experts_per_tok=6,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, v41_vision_enabled=vision,
    )


def _replace(router):
    return DeepseekV41MegaGate(
        module=router, module_fqn="model.layers.0.mlp.gate", context={},
    )


def _router(device, vision=True, bias_dtype=torch.float32):
    router = DeepseekV41TopKRouter(_config(vision)).to(device)
    with torch.no_grad():
        _initialize_v41_owned_module(router, 0.02)
        router.weight.data = router.weight.to(torch.bfloat16)
        text_bias = router.bias
        assert isinstance(text_bias, torch.Tensor)
        text_bias.data = torch.linspace(-0.2, 0.2, 16, device=device).to(bias_dtype)
        if vision:
            vision_bias = router.bias_vl
            assert isinstance(vision_bias, torch.Tensor)
            vision_bias.data = -text_bias.detach().clone()  # pylint: disable=not-callable
    return router


def _close(actual, expected, *, gradient=False):
    torch.testing.assert_close(
        actual, expected, rtol=1e-2 if gradient else 1e-5,
        atol=1e-4 if gradient else 1e-6,
    )


def _run_router(module, hidden, mask):
    # HSDPModule exposes the no-argument form; both paths clear gradients to None.
    module.zero_grad()
    inputs = hidden.detach().clone().requires_grad_(True)
    logits, weights, indices = module(inputs, image_mask=mask)
    logits.retain_grad()
    # Expert-dependent upstream gradients remain equivalent when TopK order differs.
    loss = (weights * (indices.float() + 1) / module.num_experts).square().mean()
    loss.backward()
    torch.npu.synchronize()
    order = indices.argsort(dim=-1)
    return (
        logits.detach(), weights.gather(1, order).detach(), indices.gather(1, order),
        logits.grad.detach().clone(), inputs.grad.detach().clone(),
    )


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _compare_router_runs(expected, actual, golden, candidate):
    for result, baseline in zip(actual[:4], expected[:4]):
        _close(result, baseline)
    _close(actual[4], expected[4], gradient=True)
    _close(_local(candidate.weight.grad), _local(golden.weight.grad), gradient=True)
    assert candidate.bias.grad is None
    if candidate.bias_vl is not None:
        assert candidate.bias_vl.grad is None


def test_router_parity():
    """Compare actual V4.1 routers across visual masks and FSDP bias dtypes."""
    _, local_rank = init_dist()
    device = torch.device("npu", local_rank)
    torch.manual_seed(73)
    for bias_dtype in (torch.float32, torch.bfloat16):
        for vision in (False, True):
            golden = _router(device, vision, bias_dtype)
            candidate = _replace(deepcopy(golden))
            hidden = torch.randn(1, 257, 128, device=device, dtype=torch.bfloat16)
            mask_modes = (None, 0, 1, 4) if vision else (None,)
            for mask_mode in mask_modes:
                mask = None if mask_mode is None else (
                    torch.arange(257, device=device).remainder(max(mask_mode, 1)) == 0
                ).view(1, 257)
                if mask_mode == 0:
                    mask.zero_()
                expected = _run_router(golden, hidden, mask)
                with patch.object(MegaGate, "_torch_forward", side_effect=AssertionError("native required")):
                    actual = _run_router(candidate, hidden, mask)
                _compare_router_runs(expected, actual, golden, candidate)
            with torch.no_grad():
                golden.bias.add_(0.5)
                candidate.bias.copy_(golden.bias)
                if vision:
                    golden.bias_vl.sub_(0.5)
                    candidate.bias_vl.copy_(golden.bias_vl)
            expected = _run_router(golden, hidden, mask)
            with patch.object(MegaGate, "_torch_forward", side_effect=AssertionError("native required")):
                actual = _run_router(candidate, hidden, mask)
            _compare_router_runs(expected, actual, golden, candidate)


def test_fsdp_bf16_parity():
    """Use the recipe's BF16 parameter policy on two ranks without bypassing native."""
    rank, local_rank = init_dist()
    device = torch.device("npu", local_rank)
    torch.manual_seed(73)
    golden = _router(device).float()
    candidate = _replace(deepcopy(golden))
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
    )
    for module in (golden, candidate):
        fully_shard(module, mesh=mesh, mp_policy=policy)
    optimizers = [torch.optim.AdamW(module.parameters(), lr=1e-4) for module in (golden, candidate)]
    torch.manual_seed(100 + rank)
    hidden = torch.randn(1, 257, 128, device=device, dtype=torch.bfloat16)
    mask = (torch.arange(257, device=device).remainder(4) == rank % 4).view(1, 257)
    for _ in range(2):
        expected = _run_router(golden, hidden, mask)
        with patch.object(MegaGate, "_torch_forward", side_effect=AssertionError("native required")):
            actual = _run_router(candidate, hidden, mask)
        _compare_router_runs(expected, actual, golden, candidate)
        # FSDP optimizer updates act on local parameter shards, as in the trainer.
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            for optimizer in optimizers:
                optimizer.step()
        torch.npu.synchronize()
        _close(_local(candidate.weight), _local(golden.weight), gradient=True)


def _moe(device):
    config = DeepseekV4Config(
        hidden_size=128, moe_intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=1, head_dim=16, q_lora_rank=64,
        n_routed_experts=16, num_experts_per_tok=6, n_shared_experts=1,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"],
        o_groups=2, o_lora_rank=64, index_n_heads=4, index_head_dim=16,
    )
    module = DeepseekV4DecoderLayer(config, 0).mlp.to(device=device, dtype=torch.bfloat16)
    module.gate = _router(device)
    module.forward = MethodType(_v41_sparse_moe_forward, module)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not name.startswith("gate."):
                parameter.normal_(0, 0.02)
    return module


def _run_moe(module, hidden, mask, compute):
    module.zero_grad(set_to_none=True)
    inputs = hidden.detach().clone().requires_grad_(True)
    output = compute(module, inputs, image_mask=mask)
    output.float().square().mean().backward()
    torch.npu.synchronize()
    return output.detach(), inputs.grad.detach().clone(), module.gate.weight.grad.detach().clone()


def test_moe_parity():
    """Exercise V4.1's actual expert container and visual dispatch/combine."""
    _, local_rank = init_dist()
    device = torch.device("npu", local_rank)
    torch.manual_seed(73)
    golden = _moe(device)
    candidate = deepcopy(golden)
    candidate.gate = _replace(candidate.gate)
    hidden = torch.randn(1, 257, 128, device=device, dtype=torch.bfloat16)
    mask = (torch.arange(257, device=device).remainder(4) == 0).view(1, 257)
    expected = _run_moe(golden, hidden, mask, _v41_sparse_moe_forward)
    with patch.object(MegaGate, "_torch_forward", side_effect=AssertionError("native required")):
        actual = _run_moe(candidate, hidden, mask, _v41_sparse_moe_forward)
    for result, baseline in zip(actual, expected):
        _close(result, baseline, gradient=True)


def test_ep_parity():
    """Replace only the Gate while retaining the V4.1 model-owned EP compute factory."""
    rank, local_rank = init_dist()
    device = torch.device("npu", local_rank)
    world_size = dist.get_world_size()
    torch.manual_seed(73)
    golden = _moe(device)
    candidate = deepcopy(golden)
    candidate.gate = _replace(candidate.gate)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("ep",))
    computes = []
    for module in (golden, candidate):
        # Represent the local expert block that the model planner gives the compute factory.
        for name, parameter in tuple(module.experts.named_parameters(recurse=False)):
            assert parameter.shape[0] == 16
            shard = parameter.detach().chunk(world_size, dim=0)[rank].contiguous()
            setattr(module.experts, name, torch.nn.Parameter(shard))
        compute = deepseek_v41_ep_compute_fn(
            module=module, mesh=mesh, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
        )
        computes.append(compute)
    torch.manual_seed(100 + rank)
    hidden = torch.randn(1, 257, 128, device=device, dtype=torch.bfloat16)
    mask = (torch.arange(257, device=device).remainder(4) == rank % 4).view(1, 257)
    expected = _run_moe(golden, hidden, mask, computes[0])
    with patch.object(MegaGate, "_torch_forward", side_effect=AssertionError("native required")):
        actual = _run_moe(candidate, hidden, mask, computes[1])
    for result, baseline in zip(actual, expected):
        _close(result, baseline, gradient=True)
