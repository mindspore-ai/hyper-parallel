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
"""Real HCCL MLA CP regression workers; launch through test_mla_cp.py."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.context_parallel.mla_context_parallel import (
    MLADimensions, MLA_CP_STRATEGIES, _apply_mla_rope, mla_all_gather,
)
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.trainer.config import Target
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.mla_cp_utils import (
    TestCPMesh, attention_options, make_mla, mathematical_reference, position_embeddings,
    make_deepseek_model, replace_and_load_mla,
)


def _npu_device() -> torch.device:
    # Load the optional hardware dependency only when running NPU cases.
    import torch_npu  # pylint: disable=C0415,unused-import

    if not torch.npu.is_available():
        pytest.skip("MLA HCCL tests require visible NPU devices")
    index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(index)
    return torch.device("npu", index)


def _compare(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, label: str) -> float:
    """Assert dtype-specific elementwise tolerance and return the maximum absolute error."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    tolerances = {
        torch.float64: (1e-10, 1e-9),
        torch.float16: (5e-4, 0.012),
        torch.bfloat16: (4e-3, 0.06),
    }
    atol, rtol = tolerances[dtype]
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{label}: {message}",
    )
    return (actual - expected).abs().max().item()


def _check_all_gather_gradients(device: torch.device, mesh: Any = None) -> None:
    """Check token order, noncontiguous inputs, live outputs and consumer SUM gradients."""
    mesh = TestCPMesh() if mesh is None else mesh
    degree, rank = mesh.size(), mesh.get_local_rank()
    for sequence_dim in (0, 1, -1):
        local = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4) + rank * 100
        local = local.transpose(0, 1).detach().requires_grad_()
        second = (local.detach() + 7).requires_grad_()
        first_output = mla_all_gather(local, sequence_dim, mesh)
        second_output = mla_all_gather(second, sequence_dim, mesh)
        expected = torch.cat([local.detach() + (peer - rank) * 100 for peer in range(degree)], dim=sequence_dim)
        torch.testing.assert_close(first_output, expected, atol=0, rtol=0)
        torch.testing.assert_close(second_output, expected + 7, atol=0, rtol=0)
        coefficient = rank + 1
        # Integer-valued upstream gradients make the collective's SUM semantics exactly checkable.
        torch.autograd.backward(
            (first_output, second_output),
            (coefficient * first_output.detach(), 3 * coefficient * second_output.detach()),
        )
        consumer_sum = degree * (degree + 1) // 2
        torch.testing.assert_close(local.grad, consumer_sum * local.detach(), atol=0, rtol=0)
        torch.testing.assert_close(second.grad, 3 * consumer_sum * second.detach(), atol=0, rtol=0)


def test_mla_npu_bfloat16() -> None:
    """
    Feature: BF16 MLA context parallelism with native QK192/V128 attention.
    Description: Execute CP8 expanded and latent-KV strategies against the original non-CP module.
    Expectation: Outputs, loss, input gradients and parameter gradients satisfy the existing BF16 tolerances.
    """
    _tp_suite(_npu_device(), torch.bfloat16, ((1, int(os.environ["WORLD_SIZE"]), False),), "npu")


def test_mla_npu_single_card() -> None:
    """
    Feature: Fused RoPE and BNSD/TND attention semantics.
    Description: Compare BF16 and FP16 forward and backward with CPU math and include fully masked query rows.
    Expectation: Outputs and gradients match within tolerance and fully masked rows remain exactly zero.
    """
    device = _npu_device()
    _check_npu_rope(device)
    torch.manual_seed(913)
    mask = torch.ones(32, 32, dtype=torch.bool).tril()
    mask[3] = False
    cases = (attention_options(), attention_options((3, 17, 32)), attention_options((3, 17, 32), False),
             {**attention_options(causal=False), "attention_mask": mask.to(device)})
    for dtype in (torch.float16, torch.bfloat16):
        for metadata in cases:
            query = torch.randn(1, 4, 32, 192, dtype=dtype, device="cpu").to(device).requires_grad_()
            key = torch.randn(1, 4, 32, 192, dtype=dtype, device="cpu").to(device).requires_grad_()
            value = torch.randn(1, 4, 32, 128, dtype=dtype, device="cpu").to(device).requires_grad_()
            reference_inputs = [part.detach().cpu().float().requires_grad_() for part in (query, key, value)]
            allowed = torch.ones(32, 32, dtype=torch.bool)
            if metadata.get("actual_seq_len"):
                ids = torch.cat((torch.zeros(3), torch.ones(14), torch.full((15,), 2)))
                allowed &= ids[:, None] == ids[None, :]
            if metadata.get("is_causal", True):
                allowed &= allowed.tril()
            if metadata.get("attention_mask") is not None:
                allowed &= metadata["attention_mask"].cpu()
            ref_query, ref_key, ref_value = reference_inputs
            scale = 0.73 / 192**0.5
            scores = (ref_query @ ref_key.transpose(-2, -1)) * scale
            valid = allowed.any(dim=-1, keepdim=True)
            scores = torch.where(valid, scores.masked_fill(~allowed, -torch.inf), 0.0)
            expected = (scores.softmax(-1).masked_fill(~valid, 0.0) @ ref_value).transpose(1, 2)
            options = dict(metadata)
            actual = npu_fusion_attention_forward(
                torch.nn.Module(), query, key, value, options.pop("attention_mask", None), scaling=scale, **options,
            )[0]
            _compare(actual, expected, dtype, "single-card kernel output")
            target = torch.randn_like(expected)
            (expected * target).mean().backward()
            (actual.float() * target.to(device)).mean().backward()
            for name, tensor, reference_tensor in zip(("Q", "K", "V"), (query, key, value), reference_inputs):
                _compare(tensor.grad, reference_tensor.grad, dtype, f"single-card {name} gradient")
            if metadata.get("attention_mask") is not None:
                torch.testing.assert_close(actual[:, 3], torch.zeros_like(actual[:, 3]), atol=0, rtol=0)
                torch.testing.assert_close(query.grad[:, :, 3], torch.zeros_like(query.grad[:, :, 3]), atol=0, rtol=0)
            print(json.dumps({"single_card": True, "dtype": str(dtype), "packed": metadata.get("actual_seq_len"),
                              "causal": metadata["is_causal"], "empty_row": "attention_mask" in metadata}), flush=True)


def _check_npu_rope(device: torch.device) -> None:
    """Check shared K, per-batch positions and both fused RoPE layouts and gradients."""
    torch.manual_seed(811)
    positions = position_embeddings(14, 64, attention_options(), dtype=torch.float32)
    positions = tuple(part.reshape(2, 7, 64).to(device) for part in positions)
    for dtype in (torch.bfloat16, torch.float16):
        for interleaved in (False, True):
            inputs = [torch.randn(2, 7, heads, 64, dtype=dtype).to(device).requires_grad_()
                      for heads in (4, 1)]
            references = [part.detach().cpu().double().requires_grad_() for part in inputs]
            cos, sin = (part.to(dtype).cpu().double().unsqueeze(2) for part in positions)
            expected = []
            for reference in references:
                ordered = (torch.cat((reference[..., 0::2], reference[..., 1::2]), -1)
                           if interleaved else reference)
                first, second = ordered.chunk(2, -1)
                expected.append(ordered * cos + torch.cat((-second, first), -1) * sin)
            actual = _apply_mla_rope(*inputs, positions, interleaved=interleaved)
            for name, tensor, reference, source, source_reference in zip(
                    ("Q", "K"), actual, expected, inputs, references):
                _compare(tensor, reference, dtype, f"fused RoPE {name}")
                target = torch.randn_like(reference)
                tensor.backward(target.to(device=device, dtype=dtype))
                reference.backward(target.to(dtype).double())
                _compare(source.grad, source_reference.grad, dtype, f"fused RoPE {name} gradient")
            print(json.dumps({"fused_rope": True, "dtype": str(dtype), "interleaved": interleaved}), flush=True)


class _MLAModel(nn.Module):
    """An attention boundary discovered by the actual model-family planner."""

    def __init__(self, attention: nn.Module) -> None:
        """Expose a discoverable attention boundary and its family config."""
        super().__init__()
        self.self_attn = attention
        self.config = attention.config
        self.config.model_type = "deepseek_v3"
        self.config.architectures = ["DeepseekV3ForCausalLM"]

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Keep the external attention output local to the CP token owner."""
        return self.self_attn(hidden_states, **kwargs)[0]


def _tp_shard(tensor: torch.Tensor, name: str, degree: int, rank: int) -> torch.Tensor:
    """Independent parameter oracle: shard only up-projection rows and Wo columns."""
    if ".q_b_proj." in name or ".kv_b_proj." in name:
        return tensor.chunk(degree, dim=0)[rank]
    if name.endswith("o_proj.weight"):
        return tensor.chunk(degree, dim=1)[rank]
    return tensor


def _replicated_on_tp(name: str) -> bool:
    return not (".q_b_proj." in name or ".kv_b_proj." in name or name.endswith("o_proj.weight"))


def _sync_tp_cp_gradients(parallel, reference, tp_mesh, cp_mesh, tp, tp_rank, dtype):
    """Reconstruct logical gradients according to independent TP parameter ownership."""
    gradients, expected_gradients = [], []
    for name, parameter in parallel.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        if cp_mesh is not None:
            dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
        if tp_mesh is not None and _replicated_on_tp(name):
            dist.all_reduce(parameter.grad, group=tp_mesh.get_group())
        expected_grad = _tp_shard(reference.get_parameter(name).grad, name, tp, tp_rank)
        _compare(parameter.grad, expected_grad, dtype, f"TP CP gradient {name}")
        gradients.append(parameter.grad.detach().cpu().double().flatten())
        expected_gradients.append(expected_grad.detach().cpu().double().flatten())
    actual_gradient, expected_gradient = torch.cat(gradients), torch.cat(expected_gradients)
    relative = (actual_gradient - expected_gradient).norm() / expected_gradient.norm().clamp_min(1e-12)
    bound = {torch.float64: 1e-9, torch.float16: 0.006, torch.bfloat16: 0.025}[dtype]
    assert relative < bound, f"TP CP gradient relative L2={relative.item()}, limit={bound}"
    return relative


def _build_composition(mesh, tp, cp, strategy, metadata, device, dtype, backend):
    """Apply the production plan and assert the physical TP layout before training."""
    tp_rank = mesh["tp"].get_local_rank() if tp > 1 else 0
    torch.manual_seed(409)
    dimensions = (MLADimensions(8, 32, 32, 128, 64, 128) if device.type == "npu"
                  else MLADimensions(8, 5, 6, 3, 2, 4))
    reference = _MLAModel(make_mla(
        dimensions, dtype=dtype, interleaved=dtype == torch.float16 or metadata.get("actual_seq_len") is not None,
    ))
    reference.to(device)
    parallel = deepcopy(reference)
    spec = ModuleShardingSpec(
        inner_target="self", region_dispatch=False,
        inner_wrapper=Target(
            mla_cp_wrapper,
            target_path="hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel.mla_cp_wrapper",
            strategy=strategy, backend=backend,
        ),
    )
    plan = ShardingPlanner(plan_overrides={"self_attn": spec}).plan(
        parallel, mesh, tp_size=tp, cp_size=cp, sequence_parallel=False,
    )
    parallel, source_info = apply_sharding_plan(parallel, plan, mesh)
    if tp > 1:
        assert source_info, "TP parameter shards must retain source layouts for the reducer"
    assert parallel.self_attn.num_heads == dimensions.heads // tp
    assert parallel.self_attn.mla_cp_runtime.plan.compute_heads == dimensions.heads // (tp * cp)
    for name, parameter in parallel.named_parameters():
        _compare(parameter, _tp_shard(reference.get_parameter(name), name, tp, tp_rank), dtype, f"TP layout {name}")
    return reference, parallel, dimensions


def _gradient_composition(mesh, tp: int, cp: int, strategy: str, metadata: dict,
                       device: torch.device, dtype: torch.dtype, backend: str) -> None:
    """Compare recomputed TP/CP outputs and accumulated gradients to the unsharded reference."""
    tp_mesh = mesh["tp"] if tp > 1 else None
    cp_mesh = mesh["cp"] if cp > 1 else None
    tp_rank = tp_mesh.get_local_rank() if tp_mesh is not None else 0
    cp_rank = cp_mesh.get_local_rank() if cp_mesh is not None else 0
    reference, parallel, dimensions = _build_composition(mesh, tp, cp, strategy, metadata, device, dtype, backend)
    lengths = metadata.get("actual_seq_len")
    sequence = lengths[-1] if lengths else (64 if device.type == "npu" else 16)
    begin, end = cp_rank * sequence // cp, (cp_rank + 1) * sequence // cp
    positions = position_embeddings(sequence, dimensions.rope_dim, metadata, dtype=dtype, device=device)
    local_positions = tuple(part[:, begin:end].contiguous() for part in positions)
    batch = 2 if tp == 1 and lengths is None else 1
    relative = None
    parallel.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    for microbatch in range(2):
        torch.manual_seed(1000 + microbatch)
        hidden = torch.randn(batch, sequence, reference.self_attn.linear_qkv.in_features, dtype=dtype).to(device)
        hidden.requires_grad_()
        local_hidden = hidden.detach()[:, begin:end].contiguous().requires_grad_()
        target = torch.randn_like(hidden)
        if device.type == "npu":
            expected = reference(
                hidden, position_embeddings=positions, **metadata,
            )
        else:
            expected = mathematical_reference(
                reference.self_attn, hidden, positions, metadata, metadata.get("attention_mask"),
            )

        def forward(inputs: torch.Tensor) -> torch.Tensor:
            """Execute the local CP shard through the TP boundary."""
            return parallel(inputs, position_embeddings=local_positions, **metadata)

        actual = checkpoint(forward, local_hidden, use_reentrant=False)
        _compare(actual, expected[:, begin:end], dtype, "TP CP output")
        loss_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        expected_loss = (expected.to(loss_dtype) - target.to(loss_dtype)).square().mean() / 2
        # The boundary all-reduce sums backward contributions. Divide the
        # repeated loss over TP; CP owns distinct tokens and sums them.
        local_loss = (actual.to(loss_dtype) - target[:, begin:end].to(loss_dtype)).square().sum()
        local_loss = local_loss / (hidden.numel() * tp * 2)
        total_loss = local_loss.detach().clone()
        dist.all_reduce(total_loss)
        _compare(total_loss, expected_loss, dtype, "TP CP loss")
        expected_loss.backward()
        local_loss.backward()
        logical_input_grad = local_hidden.grad.clone()
        if tp_mesh is not None:
            dist.all_reduce(logical_input_grad, group=tp_mesh.get_group())
        _compare(logical_input_grad, hidden.grad[:, begin:end], dtype, "TP CP logical dX")
    relative = _sync_tp_cp_gradients(parallel, reference, tp_mesh, cp_mesh, tp, tp_rank, dtype)
    if dist.get_rank() == 0:
        print(json.dumps({"tp": tp, "cp": cp, "mesh_axes": mesh.mesh_dim_names, "strategy": strategy,
                          "backend": backend, "dtype": str(dtype), "packed": metadata.get("actual_seq_len"),
                          "causal": metadata.get("is_causal", True),
                          "gradient_relative_l2": relative.item(), "backward_passes": 2}), flush=True)


def _tp_suite(device: torch.device, dtype: torch.dtype, configurations: tuple, backend: str) -> None:
    """Exercise both strategies and supported masks over the requested TP/CP meshes."""
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        for tp, cp, reverse in configurations:
            axes, shape = (("cp", "tp"), (cp, tp)) if reverse else (("tp", "cp"), (tp, cp))
            mesh = init_device_mesh(device.type, shape, mesh_dim_names=axes)
            packed = (3, 73, 129, 256)
            if cp > 1:
                _check_all_gather_gradients(device, mesh["cp"])
            cases = [attention_options()]
            if cp > 1 and dtype != torch.float16:
                cases.append(attention_options(packed))
            if tp == 1 and dtype == torch.bfloat16:
                sequence = 64
                indices = torch.arange(sequence, device=device)
                mask = (indices[:, None] >= indices[None, :]) & (indices[:, None] - indices[None, :] < 5)
                mask[3] = False
                cases.append({**attention_options(causal=False), "attention_mask": mask})
            for strategy in MLA_CP_STRATEGIES:
                for metadata in cases:
                    _gradient_composition(mesh, tp, cp, strategy, metadata, device, dtype, backend)
    finally:
        dist.destroy_process_group()


def test_mla_tp_cp_npu_bfloat16() -> None:
    """
    Feature: BF16 MLA tensor and context parallel composition.
    Description: Run both TP2 CP4 mesh orders and strategies with ordinary and cross-shard packed sequences.
    Expectation: Outputs and reconstructed TP/CP gradients agree with the native non-CP module.
    """
    _tp_suite(_npu_device(), torch.bfloat16, ((2, 4, False), (2, 4, True)), "npu")


def _model_fsdp_case(mesh: MeshContext, strategy: str, device: torch.device, *, sequence_parallel: bool) -> None:
    """Compare two real decoder layers and reconstructed FSDP gradients with SP on or off."""
    dtype = torch.bfloat16
    torch.manual_seed(314)
    parallel, _ = replace_and_load_mla(make_deepseek_model(dtype=dtype, npu_dimensions=True))
    parallel.to(device)
    reference = deepcopy(parallel)
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            inner_target="self", region_dispatch=False,
            inner_wrapper=Target(
                mla_cp_wrapper,
                target_path="hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel.mla_cp_wrapper",
                strategy=strategy, backend="npu",
            ),
        ),
    })
    plan = planner.plan(parallel, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size,
                        sequence_parallel=sequence_parallel)
    parallel, source_info = apply_sharding_plan(parallel, plan, mesh)
    manager = FSDP2Manager(FSDP2Config(
        dp_shard_size=mesh.dp_shard_size, reshard_after_forward=True,
        mix_precision=FSDP2MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32",
                                               output_dtype="bfloat16"),
    ), mesh)
    manager.parallelize(parallel, source_info)
    sequence = 32
    cp_rank = mesh.device_mesh["cp"].get_local_rank()
    dp_rank = mesh.device_mesh["dp"].get_local_rank()
    begin, end = cp_rank * sequence // mesh.cp_size, (cp_rank + 1) * sequence // mesh.cp_size
    allowed = torch.ones(1, 1, sequence, sequence, dtype=torch.bool, device=device).tril()
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    for microbatch in range(2):
        torch.manual_seed(204 + microbatch)
        tokens = torch.randint(0, 128, (mesh.dp_size, sequence)).to(device)
        targets = torch.randn(mesh.dp_size, sequence, 128).to(device)
        expected = reference(input_ids=tokens, attention_mask=allowed, position_ids=positions,
                             use_cache=False).logits
        expected_loss = (expected.float() - targets).square().mean() / 2
        expected_loss.backward()
        local_tokens = tokens[dp_rank:dp_rank + 1, begin:end].contiguous()
        with SkipDTensorDispatch():
            actual = parallel(input_ids=local_tokens, attention_mask=allowed,
                              position_ids=positions[:, begin:end].contiguous(), use_cache=False).logits
            local_mean = (actual.float() - targets[dp_rank:dp_rank + 1, begin:end]).square().mean()
            counts = {"foundation_tokens": local_tokens.new_tensor(local_tokens.numel())}
            step_counts = {"foundation_tokens": counts["foundation_tokens"] * 2}
            loss = mean_global_loss(local_mean, counts, step_counts, mesh)["foundation_loss"]
            loss.backward()
        _compare(actual, expected[dp_rank:dp_rank + 1, begin:end], dtype, "decoder FSDP logits")
        _compare(loss, expected_loss, dtype, "decoder FSDP global loss")
    gradients, reference_gradients = [], []
    for name, parameter in parallel.named_parameters():
        actual_gradient = parameter.grad.full_tensor()
        expected_gradient = reference.get_parameter(name).grad
        _compare(actual_gradient, expected_gradient, dtype, f"decoder FSDP gradient {name}")
        gradients.append(actual_gradient.detach().cpu().double().flatten())
        reference_gradients.append(expected_gradient.detach().cpu().double().flatten())
    actual_gradient, expected_gradient = torch.cat(gradients), torch.cat(reference_gradients)
    relative = (actual_gradient - expected_gradient).norm() / expected_gradient.norm().clamp_min(1e-12)
    assert relative < 0.025, f"Decoder gradient relative L2={relative.item()}"
    parallel.reset_iter_state()
    if dist.get_rank() == 0:
        print(json.dumps({"real_model": "DeepseekV3ForCausalLM", "layers": 2, "tp": mesh.tp_size,
                          "cp": mesh.cp_size, "dp": mesh.dp_size, "strategy": strategy,
                          "sequence_parallel": sequence_parallel,
                          "fsdp": "FSDP2Manager source layouts", "microbatches": 2,
                          "gradient_relative_l2": relative.item()}), flush=True)


def _run_fsdp(dp: int) -> None:
    """Run both CP strategies and SP settings with independently seeded DP batches."""
    device = _npu_device()
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = MeshContext(tp_size=2, cp_size=2, dp_size=dp, dp_shard_size=dp * 2)
        mesh.build_meshs("npu", dist.get_world_size())
        for strategy in MLA_CP_STRATEGIES:
            for sequence_parallel in (False, True):
                _model_fsdp_case(mesh, strategy, device, sequence_parallel=sequence_parallel)
    finally:
        dist.destroy_process_group()


def test_mla_model_dp2_tp2_cp2_fsdp() -> None:
    """
    Feature: Two-layer MLA under DP2 TP2 CP2, FSDP and SP.
    Description: Use distinct DP data, accumulate two microbatches, and compare both strategies with SP on and off.
    Expectation: Full logical gradients and global losses agree with the non-CP model within BF16 tolerances.
    """
    _run_fsdp(2)
