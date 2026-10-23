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
"""Single-device MXFP8 acceptance; no EP or HSDP modifications required."""
import copy
import json
import os
import time
import statistics
from typing import Any
import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from hyper_parallel.components.quantization.ops.npu_mxfp8 import MXFP8NpuOps
from hyper_parallel.components.quantization.tensor import QuantizedTensor
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import _fused_quant
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts


class CountingMXOps(MXFP8NpuOps):
    """Count genuine float8 GMM operator calls without replacing arithmetic."""

    def __init__(self, backend: Any) -> None:
        """Validate the backend and start a fresh operator-call record."""
        super().__init__(backend)
        self.calls = []
        self.fused_forward = 0
        self.fused_backward = 0

    def quant_grouped_matmul(self, x1: torch.Tensor, x2: torch.Tensor, x2_scale: torch.Tensor,
                            **kwargs: Any) -> torch.Tensor:
        """Record float8 operands and execute the real device operator.

        Args:
            x1: Quantized left operand.
            x2: Quantized right operand.
            x2_scale: Right operand scales.
            **kwargs: Unmodified grouped operator options and left scales.
        """
        self.calls.append((str(x1.dtype), str(x2.dtype), kwargs["group_type"]))
        assert x1.dtype == torch.float8_e4m3fn and x2.dtype == torch.float8_e4m3fn
        return super().quant_grouped_matmul(x1, x2, x2_scale, **kwargs)


    def swiglu_mx_quant_dual_axis(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Count genuine fused forward calls."""
        self.fused_forward += 1
        return super().swiglu_mx_quant_dual_axis(*args, **kwargs)

    def swiglu_backward_mx_quant_dual_axis(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Count genuine fused backward calls."""
        self.fused_backward += 1
        return super().swiglu_backward_mx_quant_dual_axis(*args, **kwargs)

def _pack_physical(tensor):
    """Reject saved carriers and exercise restoration from an independent allocation."""
    assert not isinstance(tensor, QuantizedTensor)
    return tensor.detach().clone()

def _device():
    """Skip explicitly if real MXFP8 hardware/operators are unavailable."""
    backend = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Real NPU unavailable; CPU contracts do not validate MXFP8 numerics")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not any(marker in torch.npu.get_device_name() for marker in ("Ascend950", "Ascend910_95")):
        pytest.skip("MXFP8 requires A5 hardware")
    return torch.device("npu", local_rank), backend

def _metric(name, actual, reference):
    """Report absolute and normalized error using a predeclared acceptance rule."""
    diff = (actual.float() - reference.float()).abs()
    rms = reference.float().square().mean().sqrt().item() if reference.numel() else 0.
    error = diff.square().mean().sqrt().item() if diff.numel() else 0.
    maximum = diff.max().item() if diff.numel() else 0.
    ref_max = reference.float().abs().max().item() if reference.numel() else 0.
    result = {"name": name, "max_abs": maximum, "nrmse": error / max(rms, 1e-12),
              "p99_abs": torch.quantile(diff.flatten(), .99).item() if diff.numel() else 0.}
    print(json.dumps(result), flush=True)
    assert torch.isfinite(actual).all()
    assert error <= .10 * rms + 1e-6, result
    assert maximum <= .5 * ref_max + 1e-6, result

def _dequantize_rows(operand):
    """Decode valid row E8M0 blocks on CPU, excluding physical padding."""
    payload = operand.row_data.cpu().float()
    exponent = operand.row_scale.cpu().view(torch.uint8).reshape(payload.shape[0], -1).float()
    scale = torch.pow(2., exponent - 127).repeat_interleave(32, dim=-1)[:, :payload.shape[-1]]
    return payload * scale

def _dequantize_columns(operand, counts):
    """Decode grouped E8M0 scales, excluding per-expert and allocation padding."""
    payload = operand.col_data.cpu().float()
    exponents = operand.col_scale.cpu().view(torch.uint8).float()
    result = torch.empty_like(payload)
    token_offset = 0
    for expert_index, count in enumerate(counts.cpu().tolist()):
        # The kernel reserves floor(total_tokens / 64) + num_experts packs;
        # each expert starts at floor(its token offset / 64) + expert index.
        scale_offset = token_offset // 64 + expert_index
        packed_blocks = (count + 63) // 64
        exponent = exponents[scale_offset:scale_offset + packed_blocks].permute(0, 2, 1)
        exponent = exponent.reshape(-1, payload.shape[1]).repeat_interleave(32, dim=0)[:count]
        result[token_offset:token_offset + count] = (
            payload[token_offset:token_offset + count] * torch.pow(2., exponent - 127))
        token_offset += count
    return result

def test_fused_boundary():
    """Compare fixed SwiGLU inputs before any GMM or communication can differ."""
    device, _ = _device()
    torch.manual_seed(793862)
    quantizer = MXFP8Quantizer()
    counts = torch.tensor([1, 0, 31, 32, 33], device=device, dtype=torch.int64)
    exact = torch.cat([torch.full((count, 64), float(2 ** (index * 3)), device=device, dtype=torch.bfloat16)
                       for index, count in enumerate(counts.cpu().tolist())])
    exact_q = quantizer.quantize(exact, group_list=counts, group_list_type=1, colwise=True)
    torch.testing.assert_close(_dequantize_columns(exact_q, counts), exact.cpu().float(), rtol=0, atol=0)
    projected = torch.randn(97, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
    gate, up = projected.chunk(2, -1)
    activated = F.silu(gate) * up
    upstream = torch.randn_like(activated)
    derivative, = torch.autograd.grad(activated, projected, upstream)
    for name, reference, gradient in (("forward", activated, None), ("backward", derivative, upstream)):
        plain = quantizer.quantize(reference, group_list=counts, group_list_type=1, rowwise=True, colwise=True)
        fused = _fused_quant(projected.detach(), quantizer, counts, gradient)
        _metric(f"boundary_{name}_row_dequant", _dequantize_rows(fused), _dequantize_rows(plain))
        _metric(f"boundary_{name}_vs_bf16", _dequantize_rows(fused), reference.detach().cpu())
        _metric(f"boundary_{name}_column_dequant", _dequantize_columns(fused, counts),
                _dequantize_columns(plain, counts))
        _metric(f"boundary_{name}_column_vs_bf16", _dequantize_columns(fused, counts), reference.detach().cpu())
        for field in ("row_data", "row_scale", "col_data", "col_scale"):
            actual = getattr(fused, field).cpu().view(torch.uint8)
            expected = getattr(plain, field).cpu().view(torch.uint8)
            assert actual.shape == expected.shape
            # Column scale storage contains padding; its raw diff is diagnostic, not a numerical criterion.
            print(json.dumps({"boundary": name, "buffer": field, "shape": list(actual.shape),
                              "different_bytes_including_padding": int((actual != expected).sum())}), flush=True)

def test_fused_offload_checkpoint():
    """Exercise real fusion with CPU saved buffers and non-reentrant recomputation."""
    device, backend = _device()
    torch.manual_seed(862793)
    model = TinyMoE(dtype=torch.bfloat16).to(device)
    replace_experts(model, family_config(use_mxfp8=True, use_fused_mxfp8=True))
    expert = model.mlp.experts
    expert.quantizer = MXFP8Quantizer(npu_ops=CountingMXOps(backend))
    counts = torch.tensor([1, 0, 31, 33], device=device, dtype=torch.int64)
    inputs = torch.randn(65, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(inputs)
    variables = (inputs, expert.gate_up_proj, expert.down_proj)
    reference = expert.forward_expert_major(inputs, counts)
    expected = torch.autograd.grad(reference, variables, upstream)
    for recompute in (False, True):
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            output = (checkpoint(expert.forward_expert_major, inputs, counts, use_reentrant=False)
                      if recompute else expert.forward_expert_major(inputs, counts))
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        for _ in range(2):
            gradients = torch.autograd.grad(output, variables, upstream, retain_graph=True)
            for actual, target in zip(gradients, expected):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert expert.quantizer.npu_ops.fused_forward > 3
    assert expert.quantizer.npu_ops.fused_backward == 5

def _bf16_local(experts, x, counts):
    """Plain reference with identical expert token groups and BF16 boundaries."""
    outputs = []
    for block, w1, w2 in zip(x.split(counts), experts.gate_up_proj, experts.down_proj):
        gate, up = (block @ w1).chunk(2, -1)
        outputs.append((F.silu(gate) * up) @ w2)
    return torch.cat(outputs)

def _check_local_case(expert, baseline, counts, frozen, device):
    """Compare one gradient-requirement/grouping combination with BF16."""
    for name, weight in expert.named_parameters():
        weight.requires_grad_(frozen not in (name, "all_weights"))
    reference = copy.deepcopy(baseline)
    for name, weight in reference.named_parameters():
        weight.requires_grad_(frozen not in (name, "all_weights"))
    x_low = torch.randn(sum(counts), 128, dtype=torch.bfloat16).to(device).requires_grad_(frozen != "input")
    x_ref = x_low.detach().clone().requires_grad_(x_low.requires_grad)
    groups = torch.tensor(counts, device=device, dtype=torch.int64)
    out = expert.forward_expert_major(x_low, groups)
    ref = _bf16_local(reference, x_ref, counts)
    upstream = torch.randn_like(out)
    named_variables = [(name, p) for name, p in [("input", x_low), *expert.named_parameters()]
                       if p.requires_grad]
    variables = [p for _, p in named_variables]
    ref_variables = [p for p in (x_ref, *reference.parameters()) if p.requires_grad]
    grads = torch.autograd.grad(out, variables, upstream, retain_graph=True)
    repeated = torch.autograd.grad(out, variables, upstream)
    expected = torch.autograd.grad(ref, ref_variables, upstream)
    label = f"counts={counts}/frozen={frozen}"
    _metric(label + "/output", out, ref)
    for (name, _), actual, again, wanted in zip(named_variables, grads, repeated, expected):
        torch.testing.assert_close(actual, again, rtol=0, atol=0)
        _metric(label + f"/{name}_grad", actual, wanted)
    if sum(counts) == 0:
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)

@pytest.mark.parametrize("fused", [False, True])
def test_single_device(fused):
    """Fixed routing, output/dX/dW, empty/frozen/repeated backward and three updates.

    Args:
        fused: Whether to exercise fused activation and quantization.
    """
    device, backend = _device()
    torch.manual_seed(793)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    baseline, low = copy.deepcopy(source), copy.deepcopy(source)
    replace_experts(baseline, family_config())
    replace_experts(low, family_config(use_mxfp8=True, use_fused_mxfp8=fused))
    ops = CountingMXOps(backend)
    low.mlp.experts.quantizer = MXFP8Quantizer(npu_ops=ops)
    tokens = 40
    indices = torch.tensor([[i % 4, (i + 1) % 4] for i in range(tokens)], device=device)
    scores = torch.tensor([[.25, .75]] * tokens, device=device, dtype=torch.bfloat16)
    x = torch.randn(1, tokens, 128, dtype=torch.bfloat16).to(device)
    for model in (source, baseline, low):
        model.mlp.gate.indices, model.mlp.gate.scores = indices, scores
    source_out, base_out, low_out = source(x), baseline(x), low(x)
    torch.testing.assert_close(base_out, source_out, rtol=1e-2, atol=1e-4)
    _metric("routed_output", low_out, base_out)
    assert len(ops.calls) == 2

    for counts in ([1, 0, 31, 33], [0, 0, 0, 0], [0, 32, 1, 31]):
        for frozen in ("none", "gate_up_proj", "down_proj", "all_weights", "input"):
            _check_local_case(low.mlp.experts, baseline.mlp.experts, counts, frozen, device)

    empty = torch.empty(1, 0, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    low.mlp.gate.indices = torch.empty(0, 2, device=device, dtype=torch.int64)
    low.mlp.gate.scores = torch.empty(0, 2, device=device, dtype=torch.bfloat16)
    low.zero_grad(set_to_none=True)
    low(empty).sum().backward()
    assert empty.grad.shape == empty.shape
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in low.parameters())
    low.mlp.gate.indices, low.mlp.gate.scores = indices, scores

    expert = low.mlp.experts
    expert.requires_grad_(True)
    optimizer = torch.optim.SGD(expert.parameters(), lr=.1)
    initial = [p.detach().clone() for p in expert.parameters()]
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        step_input = x.detach().clone().requires_grad_(True)
        output = low(step_input)
        loss = output.float().square().sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        assert step_input.grad is not None and torch.isfinite(step_input.grad).all()
    assert all(not torch.equal(before, after) for before, after in zip(initial, expert.parameters()))
    assert any(call[2] == 2 for call in ops.calls)
    if fused:
        assert ops.fused_forward > 0 and ops.fused_backward > 0
    torch.npu.synchronize()
    print(json.dumps({"fused": fused, "fusion_calls": [ops.fused_forward, ops.fused_backward],
                      "single_device": "passed", "gmm_calls": len(ops.calls),
                      "three_step_losses": losses}), flush=True)

def test_fused_comparison():
    """Compare fused/unfused output and all gradients before timing identical work."""
    device, backend = _device()
    torch.manual_seed(862)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    models = [copy.deepcopy(source) for _ in range(2)]
    for fused, model in zip((False, True), models):
        replace_experts(model, family_config(use_mxfp8=True, use_fused_mxfp8=fused))
        model.mlp.experts.quantizer = MXFP8Quantizer(npu_ops=CountingMXOps(backend))
    for counts in ([1, 0, 31, 33], [0, 0, 0, 0], [0, 32, 1, 31]):
        groups = torch.tensor(counts, device=device, dtype=torch.int64)
        x = torch.randn(sum(counts), 128, dtype=torch.bfloat16, device=device)
        upstream = torch.randn_like(x)
        results = []
        for model in models:
            expert = model.mlp.experts
            inputs = x.detach().clone().requires_grad_()
            # Saved-tensor hooks see physical tensors, including retained backward.
            with torch.autograd.graph.saved_tensors_hooks(_pack_physical, lambda t: t):
                output = expert.forward_expert_major(inputs, groups)
                grads = torch.autograd.grad(output, (inputs, *expert.parameters()), upstream, retain_graph=True)
                repeated = torch.autograd.grad(output, (inputs, *expert.parameters()), upstream)
            for actual, again in zip(grads, repeated):
                torch.testing.assert_close(actual, again, rtol=0, atol=0)
            results.append((output, *grads))
        for name, plain, fused in zip(("output", "dx", "dgate_up", "ddown"), *results):
            _metric(f"fused_vs_unfused/{counts}/{name}", fused, plain)
    groups = torch.tensor([256] * 4, device=device, dtype=torch.int64)
    x = torch.randn(1024, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    for fused, model in zip((False, True), models):
        expert = model.mlp.experts
        times = []
        for step in range(13):
            expert.zero_grad(set_to_none=True)
            x.grad = None
            torch.npu.synchronize()
            start = time.perf_counter()
            expert.forward_expert_major(x, groups).float().square().mean().backward()
            torch.npu.synchronize()
            if step >= 3:
                times.append((time.perf_counter() - start) * 1000)
        ops = expert.quantizer.npu_ops
        print(json.dumps({"fused": fused, "train_ms_median": statistics.median(times),
                          "samples": len(times), "fusion_calls": [ops.fused_forward, ops.fused_backward]}), flush=True)


def _loss_update(model, optimizer, inputs):
    """Perform one finite-gradient SGD update and retain its comparison values."""
    optimizer.zero_grad(set_to_none=True)
    inputs = inputs.detach().clone().requires_grad_()
    output = model(inputs)
    loss = output.float().square().sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    return output.detach(), loss.item()

def test_three_way_loss():
    """Compare BF16/unfused/fused loss trajectories from identical initial weights."""
    device, backend = _device()
    torch.manual_seed(793)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    names = ("bf16", "mxfp8", "fused_mxfp8")
    models = [copy.deepcopy(source) for _ in names]
    for index, model in enumerate(models):
        replace_experts(model, family_config(use_mxfp8=index > 0, use_fused_mxfp8=index == 2))
        if index:
            model.mlp.experts.quantizer = MXFP8Quantizer(npu_ops=CountingMXOps(backend))
    x = torch.randn(1, 40, 128, dtype=torch.bfloat16).to(device)
    indices = torch.tensor([[i % 4, (i + 1) % 4] for i in range(40)], device=device)
    scores = torch.tensor([[.25, .75]] * 40, device=device, dtype=torch.bfloat16)
    for model in models:
        model.mlp.gate.indices, model.mlp.gate.scores = indices, scores
        for ref, actual in zip(models[0].parameters(), model.parameters()):
            torch.testing.assert_close(ref, actual, rtol=0, atol=0)
    optimizers = [torch.optim.SGD(model.parameters(), lr=.1) for model in models]
    trajectories = {name: [] for name in names}
    for step in range(10):
        outputs = []
        for name, model, optimizer in zip(names, models, optimizers):
            output, loss = _loss_update(model, optimizer, x)
            trajectories[name].append(loss)
            outputs.append(output)
        for index in (1, 2):
            _metric(f"loss_step={step + 1}/{names[index]}_output_vs_bf16", outputs[index], outputs[0])
        _metric(f"loss_step={step + 1}/fused_output_vs_mxfp8", outputs[2], outputs[1])
    for model in models[1:]:
        assert len(model.mlp.experts.quantizer.npu_ops.calls) > 0
    fused_ops = models[2].mlp.experts.quantizer.npu_ops
    assert fused_ops.fused_forward == 10 and fused_ops.fused_backward == 10
    print(json.dumps({"three_way_losses": trajectories, "loss_definition": "sum(output.float()**2)",
                      "steps": 10, "learning_rate": .1}), flush=True)
