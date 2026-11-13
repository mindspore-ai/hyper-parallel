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
"""Opt-in native V4 numerical gradients and tiny shared Qwen3 optimizer checks."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch_npu  # pylint: disable=unused-import
from transformers import Qwen3Config

from rl.unified_execution.ops.attention.flash_attention_npu_v4 import dense_attention, varlen_attention
from rl.unified_execution.runtimes.hyper_parallel.builder import normalized_compute_setup
from rl.unified_execution.runtimes.hyper_parallel.training import install_training_attention
from rl.workers.adapters.qwen3 import Qwen3AutoModel


def _reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Calculate causal GQA in CPU float64 using explicit matrix arithmetic."""
    repeats = query.shape[-2] // key.shape[-2]
    query = query.transpose(1, 2)
    key = key.repeat_interleave(repeats, dim=2).transpose(1, 2)
    value = value.repeat_interleave(repeats, dim=2).transpose(1, 2)
    scores = query @ key.transpose(-2, -1) * query.shape[-1] ** -0.5
    allowed = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    return (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ value).transpose(1, 2)


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Apply fixed tolerances to original outputs and gradients."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    difference = (actual - expected).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


def _operator_check(device: torch.device, packed: bool) -> dict[str, Any]:
    """Compare native dense/packed GQA forward and all three gradients to FP64."""
    lengths = (0, 73, 111)
    shapes = [(sum(lengths), heads, 64) if packed else (2, 128, heads, 64) for heads in (4, 2, 2)]
    states = [(torch.randn(shape) * 0.25).to(torch.bfloat16) for shape in shapes]
    gradient = (torch.randn(shapes[0]) * 0.25).to(torch.bfloat16)
    reference_states = [state.double().requires_grad_(True) for state in states]
    if packed:
        outputs, offset = [], 0
        for length in lengths:
            request = [state[offset:offset + length].unsqueeze(0) for state in reference_states]
            outputs.append(_reference(*request).squeeze(0))
            offset += length
        reference = torch.cat(outputs)
    else:
        reference = _reference(*reference_states)
    reference.backward(gradient.double())
    runs = []
    for _ in range(2):
        native_states = [state.to(device).detach().requires_grad_(True) for state in states]
        if packed:
            cumulative = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
            output = varlen_attention(
                *native_states, cu_seqlens_q=cumulative, cu_seqlens_k=cumulative,
                max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
            )
        else:
            output = dense_attention(*native_states)
        output.backward(gradient.to(device))
        torch.npu.synchronize()
        if any(state.grad is None for state in native_states):
            raise AssertionError("Native V4 must produce all Q/K/V gradients")
        runs.append([tensor.detach().cpu() for tensor in (output, *(state.grad for state in native_states))])
    names = ("output", "dq", "dk", "dv")
    for name, first, second in zip(names, *runs):
        if not torch.equal(first, second):
            raise AssertionError(f"Native deterministic {name} changed across repeated calls")
    expected = (reference, *(state.grad for state in reference_states))
    return {name: _compare(actual, target) for name, actual, target in zip(names, runs[0], expected)}


def _model_check(device: torch.device, replace_gqa: bool) -> dict[str, Any]:
    """Build through HP, preserve parameter identities and execute an optimizer update."""
    config = Qwen3Config.from_dict({
        "vocab_size": 128, "hidden_size": 256, "intermediate_size": 384, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "attention_dropout": 0.0,
    })
    plan = []
    if replace_gqa:
        plan = [{"match": "*.self_attn", "replace": {
            "kind": "module", "source_type": "transformers.models.qwen3.modeling_qwen3.Qwen3Attention",
            "implementation": {
                "_target_": "hyper_parallel.models.qwen3.adapter.conversion.module_replacement."
                            "replace_qwen3_flash_attention",
            },
        }}]
    model = Qwen3AutoModel.from_config(
        config, distributed_setup=normalized_compute_setup(plan), torch_dtype=torch.bfloat16,
        attn_implementation="eager", validate_placement=False,
    ).to(device).train()
    identities = [id(parameter) for parameter in model.parameters()]
    install_training_attention(model)
    if identities != [id(parameter) for parameter in model.parameters()]:
        raise AssertionError("Training attention installation replaced model parameters")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.model.embed_tokens.weight.detach().clone()
    ids = torch.randint(1, config.vocab_size, (2, 128), device=device)
    mask = torch.ones((2, 128), dtype=torch.int64, device=device)
    mask[0, :27] = 0
    mask[1, 91:] = 0
    logits = model(ids, attention_mask=mask, use_cache=False).logits
    loss = logits[mask.bool()].float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise AssertionError("Tiny Qwen3 must produce finite gradients")
    norm = sum(float(gradient.float().square().sum()) for gradient in gradients) ** 0.5
    if norm == 0:
        raise AssertionError("Tiny Qwen3 must produce nonzero gradients")
    optimizer.step()
    torch.npu.synchronize()
    if torch.equal(before, model.model.embed_tokens.weight):
        raise AssertionError("Tiny Qwen3 optimizer step did not change embedding weights")
    return {"loss": float(loss.detach()), "gradient_norm": norm, "parameters_preserved": True,
            "embedding_updated": True, "attention_class": type(model.model.layers[0].self_attn).__name__}


def main() -> None:
    """Run on the caller-selected free NPU and write detailed JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.manual_seed(6107)
    report = {
        "dense_gqa": _operator_check(device, False),
        "packed_gqa": _operator_check(device, True),
        "hf_qwen3_update": _model_check(device, False),
        "hp_gqa_update": _model_check(device, True),
        "scope": "Native operator gradients and tiny model updates; not full RL or batch invariance acceptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
