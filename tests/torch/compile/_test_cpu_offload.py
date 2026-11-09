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
"""Worker-side graph-mode attention CPU offload loss regression.

Runs on a single accelerator. The same toy attention LM is trained three ways
from identical weights:

- eager (uncompiled reference),
- joint graph with ``activation_swap="none"`` (graph baseline),
- joint graph with ``activation_swap="attention"`` (host offload enabled).

Loss, gradients and updated parameters must agree, proving the ``ao::``
offload/reload round trip does not change the numerics.
"""

import copy
import os

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.passes import cpu_offload as offload_module
from tests.torch.utils import get_device_type

_BATCH = 16
_SEQ = 128
_DIM = 256
_HEADS = 4
_HIDDEN = 512
_VOCAB = 1024
_LAYERS = 2
_STEPS = 3


class _Attention(nn.Module):
    """Multi-head self-attention registered as ``self_attn`` by its block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Build the Q/K/V/O projections."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, seq, dim)`` to ``(batch, heads, seq, head_dim)``."""
        batch, seq, _ = tensor.shape
        return tensor.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run scaled dot-product attention and project the result."""
        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))
        scores = torch.softmax(
            query @ key.transpose(-1, -2) * (self.head_dim**-0.5), dim=-1
        )
        context = (scores @ value).transpose(1, 2).reshape(hidden.shape)
        return self.o_proj(context)


class _Block(nn.Module):
    """Transformer-style attention + MLP residual block."""

    def __init__(self, dim: int, hidden_dim: int, num_heads: int) -> None:
        """Build the attention and MLP sublayers."""
        super().__init__()
        self.self_attn = _Attention(dim, num_heads)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run attention then MLP with pre-norm residuals."""
        hidden = hidden + self.self_attn(self.attn_norm(hidden))
        return hidden + self.fc2(self.act(self.fc1(self.mlp_norm(hidden))))


class _TinyLM(nn.Module):
    """Embedding + repeated attention blocks + causal-LM cross entropy."""

    def __init__(
        self, vocab: int, dim: int, hidden_dim: int, num_heads: int, num_layers: int
    ) -> None:
        """Build the toy causal LM."""
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            _Block(dim, hidden_dim, num_heads) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return next-token cross entropy for one batch."""
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.head(self.norm(hidden)).float()
        return F.cross_entropy(logits.reshape(-1, self.vocab), labels.reshape(-1))


def _train_fn(
    model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Trainer-style forward returning the scalar loss."""
    return model(input_ids, labels)


def _make_inputs(step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic next-token batch on the target device."""
    generator = torch.Generator().manual_seed(1234 + step)
    input_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator).to(device)
    return input_ids, torch.roll(input_ids, shifts=-1, dims=1)


def _zero_grads(*models: nn.Module) -> None:
    """Clear gradients on every model."""
    for model in models:
        model.zero_grad(set_to_none=True)


def _tolerances(dtype: torch.dtype) -> dict:
    """Return dtype-appropriate assert_close tolerances."""
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-6}
    return {"rtol": 1e-2, "atol": 1e-3}


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, label: str
) -> None:
    """Compare one loss/gradient/parameter value with a named failure message."""
    torch.testing.assert_close(
        actual,
        expected,
        **_tolerances(dtype),
        msg=lambda message: f"{label}: {message}",
    )


def _ao_targets(compiler: GraphCompiler) -> list[str]:
    """Return the inserted ``ao.*`` op target names of a compiled graph."""
    graph = compiler._joint_graph.graph_module.graph  # pylint: disable=protected-access
    return [
        str(node.target)
        for node in graph.nodes
        if node.op == "call_function" and str(node.target).startswith("ao.")
    ]


def _run_graph_step(
    compiler: GraphCompiler,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run one graph forward/backward and return the scalar loss."""
    loss, _ = compiler.forward_backward(input_ids=input_ids, labels=labels)
    return loss.detach()


def _compare_gradients(
    offload: nn.Module,
    baseline: nn.Module,
    reference: nn.Module,
    dtype: torch.dtype,
    step: int,
) -> None:
    """Compare every offload gradient with the graph and eager baselines."""
    for (name, offload_param), (_, baseline_param), (_, ref_param) in zip(
        offload.named_parameters(),
        baseline.named_parameters(),
        reference.named_parameters(),
    ):
        _assert_close(
            offload_param.grad,
            baseline_param.grad,
            dtype,
            f"step {step} {name} offload-vs-baseline grad",
        )
        _assert_close(
            offload_param.grad,
            ref_param.grad,
            dtype,
            f"step {step} {name} offload-vs-eager grad",
        )


def _compare_parameters(
    offload: nn.Module,
    baseline: nn.Module,
    reference: nn.Module,
    dtype: torch.dtype,
    step: int,
) -> None:
    """Compare every updated offload parameter with both baselines."""
    for (name, offload_param), (_, baseline_param), (_, ref_param) in zip(
        offload.named_parameters(),
        baseline.named_parameters(),
        reference.named_parameters(),
    ):
        _assert_close(
            offload_param,
            baseline_param,
            dtype,
            f"step {step} {name} offload-vs-baseline param",
        )
        _assert_close(
            offload_param,
            ref_param,
            dtype,
            f"step {step} {name} offload-vs-eager param",
        )


def _run_case(dtype: torch.dtype) -> None:
    """Compare offload, no-offload graph and eager training over several steps."""
    if not offload_module._AO_AVAILABLE:  # pylint: disable=protected-access
        pytest.skip("torch build does not ship the ao:: offload custom ops")
    device_type = get_device_type()
    if device_type != "npu":
        pytest.skip("graph-mode CPU offload requires an Ascend NPU")

    torch.manual_seed(0)
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device(device_type)
    model = _TinyLM(_VOCAB, _DIM, _HIDDEN, _HEADS, _LAYERS).to(
        device=device, dtype=dtype
    )
    reference = copy.deepcopy(model)
    baseline = copy.deepcopy(model)
    offload = copy.deepcopy(model)

    def _compiler(target: nn.Module, swap: str) -> GraphCompiler:
        """Build a single-card joint-graph compiler for ``target``."""
        return GraphCompiler(
            target,
            _train_fn,
            pass_config=PassConfig(
                fsdp_enabled=False,
                enable_overlap=False,
                activation_swap=swap,
                offload_prefetch_n_layers=1,
            ),
            device=device,
        )

    baseline_compiler = _compiler(baseline, "none")
    offload_compiler = _compiler(offload, "attention")

    optimizers = [
        torch.optim.SGD(target.parameters(), lr=0.01)
        for target in (reference, baseline, offload)
    ]
    for step in range(_STEPS):
        input_ids, labels = _make_inputs(step, device)
        _zero_grads(reference, baseline, offload)

        ref_loss = _train_fn(reference, input_ids, labels)
        ref_loss.backward()
        baseline_loss = _run_graph_step(baseline_compiler, input_ids, labels)
        offload_loss = _run_graph_step(offload_compiler, input_ids, labels)

        _assert_close(
            offload_loss, baseline_loss, dtype, f"step {step} offload-vs-baseline loss"
        )
        _assert_close(
            offload_loss, ref_loss.detach(), dtype, f"step {step} offload-vs-eager loss"
        )
        _compare_gradients(offload, baseline, reference, dtype, step)

        for optimizer in optimizers:
            optimizer.step()
        _compare_parameters(offload, baseline, reference, dtype, step)

    offload_targets = _ao_targets(offload_compiler)
    baseline_targets = _ao_targets(baseline_compiler)
    offloads = offload_targets.count("ao.offload.default")
    reloads = offload_targets.count("ao.reload.default")
    waits = offload_targets.count("ao.wait_tensor.default")
    assert offloads > 0, f"offload pass inserted no transfers: {offload_targets}"
    assert offloads == reloads, f"offload/reload mismatch: {offload_targets}"
    assert waits == 2 * offloads, f"wait mismatch: {offload_targets}"
    assert not baseline_targets, f"baseline unexpectedly offloaded: {baseline_targets}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_attention_cpu_offload_loss(dtype: torch.dtype) -> None:
    """Feature: Graph-mode attention activation CPU offload.
    Description: Train the same toy LM eager, as a plain joint graph, and with
        ``activation_swap="attention"`` from identical weights.
    Expectation: Offload loss, gradients and updated parameters match the
        no-offload graph and eager baselines; the offload graph contains paired
        ``ao.offload``/``ao.reload``/``ao.wait_tensor`` nodes.
    """
    _run_case(dtype)
