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
"""CSA shared-chain parity for all-gather and all-to-all-v raw KV."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAAttention, SharedCompressedAttentionState, SharedCompressedPackedSequence,
    SharedCompressedAttentionCPContext,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Attention
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    _build_shared_attention_cp_context as _production_cp_context,
)
from tests.torch.context_parallel._test_deepseek_v41_dsa_cp import _config
from tests.torch.context_parallel._test_sequence_halo import WorldMesh, init_group


def _build_shared_attention_cp_context(mesh, mode):
    """Keep the all-gather oracle confined to the test harness."""
    context = _production_cp_context(mesh)
    return replace(context, launch_raw_halo=None) if mode == "all_gather" else context


def make_chain(
        ratio: int | str, device: torch.device, length: int, sparse: bool = False,
) -> tuple[torch.nn.ModuleList, int]:
    """Use two sources plus Reindex/Reuse without post-training candidates.

    Args:
        ratio: Compression ratio, or mixed for the r2-to-r1 shared chain.
        device: Device on which tensors and prepared geometry are consumed.
        length: Global sequence length of the test case.
        sparse: Whether to require TopK=4 on the NPU fixture.
    """
    config = _config()
    count = 6 if ratio == "mixed" else (1 if ratio == 0 else 5)
    config.num_hidden_layers = count
    config.layer_types = ["sliding_attention"] * count
    config.v41_compress_ratios = [0, 2, 2, 1, 1, 1] if ratio == "mixed" else [ratio] * count
    config.v41_kv_source_layer_ids = [] if ratio == 0 else [0, 3]
    config.v41_index_source_layer_ids = [] if ratio == 0 else [0, 2, 3]
    if ratio == "mixed":
        config.v41_kv_source_layer_ids = [1, 3]
        config.v41_index_source_layer_ids = [1, 3, 4]
    config.v41_candidate_source_layer_id = -1
    config.v41_indexer_loss_coeff = .001
    # CPU exercises genuinely sparse top-k. The NPU parity fixture selects
    # all eligible keys to avoid BF16 ties changing the discrete selection.
    config.index_topk = length if device.type == "npu" and not sparse else 4
    if device.type == "npu":
        config.head_dim = 512
        config.num_attention_heads = 8
        config.qk_rope_head_dim = 64
        config.partial_rotary_factor = .125
        config.index_head_dim = 128
        config.index_n_heads = 8
    modules = torch.nn.ModuleList([
        SharedCompressedDSAAttention(DeepseekV41Attention(config, idx)) for idx in range(count)
    ])
    with torch.no_grad():
        for name, parameter in modules.named_parameters():
            if name.endswith("sinks"):
                parameter.zero_()
            elif parameter.ndim == 1:
                parameter.fill_(1.)
            else:
                parameter.normal_(0, .03)
    modules = modules.to(device=device, dtype=torch.bfloat16 if device.type == "npu" else torch.float32)
    return modules, config.qk_rope_head_dim


def forward_chain(
        modules: torch.nn.ModuleList, hidden: torch.Tensor, start: int, length: int,
        boundaries: list[int], rope_dim: int, context: SharedCompressedAttentionCPContext | None,
) -> torch.Tensor:
    """Share state across Full/Reuse/Reindex consumers and preserve packed geometry.

    Args:
        modules: Attention layers that share compressed KV and Indexer state.
        hidden: Hidden-state tensor for the query interval.
        start: Global starting token position of the query interval.
        length: Global sequence length of the test case.
        boundaries: Global packed sample boundaries, including the terminal offset.
        rope_dim: Number of rotary channels.
        context: CP transport callbacks, or None for the single-rank reference.
    """
    positions = torch.arange(start, start+hidden.shape[1], device=hidden.device).unsqueeze(0)
    frequencies = torch.arange(rope_dim//2, device=hidden.device).float()
    angles = positions.float().unsqueeze(-1) * (1. / (10000 ** (frequencies/(rope_dim//2))))
    embeddings = {"main": (angles.cos(), angles.sin()), "compress": (angles.cos(), angles.sin())}
    packed = SharedCompressedPackedSequence(torch.tensor(boundaries), start, hidden.shape[1], length)
    packed = packed.prepare(hidden.device, tuple(module.compress_ratio for module in modules))
    state = SharedCompressedAttentionState()
    for module in modules:
        output, _ = module(hidden, position_embeddings=embeddings, position_ids=positions,
            attention_mask=None, shared_attention_state=state,
            shared_attention_cp_context=context, packed_seq_params=packed)
        hidden = hidden + .125 * output
    return hidden


def _report_chain(backend: str, size: int, ratio: int | str, boundaries: list[int], metrics: list) -> None:
    """Report the completed case and its largest measured errors on rank zero."""
    if metrics:
        print(f"BF16 worst relative errors: {sorted(metrics, reverse=True)[:3]}", flush=True)
    scope = ("CP1/AG/halo output and gradients" if backend == "gloo" else
             "CP1 output + exact halo/AG gradients; CP1 BF16 gradients are diagnostic")
    print(f"PASS {backend} CP{size} ratio={ratio} packed={boundaries}: {scope}", flush=True)


def run_chains(backend: str) -> None:
    """Compare CP1 and both distributed transports, including KL gradients.

    Args:
        backend: Distributed backend, gloo or hccl.
    """
    if backend == "hccl":
        pytest.importorskip("omni_training_custom_ops")
    device = init_group(backend)
    torch.use_deterministic_algorithms(True)
    size, rank, length = dist.get_world_size(), dist.get_rank(), 64
    part = slice(rank*(length//size), (rank+1)*(length//size))
    tolerance = {"rtol": 3e-4, "atol": 3e-6}
    def compare(actual: torch.Tensor, expected: torch.Tensor, name: str, metrics: list) -> None:
        """Check exact-target CPU or scale-aware BF16 numerical parity.

        Args:
            actual: Result from the distributed path under test.
            expected: Reference result for the same global data.
            name: Tensor name included in assertion failures.
            metrics: List to which numerical comparison measurements are appended.
        """
        if backend != "hccl":
            torch.testing.assert_close(actual, expected, **tolerance)
            return
        actual_fp32, expected_fp32 = actual.detach().float(), expected.detach().float()
        error = actual_fp32 - expected_fp32
        reference_norm = float(expected_fp32.norm())
        relative = float(error.norm()) / max(reference_norm, 1e-12)
        maximum = float(error.abs().max()) / max(float(expected_fp32.abs().max()), 1e-12)
        metrics.append((relative, maximum, name, reference_norm))
        assert bool(actual.isfinite().all()) and bool(expected.isfinite().all()), name
        # CP1 sums projection gradients before BF16 rounding, whereas CP8 sums
        # rounded local GEMMs. The original AR path also fails elementwise CP1
        # checks near cancellation. Record that diagnostic, and require exact
        # transport parity below; CPU checks still use the CP1 gradient oracle.
        if name.startswith("halo_vs_ag") or name.endswith(".output"):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                       msg=lambda message: f"{name}: {message}")

    try:
        for ratio in (2, 0, 1, "mixed"):
            for boundaries in ([0, 6, 34, length], [0, length]):
                torch.manual_seed(1903 + (3 if ratio == "mixed" else ratio))
                reference, rope_dim = make_chain(ratio, device, length)
                hidden = torch.randn(1, length, 32).to(device=device, dtype=next(reference.parameters()).dtype)
                full = hidden.detach().clone().requires_grad_()
                set_aux_loss_scale(torch.tensor(1., device=device))
                expected = forward_chain(reference, full, 0, length, boundaries, rope_dim, None)
                weights = torch.arange(1, size+1, device=device).repeat_interleave(length//size).view(1, length, 1)
                (expected.float()*weights).sum().backward()
                parameters = dict(reference.named_parameters())
                ag_result = {}
                metrics = []
                for mode in ("all_gather", "halo_alltoallv"):
                    parallel = deepcopy(reference)
                    parallel.zero_grad(set_to_none=True)
                    local = hidden[:, part].detach().clone().requires_grad_()
                    # KL averages local queries. Under SUM parameter reduction,
                    # 1/CP matches the full-query mean without changing trainer code.
                    set_aux_loss_scale(torch.tensor(1./size, device=device))
                    context = _build_shared_attention_cp_context(WorldMesh(), mode)
                    actual = forward_chain(parallel, local, part.start, length, boundaries, rope_dim, context)
                    (actual.float()*(rank+1)).sum().backward()
                    compare(actual, expected[:, part], f"{mode}.output", metrics)
                    compare(local.grad, full.grad[:, part], f"{mode}.dHidden", metrics)
                    gradients = {}
                    for name, parameter in parallel.named_parameters():
                        ref_grad = parameters[name].grad
                        if ref_grad is None or parameter.grad is None:
                            assert ref_grad is None and parameter.grad is None, f"gradient presence differs: {name}"
                            continue
                        dist.all_reduce(parameter.grad)
                        compare(parameter.grad, ref_grad, f"{mode}.{name}", metrics)
                        gradients[name] = parameter.grad.detach().clone()
                    if mode == "all_gather":
                        ag_result = {"output": actual.detach().clone(), "hidden_grad": local.grad.detach().clone(),
                                     "parameters": gradients}
                    else:
                        if not ag_result:
                            raise RuntimeError("all-gather comparison must run before halo")
                        compare(actual, ag_result["output"], "halo_vs_ag.output", metrics)
                        compare(local.grad, ag_result["hidden_grad"], "halo_vs_ag.dHidden", metrics)
                        for name, gradient in gradients.items():
                            compare(gradient, ag_result["parameters"][name], f"halo_vs_ag.{name}", metrics)
                if rank == 0:
                    _report_chain(backend, size, ratio, boundaries, metrics)
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()


def test_csa_halo_shared_chain_gloo():
    """CPU sparse shared-chain parity."""
    run_chains("gloo")


def test_csa_halo_shared_chain_hccl():
    """Real Omni/CANN shared-chain parity on eight NPUs."""
    run_chains("hccl")
