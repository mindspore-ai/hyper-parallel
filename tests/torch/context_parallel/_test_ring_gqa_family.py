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
"""Distributed Ring GQA regression against the local GQA reference."""

from datetime import timedelta
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules import GroupedQueryAttention
from hyper_parallel.distributed.context_parallel.cp_context import CPContext


def _run_ring_gqa_case(
    ulysses_degree: int,
    *,
    packed: bool = False,
    load_balance: str = "none",
    prune_documents: bool = False,
    cp_algorithm: str = "ring",
) -> None:
    """Check Ring outputs and gradients against full-sequence GQA."""
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        cp_size = dist.get_world_size()
        cp_mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))
        config = SimpleNamespace(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            v_head_dim=64,
            attention_bias=False,
            o_proj_bias=False,
            qk_norm=False,
            rms_norm_eps=1e-6,
            dtype=torch.bfloat16,
            sliding_window=None,
            is_causal=True,
            attention_dropout=0.0,
        )
        torch.manual_seed(53)
        reference = GroupedQueryAttention(config=config).to(device=device)
        ring = GroupedQueryAttention(
            config=config, context={"cp_size": cp_size}, ulysses_degree=ulysses_degree,
            cp_algorithm=cp_algorithm,
            load_balance=load_balance, prune_documents=prune_documents,
        ).to(device=device)
        ring.load_state_dict(reference.state_dict())
        ring.bind_parallel(SimpleNamespace(cp_context=CPContext(cp_mesh)))

        torch.manual_seed(67)
        tokens_per_rank = 8 if prune_documents else 4
        global_tokens = cp_size * tokens_per_rank
        global_hidden = torch.randn(1, global_tokens, 256, dtype=torch.bfloat16, device="cpu").to(device)
        attention_options = {}
        if packed:
            attention_options["actual_seq_len"] = (
                (5, global_tokens) if prune_documents else (3, global_tokens)
            )
        reference_input = global_hidden.detach().requires_grad_()
        reference_output, _ = reference(reference_input, **attention_options)
        reference_output.float().square().mean().backward()

        rank = cp_mesh.get_local_rank()
        local_input = global_hidden.chunk(cp_size, dim=1)[rank].contiguous().detach().requires_grad_()
        ring_output, _ = ring(local_input, **attention_options)
        expected_output = reference_output.chunk(cp_size, dim=1)[rank]
        torch.testing.assert_close(ring_output, expected_output, atol=1.2e-2, rtol=8.0e-2)
        (ring_output.float().square().mean() / cp_size).backward()

        expected_input_grad = reference_input.grad.chunk(cp_size, dim=1)[rank]
        torch.testing.assert_close(local_input.grad, expected_input_grad, atol=2e-2, rtol=1.2e-1)
        for name, parameter in ring.named_parameters():
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
            torch.testing.assert_close(
                parameter.grad, reference.get_parameter(name).grad, atol=2e-2, rtol=1.2e-1,
            )
    finally:
        dist.destroy_process_group()


def test_ring_gqa_family_bfloat16() -> None:
    """Check pure Ring outputs and gradients on two ranks."""
    _run_ring_gqa_case(ulysses_degree=1)


def test_gqa_ulysses_cp_bfloat16() -> None:
    """Check the existing Ulysses GQA CP member on four ranks."""
    _run_ring_gqa_case(ulysses_degree=2, cp_algorithm="ulysses")


def test_ring_gqa_packed_tnd_bfloat16() -> None:
    """Check packed TND output and gradients with a document boundary between ranks."""
    _run_ring_gqa_case(ulysses_degree=1, packed=True)


def test_ring_gqa_pruned_zigzag_bfloat16() -> None:
    """Check document-subring pruning plus balanced ownership on multiple ranks."""
    _run_ring_gqa_case(
        ulysses_degree=1, packed=True, load_balance="zigzag", prune_documents=True,
    )


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "test_ring_gqa_family_bfloat16"
    test_cases = {
        "test_ring_gqa_family_bfloat16": test_ring_gqa_family_bfloat16,
        "test_gqa_ulysses_cp_bfloat16": test_gqa_ulysses_cp_bfloat16,
        "test_ring_gqa_packed_tnd_bfloat16": test_ring_gqa_packed_tnd_bfloat16,
        "test_ring_gqa_pruned_zigzag_bfloat16": test_ring_gqa_pruned_zigzag_bfloat16,
    }
    if test_name not in test_cases:
        raise ValueError(f"unknown Ring GQA test case: {test_name}")
    test_cases[test_name]()
