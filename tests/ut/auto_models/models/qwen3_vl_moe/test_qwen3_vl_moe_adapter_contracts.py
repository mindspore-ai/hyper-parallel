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
"""CPU tests for model-owned Qwen3-VL-MoE distributed adapter contracts."""
# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.models import get_model_adapter
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_vl_moe.adapter.distributed import (
    context_parallel,
    expert_parallel,
)
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_model_adapter_registers_its_distributed_providers() -> None:
    """The family registry resolves Qwen3-VL-MoE's own CP and EP providers."""
    spec = get_model_adapter("qwen3_vl_moe")

    assert isinstance(spec, ModelAdapterSpec)
    assert spec.architecture == "Qwen3VLMoeForConditionalGeneration"
    assert spec.context_parallel is not None
    assert spec.expert_parallel is not None


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_cp_keeps_global_2d_padding_mask() -> None:
    """Retain global padding metadata and restore variable-length kernel output."""
    query = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    key = torch.empty(2, 1, 4, 4)
    padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    actual = context_parallel._prepare_attention_mask(padding_mask, query, key)

    torch.testing.assert_close(actual, padding_mask.bool())
    module = SimpleNamespace(training=True, attention_dropout=0.0, scaling=0.5)
    kernel = Mock(side_effect=lambda *args, **kwargs: (args[0],))
    with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
        output, _ = context_parallel._run_flash_attention(module, query, key, query, actual)
    expected = query.transpose(1, 2) * padding_mask[:, :, None, None]
    torch.testing.assert_close(output, expected)
    assert kernel.call_args.args[4] == "TND", f"Expected TND: {kernel.call_args}"
    assert kernel.call_args.kwargs["actual_seq_qlen"] == (3, 5), f"Expected Q ends (3, 5): {kernel.call_args}"
    assert kernel.call_args.kwargs["actual_seq_kvlen"] == (3, 5), f"Expected KV ends (3, 5): {kernel.call_args}"
    assert kernel.call_args.kwargs["sparse_mode"] == 3, f"Expected causal mode 3: {kernel.call_args}"
    with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
        direct, _ = npu_fusion_attention_forward(
            module, query, key, query, padding_mask, dropout=0.1, scaling=0.25,
        )
    torch.testing.assert_close(direct, expected)
    assert kernel.call_args.kwargs["keep_prob"] == 0.9, f"Expected keep_prob=0.9: {kernel.call_args}"
    assert kernel.call_args.kwargs["scale"] == 0.25, f"Expected scale=0.25: {kernel.call_args}"
    for mask in (
        torch.tensor([[False, True, True, True]]),
        torch.tensor([[True, False, True, False]]),
        torch.zeros(1, 4, dtype=torch.bool),
    ):
        kernel.reset_mock()
        with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
            output, _ = context_parallel._run_flash_attention(module, query, key, query, mask)
        torch.testing.assert_close(output, query.transpose(1, 2) * mask[:, :, None, None])
        if mask.any():
            assert kernel.call_args.args[4] == "TND", f"Expected unpadding for {mask}"
        else:
            kernel.assert_not_called()


def test_text_cp_window_uses_compressed_band_mask() -> None:
    """Unpad contiguous padding and pass the causal window to the NPU band mode."""
    query = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    module = SimpleNamespace(training=True, attention_dropout=0.0, scaling=0.5)
    kernel = Mock(side_effect=lambda *args, **kwargs: (args[0],))
    for mask in (
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]),
        torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]]),
        torch.zeros(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
        None,
    ):
        for options in ({"sliding_window": 2}, {"pre_tokens": 1}):
            kernel.reset_mock()
            with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
                output, _ = context_parallel._run_flash_attention(module, query, query, query, mask, **options)
            expected = query.transpose(1, 2)
            if mask is not None:
                expected = expected * mask[:, :, None, None]
            torch.testing.assert_close(output, expected)
            if mask is not None and not mask.any():
                kernel.assert_not_called()
                continue
            assert kernel.call_args.kwargs["sparse_mode"] == 4, f"Expected band mode: {kernel.call_args}"
            assert kernel.call_args.kwargs["pre_tockens"] == 1, f"Expected one preceding token: {kernel.call_args}"
            assert kernel.call_args.kwargs["next_tockens"] == 0, f"Expected no future tokens: {kernel.call_args}"
            actual_mask = kernel.call_args.kwargs["atten_mask"]
            assert actual_mask.shape == (2048, 2048), f"Expected compressed mask: {actual_mask.shape}"
            if mask is not None and not mask.all():
                assert kernel.call_args.args[4] == "TND", f"Expected TND: {kernel.call_args}"
                ends = tuple(mask.sum(dim=-1)[mask.any(dim=-1)].cumsum(0).tolist())
                assert kernel.call_args.kwargs["actual_seq_qlen"] == ends, f"Expected sequence ends {ends}"


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_input_cp_wrapper_slices_after_visual_injection() -> None:
    """The VLM wrapper shards text tensors while retaining matching visual features."""
    calls = []
    target = nn.Module()
    target.forward = lambda **kwargs: calls.append(kwargs) or kwargs
    cp_mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    request = context_parallel.qwen3_vl_moe_text_input_cp_wrapper(
        target, None, None, cp_mesh, None
    )
    assert any(getattr(module, "_hp_cp_shards_inputs", False) for module in target.modules())

    inputs_embeds = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    position_ids = torch.arange(24).reshape(3, 1, 8)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    visual_mask = torch.tensor([[False, True, True, False, False, True, False, True]])
    deepstack = [torch.tensor([[10.0], [11.0], [12.0], [13.0]])]

    request.forward(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=deepstack,
    )

    forwarded = calls[0]
    torch.testing.assert_close(forwarded["inputs_embeds"], inputs_embeds[:, 4:8])
    torch.testing.assert_close(forwarded["position_ids"], position_ids[..., 4:8])
    assert forwarded["attention_mask"] is attention_mask
    torch.testing.assert_close(forwarded["visual_pos_masks"], visual_mask[:, 4:8])
    torch.testing.assert_close(forwarded["deepstack_visual_embeds"][0], torch.tensor([[12.0], [13.0]]))


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vision_cp_wrapper_keeps_replicated_forward() -> None:
    """Vision compute remains replicated before the language sequence is sharded."""
    class _VisionModule(nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1

    target = _VisionModule()
    cp_mesh = SimpleNamespace(size=lambda: 2)

    request = context_parallel.qwen3_vl_moe_replicated_vision_cp_wrapper(
        target, None, None, cp_mesh, None
    )

    inputs = torch.zeros(2, 4, 8)
    torch.testing.assert_close(request.forward(inputs), inputs + 1)
    torch.testing.assert_close(target(inputs), inputs + 1)


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_cp_wrapper_accepts_qwen3_vl_contract_without_sliding_window() -> None:
    """Qwen3-VL-MoE attention has a model-owned contract without sliding_window."""
    target = nn.Module()
    target.q_proj = nn.Linear(4, 4)
    target.k_proj = nn.Linear(4, 4)
    target.v_proj = nn.Linear(4, 4)
    target.o_proj = nn.Linear(4, 4)
    target.q_norm = nn.LayerNorm(4)
    target.k_norm = nn.LayerNorm(4)
    target.head_dim = 2
    target.scaling = 2**-0.5
    target.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
    cp_mesh = SimpleNamespace(size=lambda: 2)

    request = context_parallel.qwen3_vl_moe_async_ulysses_cp_wrapper(
        target, None, None, cp_mesh, None
    )

    assert isinstance(request, _ForwardRewriteRequest)
    assert request.target is target


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ep_router_uses_qwen3_vl_model_outputs() -> None:
    """The VL EP adapter preserves the model-native router weights and indices."""
    weights = torch.tensor([[0.75, 0.25]])
    indices = torch.tensor([[3, 1]])
    module = SimpleNamespace(gate=lambda hidden_states: (None, weights, indices))

    actual_indices, actual_weights = expert_parallel._qwen3_vl_moe_router(
        module, torch.zeros(1, 4)
    )

    assert actual_indices is indices
    assert actual_weights is weights
