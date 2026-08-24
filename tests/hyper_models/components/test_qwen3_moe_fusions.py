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
"""Unit tests for Qwen3-MoE fused attention replacement contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.auto_models.components.datasets.batching.attention_runtime import (
    build_dense_attention_masks,
)
from hyper_parallel.auto_models.components.distributed import cp_wrappers
from hyper_parallel.auto_models.components.models import qwen3_moe_fusions


class _Attention(nn.Module):
    """Minimal attention module exposing the replacement forward contract."""

    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Identity()
        self.attention_dropout = 0.0
        self.scaling = 1.0
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del position_embeddings, attention_mask, past_key_values, kwargs
        return hidden_states, None


class _CpMesh:
    """Small CP mesh stub used by the apply-time wrapper."""

    @staticmethod
    def size() -> int:
        return 2

    @staticmethod
    def get_local_rank() -> int:
        return 1


def test_packed_dense_mask_uses_true_for_allowed_positions() -> None:
    """Packed online masks follow Transformers and PyTorch SDPA semantics."""
    attention_mask, _ = build_dense_attention_masks(
        cu_seq_lens=torch.tensor([0, 2, 4]),
        micro_batch_size=1,
        seq_length=4,
        device=torch.device("cpu"),
        reset_attention_mask=True,
        sliding_window=None,
    )
    attention_mask = attention_mask[0, 0]

    assert attention_mask[1, 0]
    assert attention_mask[3, 2]
    assert not attention_mask[3, 1]
    assert not attention_mask[1, 2]


def test_cp_wrapper_rejects_attention_without_required_base_replacement() -> None:
    """The CP implementation cannot be paired with an arbitrary forward."""
    with pytest.raises(ValueError, match="can only replace"):
        qwen3_moe_fusions.qwen3_moe_flash_attention_cp_wrapper(
            _Attention(),
            mesh=None,
            tp_mesh=None,
            cp_mesh=_CpMesh(),
            ep_mesh=None,
        )


def test_cp_wrapper_replaces_only_the_fused_base_forward(monkeypatch) -> None:
    """The CP forward gathers K/V and builds a rank-offset causal mask."""
    attention = qwen3_moe_fusions.replace_qwen3_moe_flash_attention(
        module=_Attention(),
        module_fqn="model.layers.0.self_attn",
        context={},
    )
    assert (
        attention.forward.__func__
        is qwen3_moe_fusions.qwen3_moe_flash_attention_forward
    )

    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    captured = SimpleNamespace()

    def fake_prepare(module, hidden_states, position_embeddings, past_key_values):
        del module, position_embeddings, past_key_values
        return hidden_states.shape[:-1], query, key, value

    def fake_allgather(local_key, local_value, dim, cp_mesh):
        del cp_mesh
        captured.gather_dim = dim
        return (
            torch.cat((local_key, local_key), dim=dim),
            torch.cat((local_value, local_value), dim=dim),
        )

    def fake_mask(query_length, key_length, query_offset, device):
        captured.mask_args = (query_length, key_length, query_offset, device)
        return torch.ones(query_length, key_length, dtype=torch.bool, device=device)

    def fake_flash(module, query_states, key_states, value_states, attention_mask, **kwargs):
        del module, value_states, kwargs
        captured.key_length = key_states.shape[-2]
        captured.attention_mask = attention_mask
        return query_states.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(
        qwen3_moe_fusions,
        "_prepare_qwen3_moe_attention_states",
        fake_prepare,
    )
    monkeypatch.setattr(qwen3_moe_fusions, "flex_cp_allgather", fake_allgather)
    monkeypatch.setattr(qwen3_moe_fusions, "_cp_offset_causal_mask", fake_mask)
    monkeypatch.setattr(
        qwen3_moe_fusions,
        "_run_qwen3_moe_flash_attention",
        fake_flash,
    )

    qwen3_moe_fusions.qwen3_moe_flash_attention_cp_wrapper(
        attention,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_CpMesh(),
        ep_mesh=None,
    )
    hidden_states = torch.randn(1, 2, 4)
    output, weights = attention(
        hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        attention_mask=None,
    )

    torch.testing.assert_close(output, query.transpose(1, 2).reshape(1, 2, 4))
    assert weights is None
    assert captured.gather_dim == 2
    assert captured.key_length == 4
    assert captured.mask_args[:3] == (2, 4, 2)
    assert captured.attention_mask.shape == (2, 4)


def test_cp_wrapper_rejects_external_attention_mask() -> None:
    """The implicit-mask wrapper rejects an external mask before projections."""
    attention = qwen3_moe_fusions.replace_qwen3_moe_flash_attention(
        module=_Attention(),
        module_fqn="model.layers.0.self_attn",
        context={},
    )
    qwen3_moe_fusions.qwen3_moe_flash_attention_cp_wrapper(
        attention,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_CpMesh(),
        ep_mesh=None,
    )

    with pytest.raises(ValueError, match="cp_mask_wrapper"):
        attention(
            torch.randn(1, 2, 4),
            position_embeddings=(torch.empty(0), torch.empty(0)),
            attention_mask=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
        )


def test_cp_mask_wrapper_slices_external_global_allowed_mask(monkeypatch) -> None:
    """The external-mask wrapper selects this CP rank's global query rows."""
    attention = qwen3_moe_fusions.replace_qwen3_moe_flash_attention(
        module=_Attention(),
        module_fqn="model.layers.0.self_attn",
        context={},
    )
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    captured = SimpleNamespace()

    def fake_prepare(module, hidden_states, position_embeddings, past_key_values):
        del module, position_embeddings, past_key_values
        return hidden_states.shape[:-1], query, key, value

    def fake_allgather(local_key, local_value, dim, cp_mesh):
        del dim, cp_mesh
        return (
            torch.cat((local_key, local_key), dim=2),
            torch.cat((local_value, local_value), dim=2),
        )

    def fake_flash(module, query_states, key_states, value_states, attention_mask, **kwargs):
        del module, key_states, value_states
        captured.attention_mask = attention_mask
        captured.flash_kwargs = kwargs
        return query_states.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(
        qwen3_moe_fusions,
        "_prepare_qwen3_moe_attention_states",
        fake_prepare,
    )
    monkeypatch.setattr(qwen3_moe_fusions, "flex_cp_allgather", fake_allgather)
    monkeypatch.setattr(
        qwen3_moe_fusions,
        "_run_qwen3_moe_flash_attention",
        fake_flash,
    )

    qwen3_moe_fusions.qwen3_moe_flash_attention_cp_mask_wrapper(
        attention,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_CpMesh(),
        ep_mesh=None,
    )
    global_allowed_mask = torch.tril(
        torch.ones(1, 1, 4, 4, dtype=torch.bool),
    )
    output, weights = attention(
        torch.randn(1, 2, 4),
        position_embeddings=(torch.empty(0), torch.empty(0)),
        attention_mask=global_allowed_mask,
    )

    torch.testing.assert_close(output, query.transpose(1, 2).reshape(1, 2, 4))
    torch.testing.assert_close(
        captured.attention_mask,
        global_allowed_mask.narrow(-2, 2, 2),
    )
    assert weights is None
    assert "attention_mask_is_blocked" not in captured.flash_kwargs


def test_async_cp_runner_uses_transformers_allowed_mask(monkeypatch) -> None:
    """Qwen3 async CP keeps the model-facing True=allowed convention."""
    attention = _Attention()
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 4)
    allowed_mask = torch.tril(
        torch.ones(1, 1, 2, 4, dtype=torch.bool),
    )
    captured = SimpleNamespace()

    def fake_flash(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **kwargs,
    ):
        del module, key_states, value_states
        captured.attention_mask = attention_mask
        captured.flash_kwargs = kwargs
        return query_states.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(
        cp_wrappers,
        "_run_qwen3_moe_flash_attention",
        fake_flash,
    )
    cp_wrappers._run_qwen3_moe_fused_attention(
        attention,
        query,
        key,
        value,
        allowed_mask,
        {},
    )

    assert captured.attention_mask is allowed_mask
    assert "attention_mask_is_blocked" not in captured.flash_kwargs
