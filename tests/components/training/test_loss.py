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
"""Tests for MaskedCrossEntropy and calculate_loss dispatcher (design doc 03 §10)."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import hyper_models.components.loss.masked_ce as masked_ce_mod
from hyper_models.components.loss.masked_ce import MaskedCrossEntropy
from hyper_models.components.loss.utils import calculate_loss

try:
    from hyper_models.components.loss.linear_ce import FusedLinearCrossEntropy
    HAS_FUSED = True
except ImportError:  # pragma: no cover
    FusedLinearCrossEntropy = None
    HAS_FUSED = False


def _make_logits_labels(batch=1, seq=4, vocab=8):
    torch.manual_seed(0)
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    return logits, labels


# ── MaskedCrossEntropy ──

def test_masked_ce_sum():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="sum")
    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits.view(-1, 8).float(), labels.view(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_masked_ce_ignore_index():
    logits, labels = _make_logits_labels()
    labels = labels.clone()
    labels.view(-1)[0] = -100
    loss_fn = MaskedCrossEntropy(reduction="sum")
    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits.view(-1, 8).float(), labels.view(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_masked_ce_fp32_upcast():
    logits, labels = _make_logits_labels()
    logits = logits.bfloat16()
    captured = {}
    real_ce = F.cross_entropy

    def spy(*args, **kwargs):
        captured["dtype"] = args[0].dtype
        return real_ce(*args, **kwargs)

    loss_fn = MaskedCrossEntropy(fp32_upcast=True)
    with patch.object(masked_ce_mod.F, "cross_entropy", side_effect=spy):
        loss_fn(logits, labels)
    assert captured["dtype"] == torch.float32


def test_masked_ce_fp32_upcast_disabled():
    logits, labels = _make_logits_labels()
    logits = logits.bfloat16()
    captured = {}
    real_ce = F.cross_entropy

    def spy(*args, **kwargs):
        captured["dtype"] = args[0].dtype
        return real_ce(*args, **kwargs)

    loss_fn = MaskedCrossEntropy(fp32_upcast=False)
    with patch.object(masked_ce_mod.F, "cross_entropy", side_effect=spy):
        loss_fn(logits, labels)
    assert captured["dtype"] == torch.bfloat16


def test_masked_ce_default_ignore_index():
    assert MaskedCrossEntropy().ignore_index == -100


def test_masked_ce_custom_ignore_index():
    logits, labels = _make_logits_labels()
    labels = labels.clone()
    labels.view(-1)[0] = -1
    loss_fn = MaskedCrossEntropy(ignore_index=-1, reduction="sum")
    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits.view(-1, 8).float(), labels.view(-1),
        ignore_index=-1, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_masked_ce_all_ignored():
    logits, labels = _make_logits_labels()
    labels = torch.full_like(labels, -100)
    loss_fn = MaskedCrossEntropy(reduction="sum")
    assert loss_fn(logits, labels).item() == 0.0


def test_masked_ce_num_label_tokens():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="sum")
    full = loss_fn(logits, labels)
    normed = loss_fn(logits, labels, num_label_tokens=10)
    assert normed.item() == pytest.approx(full.item() / 10)


def test_masked_ce_num_label_tokens_zero():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="sum")
    assert loss_fn(logits, labels, num_label_tokens=0).item() == 0.0


def test_masked_ce_num_label_tokens_requires_sum():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="mean")
    with pytest.raises(ValueError, match="sum"):
        loss_fn(logits, labels, num_label_tokens=10)


def test_masked_ce_reduction_mean():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="mean")
    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits.view(-1, 8).float(), labels.view(-1),
        ignore_index=-100, reduction="mean",
    )
    assert loss.item() == pytest.approx(expected.item())


# ── calculate_loss dispatcher ──

def test_calculate_loss_logit_based():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="sum")
    loss = calculate_loss(loss_fn, logits=logits, labels=labels)
    # shift 后：logits[..., :-1, :] vs labels[..., 1:]
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_calculate_loss_loss_aggregation_token_weighted():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="sum")
    loss = calculate_loss(
        loss_fn, logits=logits, labels=labels, loss_aggregation="token_weighted",
    )
    # token_weighted：raw ce_sum（reduction="sum"，不除 N）
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_calculate_loss_loss_aggregation_rank_average():
    logits, labels = _make_logits_labels()
    loss_fn = MaskedCrossEntropy(reduction="mean")
    loss = calculate_loss(
        loss_fn, logits=logits, labels=labels, loss_aggregation="rank_average",
    )
    # rank_average：mean 尺度 loss
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="mean",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_calculate_loss_shift_labels():
    # 等长 batch 场景：CE 只覆盖 shift 后的位置（seq-1 个位置）
    logits, labels = _make_logits_labels(batch=2, seq=5, vocab=8)
    loss_fn = MaskedCrossEntropy(reduction="sum")
    loss = calculate_loss(loss_fn, logits=logits, labels=labels)
    manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(manual.item())


@pytest.mark.skipif(not HAS_FUSED, reason="FusedLinearCrossEntropy stub unavailable")
def test_calculate_loss_fused_linear():
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 6)
    labels = torch.randint(0, 8, (1, 4))
    lm_weight = torch.randn(8, 6)
    loss_fn = FusedLinearCrossEntropy()
    loss = calculate_loss(
        loss_fn, labels=labels, hidden_states=hidden, lm_weight=lm_weight,
    )
    # 融合路径：matmul(hidden, lm_weight.T) → CE sum（dispatcher 不做 shift）
    logits = torch.matmul(hidden.float(), lm_weight.t()).view(-1, 8)
    expected = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.skipif(not HAS_FUSED, reason="FusedLinearCrossEntropy stub unavailable")
def test_calculate_loss_fused_linear_fallback():
    logits, labels = _make_logits_labels()
    loss_fn = FusedLinearCrossEntropy()
    # 缺 hidden_states → 降级到 logits 路径（stub 中 lm_weight=None → hidden 视为 logits）
    loss = calculate_loss(loss_fn, logits=logits, labels=labels)
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.skipif(not HAS_FUSED, reason="FusedLinearCrossEntropy stub unavailable")
def test_calculate_loss_fused_linear_no_inputs_raises():
    loss_fn = FusedLinearCrossEntropy()
    with pytest.raises(ValueError, match="hidden_states"):
        calculate_loss(loss_fn, labels=torch.randint(0, 8, (1, 4)))
