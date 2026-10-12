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
"""Unit tests for FlashMLA sparse-attention modules."""

import pytest

from hyper_parallel.components import modules
from hyper_parallel.components.modules import deepseek_sparse_attention, flash_mla


def _assert_equal(actual, expected, context):
    assert actual == expected, f"{context}: expected={expected!r}, actual={actual!r}"


@pytest.mark.parametrize("previous,canonical", [
    ("DeepseekV41SparsePrefillAttention", "FlashMLASparsePrefill"),
    ("DeepseekV41SparseDecodeAttention", "FlashMLASparseDecode"),
])
def test_flashmla_compatibility_exports_preserve_identity(previous, canonical):
    """Old import paths resolve to the canonical class, not a wrapper subclass."""
    implementation = getattr(flash_mla, canonical)
    assert getattr(deepseek_sparse_attention, previous) is implementation, (
        f"Legacy module export {previous} differs from {canonical}"
    )
    assert getattr(modules, previous) is implementation, f"Legacy public export {previous} differs from {canonical}"
    assert getattr(modules, canonical) is implementation, f"Canonical public export {canonical} changed identity"


def test_public_exports_match_lazy_module_namespace():
    """Every sparse-attention module is reachable through the lazy public namespace."""
    missing_exports = set(flash_mla.__all__) - set(modules.__all__)
    _assert_equal(missing_exports, set(), "DeepSeek exports missing from module namespace")
    matching_exports = all(
        getattr(modules, name) is getattr(flash_mla, name)
        for name in flash_mla.__all__
    )
    _assert_equal(matching_exports, True, "Lazy module exports resolve to their implementations")


def test_prefill_module_forwards_configuration(monkeypatch):
    """The prefill module forwards static and per-call options."""
    recorded = {}

    def _sparse_prefill(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "output"

    monkeypatch.setattr(flash_mla, "deepseek_flash_mla_sparse_prefill", _sparse_prefill)
    module = flash_mla.FlashMLASparsePrefill(0.125, value_head_dim=512)
    _assert_equal(module("q", "kv", "indices", attention_sink="sink"), "output", "Prefill module result")
    _assert_equal(recorded["args"], ("q", "kv", "indices", 0.125), "Prefill positional forwarding")
    _assert_equal(recorded["kwargs"]["attention_sink"], "sink", "Prefill attention-sink forwarding")
    with pytest.raises(ValueError, match="value_head_dim=512"):
        flash_mla.FlashMLASparsePrefill(0.125, value_head_dim=128)
    with pytest.raises(ValueError, match="value_head_dim=512"):
        flash_mla.FlashMLASparseDecode(value_head_dim=128)


def test_decode_module_reuses_and_resets_scheduler(monkeypatch):
    """Decode metadata is reused until an explicit or configuration-driven reset."""
    metadata = iter((("meta-1", None), ("meta-2", None), ("meta-3", None), ("meta-4", None)))
    seen = []
    monkeypatch.setattr(flash_mla, "deepseek_flash_mla_metadata", lambda: next(metadata))

    def _sparse_decode(query, cache, indices, scheduler, **kwargs):
        seen.append((query, cache, indices, scheduler, kwargs))
        return "output"

    monkeypatch.setattr(flash_mla, "deepseek_flash_mla_sparse_decode", _sparse_decode)
    module = flash_mla.FlashMLASparseDecode(enable_batch_invariant=True)

    _assert_equal(module("q1", "cache", "indices"), "output", "First decode result")
    _assert_equal(module("q2", "cache", "indices"), "output", "Second decode result")
    module.reset_scheduler()
    _assert_equal(module("q3", "cache", "indices"), "output", "Decode result after explicit reset")
    module.to("cpu")
    _assert_equal(module("q4", "cache", "indices"), "output", "Decode result after device transform")
    module.enable_batch_invariant = False
    _assert_equal(module("q5", "cache", "indices"), "output", "Decode result after scheduler option change")
    with pytest.raises(ValueError, match="enable_batch_invariant must be bool"):
        module.enable_batch_invariant = 1

    _assert_equal(
        [item[3] for item in seen],
        ["meta-1", "meta-1", "meta-2", "meta-3", "meta-4"],
        "Decode scheduler reuse and reset sequence",
    )
    _assert_equal(
        [item[4]["enable_batch_invariant"] for item in seen],
        [True, True, True, True, False],
        "Decode batch-invariant forwarding",
    )
