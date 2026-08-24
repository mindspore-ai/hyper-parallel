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
"""Dependency-free guards for the built-in MLA/DSA CP wrapper.

The toy modules intentionally mimic only the class names and attributes used
by the wrapper. This test has no model-library dependency, so the generic
Hyper-Parallel test environment can run it.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    _slice_sequence,
    mla_dsa_ulysses_cp_wrapper,
)


class _FakeCPMesh:
    """Minimal CP mesh used to exercise wrapper validation and injection."""

    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank
        self._group = object()

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self):
        return self._group


class _FakeCPContext:
    size = 2
    rank = 1


def _text_forward(self, inputs_embeds=None, **kwargs):
    del self, kwargs
    return inputs_embeds


ToyTextModel = type(
    "ToyTextModel",
    (nn.Module,),
    {"__module__": __name__, "forward": _text_forward},
)


ToyMLAAttention = type(
    "ToyMLAAttention",
    (nn.Module,),
    {"__module__": __name__, "attention_type": "mla"},
)


class _ToyModel(nn.Module):
    """Small module tree matching the MLA/DSA discovery contract."""

    def __init__(self, *, heads=8, index_heads=4):
        super().__init__()
        self.text_model = ToyTextModel()
        self.attention = ToyMLAAttention()
        self.config = SimpleNamespace(text_config=SimpleNamespace(
            num_attention_heads=heads,
            index_num_attention_heads=index_heads,
            dsa_dense_warm_up=False,
            apply_FA_rescale=True,
            use_fused_sink_fa=False,
        ))

    def forward(self, inputs_embeds=None):
        return self.text_model.forward(inputs_embeds=inputs_embeds)


class _ToyKLLoss:
    @staticmethod
    def apply(*args):
        return args[0]


def _define_attention_symbols(monkeypatch):
    """Define model-module symbols adapted by the MLA/DSA wrapper."""
    module = sys.modules[__name__]

    def apply_mome(hidden_states, mome_mask, conv, use_fused):
        del mome_mask, conv, use_fused
        return hidden_states

    def mla_backend(module, query, key, value, attention_mask, **kwargs):
        del module, key, value, attention_mask, kwargs
        return query

    def sparse_backend(module, query, key, value, attention_mask, **kwargs):
        del module, key, value, attention_mask, kwargs
        return query, None, None

    def indexer(module, index_query, index_key, merge_weight,
                actual_q_len, actual_kv_len):
        del module, index_key, merge_weight, actual_q_len, actual_kv_len
        return index_query

    monkeypatch.setattr(module, "_apply_mome", apply_mome, raising=False)
    monkeypatch.setattr(module, "ATTENTION_FUNCTIONS", {
        "npu_fa_rescale": mla_backend,
        "dsa_sparse_attention": sparse_backend,
    }, raising=False)
    monkeypatch.setattr(
        module, "dsa_lightning_indexer_forward", indexer, raising=False)
    monkeypatch.setattr(
        module, "SparseLightningIndexerKLLossTrainFunction", _ToyKLLoss,
        raising=False)
    return module, apply_mome, mla_backend, sparse_backend, indexer


def test_mla_dsa_wrapper_is_registered():
    assert INNER_WRAPPER_REGISTRY["mla_dsa_ulysses"] is (
        mla_dsa_ulysses_cp_wrapper)
    injection_meta = getattr(mla_dsa_ulysses_cp_wrapper, "_injection_meta")
    assert injection_meta.kind == "inner_wrapper"


@pytest.mark.parametrize("cp_mesh", [None, _FakeCPMesh(size=1)])
def test_mla_dsa_wrapper_requires_active_cp_mesh(cp_mesh):
    with pytest.raises(ValueError, match="active CP mesh"):
        mla_dsa_ulysses_cp_wrapper(
            _ToyModel(), None, None, cp_mesh, None)


def test_mla_dsa_wrapper_validates_head_divisibility():
    with pytest.raises(ValueError, match="num_attention_heads"):
        mla_dsa_ulysses_cp_wrapper(
            _ToyModel(heads=7), None, None, _FakeCPMesh(), None)


def test_mla_dsa_wrapper_requires_config():
    model = _ToyModel()
    del model.config
    with pytest.raises(ValueError, match="target_module.config"):
        mla_dsa_ulysses_cp_wrapper(
            model, None, None, _FakeCPMesh(), None)


def test_mla_dsa_wrapper_requires_non_none_text_config():
    model = _ToyModel()
    model.config.text_config = None
    with pytest.raises(ValueError, match="non-None text_config"):
        mla_dsa_ulysses_cp_wrapper(
            model, None, None, _FakeCPMesh(), None)


@pytest.mark.parametrize(
    "missing_name", ["num_attention_heads", "index_num_attention_heads"])
def test_mla_dsa_wrapper_requires_head_config(missing_name):
    model = _ToyModel()
    delattr(model.config.text_config, missing_name)
    with pytest.raises(ValueError, match=missing_name):
        mla_dsa_ulysses_cp_wrapper(
            model, None, None, _FakeCPMesh(), None)


def test_slice_sequence_uses_cp_rank():
    tensor = torch.arange(16).reshape(2, 8)
    actual = _slice_sequence(tensor, 1, _FakeCPContext())
    torch.testing.assert_close(actual, tensor[:, 4:])
    assert actual.is_contiguous()


def test_wrapper_configures_all_model_side_adaptations(monkeypatch):
    """Verify that one wrapper call configures every model-side adaptation."""
    module, old_mome, old_mla, old_sparse, old_indexer = (
        _define_attention_symbols(monkeypatch))
    model = _ToyModel()
    original_model_forward = model.forward
    original_text_forward = model.text_model.forward

    mla_dsa_ulysses_cp_wrapper(
        model, None, None, _FakeCPMesh(), None)

    assert model.forward != original_model_forward
    assert model.text_model.forward != original_text_forward
    assert getattr(module, "_apply_mome") is not old_mome
    assert module.ATTENTION_FUNCTIONS["npu_fa_rescale"] is not old_mla
    assert module.ATTENTION_FUNCTIONS["dsa_sparse_attention"] is not old_sparse
    assert module.dsa_lightning_indexer_forward is not old_indexer
    assert module.SparseLightningIndexerKLLossTrainFunction is not _ToyKLLoss
    context = getattr(model, "_hyper_ulysses_context")
    assert context.cp_mesh.size() == 2

    inputs = torch.randn(1, 8, 4)
    output = model.text_model.forward(inputs_embeds=inputs, use_cache=False)
    torch.testing.assert_close(output, inputs[:, :4])
