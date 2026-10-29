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
"""Tests for Trainer-owned replacements on copied codegen model classes."""

from types import SimpleNamespace

import pytest
from torch import nn

from hyper_parallel.models._transformers.model_builder import (
    _rebind_codegen_replacement_specs,
    _symbol_path,
)
from hyper_parallel.models.replacement import ModuleReplacementSpec, module_replacement


class SourceBlock(nn.Module):
    """Native source type referenced by the Trainer replacement rule."""


@module_replacement
def _replace_block(module, module_fqn, context):
    del module_fqn, context
    return module


def _generated_model(source_type=None):
    copied_type = source_type or type(
        "SourceBlock",
        (nn.Module,),
        {"__module__": "generated_modeling", "__qualname__": SourceBlock.__qualname__},
    )
    root_type = type("GeneratedModel", (nn.Module,), {"__module__": "generated_modeling"})
    model = root_type()
    model.block = copied_type()
    return model, copied_type


def _meta(module_fqn="block", source_type=None):
    return SimpleNamespace(emission_patches={
        "module_replacements": [{
            "module_fqn": module_fqn,
            "source_type": source_type or _symbol_path(SourceBlock),
            "replacement_type": "tests:ReplacementBlock",
        }],
    })


def _spec(exact_type=False):
    return ModuleReplacementSpec(("block",), _replace_block, SourceBlock, exact_type)


def test_rebind_codegen_replacement_uses_observed_generated_type(monkeypatch):
    model, copied_type = _generated_model()
    monkeypatch.setattr("hyper_parallel.codegen.runtime.load_codegen_meta", lambda _: _meta())

    rebound = _rebind_codegen_replacement_specs(model, (_spec(exact_type=True),), "artifact")

    assert rebound[0].module_type is copied_type
    assert rebound[0].exact_type is True
    assert rebound[0].factory is _replace_block


def test_rebind_codegen_replacement_leaves_native_build_unchanged():
    spec = _spec()

    assert _rebind_codegen_replacement_specs(nn.Module(), (spec,), None) == (spec,)


@pytest.mark.parametrize(
    "meta",
    (
        _meta(source_type="other.module:SourceBlock"),
        _meta(module_fqn="other"),
    ),
)
def test_rebind_codegen_replacement_rejects_metadata_drift(monkeypatch, meta):
    model, _ = _generated_model()
    monkeypatch.setattr("hyper_parallel.codegen.runtime.load_codegen_meta", lambda _: meta)

    with pytest.raises(RuntimeError):
        _rebind_codegen_replacement_specs(model, (_spec(),), "artifact")


def test_rebind_codegen_replacement_rejects_same_name_impostor(monkeypatch):
    impostor = type(
        "SourceBlock",
        (nn.Module,),
        {"__module__": "unrelated_modeling", "__qualname__": SourceBlock.__qualname__},
    )
    model, _ = _generated_model(impostor)
    monkeypatch.setattr("hyper_parallel.codegen.runtime.load_codegen_meta", lambda _: _meta())

    with pytest.raises(RuntimeError, match="not the generated equivalent"):
        _rebind_codegen_replacement_specs(model, (_spec(),), "artifact")
