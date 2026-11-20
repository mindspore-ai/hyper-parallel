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
"""Contracts for recording native module replacement decisions."""

import pytest
import torch
from torch import nn

from hyper_parallel.codegen.manager import _preinstalled_records
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models import replacement as replacement_api
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.preparation_manifest import collect_replacement_applications, get_preparation_manifest
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    module_replacement,
)
from hyper_parallel.trainer.config import (
    PlanOverride,
    ReplaceAction,
    Target,
    entries_to_module_replacements,
)


class _SourceBlock(nn.Module):
    """Minimal source module used by replacement-recording tests."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        return inputs * self.weight


class _ReplacementBlock(nn.Module):
    """Structure-preserving shell that reuses the source parameter."""

    def __init__(self, source):
        super().__init__()
        self.register_parameter("weight", source.weight)
        self.training = source.training

    def forward(self, inputs):
        return inputs * self.weight


@module_replacement
def _replace_source_block(*, module, module_fqn, context):
    del module_fqn, context
    return _ReplacementBlock(module)


def test_native_replacement_records_declaration_and_application_once():
    """Record each native replacement declaration and application once."""
    calls = []

    @module_replacement
    def replace_block(*, module, module_fqn, context):
        calls.append((module_fqn, dict(context)))
        return _ReplacementBlock(module)

    source = _SourceBlock()
    model = nn.Module()
    model.primary = source
    model.alias = source
    spec = ModuleReplacementSpec(
        match=("primary",),
        factory=replace_block,
        module_type=_SourceBlock,
        exact_type=True,
        declaration_id="plan_override:3:replace_module",
        replacement_target="tests.replace_block",
        static_arguments={"mode": "pipelined"},
    )
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})

    plan = compile_module_replacements(model, (spec,))
    with collect_replacement_applications() as applications:
        apply_module_replacements(model, plan, context={"ep": True})
    session.record_replacements(
        declarations=replacement_api.describe_replacement_declarations(plan.specs),
        applications=[application.to_legacy_dict() for application in applications],
    )
    session.mark_not_required("replacement_only_test")
    snapshot = session.finish().to_dict()

    assert calls == [("alias", {"ep": True})]
    assert model.primary is model.alias
    assert snapshot["replacements"]["declarations"] == [
        {
            "exact_type": True,
            "match": ["primary"],
            "order": 0,
            "replacement_target": "tests.replace_block",
            "rule_id": "plan_override:3:replace_module",
            "source_type": f"{_SourceBlock.__module__}:{_SourceBlock.__qualname__}",
            "static_arguments": {"mode": "pipelined"},
        }
    ]
    application = snapshot["replacements"]["applications"][0]
    assert application["rule_id"] == "plan_override:3:replace_module"
    assert application["module_fqns"] == ["alias", "primary"]
    assert application["primary_fqn"] == "alias"
    assert application["source_type"].endswith(":_SourceBlock")
    assert application["replacement_type"].endswith(":_ReplacementBlock")
    assert application["weight_transforms"] == []
    assert application.get("replay") is None


def test_python_replacement_rule_gets_stable_order_fallback():
    """Python replacement declarations receive deterministic fallback order."""

    @module_replacement
    def replace_block(*, module, module_fqn, context):
        del module_fqn, context
        return _ReplacementBlock(module)

    model = nn.Module()
    model.block = _SourceBlock()
    spec = ModuleReplacementSpec(("block",), replace_block, _SourceBlock, True)

    plan = compile_module_replacements(model, (spec,))
    with collect_replacement_applications() as applications:
        apply_module_replacements(model, plan)

    declarations = replacement_api.describe_replacement_declarations(plan.specs)
    assert declarations[0]["rule_id"] == "python-replacement-0000"
    assert applications[0].declaration_id == "python-replacement-0000"


def test_inferred_source_type_is_recorded_from_native_application():
    """A declaration without source_type uses the actually matched module type."""
    model = nn.Module()
    model.block = _SourceBlock()
    spec = ModuleReplacementSpec(
        match=("block",),
        factory=_replace_source_block,
        module_type=None,
        replacement_target="tests.replace_source_block",
    )
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})

    plan = compile_module_replacements(model, (spec,))
    with collect_replacement_applications() as applications:
        apply_module_replacements(model, plan)
    session.record_replacements(
        declarations=replacement_api.describe_replacement_declarations(plan.specs),
        applications=[application.to_legacy_dict() for application in applications],
    )
    session.mark_not_required("replacement_only_test")
    snapshot = session.finish()

    replacements = snapshot.to_dict()["replacements"]
    assert replacements["declarations"][0]["source_type"] is None
    assert replacements["applications"][0]["source_type"] == (
        f"{_SourceBlock.__module__}:{_SourceBlock.__qualname__}"
    )
    replay = replacements["applications"][0]["replay"]
    assert replay["constructor"] == f"{_replace_source_block.__module__}:{_replace_source_block.__qualname__}"


def test_yaml_replacement_preserves_declaration_order_and_static_arguments():
    """YAML replacements preserve declaration order and static arguments."""
    target = Target(
        _replace_source_block,
        target_path="tests.replace_source_block",
        mode="pipelined",
    )
    entries = [
        PlanOverride(match="ignored", params={}),
        PlanOverride(
            match="block",
            replace=ReplaceAction(
                kind="module",
                implementation=target,
                source_type=f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}",
                exact_type=True,
            ),
        ),
    ]

    spec = entries_to_module_replacements(entries)[0]

    assert spec.declaration_id == "plan_override:1:replace"
    assert spec.replacement_target == "tests.replace_source_block"
    assert spec.static_arguments == {"mode": "pipelined"}


def test_unmatched_rule_fails_before_preparation_publishes_a_manifest():
    """An unmatched native rule fails before publishing any application."""
    spec = ModuleReplacementSpec(
        match=("missing",),
        factory=_replace_source_block,
        module_type=_SourceBlock,
        declaration_id="missing-rule",
    )
    model = nn.Module()
    with pytest.raises(ValueError, match="matched no final module"):
        prepare_model_components(model, {**build_module_context(None), "module_replacements": (spec,)})

    with pytest.raises(RuntimeError, match="have not produced a PreparationManifest"):
        get_preparation_manifest(model)


def test_missing_constructor_inputs_are_rejected_at_exact_fqn():
    """A factory identity alone does not describe the final model construction."""
    payload = {"replacements": {"applications": [
        {"primary_fqn": "layers.0.attention", "replay": {"constructor": "demo:Attention"}},
        {"primary_fqn": "layers.0.experts"},
    ]}}
    with pytest.raises(ValueError, match=r"no concrete construction inputs.*layers\.0\.experts"):
        _preinstalled_records(payload)
