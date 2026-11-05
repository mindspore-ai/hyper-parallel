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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.inline.framework_spec import replacement_spec_for
from hyper_parallel.codegen.model_build_overlay import (
    _observe_replacement_installations,
    _observe_replacements,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models import replacement as replacement_api
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
class _ObservedReplacementBlock(_ReplacementBlock):
    """Model-owned replacement with required native executor arguments."""

    def __init__(self, *, module, module_fqn, context):
        del module_fqn, context
        super().__init__(module)


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
    with _observe_replacement_installations(model, plan.specs) as applications:
        apply_module_replacements(model, plan, context={"ep": True})
    session.record_replacements(
        declarations=replacement_api.describe_replacement_declarations(plan.specs),
        applications=applications,
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
    assert replacement_records_from_snapshot(session.finish()) == (
        {
            "exact_type": True,
            "factory": f"{_ReplacementBlock.__module__}.{_ReplacementBlock.__qualname__}",
            "fqn": "alias",
            "fqns": ["alias", "primary"],
            "match": ["primary"],
            "module_type": f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}",
            "replacement_target": "tests.replace_block",
            "target_config": {"mode": "pipelined"},
        },
    )


def test_python_replacement_rule_gets_stable_order_fallback():
    """Python replacement declarations receive deterministic fallback order."""

    @module_replacement
    def replace_block(*, module, module_fqn, context):
        del module_fqn, context
        return _ReplacementBlock(module)

    model = nn.Module()
    model.block = _SourceBlock()
    spec = ModuleReplacementSpec(("block",), replace_block, _SourceBlock, True)
    applications = []

    plan = compile_module_replacements(model, (spec,))
    with _observe_replacement_installations(model, plan.specs) as applications:
        apply_module_replacements(model, plan)

    declarations = replacement_api.describe_replacement_declarations(plan.specs)
    assert declarations[0]["rule_id"] == "python-replacement-0000"
    assert applications[0]["rule_id"] == "python-replacement-0000"


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
    with _observe_replacement_installations(model, plan.specs) as applications:
        apply_module_replacements(model, plan)
    session.record_replacements(
        declarations=replacement_api.describe_replacement_declarations(plan.specs),
        applications=applications,
    )
    session.mark_not_required("replacement_only_test")
    snapshot = session.finish()

    replacements = snapshot.to_dict()["replacements"]
    assert replacements["declarations"][0]["source_type"] is None
    assert replacements["applications"][0]["source_type"] == (
        f"{_SourceBlock.__module__}:{_SourceBlock.__qualname__}"
    )
    assert replacement_records_from_snapshot(snapshot)[0]["module_type"] == (
        f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}"
    )


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


def test_unmatched_rule_is_recorded_before_native_error_is_raised():
    """Capture an unmatched rule before its native error propagates."""
    spec = ModuleReplacementSpec(
        match=("missing",),
        factory=_replace_source_block,
        module_type=_SourceBlock,
        declaration_id="missing-rule",
    )
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})

    with pytest.raises(ValueError, match="matched no final module"):
        _observe_replacements(
            nn.Module(),
            SimpleNamespace(module_replacements=(spec,), mesh_context=None),
            None,
            False,
            session,
        )

    session.mark_not_required("expected_native_error_probe")
    unmatched = session.finish().to_dict()["replacements"]["unmatched"]
    assert unmatched == [
        {
            "pattern": "missing",
            "reason": "matched_no_final_module",
            "rule_id": "missing-rule",
        }
    ]


def test_replacement_observer_restores_native_installer_after_error():
    original = replacement_api._install_replacement_target

    with pytest.raises(RuntimeError, match="probe"):
        with _observe_replacement_installations(nn.Module(), ()):
            assert replacement_api._install_replacement_target is not original
            raise RuntimeError("probe")

    assert replacement_api._install_replacement_target is original


def test_observed_replacement_class_needs_no_codegen_declaration():
    """Avoid duplicate declarations for an observed replacement class."""
    replacement_type = (
        f"{_ObservedReplacementBlock.__module__}."
        f"{_ObservedReplacementBlock.__qualname__}"
    )
    source_type = f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}"

    spec = replacement_spec_for(replacement_type, module_type=source_type)

    assert spec is not None
    assert spec.old_ctor == "_SourceBlock"
    assert spec.new_ctor == "_ObservedReplacementBlock"
    assert spec.mode == "wrap_source"
    assert spec.keyword_args == ("module_fqn=''", "context={}")
