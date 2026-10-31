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

from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    _apply_module_replacement_actions,
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
    apply_module_replacements(
        model,
        plan,
        context={"ep": True},
        observation_callback=session.record_replacements,
    )
    session.mark_not_required("replacement_only_test")
    snapshot = session.finish().to_dict()

    assert calls == [("alias", {"ep": True})]
    assert model.primary is model.alias
    assert snapshot["replacements"]["declarations"] == [{
        "exact_type": True,
        "match": ["primary"],
        "order": 0,
        "replacement_target": "tests.replace_block",
        "rule_id": "plan_override:3:replace_module",
        "source_type": f"{_SourceBlock.__module__}:{_SourceBlock.__qualname__}",
        "static_arguments": {"mode": "pipelined"},
    }]
    application = snapshot["replacements"]["applications"][0]
    assert application["rule_id"] == "plan_override:3:replace_module"
    assert application["module_fqns"] == ["alias", "primary"]
    assert application["primary_fqn"] == "alias"
    assert application["source_type"].endswith(":_SourceBlock")
    assert application["replacement_type"].endswith(":_ReplacementBlock")
    assert application["weight_transforms"] == []
    assert replacement_records_from_snapshot(session.finish()) == ({
        "exact_type": True,
        "factory": "tests.replace_block",
        "fqn": "alias",
        "fqns": ["alias", "primary"],
        "match": ["primary"],
        "module_type": f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}",
        "target_config": {"mode": "pipelined"},
    },)


def test_python_replacement_rule_gets_stable_order_fallback():
    @module_replacement
    def replace_block(*, module, module_fqn, context):
        del module_fqn, context
        return _ReplacementBlock(module)

    model = nn.Module()
    model.block = _SourceBlock()
    spec = ModuleReplacementSpec(("block",), replace_block, _SourceBlock, True)
    records = {}

    plan = compile_module_replacements(model, (spec,))
    apply_module_replacements(
        model,
        plan,
        observation_callback=lambda **values: records.update(values),
    )

    assert records["declarations"][0]["rule_id"] == "python-replacement-0000"
    assert records["applications"][0]["rule_id"] == "python-replacement-0000"


def test_yaml_replacement_preserves_declaration_order_and_static_arguments():
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

    assert spec.declaration_id == "plan_override:1:replace_module"
    assert spec.replacement_target == "tests.replace_source_block"
    assert spec.static_arguments == {"mode": "pipelined"}


def test_unmatched_rule_is_recorded_before_native_error_is_raised():
    spec = ModuleReplacementSpec(
        match=("missing",),
        factory=_replace_source_block,
        module_type=_SourceBlock,
        declaration_id="missing-rule",
    )
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})

    with pytest.raises(ValueError, match="matched no final module"):
        _apply_module_replacement_actions(
            nn.Module(),
            (spec,),
            observation_callback=session.record_replacements,
        )

    session.mark_not_required("expected_native_error_probe")
    unmatched = session.finish().to_dict()["replacements"]["unmatched"]
    assert unmatched == [{
        "pattern": "missing",
        "reason": "matched_no_final_module",
        "rule_id": "missing-rule",
    }]
