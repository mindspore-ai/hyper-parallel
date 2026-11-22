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

import ast
from copy import deepcopy
from importlib import import_module, util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.inline.framework_spec import observed_constructor_spec, replacement_spec_for
from hyper_parallel.codegen.inline.component_source import collapse_unmodified_components
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.inline.ir import InlinePatchSet, SourceObservation
from hyper_parallel.codegen.inline.observations import collect_source_observations
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.replacement_pass import build_replacement_patches
from hyper_parallel.codegen.model_build_overlay import (
    _observe_replacement_installations,
    _observe_replacements,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.models.module_build import build_module_context
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
    with _observe_replacement_installations(plan.specs) as applications:
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
            "construction": {
                "implementation": f"{_ReplacementBlock.__module__}:{_ReplacementBlock.__qualname__}",
                "source_fqns": ["alias", "primary"],
            },
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
    with _observe_replacement_installations(plan.specs) as applications:
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
    with _observe_replacement_installations(plan.specs) as applications:
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
    """Failed observation restores the native installer and leaves no partial replacement record."""
    original_prepare = replacement_api._prepare_replacement_target
    original_install = replacement_api._install_module_candidates

    with pytest.raises(RuntimeError, match="probe"):
        with _observe_replacement_installations(()):
            assert replacement_api._prepare_replacement_target is not original_prepare
            assert replacement_api._install_module_candidates is not original_install
            raise RuntimeError("probe")

    assert replacement_api._prepare_replacement_target is original_prepare
    assert replacement_api._install_module_candidates is original_install


@pytest.mark.parametrize("separator", [".", ":"])
def test_observed_replacement_class_needs_no_codegen_declaration(separator):
    """Avoid duplicate declarations for an observed replacement class."""
    replacement_type = (
        f"{_ObservedReplacementBlock.__module__}{separator}"
        f"{_ObservedReplacementBlock.__qualname__}"
    )
    source_type = f"{_SourceBlock.__module__}.{_SourceBlock.__qualname__}"

    spec = replacement_spec_for(replacement_type, module_type=source_type)

    assert spec is not None
    assert spec.old_ctor == "_SourceBlock"
    assert spec.new_ctor == "_ObservedReplacementBlock"
    assert spec.mode == "wrap_source"
    assert spec.keyword_args == ("module_fqn=''", "context={}")


class TestNativeStageSource(unittest.TestCase):
    """Generated family classes expose native stages without constructing resources."""

    def setUp(self):
        """Resolve optional-backed components after the CPU stub fixture starts."""
        self.cp_type = import_module(
            "hyper_parallel.components.modules.attention.grouped_query_attention.cp_grouped_query_attention"
        ).CPGroupedQueryAttentionModule
        self.ep_type = import_module(
            "hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts"
        ).EPGroupedExperts

    @staticmethod
    def _render(component, source_text="from torch import nn\n", plan=None):
        """Render and import native stages through the production patch pipeline."""
        rule = SourceObservation(
            match=("model.block",), module_type="demo.Block",
            construction={"implementation": f"{component.__module__}:{component.__name__}"},
        )
        spec = observed_constructor_spec(rule)
        source = apply_patch_set(source_text, InlinePatchSet(
            imports=list(spec.imports) * 2, module_snippets=list(spec.snippets) * 2,
        ))
        if plan is not None:
            source = lower_forward_boundaries(source, plan, boundary_classes={"model.block": component.__name__})
        source = collapse_unmodified_components(source)
        import_lines = [ast.unparse(node) for node in ast.parse(source).body
                        if isinstance(node, (ast.Import, ast.ImportFrom))]
        if len(import_lines) != len(set(import_lines)):
            raise AssertionError("Repeated native stages duplicated artifact imports")
        name = "rendered_family_stage_test"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "modeling.py"
            path.write_text(source, encoding="utf-8")
            module_spec = util.spec_from_file_location(name, path)
            module = util.module_from_spec(module_spec)
            with patch.dict(sys.modules, {name: module}):
                module_spec.loader.exec_module(module)
        return module, source

    def test_ep_source_keeps_inherited_forward_and_native_transport(self):
        """EP dispatch and combine still delegate to the borrowed native context."""
        module, source = self._render(self.ep_type)
        config = import_module(self.ep_type.__module__).GroupedExpertsConfig(
            num_experts=2, hidden_size=2, intermediate_size=2, num_layers=1,
        )
        generated = getattr(module, "EPGroupedExperts")(config=config)
        context = Mock()
        generated.ep_context = context
        hidden, ids, counts, weights = (torch.ones(1) for _ in range(4))
        result = generated.dispatch(hidden, ids, counts, weights)
        self.assertIs(result, context.dispatch.return_value)
        context.dispatch.assert_called_once_with(hidden, ids, weights)
        result = generated.combine(hidden, [1], [1], ids)
        self.assertIs(result, context.combine.return_value)
        context.combine.assert_called_once_with(hidden, [1], [1], ids)
        self.assertIs(type(generated), self.ep_type)
        self.assertIs(generated.forward.__func__, self.ep_type.forward)
        names = [node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
        self.assertNotIn("GroupedExpertsModule", names)
        self.assertNotIn("EPGroupedExperts", names)
        self.assertNotIn("def compute_experts(", source)
        self.assertNotIn("def _packed_parameters(", source)
        self.assertNotIn("test_ep_source", source)
        native = import_module(self.ep_type.__module__)
        self.assertIs(generated.parallel_module_spec().validate, native.validate_ep_layout)
        self.assertNotIn("def validate_ep_layout(", source)
        with self.assertRaisesRegex(ValueError, "complete canonical expert ownership"):
            generated.parallel_module_spec().validate(
                module=generated, sharding=SimpleNamespace(params={}), context=None)

    def test_expert_constructor_preserves_native_depth_and_nearest_config(self):
        """Recorded native depth reaches the emitted expert constructor without config or mesh objects."""
        hf = import_module("transformers.models.qwen3_moe.modeling_qwen3_moe")
        factory = import_module("hyper_parallel.components.modules").GroupedExperts
        source = ("from torch import nn\n"
                  "from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts\n\n"
                  "class Model(nn.Module):\n"
                  "    def __init__(self, config):\n"
                  "        super().__init__()\n"
                  "        self.experts = Qwen3MoeExperts(config)\n")
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("tp",), (2,)), tp_size=2, ep_size=2,
                           fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (1, 2)))
        for root_depth, nearest_depth in ((4, None), (8, 3)):
            with self.subTest(root_depth=root_depth, nearest_depth=nearest_depth), torch.device("meta"):
                config = hf.Qwen3MoeConfig(num_hidden_layers=root_depth, hidden_size=8,
                                          moe_intermediate_size=8, num_local_experts=4, num_experts_per_tok=2)
                model = nn.Module()
                model.config = config
                model.block = nn.Module()
                expert_config = deepcopy(config)
                if nearest_depth is not None:
                    expert_config.num_hidden_layers = nearest_depth
                model.block.config = expert_config
                model.block.experts = hf.Qwen3MoeExperts(expert_config)
                spec = ModuleReplacementSpec(match=("block.experts",), factory=factory)
                plan = compile_module_replacements(model, (spec,))
                context = build_module_context(DistributedSetup(mesh_context=mesh), config=config)
                with _observe_replacement_installations(plan.specs) as applications:
                    model, _ = apply_module_replacements(model, plan, context=context, weights_mapping=[])
                session = CodegenBuildSession()
                session.record_source({"model_type": "qwen3_moe"})
                session.record_replacements(
                    declarations=replacement_api.describe_replacement_declarations(plan.specs),
                    applications=applications,
                )
                session.mark_not_required("constructor_contract")
                records = replacement_records_from_snapshot(session.finish())
                recorded = records[0]["target_config"]["context"]
                self.assertEqual(recorded["num_layers"], nearest_depth or root_depth)
                self.assertEqual(set(recorded), {"tp_size", "cp_size", "ep_size", "pp_size", "num_layers"})
                rules = collect_source_observations(SimpleNamespace(module_overrides=records))
                rendered = apply_patch_set(source, build_replacement_patches(rules))
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "modeling.py"
                    path.write_text(rendered, encoding="utf-8")
                    module_spec = util.spec_from_file_location("generated_expert_depth_contract", path)
                    module = util.module_from_spec(module_spec)
                    with patch.dict(sys.modules, {module.__name__: module}):
                        module_spec.loader.exec_module(module)
                        generated = module.Model(expert_config).experts
                self.assertEqual(generated.config.num_layers, model.block.experts.config.num_layers)
                self.assertEqual(generated.config.init_method_std, model.block.experts.config.init_method_std)
                self.assertEqual(type(generated).__name__, "EPGroupedExperts")

    def test_cp_source_keeps_parent_forward_and_overridden_stages(self):
        """An unchanged CP class retains native identity and helper globals."""
        module, source = self._render(self.cp_type)
        generated = getattr(module, "CPGroupedQueryAttentionModule")
        self.assertIs(generated, self.cp_type)
        for method in ("forward", "prepare_attention_inputs", "exchange_attention", "restore_attention"):
            actual, expected = getattr(generated, method), getattr(self.cp_type, method)
            self.assertIs(actual, expected)
        self.assertNotIn("class CPGroupedQueryAttentionModule(", source)
        self.assertNotIn("class GroupedQueryAttentionModule(", source)
        native = import_module(self.cp_type.__module__)
        hidden = torch.ones(1, 2, 4)
        probe = SimpleNamespace(cp_context=SimpleNamespace(size=2), fuse_qkv=False,
                                project_query_state=Mock(return_value=(hidden, None)),
                                project_key=Mock(return_value=hidden),
                                project_value=Mock(return_value=hidden),
                                _transfer_input=Mock(side_effect=lambda value, **_: (value, None)))
        prepared = generated.prepare_attention_inputs(probe, hidden, None)
        self.assertIs(type(prepared), native.CPGQAPreparedInputs)
        self.assertIs(prepared.query, hidden)

    def test_cp_source_preserves_source_model_rotary_helper(self):
        """A copied baseline must not replace the source model's identically named math helper."""
        module, _ = self._render(self.cp_type, "from torch import nn\n\ndef rotate_half(x):\n    return x\n")
        self.assertEqual(getattr(module, "rotate_half")(7), 7)

    def test_tp_forward_keeps_native_cp_stages_and_executes_collectives(self):
        """Only the added TP boundary is emitted; native CP methods retain identity."""
        plan = {"mesh_dim_names": ("tp",), "param_plan": {"model.block": {
            "is_boundary": True,
            "in_src": {"hidden_states": {"tp": "S(1)"}},
            "in_dst": {"hidden_states": {"tp": "R"}},
            "out_src": {"output": {"tp": "P(sum)"}},
            "out_dst": {"output": {"tp": "S(1)"}},
        }}}
        module, source = self._render(self.cp_type, plan=plan)
        generated = module.CPGroupedQueryAttentionModule
        self.assertIs(generated.__bases__[0], self.cp_type)
        for method in ("__init__", "bind_parallel", "exchange_attention", "restore_attention"):
            self.assertIs(getattr(generated, method), getattr(self.cp_type, method))
        self.assertIn("self._hp_tp.all_gather(", source)
        self.assertIn("self._hp_tp.reduce_scatter(", source)
        self.assertNotIn("def prepare_attention_inputs(", source)
        config = import_module("transformers.models.qwen3_moe.configuration_qwen3_moe").Qwen3MoeConfig(
            hidden_size=4, num_attention_heads=1, num_key_value_heads=1, head_dim=4,
        )
        probe = generated(config=config)
        probe._hp_tp = Mock(all_gather=Mock(side_effect=lambda value, **_: value * 2),
                            reduce_scatter=Mock(side_effect=lambda value, **_: value / 3))
        hidden = torch.ones(1, 2, 4)
        with patch.object(self.cp_type, "forward", autospec=True, return_value=(hidden * 2 + 1, None)) as native:
            result = probe(hidden)
        torch.testing.assert_close(result[0], hidden)
        self.assertIsNone(result[1])
        native.assert_called_once()
        torch.testing.assert_close(native.call_args.args[1], hidden * 2)
        self.assertEqual(collapse_unmodified_components(source), source)

    def test_component_collapse_keeps_modified_native_delegation(self):
        """A subclass that changes arguments or output must never collapse to an import."""
        native = f"{self.cp_type.__module__}.{self.cp_type.__name__}"
        for body in ("return super().forward(hidden_states * 2)",
                     "return super().forward(hidden_states) * 2"):
            with self.subTest(body=body):
                source = (f"import {self.cp_type.__module__}\nclass {self.cp_type.__name__}({native}):\n"
                          f"    def forward(self, hidden_states):\n        {body}\n")
                self.assertEqual(collapse_unmodified_components(source), source)

    def test_inherited_forward_with_super_is_not_silently_rebound(self):
        """Imported native methods retain their defining-class closure."""
        class Parent(nn.Module):
            """Supply a forward that captures its defining class."""

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                """Keep the defining-class reference in the code object."""
                return super().forward(value) + value

        with patch.object(self.cp_type.__bases__[0], "forward", Parent.forward):
            module, _ = self._render(self.cp_type)
            self.assertIs(module.CPGroupedQueryAttentionModule.forward, Parent.forward)
