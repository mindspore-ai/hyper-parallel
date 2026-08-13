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
"""YAML transport contracts for generic module replacement."""

import unittest
from unittest.mock import patch

from hyper_models.components.model_transform import module_replacement
from hyper_models.config.resolver import resolve_root
from hyper_models.trainer.config import (
    PlanOverride,
    Target,
    _import_module_type,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)


def _model_target():
    return None


def _tokenizer_target():
    return None


def _optimizer_target():
    return None


def _replacement(*, module, module_fqn, context):
    del module_fqn, context
    return module


@module_replacement
def _declared_replacement(*, module, module_fqn, context):
    return _replacement(module=module, module_fqn=module_fqn, context=context)


def _root(plan_overrides):
    return {
        "model": {"_target_": f"{__name__}._model_target"},
        "tokenizer": {"_target_": f"{__name__}._tokenizer_target"},
        "optimizer": {"_target_": f"{__name__}._optimizer_target"},
        "plan_overrides": plan_overrides,
    }


class TestModuleReplacementYaml(unittest.TestCase):
    """Replacement actions share the existing plan_overrides YAML transport."""

    def test_yaml_replacement_desugars_to_generic_rule(self):
        config = resolve_root(_root([
            {
                "match": ["encoder.*", "decoder.*"],
                "module_type": "torch.nn.Linear",
                "exact_type": True,
                "replace_module": {
                    "_target_": f"{__name__}._declared_replacement",
                },
            },
        ]))

        rules = entries_to_module_replacements(config.plan_overrides)

        self.assertEqual(rules[0].match, ("encoder.*", "decoder.*"))
        self.assertEqual(rules[0].module_type.__name__, "Linear")
        self.assertTrue(rules[0].exact_type)

    def test_yaml_replacement_rejects_undeclared_factory_contract(self):
        config = resolve_root(_root([
            {
                "match": "encoder.*",
                "module_type": "torch.nn.Linear",
                "replace_module": {"_target_": f"{__name__}._replacement"},
            },
        ]))

        with self.assertRaisesRegex(TypeError, "@module_replacement"):
            entries_to_module_replacements(config.plan_overrides)

    def test_list_match_without_replacement_is_a_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "match lists"):
            entries_to_plan_overrides([PlanOverride(match=["a", "b"])])

    def test_replacement_rejects_valid_when(self):
        entry = PlanOverride(
            match="encoder.*",
            when="ep",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _declared_replacement,
                target_path=f"{__name__}._declared_replacement",
            ),
        )

        with self.assertRaisesRegex(ValueError, "does not support 'when'"):
            entries_to_module_replacements([entry])

    def test_replacement_rejects_invalid_when(self):
        entry = PlanOverride(
            match="encoder.*",
            when="xp",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _declared_replacement,
                target_path=f"{__name__}._declared_replacement",
            ),
        )

        with self.assertRaisesRegex(ValueError, "does not support 'when'"):
            entries_to_module_replacements([entry])

    def test_module_type_import_error_has_plan_override_context(self):
        with patch(
            "hyper_models.trainer.config.importlib.import_module",
            side_effect=ImportError("optional dependency is unavailable"),
        ):
            with self.assertRaisesRegex(ValueError, "plan_overrides.module_type"):
                _import_module_type("example.module.Type")
