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
"""Generated constructors must install the observed model without patch replay."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from copy import deepcopy

from torch import nn

from hyper_parallel.codegen.emit.context import EmissionContext
from hyper_parallel.codegen.emit.modeling import emit_modeling_file, _verify_recorded_patch_materialization
from hyper_parallel.codegen.inline.construction import materialize_module_constructions


class SourceBlock(nn.Module):
    """A source block with observably different computation."""

    def forward(self, value):
        """Return the source computation."""
        return value + 1


class FinalBlock(nn.Module):
    """A concrete member supporting source conversion at construction."""

    def __init__(self, *, module=None, module_fqn="", context=None):
        super().__init__()
        del module, module_fqn, context

    def forward(self, value):
        """Return the selected member's computation."""
        return value * 2


def _replacement_record():
    return {
        "primary_fqn": "block",
        "module_fqns": ["block"],
        "source_type": f"{__name__}:SourceBlock",
        "replacement_type": f"{__name__}:FinalBlock",
        "replay": {
            "constructor": f"{__name__}:FinalBlock",
            "constructor_arguments": {"mapping": {}},
            "context": {"mapping": {}},
        },
    }


def _patch_plan():
    return SimpleNamespace(
        module_replacements=[SimpleNamespace(
            module_fqn="block", module_type=SourceBlock, replacement_type=FinalBlock,
        )],
        local_computes=(), inner_wrappers=(),
    )


class TestConstructorMaterialization(unittest.TestCase):
    """Inspect the model actually built by the generated source."""

    def test_model_builds_selected_member_without_external_installation(self):
        """A normal constructor and forward must suffice to observe replacement."""
        source = (
            f"from {__name__} import SourceBlock\n"
            "class FutureModel:\n"
            "    def __init__(self):\n"
            "        self.block = SourceBlock()\n"
            "    def forward(self, value):\n"
            "        return self.block(value)\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "modeling_future.py"
            path.write_text(source, encoding="utf-8")
            meta = EmissionContext(
                source={"file_path": str(path), "architecture": "FutureModel"},
                parallel_dims={}, status="derived", preinstalled_components=[_replacement_record()],
            )
            rendered = emit_modeling_file(meta, patch_plan=_patch_plan())
        namespace = {}
        exec(compile(rendered, "modeling_future_gen.py", "exec"), namespace)
        model = namespace["FutureModel"]()
        self.assertIsInstance(model.block, FinalBlock)
        self.assertEqual(model.forward(3), 6)
        self.assertNotIn("_hp_module_constructors", rendered)

    def test_detached_constructor_table_is_not_a_materialized_replacement(self):
        """Listing a factory cannot satisfy a patch whose model remains unchanged."""
        source = (
            f"from {__name__} import SourceBlock, FinalBlock\n"
            "class FutureModel:\n"
            "    def __init__(self):\n"
            "        self.block = SourceBlock()\n"
            "def _hp_module_constructors():\n"
            "    return {'block': lambda **kwargs: FinalBlock(**kwargs)}\n"
        )
        ast.parse(source)
        meta = SimpleNamespace(preinstalled_components=[_replacement_record()])
        with self.assertRaisesRegex(RuntimeError, "block"):
            _verify_recorded_patch_materialization(source, meta, patch_plan=_patch_plan())

    def test_edit_does_not_replace_an_unselected_attribute_of_the_same_type(self):
        """Matching a source type alone must not change unrelated children."""
        source = (
            f"from {__name__} import SourceBlock\n"
            "class FutureModel:\n"
            "    def __init__(self):\n"
            "        self.block = SourceBlock()\n"
            "        self.untouched = SourceBlock()\n"
        )
        rendered = materialize_module_constructions(source, [_replacement_record()])
        namespace = {}
        exec(rendered, namespace)
        model = namespace["FutureModel"]()
        self.assertIsInstance(model.block, FinalBlock)
        self.assertIsInstance(model.untouched, SourceBlock)

    def test_partial_layers_fail_instead_of_silently_replacing_every_layer(self):
        """A shared class body cannot claim coverage for an unselected instance."""
        source = (
            f"from {__name__} import SourceBlock\n"
            "class FutureLayer:\n"
            "    def __init__(self):\n"
            "        self.block = SourceBlock()\n"
        )
        record = _replacement_record()
        record.update(primary_fqn="layers.0.block", module_fqns=["layers.0.block"], construction_sites=[{
            "owner": "FutureLayer", "attribute": "block", "instances": ["layers.0", "layers.1"],
        }])
        with self.assertRaisesRegex(ValueError, "partial construction.*FutureLayer"):
            materialize_module_constructions(source, [record])

    def test_all_layers_with_the_same_selection_share_one_source_edit(self):
        """Repeated observations must produce one class edit, not duplicate wrappers."""
        source = (
            f"from {__name__} import SourceBlock\n"
            "class FutureLayer:\n"
            "    def __init__(self):\n"
            "        self.block = SourceBlock()\n"
        )
        records = []
        for index in range(2):
            record = deepcopy(_replacement_record())
            path = f"layers.{index}.block"
            record.update(primary_fqn=path, module_fqns=[path], construction_sites=[{
                "owner": "FutureLayer", "attribute": "block", "instances": ["layers.0", "layers.1"],
            }])
            records.append(record)
        rendered = materialize_module_constructions(source, records)
        namespace = {}
        exec(rendered, namespace)
        self.assertIsInstance(namespace["FutureLayer"]().block, FinalBlock)
        self.assertEqual(rendered.count("self.block = FinalBlock("), 1)

    def test_ambiguous_assignment_is_rejected(self):
        """Conditional construction needs explicit source coverage rather than guessing."""
        source = (
            f"from {__name__} import SourceBlock\n"
            "class FutureModel:\n"
            "    def __init__(self, condition):\n"
            "        if condition:\n"
            "            self.block = SourceBlock()\n"
            "        else:\n"
            "            self.block = SourceBlock()\n"
        )
        with self.assertRaisesRegex(ValueError, "expected one.*constructor, found 2"):
            materialize_module_constructions(source, [_replacement_record()])
