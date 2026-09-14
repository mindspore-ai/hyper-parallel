# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Contract tests for generated-model ``replace_module`` handling.

* ``_HYPER_MODULE_OVERRIDES`` literal exists in the artifact;
* ``meta.module_overrides`` FQN records agree with a generation-time
  ``compile_module_replacements`` hit (same FQN set);
* the literal's FQN set == the meta record's FQN set.

The tests exercise the complete generation path:

* building the spec via ``entries_to_module_replacements`` (YAML desugar) and
  ``compile_overrides_for_meta`` (manager's ``_fill_plan_fields`` step) —
  not a hand-built record;
* assembling the artifact with ``emit_modeling_file``, then importing it and
  building a model so the entry ``__init__`` tail actually runs
  ``hyper_apply_replacements`` and swaps the module.
"""

from __future__ import annotations

import ast
import types

import pytest

from hyper_parallel.codegen.emit.modeling import emit_modeling_file
from hyper_parallel.codegen.emit.replacement import compile_overrides_for_meta
from hyper_parallel.codegen.meta import CodegenMeta
from hyper_parallel.trainer.config import (
    PlanOverride,
    Target,
    entries_to_module_replacements,
)

from . import helpers

SOURCE_TEXT = '''\
# Copyright 2026 Huawei Technologies Co., Ltd
# ============================================================================
"""Generated-source fixture for module replacement tests."""

from torch import nn

from tests.codegen.helpers import SourceMLP


class GeneratedModel(nn.Module):
    """The entry class the artifact rewrite calls ``hyper_apply_replacements`` in."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(SourceMLP())

    def forward(self, x):
        return self.model(x)
'''


@pytest.fixture(scope="module")
def source_path(tmp_path_factory) -> str:
    """Write the SourceMLP modeling source to a real file (manager-style)."""
    path = tmp_path_factory.mktemp("g7src") / "source_modeling.py"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(SOURCE_TEXT)
    return str(path)


def _meta(source_path: str, *, model_class: str, overrides: list[dict]) -> CodegenMeta:
    meta = CodegenMeta(
        codegen_version="v3",
        signature="test-g7",
        yaml_path="examples/training_demo/train_codegen_qwen3_moe.yaml",
        yaml_sha256="test",
        source={
            "file_path": source_path,
            "module_name": "transformers.models.mock.modeling_mock",
            "architecture": "GeneratedModel",
            "sha256": "test",
            "module_class": model_class,
        },
        parallel_dims={"tp_size": 1, "cp_size": 1, "ep_size": 1},
    )
    meta.model_class = model_class
    meta.module_overrides = overrides
    return meta


def _meta_model() -> helpers.MetaModel:
    return helpers.MetaModel()


def _spec_entries():
    """A real YAML-shaped ``replace_module`` entry (module_type separate)."""
    return [
        PlanOverride(
            match="model.0",
            module_type="tests.codegen.helpers.SourceMLP",
            exact_type=False,
            replace_module=Target(
                helpers._replace_mlp,
                target_path="tests.codegen.helpers._replace_mlp",
            ),
        )
    ]


def test_g7_meta_records_match_via_desugar_path(source_path):
    """YAML desugar -> compile-over-meta yields a record with the FQN set."""
    specs = entries_to_module_replacements(_spec_entries())
    records, skipped = compile_overrides_for_meta(
        _meta_model(),
        specs,
        factory_paths=["tests.codegen.helpers._replace_mlp"],
    )

    assert skipped == ()
    assert len(records) == 1
    record = records[0]
    # The match pattern is carried verbatim.
    assert record["match"] == ["model.0"]
    # The FQN set is exactly what the compile hit against the meta model aliases.
    assert record["fqns"] == ["model.0"]
    assert record["fqn"] == "model.0"
    # The factory must serialize by import path (a closure is not a literal).
    assert record["factory"] == "tests.codegen.helpers._replace_mlp"
    assert record["module_type"] == "tests.codegen.helpers.SourceMLP"
    assert record["exact_type"] is False
    # The factory path survives to the record (the closure serialization).
    assert record["match"] and record["fqns"]


def test_g7_skipped_partition_no_match(source_path):
    """A rule that matches nothing is recorded, not dropped, and never compiled."""
    spec = PlanOverride(
        match="model.99",
        module_type="tests.codegen.helpers.SourceMLP",
        exact_type=False,
        replace_module=Target(
            helpers._replace_mlp,
            target_path="tests.codegen.helpers._replace_mlp",
        ),
    )
    specs = entries_to_module_replacements([spec])
    records, skipped = compile_overrides_for_meta(
        _meta_model(),
        specs,
        factory_paths=["tests.codegen.helpers._replace_mlp"],
    )

    assert records == ()
    assert skipped == ({"match": ["model.99"], "reason": "no_match"},)


def test_g7_emit_literal_and_init_call(source_path):
    """Emit the artifact and check the module replacement surface."""
    meta = _meta(
        source_path,
        model_class="GeneratedModel",
        overrides=[
            {
                "match": ["model.0"],
                "fqn": "model.0",
                "fqns": ["model.0"],
                "module_type": "tests.codegen.helpers.SourceMLP",
                "factory": "tests.codegen.helpers._replace_mlp",
                "exact_type": False,
            }
        ],
    )
    text = emit_modeling_file(meta)

    assert "_HYPER_MODULE_OVERRIDES = " in text
    # The entry __init__ tail must invoke the runtime apply with the literal.
    assert "hyper_apply_replacements(self, _HYPER_MODULE_OVERRIDES)" in text


def test_g7_literal_fqn_set_equals_meta_fqn_set(source_path):
    """The literal's FQN set equals the metadata record set."""
    record = {
        "match": ["model.0"],
        "fqn": "model.0",
        "fqns": ["model.0"],
        "module_type": "tests.codegen.helpers.SourceMLP",
        "factory": "tests.codegen.helpers._replace_mlp",
        "exact_type": False,
    }
    meta = _meta(source_path, model_class="GeneratedModel", overrides=[record])
    text = emit_modeling_file(meta)

    # Re-parse the literal FQN set independently of the meta record path.
    literal_fqn_set = _literal_fqn_set(text)
    # The emitted literal should carry the same FQN set as the meta record.
    assert literal_fqn_set == {"model.0"}
    assert literal_fqn_set == set(record["fqns"])


def test_g7_runtime_applies_replacement(source_path):
    """End-to-end: emit, import, and confirm the entry __init__ swaps the module."""
    record = {
        "match": ["model.0"],
        "fqn": "model.0",
        "fqns": ["model.0"],
        "module_type": "tests.codegen.helpers.SourceMLP",
        "factory": "tests.codegen.helpers._replace_mlp",
        "exact_type": False,
    }
    meta = _meta(source_path, model_class="GeneratedModel", overrides=[record])
    text = emit_modeling_file(meta)

    module = types.ModuleType("_g7_generated")
    exec(compile(text, "<g7>", "exec"), module.__dict__)

    model = module.GeneratedModel()
    # The entry __init__ tail applied the replacement: SourceMLP -> ReplacementMLP.
    assert type(model.model[0]).__name__ == "ReplacementMLP"


def test_g7_no_model_class_fails_fast(source_path):
    """module_overrides present but model_class blank -> RuntimeError."""
    record = {
        "match": ["model.0"],
        "fqn": "model.0",
        "fqns": ["model.0"],
        "module_type": "tests.codegen.helpers.SourceMLP",
        "factory": "tests.codegen.helpers._replace_mlp",
        "exact_type": False,
    }
    meta = _meta(source_path, model_class="GeneratedModel", overrides=[record])
    meta.model_class = None

    # emit_modeling_file must reject the artifact before it can silently drop a rule.
    with pytest.raises(RuntimeError):
        emit_modeling_file(meta)


def _literal_fqn_set(text: str) -> set[str]:
    """Parse the FQN set out of the emitted ``_HYPER_MODULE_OVERRIDES`` literal."""
    marker = "_HYPER_MODULE_OVERRIDES = "
    start = text.index(marker) + len(marker)
    # ``render_python_literal`` may render the list across several lines, so
    # scan for the matching closing ``]`` rather than taking a single line.
    depth = 0
    end = start
    while True:
        char = text[end]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    value = ast.literal_eval(text[start:end].strip())
    return {fqn for record in value for fqn in record["fqns"]}
