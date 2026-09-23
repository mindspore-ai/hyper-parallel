# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Contract tests for generated-model ``replace_module`` handling.

``meta.module_overrides`` FQN records must agree with a generation-time
``compile_module_replacements`` hit (same FQN set).

The tests exercise the spec-building path: ``entries_to_module_replacements``
(YAML desugar) fed through ``compile_overrides_for_meta`` (the manager's
``_fill_plan_fields`` step), rather than a hand-built record.
"""

from __future__ import annotations

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
    """The entry class whose generated ``__init__`` installs the replacements."""

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
    """Build the module-override meta for the given source path."""
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
    records = compile_overrides_for_meta(
        _meta_model(),
        specs,
        factory_paths=["tests.codegen.helpers._replace_mlp"],
    )

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


def test_g7_nomatch_pattern_fails_fast(source_path):
    """A replace_module rule matching no module is a config error, not a skip.

    Mirrors the native ``compile_module_replacements`` behavior: an unmatched
    pattern raises a fail-fast :class:`ValueError` at generation time instead
    of silently recording a ``no_match`` skip that would drop the user's
    intended replacement from the artifact.
    """
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
    with pytest.raises(ValueError, match="matched no module"):
        compile_overrides_for_meta(
            _meta_model(),
            specs,
            factory_paths=["tests.codegen.helpers._replace_mlp"],
        )


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
