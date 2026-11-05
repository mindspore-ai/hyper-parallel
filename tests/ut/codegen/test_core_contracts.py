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
"""Core contract tests for codegen artifacts, plans, and publication."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from hyper_parallel.codegen.artifact import (
    clean_temp_artifacts,
    resolve_default_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.emit.bundle import _rewrite_sibling_imports
from hyper_parallel.codegen.plan.placements import (
    parse_placement,
    placement_to_string,
)
from hyper_parallel.codegen.source.resolver import resolve_remote_siblings
from hyper_parallel.trainer.config import parser as config_parser
from hyper_parallel.trainer.config.parallelism import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_component


def test_dynamic_module_source_resolves_remote_siblings(tmp_path: Path) -> None:
    """HF dynamic modules retain the same sibling closure as remote sources."""
    modeling = tmp_path / "modeling_demo.py"
    configuration = tmp_path / "configuration_demo.py"
    modeling.write_text(
        "from .configuration_demo import DemoConfig\n", encoding="utf-8"
    )
    configuration.write_text("class DemoConfig:\n    pass\n", encoding="utf-8")
    source = SimpleNamespace(
        module_name="transformers_modules.demo.digest.modeling_demo",
        file_path=str(modeling),
    )

    siblings = resolve_remote_siblings(source)

    assert siblings == {str(configuration.resolve()): str(configuration.resolve())}


def test_dynamic_module_import_retargets_generated_sibling() -> None:
    """An absolute HF cache import resolves to its generated bundle sibling."""
    source = (
        "from transformers_modules.kimi_k3.digest.modeling_kimi_linear "
        "import KimiLinearForCausalLM\n"
    )

    rendered = _rewrite_sibling_imports(
        source,
        {"modeling_kimi_linear.py": "modeling_kimi_linear_gen_npu.py"},
    )

    assert rendered == (
        "from .modeling_kimi_linear_gen_npu import KimiLinearForCausalLM\n"
    )


def test_dynamic_module_import_retargets_copied_configuration() -> None:
    """An unchanged configuration sibling is imported from the bundle."""
    source = (
        "from transformers_modules.kimi_k3.digest.configuration_kimi_k3 "
        "import KimiK3Config\n"
    )

    rendered = _rewrite_sibling_imports(
        source,
        {"configuration_kimi_k3.py": "configuration_kimi_k3.py"},
    )

    assert rendered == "from .configuration_kimi_k3 import KimiK3Config\n"


@dataclass
class _ModelTarget:
    """Minimal immutable-like model target for config preparation tests."""

    kwargs: dict[str, object] = field(default_factory=dict)

    def replace(self, **changes: object) -> "_ModelTarget":
        """Return a target carrying the supplied configured arguments."""
        return _ModelTarget({**self.kwargs, **changes})


@dataclass
class _TrainingConfig:
    """Minimal dataclass accepted by ``dataclasses.replace``."""

    model: _ModelTarget = field(default_factory=_ModelTarget)
    codegen: bool = False


def _bundle_files(layout: object) -> dict[str, str]:
    return {
        os.path.basename(layout.modeling_path): "VALUE = 1\n",
        os.path.basename(layout.diff_path): "--- source\n+++ generated\n",
        "__init__.py": "",
    }


def test_default_artifact_layout_uses_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without YAML codegen paths the artifact layout falls back to the CWD."""
    monkeypatch.chdir(tmp_path)

    layout = resolve_default_artifact_layout(model_name="Qwen3-MoE")

    assert Path(layout.artifact_dir) == tmp_path / "generated"
    assert Path(layout.modeling_path).name == "modeling_Qwen3_MoE_gen_npu.py"


def test_load_training_config_does_not_prepare_codegen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Loading the training config must not eagerly prepare codegen."""
    config = _TrainingConfig(codegen=True)
    yaml_path = tmp_path / "train.yaml"
    monkeypatch.setattr(config_parser, "_load_training_config", lambda *_args: config)

    loaded = config_parser.load_training_config(yaml_path)

    assert loaded is config
    assert loaded._yaml_path == str(yaml_path.resolve())  # pylint: disable=protected-access


def test_training_args_only_parse_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _TrainingConfig(codegen=True)
    yaml_path = tmp_path / "train.yaml"
    monkeypatch.setattr(config_parser, "_load_training_config", lambda *_args: config)

    assert config_parser.parse_training_args([str(yaml_path)]) is config


def test_deepseek_v41_engram_params_target_planner_boundary() -> None:
    """Engram parameters bind to the module while compute binds to forward."""
    params_entry = {
        "match": "model.layers.*.engram",
        "params": {
            "q_weight": {"tp": "replicate"},
            "k_weight": {"tp": "replicate"},
            "embed.weight": {"ep": "shard(0)"},
            "wkv.weight": {"tp": "replicate"},
        },
    }
    method_entry = {
        "match": "model.layers.*.engram.forward",
        "when": "ep",
        "replace": {
            "kind": "method",
            "implementation": {
                "_target_": (
                    "hyper_parallel.models.deepseek_v41.adapter.distributed."
                    "moe_engram_expert_parallel.deepseek_v41_engram_compute_fn"
                ),
            },
        },
    }

    resolved = resolve_component(params_entry, annotation=PlanOverride, path="$.plan_overrides[0]")
    overrides = entries_to_plan_overrides([resolved], ep_size=8)

    assert set(overrides) == {"model.layers.*.engram"}
    assert overrides["model.layers.*.engram"].params is not None
    assert method_entry["match"] == "model.layers.*.engram.forward"


def test_atomic_bundle_publishes_only_emitted_files(tmp_path: Path) -> None:
    """Atomic publication has no metadata sidecar or hidden completion file."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")
    write_bundle_atomic(layout, _bundle_files(layout))

    assert os.path.isfile(layout.modeling_path)
    assert sorted(os.listdir(layout.artifact_dir)) == [
        "__init__.py",
        "modeling_demo_gen_npu.py",
        "modeling_demo_gen_npu.py.diff",
    ]


def test_atomic_bundle_publishes_remote_siblings(tmp_path: Path) -> None:
    """Remote-code siblings are ordinary members of the atomic bundle."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")
    files = _bundle_files(layout)
    files["configuration_demo.py"] = "VALUE = 1\n"
    write_bundle_atomic(layout, files)
    sibling = Path(layout.artifact_dir) / "configuration_demo.py"
    assert sibling.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_each_publication_replaces_the_previous_bundle(
    tmp_path: Path,
) -> None:
    """A fresh generation never preserves stale files from an older bundle."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")
    write_bundle_atomic(layout, _bundle_files(layout))

    write_bundle_atomic(layout, {"source_helper.py": "HELPER = 1\n"})

    assert not Path(layout.modeling_path).exists()
    assert (Path(layout.artifact_dir) / "source_helper.py").read_text(
        encoding="utf-8"
    ) == "HELPER = 1\n"


@pytest.mark.parametrize(
    "name",
    ["../escape.py", "subdir/escape.py", "subdir\\escape.py", "", ".", ".."],
)
def test_bundle_rejects_paths_that_escape_staging_directory(
    tmp_path: Path,
    name: str,
) -> None:
    """Staged paths escaping the staging directory are rejected."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")

    with pytest.raises(ValueError, match="plain names"):
        write_bundle_atomic(layout, {name: "bad\n"})

    with pytest.raises(ValueError, match="plain names"):
        write_bundle_atomic(
            layout, {str(tmp_path / "absolute_escape.py"): "bad\n"}
        )


def test_complete_bundle_replacement_drops_obsolete_files(tmp_path: Path) -> None:
    """A full publication must contain exactly the newly emitted files."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")
    initial = _bundle_files(layout)
    initial["obsolete_helper.py"] = "OLD = True\n"
    write_bundle_atomic(layout, initial)

    write_bundle_atomic(layout, _bundle_files(layout))

    assert not (Path(layout.artifact_dir) / "obsolete_helper.py").exists()


def test_temp_artifact_cleanup_only_removes_codegen_siblings(tmp_path: Path) -> None:
    """Temp cleanup removes only codegen siblings, nothing else."""
    layout = resolve_default_artifact_layout(str(tmp_path / "generated"), "demo")
    parent = Path(layout.artifact_dir).parent
    (parent / ".codegen_tmp_stale").mkdir(parents=True)
    (parent / ".codegen_old_stale").mkdir()
    keep = parent / "keep"
    keep.mkdir()

    clean_temp_artifacts(layout)

    assert not (parent / ".codegen_tmp_stale").exists()
    assert not (parent / ".codegen_old_stale").exists()
    assert keep.is_dir()


@pytest.mark.parametrize("text", ["R", "S(1)", "S(-1)", "P(sum)", "SS(0,2)"])
def test_placement_strings_round_trip(text: str) -> None:
    assert placement_to_string(parse_placement(text)) == text


def test_negative_dim_parses_as_real_dim_index() -> None:
    """A negative dim in a frozen shard string is a real last-dim index.

    Feature: Frozen placement round-trip.
    Description: The frozen lm_head contract (``out_src tp: S(-1)``) used
        to deserialize as ``Shard(0, uneven_shard=True)``, which sent the
        runtime all-gather along dim 0 instead of the last dim and silently
        produced a wrongly-shaped logits tensor on the codegen arm.
    Expectation: The dim survives the freeze round-trip as a real negative
        index with ``uneven_shard`` left False.
    """
    parsed = parse_placement("S(-1)")
    assert parsed.dim == -1
    assert parsed.uneven_shard is False
    assert placement_to_string(parsed) == "S(-1)"


def test_uneven_shard_placement_refuses_to_freeze() -> None:
    """Uneven shards are FSDP-internal and never belong to a frozen plan.

    Feature: Frozen placement round-trip.
    Description: The old protocol smuggled ``uneven_shard`` through a
        negative-dim encoding, which collided with real negative dim
        indices (see ``test_negative_dim_parses_as_real_dim_index``).
    Expectation: Freezing an uneven shard placement fails fast with
        ``TypeError`` instead of silently corrupting the round-trip.
    """
    from hyper_parallel.core.dtensor.placement_types import Shard  # pylint: disable=import-outside-toplevel

    with pytest.raises(TypeError, match="uneven_shard"):
        placement_to_string(Shard(0, uneven_shard=True))


@pytest.mark.parametrize("text", ["", "Shard(0)", "S(x)", "SS(0)", "P()"])
def test_malformed_placement_strings_fail_fast(text: str) -> None:
    with pytest.raises(ValueError):
        parse_placement(text)
