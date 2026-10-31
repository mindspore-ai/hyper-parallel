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
"""Generated model files run through a scoped native Trainer overlay."""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

# The optional-dependency gate must run before importing the training stack.
# pylint: disable=wrong-import-position
from hyper_parallel.codegen.emit.bundle import _make_primary_executable
from hyper_parallel.models._generated import build_overlay
from hyper_parallel.models._generated.entry import run_generated_training
from hyper_parallel.models._transformers import auto_model, model_builder


class FutureArchitecture:
    """Unregistered class used to prove the entry has no family allowlist."""


def test_primary_generated_source_is_directly_executable() -> None:
    source = "from __future__ import annotations\n\nclass FutureArchitecture:\n    pass\n"

    rendered = _make_primary_executable(
        source,
        architecture="FutureArchitecture",
        artifact_dir="/repo/hyper_parallel/models/future_model/generated",
    )

    compile(rendered, "modeling_future_gen_npu.py", "exec")
    assert "__package__ = 'hyper_parallel.models.future_model.generated'" in rendered
    assert "run_generated_training(globals(), 'FutureArchitecture')" in rendered


def test_generated_build_overlay_restores_native_hooks_on_error() -> None:
    apply_module = build_overlay.distributed_apply
    model_class = auto_model._BaseHyperAutoModelClass  # pylint: disable=protected-access
    originals = (
        model_builder._resolve_custom_model_cls,  # pylint: disable=protected-access
        model_builder._maybe_apply_module_replacements,  # pylint: disable=protected-access
        apply_module._apply_phase_c,  # pylint: disable=protected-access
        model_class.__dict__["_from_config_parent_class"],
        model_class.__dict__["_from_pretrained_parent_class"],
    )

    with pytest.raises(RuntimeError, match="probe"):
        with build_overlay.generated_model_build(FutureArchitecture, "FutureArchitecture"):
            assert model_builder._resolve_custom_model_cls("FutureArchitecture") is FutureArchitecture
            raise RuntimeError("probe")

    restored = (
        model_builder._resolve_custom_model_cls,  # pylint: disable=protected-access
        model_builder._maybe_apply_module_replacements,  # pylint: disable=protected-access
        apply_module._apply_phase_c,  # pylint: disable=protected-access
        model_class.__dict__["_from_config_parent_class"],
        model_class.__dict__["_from_pretrained_parent_class"],
    )
    assert restored == originals


def test_generated_entry_uses_standard_config_and_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    config = SimpleNamespace(model=SimpleNamespace(_target_path="HyperAutoModelForCausalLM.from_config"))

    class FakeTrainer:
        def __init__(self, trainer_config: object) -> None:
            """Record Trainer construction."""
            events.append(("init", trainer_config))

        def train(self) -> None:
            """Record the ordinary train call."""
            events.append(("train", None))

    class FakeOverlay:
        def __enter__(self) -> None:
            """Record overlay activation."""
            events.append(("enter", None))

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            """Record overlay restoration."""
            del exc_value, traceback
            events.append(("exit", exc_type))

    monkeypatch.setattr("hyper_parallel.models._generated.entry.parse_training_args", lambda argv: config)
    monkeypatch.setattr("hyper_parallel.models._generated.entry.TextTrainer", FakeTrainer)
    monkeypatch.setattr(
        "hyper_parallel.models._generated.entry.generated_model_build",
        lambda model_class, architecture: FakeOverlay(),
    )

    run_generated_training(
        {"FutureArchitecture": FutureArchitecture},
        "FutureArchitecture",
        ["train.yaml"],
    )

    assert events == [
        ("enter", None),
        ("init", config),
        ("exit", None),
        ("train", None),
    ]
