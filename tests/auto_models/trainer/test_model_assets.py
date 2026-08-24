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
"""Tests for the single model-assets ownership path."""

from types import SimpleNamespace

import pytest

from hyper_parallel.auto_models.config.resolver import resolve_root
from hyper_parallel.auto_models.trainer.base import BaseTrainer
from hyper_parallel.auto_models.trainer.config import ModelAssetsConfig, Target, TrainerConfig
from hyper_parallel.auto_models.trainer.text_trainer import TextTrainer


def _value_target(value):
    return value


def _unused_target():
    return None


def _template_target(tokenizer):
    return tokenizer


def _config(*, datasets_type: str, tokenizer=None, chat_template=None) -> TrainerConfig:
    """Build the minimum Trainer config for model-assets tests."""
    return TrainerConfig(
        model=Target(
            _unused_target,
            target_path="tests.unused_model",
            pretrained_model_name_or_path="dummy/model",
        ),
        optimizer=Target(_unused_target, target_path="tests.unused_optimizer"),
        model_assets=ModelAssetsConfig(
            datasets_type=datasets_type,
            tokenizer=tokenizer,
            chat_template=chat_template,
        ),
    )


def _text_trainer(config: TrainerConfig) -> TextTrainer:
    """Build an uninitialized text Trainer with Base runtime state."""
    trainer = TextTrainer.__new__(TextTrainer)
    trainer.base = SimpleNamespace(config=config, model_config=object())
    return trainer


def test_base_trainer_leaves_model_assets_to_concrete_trainer() -> None:
    """Keep modality-specific asset policy out of BaseTrainer."""
    trainer = BaseTrainer.__new__(BaseTrainer)

    with pytest.raises(NotImplementedError, match="Concrete Trainer"):
        trainer._build_model_assets()


def test_text_trainer_builds_plaintext_assets_from_model_assets() -> None:
    """Ignore dataset configuration and use the model-assets tokenizer."""
    tokenizer = object()
    config = _config(
        datasets_type="plaintext",
        tokenizer=Target(
            _value_target,
            target_path="tests.value_target",
            value=tokenizer,
        ),
    )
    config.dataset = SimpleNamespace(data_config={"chat_template": "invalid"})
    trainer = _text_trainer(config)

    trainer._build_model_assets()

    assert trainer.base.tokenizer is tokenizer
    assert trainer.base.chat_template is None
    assert trainer.base.model_assets == [trainer.base.model_config, tokenizer]


def test_conversation_assets_build_template_from_model_assets(monkeypatch) -> None:
    """Build and retain tokenizer plus chat template from one config owner."""
    tokenizer = object()
    chat_template = object()
    trainer = _text_trainer(_config(
        datasets_type="conversation",
        tokenizer=Target(
            _value_target,
            target_path="tests.value_target",
            value=tokenizer,
        ),
        chat_template="tokenizer",
    ))
    calls = []

    def _build_chat_template(name, value):
        calls.append((name, value))
        return chat_template

    monkeypatch.setattr("hyper_parallel.auto_models.trainer.text_trainer.build_chat_template", _build_chat_template)

    trainer._build_model_assets()

    assert calls == [("tokenizer", tokenizer)]
    assert trainer.base.chat_template is chat_template
    assert trainer.base.model_assets == [trainer.base.model_config, tokenizer, chat_template]


def test_conversation_assets_build_template_from_target() -> None:
    """Build a concrete chat-template target with the runtime tokenizer."""
    tokenizer = object()
    chat_template = object()

    trainer = _text_trainer(_config(
        datasets_type="conversation",
        tokenizer=Target(
            _value_target,
            target_path="tests.value_target",
            value=tokenizer,
        ),
        chat_template=Target(
            lambda tokenizer: chat_template,
            target_path="tests.template_target",
        ),
    ))

    trainer._build_model_assets()

    assert trainer.base.chat_template is chat_template
    assert trainer.base.model_assets == [trainer.base.model_config, tokenizer, chat_template]


@pytest.mark.parametrize("chat_template", [None, "Janus"])
def test_model_assets_resolves_optional_template_name(chat_template) -> None:
    """Resolve omitted templates and registry names from the current schema."""
    config = resolve_root({
        "model": {"_target_": f"{__name__}._unused_target"},
        "optimizer": {"_target_": f"{__name__}._unused_target"},
        "model_assets": {
            "datasets_type": "plaintext" if chat_template is None else "conversation",
            "chat_template": chat_template,
        },
    })

    assert config.model_assets.chat_template == chat_template


def test_model_assets_resolves_chat_template_target() -> None:
    """Resolve a concrete template class target from YAML-shaped data."""
    target_path = f"{__name__}._template_target"
    config = resolve_root({
        "model": {"_target_": f"{__name__}._unused_target"},
        "optimizer": {"_target_": f"{__name__}._unused_target"},
        "model_assets": {
            "datasets_type": "conversation",
            "chat_template": {"_target_": target_path},
        },
    })

    assert isinstance(config.model_assets.chat_template, Target)
    assert config.model_assets.chat_template.to_dict() == {"_target_": target_path}


@pytest.mark.parametrize("missing_field", ["tokenizer", "chat_template"])
def test_conversation_assets_require_tokenizer_and_template(missing_field, monkeypatch) -> None:
    """Report incomplete conversation configuration at the asset boundary."""
    tokenizer = None if missing_field == "tokenizer" else object()
    tokenizer_target = None
    if tokenizer is not None:
        tokenizer_target = Target(
            _value_target,
            target_path="tests.value_target",
            value=tokenizer,
        )
    chat_template = None if missing_field == "chat_template" else "tokenizer"
    trainer = _text_trainer(_config(
        datasets_type="conversation",
        tokenizer=tokenizer_target,
        chat_template=chat_template,
    ))
    monkeypatch.setattr(
        "hyper_parallel.auto_models.trainer.text_trainer.build_chat_template",
        lambda name, value: (name, value),
    )

    with pytest.raises(ValueError, match=f"model_assets.{missing_field}"):
        trainer._build_model_assets()
