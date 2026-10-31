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
"""Standard HyperParallel training entry for generated model sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel.models._generated.build_overlay import generated_model_build
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def _entry_model_class(namespace: Mapping[str, Any], architecture: str) -> type:
    model_class = namespace.get(architecture)
    if not isinstance(model_class, type):
        raise RuntimeError(
            f"generated training entry cannot find architecture class {architecture!r}"
        )
    return model_class


def _trainer_type(config: Any) -> type:
    target_path = getattr(config.model, "_target_path", "")
    if "ImageTextToText" in target_path:
        return VLMTrainer
    return TextTrainer


def run_generated_training(
    namespace: Mapping[str, Any],
    architecture: str,
    argv: Sequence[str] | None = None,
) -> None:
    """Run standard training with the model class from a generated source.

    Args:
        namespace: Generated module globals.
        architecture: Entry architecture class emitted into that module.
        argv: Standard training CLI arguments excluding the program name.
    """
    model_class = _entry_model_class(namespace, architecture)
    config = parse_training_args(argv)
    trainer_class = _trainer_type(config)
    with generated_model_build(model_class, architecture):
        trainer = trainer_class(config)
    trainer.train()


__all__ = ["run_generated_training"]
