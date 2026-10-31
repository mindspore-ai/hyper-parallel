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
"""Scoped model-build overlay used only by executable generated sources."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from typing import Any, Iterator

from hyper_parallel.models._generated.phase_c import apply_generated_phase_c
from hyper_parallel.models._transformers import auto_model, model_builder

distributed_apply = importlib.import_module("hyper_parallel.distributed.apply")


def _skip_materialized_replacements(
    model: Any,
    distributed_setup: Any,
    low_precision_config: Any,
    load_base_model: bool,
    weights_mapping: Any,
) -> tuple[Any, Any]:
    """Keep generated constructor replacements and their conversion mapping."""
    del distributed_setup, low_precision_config, load_base_model
    return model, weights_mapping


@contextmanager
def generated_model_build(
    model_class: type,
    architecture: str,
) -> Iterator[None]:
    """Use one generated class during an otherwise native model build.

    Args:
        model_class: Entry model class defined by the generated source.
        architecture: Architecture identity expected in the HF configuration.

    Yields:
        Control while the generated model build overlay is active.
    """
    original_resolver = model_builder._resolve_custom_model_cls  # pylint: disable=protected-access
    original_replacements = model_builder._maybe_apply_module_replacements  # pylint: disable=protected-access
    original_phase_c = distributed_apply._apply_phase_c  # pylint: disable=protected-access
    auto_model_class = auto_model._BaseHyperAutoModelClass  # pylint: disable=protected-access
    original_from_config = auto_model_class.__dict__["_from_config_parent_class"]
    original_from_pretrained = auto_model_class.__dict__["_from_pretrained_parent_class"]

    def _resolve_model_class(arch_name: str) -> type | None:
        if arch_name == architecture:
            return model_class
        return original_resolver(arch_name)

    def _from_config(_cls: type, config: Any, *args: Any, **kwargs: Any) -> Any:
        constructor = getattr(model_class, "from_config", None)
        if callable(constructor):
            return constructor(config, *args, **kwargs)
        hf_constructor = getattr(model_class, "_from_config", None)
        if callable(hf_constructor):
            return hf_constructor(config, **kwargs)
        return model_class(config, *args, **kwargs)

    def _from_pretrained(_cls: type, path: str, *args: Any, **kwargs: Any) -> Any:
        return model_class.from_pretrained(path, *args, **kwargs)

    model_builder._resolve_custom_model_cls = _resolve_model_class  # pylint: disable=protected-access
    model_builder._maybe_apply_module_replacements = _skip_materialized_replacements  # pylint: disable=protected-access
    distributed_apply._apply_phase_c = apply_generated_phase_c  # pylint: disable=protected-access
    auto_model_class._from_config_parent_class = classmethod(_from_config)
    auto_model_class._from_pretrained_parent_class = classmethod(_from_pretrained)
    try:
        yield
    finally:
        model_builder._resolve_custom_model_cls = original_resolver  # pylint: disable=protected-access
        model_builder._maybe_apply_module_replacements = original_replacements  # pylint: disable=protected-access
        distributed_apply._apply_phase_c = original_phase_c  # pylint: disable=protected-access
        auto_model_class._from_config_parent_class = original_from_config
        auto_model_class._from_pretrained_parent_class = original_from_pretrained


__all__ = ["generated_model_build"]
