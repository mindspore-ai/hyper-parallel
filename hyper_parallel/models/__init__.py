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
"""Models: public entry of the model-building suite.

This package is the external interface of the former ``auto_models``
package (top-level split, migration plan §5-1): model-family adapters
organized by architecture, plus the build facade and option DTOs.

Stable entry + lazy discovery (adjust doc §7.1): importing this package
must not import Trainer/Data, read YAML, hit the network, build models or
initialize distributed. The shared data contract lives in
``adapter_spec.py``, discovery in ``registry.py`` — both lazy.

The AutoModel facade loads the full torch/Transformers runtime; keep
package initialization lightweight for config-only imports (the converged
recipe path is ``hyper_parallel.models.HyperAutoModelForCausalLM
.from_pretrained``).
"""

import importlib
from importlib import import_module as _import_module  # pylint: disable=invalid-name
from typing import TYPE_CHECKING

from hyper_parallel.models.build_options import (
    CompileConfig,
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
    ModelBuildOptions,
    normalize_build_options,
)

if TYPE_CHECKING:
    from hyper_parallel.models._transformers import HyperAutoModelForCausalLM


_LAZY_FACADE_EXPORTS = {
    "HyperAutoModelForCausalLM": "hyper_parallel.models._transformers",
}

__all__ = [
    "CompileConfig",
    "FSDP2Config",
    "FSDP2MixedPrecisionConfig",
    "HyperAutoModelForCausalLM",
    "ModelAdapterSpec",
    "ModelBuildOptions",
    "get_model_adapter",
    "normalize_build_options",
    "register_model_adapter",
]


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazy attribute access keeps the package import side-effect free."""
    if name in _LAZY_FACADE_EXPORTS:
        module = _import_module(_LAZY_FACADE_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "ModelAdapterSpec":
        return importlib.import_module(".adapter_spec", __name__).ModelAdapterSpec
    if name in ("get_model_adapter", "register_model_adapter"):
        return getattr(importlib.import_module(".registry", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pylint: disable=invalid-name
    """Include lazy facade exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_FACADE_EXPORTS))
