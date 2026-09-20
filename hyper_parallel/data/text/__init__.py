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
"""Text (LLM) dataset builders, transforms, and chat templates.

Moved from ``components/datasets/llm`` in stage 6 (05 §11.3). The public
names are re-exported lazily: ``build_dataset`` pulls in the natively built
indexed-dataset helpers, which are unavailable on a plain CPU checkout, so
importing this package must stay cheap.
"""

import importlib
from typing import Any

from hyper_parallel.data.constants import TEXT_LAZY_EXPORTS

__all__ = sorted(TEXT_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name through its owning submodule on first use."""
    module_path = TEXT_LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the lazily exported public names."""
    return sorted(__all__)
