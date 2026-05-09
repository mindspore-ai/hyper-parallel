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
"""Model spec auto-discovery — resolve ``model.name`` to a Python package.

import_module`` pattern in
```: the entry script never hard-codes any model
import; the spec is loaded on demand based on the YAML config.

Naming Convention
-----------------
``model.name`` is **both** the spec key and the import path:

- **Built-in**: ``model.name: qwen3_5`` → imports
  ``hyper_parallel.models.qwen3_5`` whose ``__init__.py`` must call
  ``register_spec("qwen3_5", ...)`` at import time.
- **External**: any name containing ``.`` is treated as a fully-qualified
  module path (e.g. ``my_org.research_models.my_moe``). The trainer
  imports it as-is so external packages can ship their own specs.

Why convention-over-config
--------------------------
- Reuses ``model.name`` as both the spec key AND the import path. No new
  YAML field, no name → path mapping table to maintain.
- Single source of truth: the package name IS the model name.
- Adding a new built-in model = create ``models/<name>/__init__.py`` that
  calls ``register_spec("<name>", ...)``. Zero changes to entry scripts
  or trainer code.
"""
import importlib
import logging
from pkgutil import iter_modules

logger = logging.getLogger(__name__)

_BUILTIN_NAMESPACE = "hyper_parallel.models"
_EXCLUDE_FROM_LIST = frozenset({"modules", "spec"})  # shared blocks / spec layer, not models

def discover_model_spec(name: str) -> str:
    """Import the package that registers ``ModelSpec`` for ``name``.

    The import side-effect is what calls ``register_spec`` and populates
    the global ``_SPEC_REGISTRY``. The trainer's later ``get_spec(name)``
    then succeeds.

    Args:
        name: Either a built-in model name (e.g. ``"qwen3_5"``) or a
            fully-qualified package path (e.g. ``"my_pkg.my_model"``).
            A name is treated as fully-qualified iff it contains ``.``.

    Returns:
        The fully-qualified module path that was imported.

    Raises:
        ImportError: With the list of available built-in models when the
            name cannot be resolved. The message tells the user how to
            register an external model.
    """
    candidate = name if "." in name else f"{_BUILTIN_NAMESPACE}.{name}"
    try:
        importlib.import_module(candidate)
    except ImportError as exc:
        available = sorted(_list_builtin_models())
        raise ImportError(
            f"Cannot import model package '{candidate}' for "
            f"model.name='{name}'. Built-in models: {available}. "
            f"For an external model, set model.name to a fully-qualified "
            f"package path (e.g. 'my_org.my_model'); the package's "
            f"__init__.py must call register_spec(...)."
        ) from exc
    logger.info("Model spec auto-discovered: model.name=%s -> %s", name, candidate)
    return candidate

def _list_builtin_models() -> list[str]:
    """List subpackages under ``hyper_parallel.models`` that look like models.

    Excludes ``common`` (shared building blocks, not a model). Used only
    for friendly error messages — no caching, called at most once per
    failed lookup.
    """
    try:
        pkg = importlib.import_module(_BUILTIN_NAMESPACE)
    except ImportError:
        return []
    return [
        m.name for m in iter_modules(pkg.__path__)
        if m.ispkg and m.name not in _EXCLUDE_FROM_LIST
    ]
