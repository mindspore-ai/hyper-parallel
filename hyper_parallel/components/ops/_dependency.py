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
"""Check optional kernel packages and symbols before selecting an adapter."""

from importlib import import_module
from types import ModuleType

from .registry import ImplementationUnavailable
from .spec import Fn


def require_package(name: str) -> ModuleType:
    """Import the optional package, preserving failures in its own imports."""
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            raise ImplementationUnavailable(f"missing optional package {name}") from exc
        raise


def require_symbol(module: object, name: str) -> None:
    """Reject a candidate if the optional operator is not registered."""
    if not callable(getattr(module, name, None)):
        raise ImplementationUnavailable(f"missing operator {name}")


def load_cann(kernel: Fn, *symbols: str) -> Fn:
    """Return a CANN adapter after checking its required operator symbols."""
    package = require_package("cann_ops_transformer")
    for symbol in symbols:
        require_symbol(package, symbol)
    return kernel
