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
"""DTensor public re-exports for the hyper_parallel package."""
from importlib import import_module as _import_module


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import ``CommDebugMode`` when accessed.

    Importing the debug subpackage eagerly pulls in ``dtensor.py`` through
    ``debug._module_tracker``. On the MindSpore platform the runtime patch
    must rebind ``_dtensor_base.DTensorBase`` *before* ``class DTensor`` is
    defined; an eager import here would run the definition first. So defer the
    debug chain to first use.
    """
    if name == "CommDebugMode":
        module = _import_module("hyper_parallel.core.dtensor.debug")
        value = getattr(module, "CommDebugMode")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pylint: disable=invalid-name
    """Include the lazy ``CommDebugMode`` export in ``dir()``."""
    return sorted(set(globals()) | {"CommDebugMode"})
