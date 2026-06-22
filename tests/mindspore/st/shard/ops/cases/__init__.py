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
"""MindSpore shard-ops case declarations.

Each ``case_*.py`` module calls ``register(OpShardCase(...))`` at import
time, so importing this package eagerly populates the framework registry.
"""
from importlib import import_module
from pkgutil import iter_modules

for _mod in iter_modules(__path__):
    if _mod.name.startswith("case_"):
        import_module(f"{__name__}.{_mod.name}")
