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
"""Emit a codegen artifact bundle from a frozen plan.

The only public entry is :func:`emit_bundle` — the manager calls it as
``emit_bundle(layout, meta, snapshot=...)`` (manager.py). The helpers are the
building blocks that ``emit_bundle`` composes.
"""
from hyper_parallel.codegen.emit.bundle import (
    emit_bundle,
    emit_diff,
    emit_init_file,
    emit_modeling_file,
)

__all__ = [
    "emit_bundle",
    "emit_diff",
    "emit_init_file",
    "emit_modeling_file",
]
