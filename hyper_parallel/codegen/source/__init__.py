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
"""Source resolution for codegen bundles.

Locate the real ``modeling_*.py`` (installed transformers or checkpoint remote
code) that a generated modeling file is produced from.
"""
from hyper_parallel.codegen.source.compat import sanitize_source_compat
from hyper_parallel.codegen.source.resolver import (
    ResolvedSource,
    import_module_file,
    resolve_hf_source,
    resolve_remote_code_file,
    resolve_remote_siblings,
    resolve_source_identity,
)

__all__ = [
    "ResolvedSource",
    "import_module_file",
    "resolve_hf_source",
    "resolve_remote_code_file",
    "resolve_remote_siblings",
    "resolve_source_identity",
    "sanitize_source_compat",
]
