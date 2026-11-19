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
"""Apply inline patch IR to copied modeling source."""

from __future__ import annotations

from hyper_parallel.codegen.astkit.edits import apply_edits
from hyper_parallel.codegen.astkit.index import build_source_index
from hyper_parallel.codegen.inline.declaration_edits import declaration_edits
from hyper_parallel.codegen.inline.forward_edits import forward_extract_edits
from hyper_parallel.codegen.inline.ir import InlinePatchSet


def apply_patch_set(source_text: str, patch_set: InlinePatchSet) -> str:
    """Insert declarations before rewriting the selected forward methods."""
    index = build_source_index(source_text)
    edits = declaration_edits(
        source_text, index.import_end, index.docstring_end, patch_set.imports, patch_set.module_snippets,
    )
    text = apply_edits(source_text, edits) if edits else source_text
    edits = forward_extract_edits(text, patch_set.forward_extracts)
    return apply_edits(text, edits) if edits else text
