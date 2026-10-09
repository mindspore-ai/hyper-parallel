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
"""Render one HotpotQA sample as independently searchable article files."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _candidate_documents(metadata: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """Read only the per-question context; no answer or gold facts are needed."""
    context = metadata.get("context_json")
    if isinstance(context, str):
        context = json.loads(context)
    if isinstance(context, Mapping):
        pairs = zip(context.get("title", []), context.get("sentences", []))
    elif isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
        pairs = context
    else:
        raise ValueError("HotpotQA context_json must contain candidate articles")
    documents = []
    for title, sentences in pairs:
        if not isinstance(title, str) or not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
            raise ValueError("Invalid HotpotQA article title or sentences")
        documents.append((title, [str(item).strip() for item in sentences]))
    if not documents:
        raise ValueError("HotpotQA context contains no candidate articles")
    return documents


def build_search_workspace(prompt: Any, workspace_dir: Path) -> None:
    """Expose only candidate articles, never the answer or supporting-fact labels."""
    articles = workspace_dir / "articles"
    articles.mkdir(parents=True, exist_ok=True)
    documents = _candidate_documents(prompt.metadata)
    manifest = []
    for index, (title, sentences) in enumerate(documents):
        filename = f"{index:03d}.txt"
        lines = [f"FILE: articles/{filename}", f"TITLE: {title}"]
        lines.extend(
            f"[REF: {title}:{sentence_id}] {sentence}"
            for sentence_id, sentence in enumerate(sentences)
        )
        (articles / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({"file": filename, "title": title})
    (workspace_dir / "articles.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
