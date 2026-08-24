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
"""Build the VLM dataset from a LLaVA-style JSON list."""

import json
import os
from typing import Any, Optional

from torch.utils.data import Dataset

from hyper_parallel.auto_models.components.datasets.contracts import SampleTransform
from hyper_parallel.auto_models.components.utils.constants import IGNORE_INDEX


class VLMDataset(Dataset):
    """Load a LLaVA-style JSON list of multimodal conversations.

    Each record is ``{"messages": [...], "images": [...]}`` where ``messages``
    content is either a Qwen3-VL content-list or a string carrying ``<image>`` /
    ``<video>`` placeholders. Image paths are resolved relative to the JSON file.
    """

    def __init__(self, data_path: str, **dataset_options: Any) -> None:
        """Load records from the JSON file and resolve media paths."""
        del dataset_options
        with open(data_path, "r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        root = os.path.dirname(os.path.abspath(data_path))
        for record in self.records:
            self._resolve_record_paths(record, root)

    def _resolve_record_paths(self, record: dict[str, Any], root: str) -> None:
        """Resolve media paths in one record against the JSON directory."""
        for key in ("images", "videos"):
            if key in record:
                record[key] = [self._resolve(root, path) for path in record[key]]
        for message in record.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                for key in ("url", "image", "video"):
                    if key in item and isinstance(item[key], str):
                        item[key] = self._resolve(root, item[key])

    @staticmethod
    def _resolve(root: str, path: str) -> str:
        """Resolve a relative media path against the JSON directory."""
        if isinstance(path, str) and not path.startswith(("http://", "https://", "/", "data:")):
            return os.path.join(root, path)
        return path

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the record at the given index."""
        return self.records[index]


class _TransformDataset(Dataset):
    """Apply one Trainer-built transform after source-specific IO.

    Records whose assistant turn was truncated away (all labels masked) are
    filtered out up front, so each index maps to exactly one valid sample
    without silently replacing or duplicating records.
    """

    def __init__(self, source: Dataset, transform: Optional[SampleTransform]) -> None:
        """Build the index of trainable samples from the source dataset."""
        self.source = source
        self.transform = transform
        self.indices = []
        for index, record in enumerate(source):
            sample = transform(record) if transform is not None else record
            labels = sample.get("labels")
            if labels is None or self._has_trainable(labels):
                self.indices.append(index)
        if not self.indices:
            raise ValueError("VLM dataset contains no samples with trainable labels")

    @staticmethod
    def _has_trainable(labels: Any) -> bool:
        if hasattr(labels, "ne"):
            return bool(labels.ne(IGNORE_INDEX).any())
        return any(value != IGNORE_INDEX for value in labels)

    def __len__(self) -> int:
        """Return the number of trainable samples."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Return the transformed sample for a trainable index."""
        record = self.source[self.indices[index]]
        return self.transform(record) if self.transform is not None else record


def build_vlm_dataset(
        *,
        data_config: dict[str, Any],
        data_path: Optional[str] = None,
        transform: Optional[SampleTransform] = None,
        tokenizer: Any = None,
        mesh_context: Any = None,
        training_config: Any = None,
        **dataset_options: Any,
) -> Any:
    """Build a transform-wrapped VLM dataset from an online source.

    Args:
        data_config: Source options; must contain ``source_type``.
        data_path: Path to the LLaVA-style JSON list.
        transform: VLM sample transform applied to each raw record.
        tokenizer: Tokenizer (accepted for the shared Trainer contract).
        mesh_context: Mesh context (accepted for the shared Trainer contract).
        training_config: Training plan (accepted for the shared Trainer contract).
        **dataset_options: Reserved source-specific options.

    Returns:
        A transform-wrapped map-style dataset.

    Raises:
        ValueError: If ``source_type`` is unsupported or ``data_path`` is missing.
    """
    del tokenizer, mesh_context, training_config, dataset_options
    if data_config.get("source_type") != "online":
        raise ValueError(f"Unsupported VLM source type: {data_config.get('source_type')!r}")
    if data_path is None:
        raise ValueError("online VLM dataset requires data_path")
    return _TransformDataset(VLMDataset(data_path), transform)


__all__ = ["VLMDataset", "build_vlm_dataset"]
