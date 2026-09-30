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
"""Decode WebDataset parts into records accepted by HP data transforms."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from hyper_parallel.data.constants import ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.online.provider import SOURCE_INFO_KEY


class NvMetaSampleAdapter:
    """Read a JSON/tensor record, selected fields, and in-shard image references.

    Args:
        record_part: JSON/NPZ/PT field dictionary, UTF-8 TXT text, or NPY token IDs.
            Defaults to json. Use None when every field is supplied by field_map.
        field_map: Output field to part name, for example ``{"text": "txt"}``
            or ``{"input_ids": "tokens.npy"}``. Mapped fields override the record.
        image_mode: ``pil`` for Transformers image blocks, or ``bytes`` for
            processors accepting ``{"type": "image", "data": bytes}``.

    Note:
        Message media references use ``part:name``, a part name, or ``sample_key.part``.
        External paths are preserved. Tokenization, label construction, and
        model-specific media processing remain owned by Text/Omni transforms.
    """

    def __init__(
        self,
        *,
        record_part: str | None = "json",
        field_map: Mapping[str, str] | None = None,
        image_mode: Literal["pil", "bytes"] = "pil",
    ) -> None:
        """Validate part selection and the processor's expected image representation."""
        self.record_part = record_part
        self.field_map = dict(field_map or {})
        self.image_mode = image_mode
        if record_part is not None and (not isinstance(record_part, str) or not record_part):
            raise ValueError("record_part must be a non-empty part name or None")
        extension = record_part.rsplit(".", 1)[-1].lower() if record_part is not None else None
        if extension not in (None, "json", "txt", "text", "npy", "npz", "pt", "pth"):
            raise ValueError("record_part must name a JSON, TXT, NPY, NPZ or PT part; use field_map for other fields")
        self._record_field = {"txt": "text", "text": "text", "npy": "input_ids"}.get(extension)
        if any(not isinstance(value, str) or not value for value in (*self.field_map, *self.field_map.values())):
            raise ValueError("field_map keys and values must be non-empty strings")
        if record_part is None and not self.field_map:
            raise ValueError("Configure record_part or field_map")
        if image_mode not in ("pil", "bytes"):
            raise ValueError("image_mode must be 'pil' or 'bytes'")
        # JSON messages may refer to variable media parts. Restrict these only
        # through the explicit reader option; field-only records have a fixed set.
        self.required_parts = None
        if extension != "json":
            self.required_parts = tuple(dict.fromkeys(
                ([record_part] if record_part is not None else []) + list(self.field_map.values())
            ))

    def __call__(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        """Decode one record without tokenizing or materializing a tar shard."""
        parts = {name.lstrip("."): payload for name, payload in raw_sample["parts"].items()}
        decoded: dict[str, Any] = {}

        def read_part(name: str) -> Any:
            """Decode a selected part once, including repeated message references."""
            normalized = name.lstrip(".")
            if normalized not in decoded:
                if normalized not in parts:
                    raise ValueError(
                        f"Sample {raw_sample.get('sample_key')!r} is missing part {name!r}; "
                        "check source.record_part or the custom sample_adapter field_map"
                    )
                decoded[normalized] = self._decode_part(normalized, parts[normalized])
            return decoded[normalized]

        sample = {}
        if self.record_part is not None:
            record = read_part(self.record_part)
            if self._record_field is not None:
                sample[self._record_field] = record
            elif isinstance(record, Mapping):
                sample.update(record)
            else:
                raise ValueError("record_part must decode to a JSON object or tensor field dictionary")
        for field, part in self.field_map.items():
            sample[field] = read_part(part)
        self._resolve_media(sample, parts, read_part, str(raw_sample.get("sample_key", "")))
        if "metadata" in raw_sample:
            sample.setdefault("metadata", raw_sample["metadata"])
        source_info = raw_sample.get(SOURCE_INFO_KEY)
        if source_info is not None:
            sample[SOURCE_INFO_KEY] = source_info
        if source_info is not None and source_info.source_path is not None:
            root = Path(source_info.source_path)
            if root.name == ".nv-meta":
                root = root.parent
            sample[ONLINE_SOURCE_PATH_KEY] = str(root / "record.json")
        return sample

    def _decode_part(self, name: str, payload: bytes) -> Any:
        extension = name.rsplit(".", 1)[-1].lower()
        if extension == "json":
            return json.loads(payload)
        if extension in ("txt", "text"):
            return payload.decode("utf-8")
        if extension == "npy":
            return np.load(io.BytesIO(payload), allow_pickle=False)
        if extension == "npz":
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                return {field: arrays[field] for field in arrays.files}
        if extension in ("pt", "pth"):
            return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if extension in ("jpg", "jpeg", "png", "webp", "bmp") and self.image_mode == "pil":
            # Pillow is optional for text-only ingestion.
            from PIL import Image  # pylint: disable=import-outside-toplevel

            with Image.open(io.BytesIO(payload)) as image:
                return image.convert("RGB")
        return payload

    def _resolve_media(
        self, sample: dict[str, Any], parts: Mapping[str, bytes], read_part: Callable[[str], Any], key: str,
    ) -> None:
        for message in sample.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                part = self._media_part(block, parts, key)
                if part is None:
                    continue
                media_type = block["type"]
                value = read_part(part)
                if media_type in ("image", "image_url"):
                    if self.image_mode == "pil" and isinstance(value, bytes):
                        raise ValueError(f"Image part {part!r} has no supported image extension")
                    if self.image_mode == "bytes" and not isinstance(value, bytes):
                        raise ValueError("image_mode='bytes' requires encoded image parts")
                    block.pop(media_type, None)
                    block["type"] = "image"
                    block["image" if self.image_mode == "pil" else "data"] = value
                else:
                    if isinstance(value, bytes):
                        raise ValueError(
                            f"Encoded {media_type} part {part!r} requires a processor-specific sample_adapter; "
                            "use decoded NPY media or an external file reference with the built-in adapter"
                        )
                    block[media_type] = value

    @staticmethod
    def _media_part(block: Mapping[str, Any], parts: Mapping[str, bytes], key: str) -> str | None:
        """Resolve an in-shard media reference while preserving external paths."""
        media_type = block.get("type")
        if media_type not in ("image", "image_url", "video", "audio"):
            return None
        reference = block.get(media_type)
        if media_type == "image_url" and isinstance(reference, dict):
            reference = reference.get("url")
        if not isinstance(reference, str):
            return None
        if reference.startswith("part:"):
            part = reference[5:].lstrip(".")
            if part not in parts:
                raise ValueError(f"Message references missing part {part!r} in sample {key!r}")
            return part
        part = reference[len(key) + 1:] if key and reference.startswith(key + ".") else reference
        return part if part in parts else None


__all__ = ["NvMetaSampleAdapter"]
