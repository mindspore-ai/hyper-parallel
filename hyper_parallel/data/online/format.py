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
"""Shared fixed-format conversion for messages, ShareGPT, and interleaved media.

Each converter accepts an explicit source contract. They do not infer arbitrary
source schemas or select training supervision. Model transforms own dataset
fields, chat templates, media encoding, labels, and training constraints.
"""

import copy
import os
import re
from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.constants import ONLINE_SOURCE_PATH_KEY


def resolve_media_path(media_path: str, source_directory: str) -> str:
    """Anchor a local media reference while preserving absolute paths and URLs.

    Args:
        media_path: Local path or HTTP, HTTPS, or data URL.
        source_directory: Directory containing the source record file.

    Returns:
        Original remote/absolute reference or normalized local path.

    Note:
        This path helper is shared by all record styles; it does not convert fields.

    Example:
        resolve_media_path("cat.jpg", "/data") -> "/data/cat.jpg"
        resolve_media_path("https://example.com/cat.jpg", "/data")
        -> "https://example.com/cat.jpg"
    """
    if os.path.isabs(media_path) or media_path.startswith(("http://", "https://", "data:")):
        return media_path

    resolved_path = os.path.normpath(os.path.join(source_directory, media_path))
    return resolved_path


def resolve_chat_media_paths(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Copy native messages and anchor media fields to their source file.

    Args:
        sample: Raw messages record with optional source-file metadata.

    Returns:
        Independent messages preserving roles, order, template fields and media.
        Without source metadata, relative message references are left unchanged.

    Raises:
        ValueError: If messages is empty or contains a non-mapping message.

    Note:
        Accepts OpenAI image_url blocks, HF typed media blocks, or native model
        messages. Each block retains its input style; only its path is resolved.
        ShareGPT from/value and Alpaca instruction/output are not parsed here.

    Example:
        HF-style input (source file /data/train.jsonl):
            {"messages": [{"role": "user", "content": [
                {"type": "image", "path": "cat.jpg"}]}]}
        HF-style output:
            [{"role": "user", "content": [
                {"type": "image", "path": "/data/cat.jpg"}]}]
        Field names stay unchanged; a final assistant reply is not required.
    """
    source_messages = sample.get("messages")
    if not isinstance(source_messages, list):
        raise ValueError("Omni sample messages must be a list")

    if not source_messages:
        raise ValueError("Omni sample must contain non-empty messages")

    messages = copy.deepcopy(source_messages)
    source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
    source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else None
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Omni sample messages must contain mapping records")

        content = message.get("content")
        if not isinstance(content, list) or source_directory is None:
            continue
        for block in content:
            _resolve_media_block_paths(block, source_directory)
    return messages


def _resolve_media_block_paths(block: Any, source_directory: str) -> None:
    """Resolve paths in place, preserving OpenAI, HF, or native media fields."""
    if not isinstance(block, dict):
        return

    block_type = block.get("type")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            block["image_url"] = resolve_media_path(image_url, source_directory)
        elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            image_url["url"] = resolve_media_path(image_url["url"], source_directory)
        return

    if block_type not in ("image", "video", "audio"):
        return

    for field in (block_type, "path", "url"):
        value = block.get(field)
        if isinstance(value, str):
            block[field] = resolve_media_path(value, source_directory)
        elif field == "video" and isinstance(value, list):
            block[field] = [
                resolve_media_path(frame, source_directory) if isinstance(frame, str) else frame
                for frame in value
            ]


def normalize_chat_media_fields(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize fixed OpenAI/HF media fields without imposing a training mode.

    Args:
        sample: Record containing OpenAI/HF-style messages and optional source metadata.

    Returns:
        HF-style messages with image_url converted to image/url blocks and string
        image/path references expressed as url. Non-string media objects and
        video frame lists retain their processor-supported representation.

    Raises:
        ValueError: If messages, content blocks, or image_url references are invalid.

    Note:
        Input: OpenAI Chat Completions or HF-style messages. Output: HF-style
        typed media blocks. Shared by DeepSeek and AutoProcessor adapters.
        Pretraining, SFT, and inference constraints belong to the caller.
        This interface does not parse ShareGPT from/value or Alpaca records.

    Example:
        OpenAI Chat Completions-style input:
            {"messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    {"type": "text", "text": "Describe it."}]},
                {"role": "assistant", "content": "A cat."}]}
        HF-style output:
            [{"role": "user", "content": [
                {"type": "image", "url": "https://example.com/cat.jpg"},
                {"type": "text", "text": "Describe it."}]},
             {"role": "assistant", "content": "A cat."}]
        HF local-file input (source file /data/train.jsonl):
            {"type": "image", "path": "cat.jpg"}
        HF-style media block output:
            {"type": "image", "url": "/data/cat.jpg"}
        HF image payloads such as {"type": "image", "image": pil_image} are kept.
        Text-only messages keep their content; no chat template is rendered.
    """
    messages = resolve_chat_media_paths(sample)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Multimodal content blocks must be mapping records")

            block_type = block.get("type")
            if block_type == "image_url":
                image_url = block.pop("image_url", None)
                reference = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(reference, str) or not reference:
                    raise ValueError("image_url content requires a nonempty URL or path")

                block["type"] = "image"
                block["url"] = reference
            elif block_type in ("image", "video", "audio") and "url" not in block:
                reference = block.get(block_type, block.get("path"))
                if isinstance(reference, str):
                    block["url"] = reference
                    block.pop(block_type, None)
                    block.pop("path", None)
    return messages


def _read_sharegpt_media(sample: Mapping[str, Any]) -> dict[str, list[str]]:
    """Read and validate ShareGPT image and video reference columns."""
    media = {"image": sample.get("images", []), "video": sample.get("videos", [])}
    for media_type, references in media.items():
        if not isinstance(references, list):
            raise ValueError(f"ShareGPT {media_type} references must be a list")

        for reference in references:
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"ShareGPT {media_type} references must be nonempty strings")

    return media


def convert_sharegpt_to_messages(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert fixed ShareGPT conversations and media columns to HF-style messages.

    Args:
        sample: conversations with from/value, optional images/videos string lists,
            and optional source-file metadata. Media placeholders consume entries
            from their corresponding lists in conversation order.

    Returns:
        Messages preserving text and media order, with resolved media references.

    Raises:
        ValueError: If roles, values, media lists, or placeholder counts are invalid.

    Example:
        ShareGPT input (source file /data/train.jsonl):
            {"conversations": [{"from": "human", "value": "<image>Describe it."},
                               {"from": "gpt", "value": "A cat."}], "images": ["cat.jpg"]}
        HF-style output:
            [{"role": "user", "content": [{"type": "image", "url": "/data/cat.jpg"},
                                           {"type": "text", "text": "Describe it."}]},
             {"role": "assistant", "content": "A cat."}]
        No turns are discarded and no training target is selected here.
    """
    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("ShareGPT requires nonempty conversations with from/value")

    media = _read_sharegpt_media(sample)
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    consumed = {"image": 0, "video": 0}
    messages = []
    for turn in conversations:
        if not isinstance(turn, dict) or not isinstance(turn.get("from"), str):
            raise ValueError("ShareGPT conversations require mapping turns with a from role")

        role = roles.get(turn["from"])
        if role is None or not isinstance(turn.get("value"), str):
            raise ValueError("ShareGPT requires human/gpt/system roles and string values")

        content = []
        for part in re.split(r"(<image>|<video>)", turn["value"]):
            if part in ("<image>", "<video>"):
                media_type = part[1:-1]
                index = consumed[media_type]
                if index >= len(media[media_type]):
                    raise ValueError(f"ShareGPT has more <{media_type}> placeholders than media references")

                content.append({"type": media_type, "url": media[media_type][index]})
                consumed[media_type] += 1
            elif part:
                content.append({"type": "text", "text": part})
        message_content = content if any(block["type"] != "text" for block in content) else turn["value"]
        messages.append({"role": role, "content": message_content})
    for media_type, references in media.items():
        if consumed[media_type] != len(references):
            raise ValueError(f"ShareGPT has unused {media_type} references without matching placeholders")

    converted_sample = dict(sample)
    converted_sample["messages"] = messages
    normalized_messages = normalize_chat_media_fields(converted_sample)
    return normalized_messages


def convert_interleaved_to_content(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert aligned text/image lists to ordered HF-style content blocks.

    Args:
        sample: Equal-length texts/images lists with exactly one non-None entry
            per position. Images are local paths or URLs. Source metadata is optional.

    Returns:
        Ordered text/image blocks without invented conversation roles or labels.

    Raises:
        ValueError: If lists, aligned entries, or image references are invalid.

    Example:
        Interleaved input (source file /data/train.jsonl):
            {"texts": ["Look: ", None, "A cat."], "images": [None, "cat.jpg", None]}
        HF-style content output:
            [{"type": "text", "text": "Look: "}, {"type": "image", "url": "/data/cat.jpg"},
             {"type": "text", "text": "A cat."}]
        The caller owns pretraining/SFT encoding and supervision.
    """
    texts, images = sample.get("texts"), sample.get("images")
    if not isinstance(texts, list) or not isinstance(images, list):
        raise ValueError("Interleaved texts/images must be lists")

    if not texts or len(texts) != len(images):
        raise ValueError("Interleaved texts/images must be nonempty lists of equal length")

    source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
    source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else None
    content = []
    for text, image in zip(texts, images):
        if (text is None) == (image is None):
            raise ValueError("Interleaved positions require exactly one text or image entry")

        if text is not None:
            if not isinstance(text, str):
                raise ValueError("Interleaved text entries must be strings")

            content.append({"type": "text", "text": text})
        else:
            if not isinstance(image, str) or not image.strip():
                raise ValueError("Interleaved image references must be nonempty strings")

            image_reference = image
            if source_directory is not None:
                image_reference = resolve_media_path(image, source_directory)
            content.append({"type": "image", "url": image_reference})
    return content
