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
"""Deterministic offline assets for the cropped Kimi-K2.6 dual-arm run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.kimi_k25.image_processing_kimi_k25 import Kimi_K25ImageProcessor
from transformers.models.kimi_k25.processing_kimi_k25 import Kimi_K25Processor
from transformers.models.kimi_k25.video_processing_kimi_k25 import Kimi_K25VideoProcessor

_IMAGE_TOKEN = "<|media_pad|>"
_VIDEO_TOKEN = "<|kimi_k25_video_placeholder|>"
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<|media_pad|>' }}"
    "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{{ '<eos>' }}{% endfor %}"
)

# Canonical hub ids for the media protocol tokens. Keeping these in the offline
# vocab makes ``KimiVLMChatTransform._media_ids`` a no-op remap and leaves the
# stream's pads at ``config.image_token_id`` (163605), matching the native
# protocol the model's ``get_placeholder_mask`` scans for.
_CANONICAL_MEDIA_TOKENS = {
    "<|media_begin|>": 163602,
    "<|media_content|>": 163603,
    "<|media_end|>": 163604,
    _IMAGE_TOKEN: 163605,
    _VIDEO_TOKEN: 163840,
}


def build_kimi_validation_processor(**_: Any) -> Kimi_K25Processor:
    """Build a small offline processor whose token ids fit the cropped model."""
    tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    tokens.extend(f"token_{index}" for index in range(4, 512))
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab.update(_CANONICAL_MEDIA_TOKENS)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=[
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_end|>",
            _IMAGE_TOKEN,
            _VIDEO_TOKEN,
        ],
        chat_template=_CHAT_TEMPLATE,
    )
    image_processor = Kimi_K25ImageProcessor(size={"max_height": 112, "max_width": 112})
    video_processor = Kimi_K25VideoProcessor(size={"max_height": 112, "max_width": 112})
    return Kimi_K25Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=_CHAT_TEMPLATE,
    )


def prepare_config(source: Path, output: Path) -> None:
    """Copy the cropped config, keeping the checkpoint's media token ids.

    Args:
        source: Directory holding the cropped checkpoint config plus the
            vendored ``configuration_kimi_k25.py`` / ``modeling_kimi_k25.py``.
        output: Destination directory for the validation config bundle.
    """
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    # image_token_id/video_token_id already equal the canonical hub ids used by
    # the offline tokenizer above, so no rewrite is needed; only the projection
    # size is normalized to the vision hidden size.
    config["projection_hidden_size"] = config["vision_config"]["hidden_size"]
    # Keep the MoE group geometry self-consistent with the cropped expert count:
    # the native gate's per-group top-2 needs >=2 experts per group.
    from hyper_parallel.models.kimi_k26.adapter.validation.cropped_model import (  # pylint: disable=import-outside-toplevel
        crop_moe_group_geometry,
    )

    text_cfg = config.get("text_config")
    if text_cfg is not None:
        experts = int(text_cfg.get("n_routed_experts", 0) or 0)
        if experts > 0:
            text_cfg["n_group"], text_cfg["topk_group"] = crop_moe_group_geometry(
                experts,
                int(text_cfg.get("n_group", 1) or 1),
                int(text_cfg.get("topk_group", 1) or 1),
                int(text_cfg.get("num_experts_per_tok", 2) or 2),
            )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name in ("configuration_kimi_k25.py", "modeling_kimi_k25.py"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)


def resolve_media_paths(data_path: Path) -> None:
    """Make synthetic media URLs independent of the training working directory.

    Args:
        data_path: Path to the ``dataset.json`` consumed by the validation run.
    """
    records = json.loads(data_path.read_text(encoding="utf-8"))
    for record in records:
        record["images"] = [str((data_path.parent / path).resolve()) for path in record["images"]]
        for message in record["messages"]:
            for item in message["content"]:
                if item.get("type") == "image":
                    item["url"] = str((data_path.parent / item["url"]).resolve())
    data_path.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Prepare a validation-only cropped config directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    prepare_config(args.source.expanduser().resolve(), args.output.expanduser().resolve())
    resolve_media_paths(args.data.expanduser().resolve())


if __name__ == "__main__":
    main()
