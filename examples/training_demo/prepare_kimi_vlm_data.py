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
"""Prepare deterministic content-list image/text data for the Kimi VLM demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

# Patch/merge-friendly default: 112 is a multiple of 14 (patch) and 28 (2x2 merge).
DEFAULT_IMAGE_SIZE = 112


def _make_image(index: int, image_size: int) -> Image.Image:
    """Return one deterministic synthetic RGB image for *index*."""
    image = Image.new("RGB", (image_size, image_size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for step in range(4):
        color = (
            (index * 37 + step * 60) % 256,
            (index * 91 + step * 30) % 256,
            (index * 53 + step * 90) % 256,
        )
        inset = step * image_size // 8
        draw.rectangle(
            (inset, inset, image_size - inset - 1, image_size - inset - 1),
            outline=color,
            width=2,
        )
    draw.line((0, image_size // 2, image_size, image_size // 2), fill=(0, 0, 0), width=2)
    return image


def prepare_kimi_vlm_data(
        output_dir: Path,
        *,
        num_samples: int = 8,
        image_size: int = DEFAULT_IMAGE_SIZE,
) -> None:
    """Write one content-list JSON plus one synthetic image per sample.

    Records follow the multimodal conversation contract consumed by
    ``hyper_parallel.data.vlm``: ``{"messages": [user, assistant], "images":
    [...]}`` where media appear as explicit ``{"type": "image", "url": ...}``
    items so their position inside the turn is preserved. Media paths are
    relative and resolved against the JSON directory by the dataset.

    Args:
        output_dir: Directory receiving ``data.json`` and ``images/``.
        num_samples: Number of deterministic conversations to generate.
        image_size: Square image edge length; keep it a multiple of 28 so the
            vision processor can patch and 2x2-merge it exactly.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if image_size <= 0 or image_size % 28 != 0:
        raise ValueError("image_size must be a positive multiple of 28")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(num_samples):
        relative_image = f"images/sample_{index:04d}.png"
        _make_image(index, image_size).save(output_dir / relative_image)
        records.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": relative_image},
                        {"type": "text", "text": "Describe the following image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": (
                            f"Sample {index} shows four nested colored rectangles "
                            "crossed by a horizontal black line."
                        ),
                    }],
                },
            ],
            "images": [relative_image],
        })

    with (output_dir / "data.json").open("w", encoding="utf-8") as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse data preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare Kimi VLM demo data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the demo conversations and their images."""
    args = _parse_args(argv)
    prepare_kimi_vlm_data(
        Path(args.output_dir).expanduser().resolve(),
        num_samples=args.num_samples,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
