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
"""Deterministic synthetic image-text pairs for the VLM training demo.

Fills the asset gap of ``run_vlm.sh``: draws simple colored shapes with PIL
and writes Omni-format conversation records (``messages`` with ``image`` and
``text`` content blocks; media paths relative to the JSONL file, matching
``hyper_parallel.data.omni.AutoProcessorTransform``). Everything is seeded,
so two runs with the same arguments produce byte-identical data — fit for
smoke training, determinism reruns, and equivalence matrices.

Example:

    python -m examples.training_demo.prepare_vlm_synthetic \
        --output-dir output/training_demo/vlm_synthetic --num-samples 64
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_COLORS = {"red": (220, 50, 47), "blue": (38, 139, 210), "green": (62, 161, 44),
           "yellow": (226, 185, 0), "purple": (138, 86, 190)}
_SHAPES = ("circle", "square", "triangle")


def _draw_sample(draw, size: int, rng: random.Random) -> list[str]:
    """Draw 1-3 shapes and return their deterministic descriptions."""
    phrases = []
    for _ in range(rng.randint(1, 3)):
        color_name = rng.choice(sorted(_COLORS))
        shape = rng.choice(_SHAPES)
        margin = size // 6
        x = rng.randint(margin, size - 2 * margin)
        y = rng.randint(margin, size - 2 * margin)
        extent = rng.randint(size // 6, size // 3)
        box = (x, y, x + extent, y + extent)
        fill = _COLORS[color_name]
        if shape == "circle":
            draw.ellipse(box, fill=fill)
        elif shape == "square":
            draw.rectangle(box, fill=fill)
        else:
            draw.polygon([(x + extent // 2, y), (x, y + extent),
                          (x + extent, y + extent)], fill=fill)
        phrases.append(f"a {color_name} {shape}")
    return phrases


def generate(output_dir: Path, num_samples: int, image_size: int,
             seed: int) -> Path:
    """Write ``images/*.png`` plus ``synthetic.jsonl``; returns the JSONL path."""
    from PIL import Image, ImageDraw  # heavy import stays local

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "synthetic.jsonl"
    rng = random.Random(seed)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as out:
        for index in range(num_samples):
            image = Image.new("RGB", (image_size, image_size), "white")
            phrases = _draw_sample(ImageDraw.Draw(image), image_size, rng)
            name = f"images/{index:05d}.png"
            image.save(image_dir / f"{index:05d}.png")
            caption = "The image contains " + ", ".join(phrases) + "."
            record = {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": name},
                    {"type": "text",
                     "text": "Describe the shapes in this image."}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": caption}]},
            ]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic VLM training data.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    path = generate(Path(args.output_dir), args.num_samples, args.image_size,
                    args.seed)
    print(f"[vlm-data] wrote {args.num_samples} samples at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
