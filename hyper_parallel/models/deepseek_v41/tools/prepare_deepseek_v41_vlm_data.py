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
"""Generate the deterministic VLM samples and processor assets used by HF validation."""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image


def prepare_vlm_data(model_dir: Path, output_dir: Path, *, num_samples: int = 256) -> Path:
    """Write the verified 84x84 image/chat recipe and matching processor settings.

    Args:
        model_dir: Local official composite config and tokenizer directory.
        output_dir: New destination for images, train.jsonl and processor assets.
        num_samples: Number of deterministic records; validation used 256.

    Returns:
        Absolute path to train.jsonl.

    Raises:
        ValueError: The sample count or input assets are invalid.
        FileExistsError: The output directory already exists.
    """
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise ValueError(f"Missing model asset: {model_dir / name}")
    source = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vision = source["vision_config"]
    output_dir.mkdir(parents=True, exist_ok=False)
    assets = output_dir / "processor_assets"
    (assets / "inference").mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(model_dir / name, assets / name)
    inference = {
        "vision_n_layers": 2, "vision_min_pixels": 1764,
        "vision_patch_size": vision["patch_size"], "vision_downsample_ratio": vision["downsample_ratio"],
        "vision_max_n_token": vision["max_image_tokens"], "vision_max_wh_ratio": vision["max_wh_ratio"],
        "image_token_id": source["image_token_id"],
    }
    (assets / "inference/config.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
    records = []
    for number in range(num_samples):
        image = Image.new("RGB", (84, 84))
        image.putdata([((x * 5 + number * 11) % 256, (y * 7 + number * 13) % 256,
                        ((x + y) * 3 + number * 17) % 256) for y in range(84) for x in range(84)])
        path = output_dir / f"image_{number:03d}.png"
        image.save(path)
        records.append({"messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": str(path)}},
                {"type": "text", "text": f"Describe the colors and pattern in image {number}."}]},
            {"role": "assistant", "content": f"The image has a colorful geometric pattern, sample {number}."},
        ]})
    data_path = output_dir / "train.jsonl"
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return data_path


def main() -> None:
    """Generate local validation samples without downloading assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    args = parser.parse_args()
    print(prepare_vlm_data(args.model_dir, args.output_dir, num_samples=args.num_samples))


if __name__ == "__main__":
    main()
