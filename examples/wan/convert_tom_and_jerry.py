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
"""Convert the Tom-and-Jerry video dataset into Wan online-training parquet."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from datasets import Dataset


def _read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _load_manifest(dataset_path: Path) -> Dataset:
    captions = _read_lines(dataset_path / "captions.txt")
    videos = _read_lines(dataset_path / "videos.txt")
    if len(captions) != len(videos):
        raise ValueError(
            f"captions.txt has {len(captions)} rows but videos.txt has {len(videos)} rows"
        )
    return Dataset.from_dict({"prompt": captions, "video": videos})


def convert_tom_and_jerry(
    *,
    dataset_path: Path,
    output_dir: Path,
    num_shards: int,
    num_proc: int,
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    dataset = _load_manifest(dataset_path)
    if len(dataset) == 0:
        raise ValueError("Tom-and-Jerry manifest is empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_size = math.ceil(len(dataset) / num_shards)
    for shard_index, start in enumerate(range(0, len(dataset), shard_size)):
        end = min(start + shard_size, len(dataset))
        shard = dataset.select(range(start, end))
        shard_num_proc = max(1, min(num_proc, len(shard)))

        def process_example(example: dict[str, str]) -> dict[str, object]:
            video_path = dataset_path / example["video"]
            return {
                "prompt": example["prompt"],
                "video_bytes": video_path.read_bytes(),
                "source": "Tom-and-Jerry-VideoGeneration-Dataset",
            }

        converted = shard.map(
            process_example,
            num_proc=shard_num_proc,
            remove_columns=shard.column_names,
            keep_in_memory=True,
            desc=f"Processing shard {shard_index}",
        )
        converted.to_parquet(str(output_dir / f"{shard_index}.parquet"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, default=30)
    parser.add_argument("--num_proc", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_tom_and_jerry(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
