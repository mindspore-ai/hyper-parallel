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

"""Reproduce the frozen GSM8K baseline splits from the full RL-format parquets."""
import argparse
import random
import tempfile
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the full train.parquet and test.parquet")
    parser.add_argument("destination", type=Path, help="Directory for the 6,400/500 split")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Choose a separate directory for the sampled splits")
    args.destination.mkdir(parents=True, exist_ok=True)
    splits = (("train.parquet", 6400), ("test.parquet", 500))
    with tempfile.TemporaryDirectory(dir=args.destination) as tmp:
        for split, count in splits:
            data = pd.read_parquet(args.source / split)
            # Historical sampling restarted the RNG for each split and preserved draw order.
            indices = random.Random(42).sample(range(len(data)), count)
            path = Path(tmp) / split
            data.iloc[indices].to_parquet(path, index=False)
        for split, _ in splits:
            (Path(tmp) / split).replace(args.destination / split)


if __name__ == "__main__":
    main()
