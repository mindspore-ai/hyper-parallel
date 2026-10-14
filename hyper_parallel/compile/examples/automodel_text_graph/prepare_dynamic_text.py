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
"""Prepare deterministic plaintext records for dynamic token-budget batching."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write locally generated text of several lengths; no download is required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/automodel_text_graph/dynamic_text.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lengths = (3, 7, 11, 17, 23, 31, 43, 59, 73, 89, 101, 113)
    records = [
        {"text": "Training example " + str(index) + ": " + " ".join(
            ["language", "models", "learn", "from", "text"][token % 5]
            for token in range(lengths[index % len(lengths)])
        )}
        for index in range(384)
    ]
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"Wrote {len(records)} variable-length records to {output}")


if __name__ == "__main__":
    main()
