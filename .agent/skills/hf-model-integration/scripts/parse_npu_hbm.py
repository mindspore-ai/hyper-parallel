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
"""Read per-die HBM from npu-smi's labelled table; reject incomplete telemetry."""

import os
import re
import sys


def parse_hbm(text: str, expected_total: int = None) -> list:
    """Return (used, total) MB pairs from the HBM column, never other ratios.

    The supported table identifies each die by a numeric Chip/Device row and
    labels its HBM-Usage(MB) column. Unknown formats fail closed.
    """
    column = None
    values = []
    for line in text.splitlines():
        fields = line.split("|")
        if "Process id" in line:
            break
        headers = [index for index, field in enumerate(fields)
                   if re.search(r"HBM[- ]Usage\s*\(MB\)", field, re.I)]
        if headers:
            if len(headers) != 1:
                raise ValueError("ambiguous HBM column")
            column = headers[0]
            continue
        if len(fields) < 2 or not re.fullmatch(r"\s*\d+\s+\d+\s*", fields[1]):
            continue
        if column is None or column >= len(fields):
            raise ValueError("device row without an HBM header/column")
        # Some releases put AICore utilization before HBM in the same cell.
        match = re.fullmatch(r"\s*(?:\d+(?:\.\d+)?\s+)?(\d+)\s*/\s*(\d+)\s*", fields[column])
        if not match:
            raise ValueError("unreadable HBM in a device row")
        used, total = map(int, match.groups())
        if total <= 0 or used > total:
            raise ValueError("invalid HBM used/total values")
        if expected_total is not None and total != expected_total:
            raise ValueError("observed HBM capacity differs from HBM_TOTAL_MB")
        values.append((used, total))
    if not values:
        raise ValueError("no labelled HBM device rows parsed")
    return values


def main() -> int:
    """Print used MB per die only after validating the entire supported table."""
    try:
        expected = os.environ.get("HBM_TOTAL_MB")
        total = int(expected) if expected else None
        if total is not None and total <= 0:
            raise ValueError("HBM_TOTAL_MB must be positive")
        values = parse_hbm(sys.stdin.read(), total)
    except ValueError as exc:
        print(f"HBM parse failed: {exc}", file=sys.stderr)
        return 2
    for used, _ in values:
        print(used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
