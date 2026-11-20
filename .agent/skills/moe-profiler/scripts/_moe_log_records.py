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
"""Read observed log contexts and apply explicit rank/step filters."""

import re
from pathlib import Path
from typing import Iterable, Optional


STEP = re.compile(r"\bstep[\s_=:]+(\d+)", re.I)
RANK = re.compile(r"\brank[\s_=:]*(\d+)", re.I)


def log_records(files: list, steps: Optional[int] = None, ranks: Optional[list] = None) -> Iterable:
    """Yield lines with observed context; unknown rank is excluded by a rank filter."""
    records = []
    for filename in files:
        rank_match = RANK.search(Path(filename).name)
        rank = int(rank_match.group(1)) if rank_match else None
        step = None
        with open(filename, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = RANK.search(line)
                if match:
                    rank = int(match.group(1))
                match = STEP.search(line)
                if match:
                    step = int(match.group(1))
                if ranks is None or rank in ranks:
                    records.append((line, step, rank))
    selected = None
    if steps is not None:
        selected = set(sorted({step for _, step, _ in records if step is not None})[:steps])
    for line, step, rank in records:
        if selected is None or step in selected:
            yield line, step, rank
