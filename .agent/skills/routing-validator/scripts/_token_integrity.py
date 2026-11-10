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
"""Validate token conservation using independent counts on the same log record."""

import re
from typing import Any, Dict, List


class TokenIntegrityChecker:
    """Check all observed routing records without borrowing stale metadata."""

    def check(self, log_files: List[str]) -> Dict[str, Any]:
        """Require input_tokens, top_k and counts on each captured log line."""
        records = []
        for filename in log_files:
            try:
                with open(filename, encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if "tokens_per_expert" not in line:
                            continue
                        result = self._check_line(line)
                        records.append({"source": f"{filename}:{number}", **result})
            except OSError as exc:
                records.append({"status": "UNVERIFIED", "source": filename, "reason": str(exc)})
        statuses = [row["status"] for row in records]
        status = "FAIL" if "FAIL" in statuses else (
            "PASS" if statuses and all(item == "PASS" for item in statuses) else "UNVERIFIED")
        return {"status": status, "details": {"records_seen": len(records),
                "records_checked": sum(item == "PASS" for item in statuses), "records": records,
                "conservation_ratio": 1.0 if status == "PASS" else None},
                "reason": "Independent input_tokens and explicit top_k must accompany every token-count record"}

    @staticmethod
    def _check_line(line: str) -> Dict:
        counts_match = re.search(r"tokens_per_expert\s*=\s*(?:tensor\()?\[([^]]*)\]", line)
        input_match = re.search(r"\binput_tokens\s*[=:]\s*(\d+)\b", line)
        topk_match = re.search(r"\b(?:top_k|num_experts_per_tok)\s*[=:]\s*(\d+)\b", line)
        if not counts_match or not input_match or not topk_match:
            return {"status": "UNVERIFIED", "reason": "Incomplete independent token capture"}
        try:
            counts = [int(item.strip()) for item in counts_match.group(1).split(",")]
        except ValueError:
            return {"status": "FAIL", "reason": "Invalid expert token counts"}
        tokens, topk = int(input_match.group(1)), int(topk_match.group(1))
        if not counts or min(counts) < 0 or topk < 1 or tokens < 1:
            return {"status": "FAIL", "reason": "Nonpositive input/top_k or negative expert count"}
        expected = tokens * topk
        return {"status": "PASS" if sum(counts) == expected else "FAIL",
                "expected_assignments": expected, "assigned_tokens": sum(counts)}
