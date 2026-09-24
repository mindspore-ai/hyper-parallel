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
"""Judge complete Python stdio test sets without executing candidates locally."""

import hashlib
import json
import re
from typing import Any, Mapping

from rl.agentic.core.types import RewardResult

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(solution: str) -> str:
    """Extract the final Python fence, or accept an unfenced Python answer."""
    answer = solution.strip()
    matches = _CODE_BLOCK.findall(answer)
    if matches:
        return matches[-1].strip()
    if answer.startswith("<think>"):
        _, separator, answer = answer.partition("</think>")
        if not separator:
            return ""
    return "" if "```" in answer else answer.strip()


def validate_tests(ground_truth: Any) -> tuple[list[str], list[str]]:
    """Reject malformed/private test contracts as data failures before execution."""
    if not isinstance(ground_truth, Mapping):
        raise ValueError("Python stdio ground_truth must be a mapping")
    unsupported = set(ground_truth) - {"inputs", "outputs", "test_version"}
    if unsupported:
        raise ValueError(f"Unsupported Python stdio test fields: {sorted(unsupported)}")
    inputs, outputs = ground_truth.get("inputs"), ground_truth.get("outputs")
    if (not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs
            or len(inputs) != len(outputs) or not all(isinstance(item, str) for item in inputs + outputs)):
        raise ValueError("Python stdio tests must be non-empty equal-length string lists")
    return inputs, outputs


async def judge_stdio(
    solution: str, ground_truth: Any, executor: Any, *, candidate_id: str, runtime_version: str,
) -> RewardResult:
    """Evaluate every declared test; task failures score zero and service failures propagate.

    Output comparison is exact after whitespace tokenization. Floating-point tolerance
    and custom checkers require a different, explicit task contract.
    """
    inputs, outputs = validate_tests(ground_truth)
    canonical = json.dumps({"inputs": inputs, "outputs": outputs}, sort_keys=True, ensure_ascii=False)
    test_version = hashlib.sha256(canonical.encode()).hexdigest()
    code = extract_python(solution)
    passed = 0
    status = "passed"
    elapsed = 0.0
    exit_code = None
    if not code or len(code.encode()) > 65536:
        status = "format_error"
    else:
        for index, (stdin, expected) in enumerate(zip(inputs, outputs)):
            result = await executor.run(code, stdin, request_id=f"{candidate_id}:{index}")
            elapsed += result.duration
            correct = result.status == "success" and result.stdout.split() == expected.split()
            passed += int(correct)
            if not correct and status == "passed":
                status = "wrong_answer" if result.status == "success" else result.status
                exit_code = result.exit_code
    success = float(passed == len(inputs))
    return RewardResult(
        value=success,
        components={"success": success, "passed": float(passed), "total": float(len(inputs))},
        metadata={"status": status, "candidate_id": candidate_id, "duration": elapsed,
                  "exit_code": exit_code, "runtime_version": runtime_version,
                  "test_version": test_version, "judge_version": "python-stdio-whitespace-v1"},
    )
