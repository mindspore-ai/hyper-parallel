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
"""Use the routing validator's shared evidence checks for profiler facts."""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load the sibling skill by file location so both CLIs use one evidence contract.
_EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "routing-validator/scripts/_routing_evidence.py"
_SPEC = importlib.util.spec_from_file_location("moe_routing_evidence", _EVIDENCE_PATH)
_EVIDENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVIDENCE)


class RoutingValidator:
    """Report observed routing, gradients and real activation history."""

    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps

    def validate(self, checkpoint_files: List[str]) -> Dict[str, Any]:
        """Validate all supplied checkpoint shards instead of choosing a filename."""
        result = {"routing": [], "gradients": [], "coverage": [], "warnings": []}
        routing_statuses, gradient_statuses = [], []
        for filename in checkpoint_files:
            try:
                capture = _EVIDENCE.load_capture(filename)
                routing = _EVIDENCE.check_routing(capture)
                gradients = _EVIDENCE.gradient_evidence(capture)
                routing_statuses.append(routing["status"])
                gradient_statuses.append(gradients["status"])
                for record in _EVIDENCE.routing_records(capture):
                    checked = _EVIDENCE.check_record(record)
                    result["routing"].append([record.get("layer_id"), record.get("step"), record.get("input_tokens"),
                                              None, checked.get("assignments_checked"), checked["status"]])
                for row in gradients["parameters"]:
                    result["gradients"].append([None, None, row["param_name"], row["has_grad"],
                                               row["grad_norm"], row["grad_zero_ratio"]])
                history = capture.get("expert_activation_history", [])
                by_layer = {}
                for row in history:
                    counts = _EVIDENCE.integers(row["tokens_per_expert"])
                    by_layer.setdefault(row["layer_id"], {})[row["step"]] = counts
                for layer, steps in by_layer.items():
                    vectors = list(steps.values())
                    if len({len(vector) for vector in vectors}) != 1:
                        raise ValueError("Inconsistent expert activation history")
                    for expert, counts in enumerate(zip(*vectors)):
                        activated = sum(count > 0 for count in counts)
                        result["coverage"].append([layer, expert, activated, activated / len(counts),
                                                   len(counts) - activated])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                result["warnings"].append(f"{filename}: {exc}")
                routing_statuses.append("UNVERIFIED")
                gradient_statuses.append("UNVERIFIED")
        for name, statuses in (("routing_status", routing_statuses), ("gradient_status", gradient_statuses)):
            result[name] = "FAIL" if "FAIL" in statuses else (
                "PASS" if statuses and all(status == "PASS" for status in statuses) else "UNVERIFIED")
        return result
