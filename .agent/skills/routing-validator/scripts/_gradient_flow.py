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
"""Validate explicitly captured gradients; weights alone are not evidence."""

from typing import Any, Dict

from _routing_evidence import gradient_evidence, load_capture


class GradientFlowChecker:
    """Check gradients for every expert parameter in a checkpoint shard."""

    def check(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load saved gradient evidence and return PASS, FAIL or UNVERIFIED."""
        try:
            return gradient_evidence(load_capture(checkpoint_path))
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            return {"status": "UNVERIFIED", "reason": str(exc)}
