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
"""Public contracts for auditable model integration and validation."""

from hyper_parallel.tools.model_integration.evidence_store import EvidenceStore
from hyper_parallel.tools.model_integration.manifest import (
    ManifestError,
    ValidationManifest,
    load_manifest,
)
from hyper_parallel.tools.model_integration.schemas import (
    ModelIntegrationReport,
    FindingSeverity,
    IntegrationState,
    ModelIntegrationFinding,
    ModelIntegrationValidationError,
)

__all__ = [
    "ModelIntegrationReport",
    "EvidenceStore",
    "FindingSeverity",
    "IntegrationState",
    "ManifestError",
    "ModelIntegrationFinding",
    "ModelIntegrationValidationError",
    "ValidationManifest",
    "load_manifest",
]
