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
"""Strict local-only manifest loading for integration and validation runs."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from hyper_parallel.tools.model_integration.schemas import SCHEMA_VERSION


class ManifestError(ValueError):
    """Raised when a model-integration manifest violates its public schema."""


def _mapping(value: Any, path: str, required: bool = True) -> dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError(f"{path} must be a mapping")
    return dict(value)


def _local_path(value: Any, base_dir: Path, field_path: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field_path} must be a non-empty local path or null")
    if value.startswith(("http://", "https://", "hf://")):
        raise ManifestError(
            f"{field_path} must be local; model-integration tools never download implicitly"
        )
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class ValidationManifest:
    """Resolved manifest with path normalization and immutable raw content."""

    path: Path
    raw: dict[str, Any]
    model: dict[str, Any]
    reference: dict[str, Any]
    matrix: dict[str, Any]
    acceptance: dict[str, Any]
    launcher: dict[str, Any]
    output_dir: Path

    @property
    def family(self) -> str:
        """Return the adapter/model family lookup key."""
        family = self.model.get("adapter") or self.model.get("model_type")
        if not isinstance(family, str) or not family:
            raise ManifestError("model.adapter or model.model_type must be a non-empty string")
        return family

    @property
    def model_path(self) -> Optional[Path]:
        """Return the local authoritative model asset path when declared."""
        return _local_path(self.model.get("id_or_path"), self.path.parent, "model.id_or_path")

    @property
    def reference_path(self) -> Optional[Path]:
        """Return the local reference repository path when declared."""
        return _local_path(self.reference.get("source_path"), self.path.parent, "reference.source_path")

    @property
    def integration_handoff_path(self) -> Optional[Path]:
        """Return the prior integration handoff required by precision validation."""
        return _local_path(
            self.raw.get("integration_handoff"),
            self.path.parent,
            "integration_handoff",
        )

    @property
    def launcher_config_path(self) -> Optional[Path]:
        """Return the Trainer recipe path relative to the validation manifest."""
        return _local_path(
            self.launcher.get("config"),
            self.path.parent,
            "launcher.config",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy safe for evidence serialization."""
        payload = copy.deepcopy(self.raw)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def import_callable(self, target: str) -> Callable[..., Any]:
        """Import an explicit dotted callable without mutating ``sys.path``."""
        if not isinstance(target, str) or "." not in target:
            raise ManifestError(f"callable target must be a dotted path, got {target!r}")
        module_name, attribute_name = target.rsplit(".", 1)
        try:
            value = getattr(importlib.import_module(module_name), attribute_name)
        except (ImportError, AttributeError) as exc:
            raise ManifestError(f"could not import callable {target!r}: {exc}") from exc
        if not callable(value):
            raise ManifestError(f"manifest target {target!r} is not callable")
        return value

    def fingerprint_paths(self) -> dict[str, Any]:
        """Hash bounded metadata from declared model and reference sources."""
        fingerprints = {}
        path_fields = {
            "model.id_or_path": self.model.get("id_or_path"),
            "reference.source_path": self.reference.get("source_path"),
        }
        for field_path, value in path_fields.items():
            path = _local_path(value, self.path.parent, field_path)
            if path is None:
                continue
            if not path.exists():
                fingerprints[field_path] = {"path": str(path), "exists": False}
                continue
            if path.is_dir():
                metadata_names = {
                    "config.json",
                    "generation_config.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                }
                metadata_files = [
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file()
                    and (candidate.name in metadata_names or candidate.name.endswith(".index.json"))
                ]
                for relative_name in (
                    "inference/model.py",
                    "inference/engram.py",
                    "inference/vision.py",
                ):
                    candidate = path / relative_name
                    if candidate.is_file():
                        metadata_files.append(candidate)
                metadata_files.sort(key=lambda candidate: str(candidate.relative_to(path)))
                fingerprints[field_path] = {
                    "path": str(path),
                    "exists": True,
                    "kind": "directory",
                    "metadata": {
                        str(candidate.relative_to(path)): self._fingerprint_file(candidate)
                        for candidate in metadata_files
                    },
                }
                continue
            fingerprints[field_path] = {
                "path": str(path),
                "exists": True,
                "kind": "file",
                **self._fingerprint_file(path),
            }
        return fingerprints

    @staticmethod
    def _fingerprint_file(path: Path) -> dict[str, Any]:
        """Return a stable SHA-256 and size for one explicit local file."""
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"sha256": digest.hexdigest(), "size": path.stat().st_size}

    def validate_for(self, command: str) -> None:
        """Validate command-specific fields without silently adding defaults."""
        _ = self.model_path
        _ = self.reference_path
        _ = self.integration_handoff_path
        if command == "parity" and not self.reference:
            raise ManifestError(f"reference is required for {command}")
        if command == "validate":
            if not self.matrix.get("baseline"):
                raise ManifestError("matrix.baseline is required for validate")
            steps = self.matrix.get("steps")
            if not isinstance(steps, int) or steps <= 0:
                raise ManifestError("matrix.steps must be a positive integer")
            trainer_config = self.launcher.get("config")
            trainer_module = self.launcher.get("module")
            if not isinstance(trainer_config, str) or not trainer_config:
                raise ManifestError("launcher.config must be a non-empty path")
            trainer_config_path = self.launcher_config_path
            if trainer_config_path is None or not trainer_config_path.is_file():
                raise ManifestError(
                    f"launcher.config does not exist: {trainer_config_path}"
                )
            if not isinstance(trainer_module, str) or not trainer_module:
                raise ManifestError("launcher.module must be a non-empty dotted module")
            devices = self.matrix.get("devices")
            if not isinstance(devices, int) or devices <= 0:
                raise ManifestError("matrix.devices must be a positive integer")
            shared_initial_checkpoint = self.matrix.get("shared_initial_checkpoint", False)
            if not isinstance(shared_initial_checkpoint, bool):
                raise ManifestError("matrix.shared_initial_checkpoint must be a bool")
            timeout = self.launcher.get("timeout_seconds")
            if timeout is not None and (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or timeout <= 0
            ):
                raise ManifestError("launcher.timeout_seconds must be positive")
            has_resume_case = bool(
                self.matrix.get("same_topology_resume")
                or self.matrix.get("cross_topology_resume")
            )
            split_step = self.matrix.get("resume_split_step")
            if has_resume_case and (
                not isinstance(split_step, int)
                or split_step <= 0
                or split_step >= steps
            ):
                raise ManifestError(
                    "resume cases require 0 < matrix.resume_split_step < matrix.steps"
                )
            handoff = self.integration_handoff_path
            if handoff is not None and not handoff.is_file():
                raise ManifestError(f"integration_handoff does not exist: {handoff}")
            if handoff is not None:
                try:
                    handoff_payload = yaml.safe_load(handoff.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as exc:
                    raise ManifestError(f"could not parse integration_handoff: {exc}") from exc
                if not isinstance(handoff_payload, Mapping):
                    raise ManifestError("integration_handoff must contain a mapping")
                if handoff_payload.get("schema_version") != SCHEMA_VERSION:
                    raise ManifestError(
                        f"integration_handoff schema_version must be {SCHEMA_VERSION}"
                    )
                if handoff_payload.get("status") != "PASS":
                    raise ManifestError("integration_handoff status must be PASS")
                if handoff_payload.get("family") != self.family:
                    raise ManifestError(
                        "integration_handoff family does not match model adapter"
                    )


def load_manifest(path: str | Path, output_dir: str | Path | None = None) -> ValidationManifest:
    """Load and validate a local YAML/JSON integration manifest.

    Args:
        path: Manifest path.
        output_dir: Optional CLI-only evidence directory override.

    Returns:
        Resolved immutable manifest.

    Raises:
        ManifestError: If schema, types, or local-only constraints are invalid.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")
    try:
        if manifest_path.suffix.lower() == ".json":
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"could not parse manifest {manifest_path}: {exc}") from exc
    raw = _mapping(loaded, "$")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    model = _mapping(raw.get("model"), "model")
    if not (model.get("adapter") or model.get("model_type")):
        raise ManifestError("model.adapter or model.model_type is required")
    reference = _mapping(raw.get("reference"), "reference", required=False)
    matrix = _mapping(raw.get("matrix"), "matrix", required=False)
    acceptance = _mapping(raw.get("acceptance"), "acceptance", required=False)
    launcher = _mapping(raw.get("launcher"), "launcher", required=False)
    configured_output = output_dir or raw.get("output_dir") or (
        manifest_path.parent / "output" / "model_integration" / str(model.get("adapter"))
    )
    resolved_output = Path(configured_output).expanduser()
    if not resolved_output.is_absolute():
        resolved_output = (manifest_path.parent / resolved_output).resolve()
    else:
        resolved_output = resolved_output.resolve()
    return ValidationManifest(
        path=manifest_path,
        raw=raw,
        model=model,
        reference=reference,
        matrix=matrix,
        acceptance=acceptance,
        launcher=launcher,
        output_dir=resolved_output,
    )


__all__ = ["ManifestError", "ValidationManifest", "load_manifest"]
