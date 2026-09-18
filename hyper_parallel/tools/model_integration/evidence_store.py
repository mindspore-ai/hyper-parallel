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
"""Evidence directory writer and deterministic Markdown report generator."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from hyper_parallel.tools.model_integration.schemas import IntegrationState, SCHEMA_VERSION


_STATE_ORDER = (
    IntegrationState.DISCOVERED,
    IntegrationState.STRUCTURE_VALIDATED,
    IntegrationState.MODULE_PARITY_PASSED,
    IntegrationState.MATRIX_PASSED,
)


def find_repository_root() -> Path:
    """Find the checkout root without depending on this module's depth."""
    for candidate in Path(__file__).resolve().parents:
        if candidate.joinpath("pyproject.toml").is_file() and candidate.joinpath(
            "hyper_parallel"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return repr(value)


class EvidenceStore:
    """Write versioned evidence without parsing free-form Trainer logs."""

    def __init__(self, root: str | Path) -> None:
        """Create an evidence writer rooted at one independent run directory."""
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str | Path) -> Path:
        """Resolve an evidence-relative path and reject directory traversal."""
        resolved = (self.root / relative_path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"evidence path escapes run directory: {relative_path}")
        return resolved

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                output_file.write(content)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        """Write sorted, UTF-8 JSON atomically."""
        output_path = self.path(relative_path)
        content = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        ) + "\n"
        self._atomic_write(output_path, content)
        return output_path

    def write_yaml(self, relative_path: str | Path, payload: Any) -> Path:
        """Write a resolved YAML object atomically."""
        output_path = self.path(relative_path)
        content = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        self._atomic_write(output_path, content)
        return output_path

    def append_jsonl(self, relative_path: str | Path, payload: Any) -> Path:
        """Append one structured record to a JSON-lines evidence stream."""
        output_path = self.path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
            output_file.flush()
        return output_path

    def write_csv(
        self,
        relative_path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        fieldnames: Sequence[str],
    ) -> Path:
        """Write a stable-column CSV evidence table."""
        output_path = self.path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        return output_path

    def capture_environment(self) -> dict[str, Any]:
        """Capture imports and revisions that determine reproducibility."""
        git_revision = None
        source_root = find_repository_root()
        try:
            git_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        environment = {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "git_revision": git_revision,
            "source_root": str(source_root),
            "sys_path": list(sys.path),
        }
        try:
            import hyper_parallel  # pylint: disable=import-outside-toplevel

            environment["hyper_parallel"] = {
                "import_path": str(Path(hyper_parallel.__file__).resolve()),
            }
        except ImportError:
            environment["hyper_parallel"] = None
        try:
            import torch  # pylint: disable=import-outside-toplevel,forbidden-backend-import

            environment["torch"] = {
                "version": torch.__version__,
                "import_path": str(Path(torch.__file__).resolve()),
            }
        except ImportError:
            environment["torch"] = None
        try:
            import transformers  # pylint: disable=import-outside-toplevel

            environment["transformers"] = {
                "version": transformers.__version__,
                "import_path": str(Path(transformers.__file__).resolve()),
            }
        except ImportError:
            environment["transformers"] = None
        self.write_json("environment.json", environment)
        return environment

    def update_state(
        self,
        state: IntegrationState,
        evidence: Iterable[str] = (),
        reason: str = "",
    ) -> Path:
        """Advance the evidence-backed workflow state without skipping gates."""
        state_path = self.path("integration_state.json")
        previous = None
        if state_path.exists():
            previous = IntegrationState(json.loads(state_path.read_text(encoding="utf-8"))["state"])
        terminal = (IntegrationState.FAILED, IntegrationState.BLOCKED)
        if previous in terminal and state not in terminal:
            raise ValueError(f"cannot advance terminal integration state {previous.value}")
        if state not in terminal and previous is not None:
            previous_index = _STATE_ORDER.index(previous)
            next_index = _STATE_ORDER.index(state)
            if next_index < previous_index:
                raise ValueError(
                    f"cannot regress integration state {previous.value} -> {state.value}"
                )
            if next_index > previous_index + 1:
                raise ValueError(
                    f"cannot skip integration state {previous.value} -> {state.value}"
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": state.value,
            "previous_state": previous.value if previous else None,
            "evidence": sorted(set(evidence)),
            "reason": reason,
        }
        return self.write_json("integration_state.json", payload)

    def current_state(self) -> IntegrationState | None:
        """Return the persisted workflow state, if this run has one."""
        state_path = self.path("integration_state.json")
        if not state_path.exists():
            return None
        return IntegrationState(json.loads(state_path.read_text(encoding="utf-8"))["state"])

    def advance_if_before(
        self,
        state: IntegrationState,
        evidence: Iterable[str] = (),
        reason: str = "",
    ) -> Path:
        """Advance one adjacent gate, or preserve an already-later valid gate.

        This is the restart-safe counterpart to :meth:`update_state`. It never
        rewrites the persisted state backwards, while ``update_state`` remains
        strict enough to catch accidental regression in ordinary callers.
        """
        current = self.current_state()
        if current in (IntegrationState.FAILED, IntegrationState.BLOCKED):
            raise ValueError(f"cannot advance terminal integration state {current.value}")
        if current is not None and _STATE_ORDER.index(current) >= _STATE_ORDER.index(state):
            return self.path("integration_state.json")
        return self.update_state(state, evidence, reason)

    def render_summary(
        self,
        status: str,
        title: str,
        sections: Sequence[tuple[str, Sequence[str]]],
        relative_path: str = "summary.md",
    ) -> Path:
        """Write a report whose first line is PASS, FAIL, or BLOCKED.

        Args:
            status: Aggregate report status.
            title: Markdown document title.
            sections: Ordered headings and bullet entries.
            relative_path: Evidence-relative report path.

        Returns:
            Path to the atomically written report.
        """
        if status not in ("PASS", "FAIL", "BLOCKED"):
            raise ValueError(f"invalid report status: {status!r}")
        lines = [status, "", f"# {title}"]
        for heading, entries in sections:
            lines.extend(("", f"## {heading}", ""))
            lines.extend(f"- {entry}" for entry in entries)
        return self._atomic_summary(relative_path, "\n".join(lines) + "\n")

    def _atomic_summary(self, relative_path: str, content: str) -> Path:
        output_path = self.path(relative_path)
        self._atomic_write(output_path, content)
        return output_path


__all__ = ["EvidenceStore", "find_repository_root"]
