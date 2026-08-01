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
"""Rank-local reporting and offline merging for precision errors."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from hyper_low_precision_observer.config import DebugOutput
from hyper_low_precision_observer.context import (
    PrecisionContext,
)
from hyper_low_precision_observer.metrics import (
    ErrorMetrics,
)


_logger = logging.getLogger(__name__)


def _rank_artifact_path(raw_root: Path, rank: int) -> Path:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return raw_root / f"rank_{rank:08d}.jsonl"


def _all_rank_artifact_paths(raw_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(raw_root.glob("rank_*.jsonl")))


def _iter_jsonl_payloads(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                _logger.warning(
                    "Ignoring incomplete trailing JSONL record in %s at line %d",
                    path,
                    line_number,
                )
                return
            try:
                payload = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSONL record in {path} at line {line_number}: "
                    f"{error}"
                ) from error
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"JSONL record must contain a mapping: "
                    f"{path}:{line_number}"
                )
            yield payload


class _RankArtifactWriter:
    """Append sampled steps to one file owned by one rank."""

    def __init__(self, raw_root: Path, rank: int) -> None:
        """Create a writer for one rank and reject resumed output directories."""
        self.path = _rank_artifact_path(raw_root, int(rank))
        if self.path.exists():
            raise FileExistsError(
                f"precision debug does not support resume: {self.path} "
                "already exists; use a new output root"
            )

    def append(self, payload: Mapping[str, Any]) -> Path:
        """Append one complete JSONL record and return its path."""
        record = (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(record)
        return self.path


def _rank_info() -> tuple[int, int]:
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not distributed:
        return 0, 1
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


class PrecisionReport:
    """Write derived metrics and mergeable raw moments for one rank."""

    def __init__(
        self,
        output: DebugOutput,
    ) -> None:
        """Initialize rank-local output paths."""
        if not output.root_dir:
            raise ValueError("precision debug output root is required")
        self.root = Path(output.root_dir)
        self.rank, self.world_size = _rank_info()
        self._artifact_writer: _RankArtifactWriter | None = None

    def flush(self, context: PrecisionContext) -> dict[str, Path] | None:
        """Write the active step and clear successfully persisted moments."""
        iteration = int(context.step)
        if not context.active():
            return None
        moments = context.error_moments()
        if not moments:
            return None

        raw = {
            name: metric.raw_dict()
            for name, metric in sorted(moments.items())
        }
        derived = {
            name: metric.to_dict()
            for name, metric in sorted(moments.items())
        }
        payload = {
            "schema_version": 3,
            "iteration": iteration,
            "rank": self.rank,
            "world_size": self.world_size,
            "error_moments": raw,
        }
        paths: dict[str, Path] = {}
        if self._artifact_writer is None:
            self._artifact_writer = _RankArtifactWriter(
                self.root / "raw_moments",
                self.rank,
            )
        artifact_path = self._artifact_writer.append(payload)

        log_path = self.root / "statistics" / f"rank_{self.rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            for name, values in derived.items():
                fields = " ".join(
                    f"{metric}={float(value):.12g}"
                    for metric, value in sorted(values.items())
                )
                stream.write(
                    f"iteration={iteration:06d} {name} {fields}\n"
                )

        paths.update({
            "statistics_log": log_path,
            "rank_artifact": artifact_path,
        })

        context.clear()
        return paths or None


def aggregate_rank_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stream rank artifacts and merge raw moments into global metrics."""
    merged: dict[str, ErrorMetrics] = {}
    iteration = 0
    artifact_count = 0
    ranks = set()
    for artifact in artifacts:
        artifact_count += 1
        iteration = max(iteration, int(artifact.get("iteration", 0)))
        rank = int(artifact.get("rank", 0))
        if rank in ranks:
            raise ValueError(f"Duplicate rank artifact: rank {rank}")
        ranks.add(rank)
        for name, raw in artifact.get("error_moments", {}).items():
            metric = ErrorMetrics.from_raw_dict(raw)
            merged[name] = (
                metric
                if name not in merged
                else merged[name].merge(metric)
            )
    derived = {
        name: metric.to_dict()
        for name, metric in sorted(merged.items())
    }
    aggregation = {
        "artifact_count": artifact_count,
        "rank_count": len(ranks),
    }
    return {
        "iteration": iteration,
        "aggregation": aggregation,
        "raw_error_moments": {
            name: metric.raw_dict()
            for name, metric in sorted(merged.items())
        },
        "metrics": derived,
    }


def _write_global_tensorboard(
    writer: Any,
    result: Mapping[str, Any],
    iteration: int,
) -> None:
    for name, values in result["metrics"].items():
        for metric, value in values.items():
            writer.add_scalar(
                f"global/{name}/{metric}",
                float(value),
                iteration,
            )
    writer.flush()


def _create_summary_writer(log_dir: Path):
    # TensorBoard is optional and imported only when report export requests it.
    tensorboard = import_module("torch.utils.tensorboard")
    return tensorboard.SummaryWriter(log_dir)


def _select_iterations(
    available: Sequence[int],
    iteration: str | int,
    *,
    raw_root: Path,
) -> list[int]:
    if not available:
        raise FileNotFoundError(f"No rank artifacts found under {raw_root}")
    if iteration == "all":
        return list(available)
    if iteration == "latest":
        return [available[-1]]
    try:
        selected = int(iteration)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "iteration must be 'all', 'latest', or an integer step"
        ) from error
    if selected not in available:
        raise FileNotFoundError(
            f"No rank artifacts found for iteration {selected} "
            f"under {raw_root}"
        )
    return [selected]


def _load_artifacts_by_step(
    raw_root: Path,
) -> dict[int, list[Mapping[str, Any]]]:
    """Load all rank artifacts into memory for the initial small-scale version."""
    paths = _all_rank_artifact_paths(raw_root)
    if not paths:
        raise FileNotFoundError(f"No rank artifacts found under {raw_root}")
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for path in paths:
        for payload in _iter_jsonl_payloads(path):
            try:
                step = int(payload["iteration"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Missing or invalid iteration in {path}"
                ) from error
            by_step.setdefault(step, []).append(payload)
    return by_step


def _iter_rank_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    iteration: int,
) -> Iterator[Mapping[str, Any]]:
    seen_ranks: set[int] = set()
    for payload in artifacts:
        try:
            schema_version = int(payload["schema_version"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing or invalid schema_version") from error
        if schema_version != 3:
            raise ValueError(
                f"Unsupported schema_version {schema_version}; expected 3"
            )
        artifact_iteration = int(payload.get("iteration", iteration))
        if artifact_iteration != iteration:
            raise ValueError(
                f"Iteration mismatch: expected {iteration}, "
                f"got {artifact_iteration}"
            )
        rank = int(payload.get("rank", -1))
        if rank < 0:
            raise ValueError("Missing or invalid rank")
        if rank in seen_ranks:
            raise ValueError(
                f"Duplicate rank {rank} in iteration {iteration}"
            )
        seen_ranks.add(rank)
        yield payload


def generate_offline_reports(
    root: str | Path,
    *,
    iteration: str | int = "all",
    tensorboard: bool = True,
) -> dict[int, dict[str, Any]]:
    """Load rank artifacts and generate global JSONL and TensorBoard reports."""
    output_root = Path(root)
    raw_root = output_root / "raw_moments"
    artifacts_by_step = _load_artifacts_by_step(raw_root)
    selected_steps = _select_iterations(
        sorted(artifacts_by_step),
        iteration,
        raw_root=raw_root,
    )

    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "global.jsonl"
    writer = None
    if tensorboard:
        try:
            writer = _create_summary_writer(
                output_root / "tensorboard" / "global"
            )
        except ImportError as error:
            _logger.warning(
                "Global TensorBoard dashboard disabled because the optional "
                "tensorboard package is unavailable: %s",
                error,
            )

    outputs: dict[int, dict[str, Any]] = {}
    try:
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for step in selected_steps:
                artifacts = _iter_rank_artifacts(
                    artifacts_by_step[step],
                    iteration=step,
                )
                result = aggregate_rank_artifacts(artifacts)
                summary = {
                    "schema_version": 3,
                    "iteration": step,
                    "aggregation": result["aggregation"],
                    "global_raw_moments": result["raw_error_moments"],
                    "global_metrics": result["metrics"],
                }
                stream.write(
                    json.dumps(
                        summary,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                if writer is not None:
                    _write_global_tensorboard(writer, result, step)
                outputs[step] = {
                    "global_summary": summary_path,
                    "tensorboard": (
                        output_root / "tensorboard" / "global"
                        if writer is not None
                        else None
                    ),
                    "rank_count": result["aggregation"]["rank_count"],
                    "observation_count": len(result["metrics"]),
                }
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    return outputs


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate global precision statistics and a TensorBoard "
            "dashboard from rank-local raw_moments."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Precision-debug output root",
    )
    parser.add_argument(
        "--iteration",
        default="all",
        help="'all' (default), 'latest', or an integer step",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Generate global statistics JSON only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline reporting command-line interface."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        outputs = generate_offline_reports(
            args.root,
            iteration=args.iteration,
            tensorboard=not args.no_tensorboard,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    for step, summary in outputs.items():
        print(
            f"iteration={step:06d} "
            f"ranks={summary['rank_count']} "
            f"observations={summary['observation_count']} "
            f"summary={summary['global_summary']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
