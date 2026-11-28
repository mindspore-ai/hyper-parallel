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
"""Reuse the prior all-rank JSONL audit format without assuming Qwen shapes or layer counts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from riser_contract import hash_file, positive, require


def read_steps(path: Path, rank: int, steps: int) -> list[dict]:
    """Require exact rank and step coverage, including warmup."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([row["step"] for row in rows] == list(range(1, steps + 1)), f"Incomplete/duplicate steps: {path}")
    require(all(row["rank"] == rank for row in rows), f"Wrong rank: {path}")
    return rows


def _check_microbatch_coverage(row: dict, expected: int, label: str) -> None:
    """Reject a matching but incomplete set of accumulated input records."""
    hashes = row["input_hashes"]
    require(isinstance(hashes, list) and len(hashes) == expected and all(hashes),
            f"Incomplete microbatch input coverage: {label}, expected {expected}")


def _audit_measurements(args: argparse.Namespace, meters: list, work: list) -> list[dict]:
    first, last = args.window
    samples = []
    for index in range(args.steps):
        rank_rows = [rows[index] for rows in meters]
        metrics = rank_rows[0]["metrics"]
        seconds = positive(metrics["performance/step_time"], "step_time")
        tokens = sum(batch["model_tokens"] for row in rank_rows for batch in row["batches"])
        require(tokens == positive(work[index]["model_tokens"], "expected model_tokens"), "Token count mismatch")
        flops = positive(work[index]["model_flops"], "expected model_flops")
        rate = flops / seconds / args.world_size / 1e12
        mfu = rate / args.peak * 100
        for row in rank_rows:
            values = row["metrics"]
            require(values["performance/step_time"] == seconds, "Ranks disagree on global MAX step time")
            require(math.isclose(values["performance/model_tflops_per_device"], rate, rel_tol=1e-6),
                    "Logged model TFLOPS disagrees with independent work")
            require(math.isclose(values["performance/mfu_percent"], mfu, rel_tol=1e-6),
                    "Logged MFU disagrees with independent work")
        if first <= index + 1 <= last:
            samples.append({"step": index + 1, "time_s": seconds, "model_tokens": tokens,
                            "model_flops": flops, "mfu_percent": mfu})
    return samples


def audit(args: argparse.Namespace) -> dict:
    """Check input/scalar parity and independent model work for every rank-step.

    expected-work is produced independently from model geometry and actual sequence
    lengths, never by reversing the measured MFU. This utility deliberately does
    not assert real-weight loading, remote source identity or tensor equivalence.
    """
    for value in (args.rtol, args.atol):
        require(math.isfinite(value) and value >= 0, "Tolerances must be finite and nonnegative")
    require(args.world_size > 0 and args.steps > 0, "Positive rank/step counts required")
    microbatches = getattr(args, "microbatches_per_update", None)
    if microbatches is not None:
        require(type(microbatches) is int and microbatches > 0, "microbatches_per_update must be a positive integer")
    positive(args.peak, "peak")
    first, last = args.window
    require(1 <= first <= last <= args.steps, "Invalid audit window")
    expected = json.loads(args.expected_work.read_text())
    require(expected.get("derivation") and expected.get("source"), "Independent work derivation/source required")
    work = expected["samples"]
    require([row["step"] for row in work] == list(range(1, args.steps + 1)), "Incomplete expected work")
    meters = []
    input_matches = loss_matches = grad_matches = 0
    source_hashes = {str(args.expected_work.resolve()): hash_file(args.expected_work)}
    for rank in range(args.world_size):
        meter_path = args.run_dir / f"meter_steps_rank{rank}.jsonl"
        audit_path = args.run_dir / f"steps_rank{rank}.jsonl"
        meter = read_steps(meter_path, rank, args.steps)
        rows = read_steps(audit_path, rank, args.steps)
        for path in (meter_path, audit_path):
            source_hashes[str(path.resolve())] = hash_file(path)
        reference = None
        if args.reference_dir:
            reference_path = args.reference_dir / f"steps_rank{rank}.jsonl"
            reference = read_steps(reference_path, rank, args.steps)
            source_hashes[str(reference_path.resolve())] = hash_file(reference_path)
        for index, row in enumerate(rows):
            require(math.isfinite(row["loss_local"]) and math.isfinite(row["grad_norm_pre_clip"]),
                    f"Nonfinite loss/gradient: rank={rank} step={index + 1}")
            require(bool(row["input_hashes"]), "Missing actual input hashes")
            if microbatches is not None:
                _check_microbatch_coverage(row, microbatches, f"candidate rank={rank} step={index + 1}")
                require(len(meter[index]["batches"]) == microbatches,
                        f"Incomplete microbatch meter coverage: rank={rank} step={index + 1}")
            if reference is not None:
                before = reference[index]
                if microbatches is not None:
                    _check_microbatch_coverage(before, microbatches, f"reference rank={rank} step={index + 1}")
                input_matches += row["input_hashes"] == before["input_hashes"]
                loss_matches += math.isclose(row["loss_local"], before["loss_local"],
                                            rel_tol=args.rtol, abs_tol=args.atol)
                grad_matches += math.isclose(row["grad_norm_pre_clip"], before["grad_norm_pre_clip"],
                                            rel_tol=args.rtol, abs_tol=args.atol)
        meters.append(meter)
    samples = _audit_measurements(args, meters, work)
    compared = args.steps * args.world_size
    parity = (input_matches == loss_matches == grad_matches == compared) if args.reference_dir else None
    unverified = ["weight loading", "runtime source identity", "module/tensor gradient parity",
                  "fusion engagement", "fallback correctness", "profiling"]
    if microbatches is None:
        unverified.append("microbatch input coverage")
    return {"schema": "mfu-riser-rank-audit/v1", "all_rank_steps": compared,
            "microbatches_per_update": microbatches,
            "all_rank_microbatches": compared * microbatches if microbatches is not None else None,
            "finite_loss_grad": True, "independent_flops": True,
            "input_matches": input_matches if args.reference_dir else None,
            "loss_matches": loss_matches if args.reference_dir else None,
            "grad_norm_matches": grad_matches if args.reference_dir else None,
            "scalar_parity": parity, "rtol": args.rtol, "atol": args.atol,
            "samples": samples, "source_sha256": source_hashes,
            "unverified": unverified}


def main(argv: list[str] | None = None) -> int:
    """Export partial acceptance evidence and canonical samples for a validation adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--expected-work", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--microbatches-per-update", type=int,
                        help="Expected accumulated microbatches per rank and optimizer update from the frozen schedule")
    parser.add_argument("--peak", type=float, required=True)
    parser.add_argument("--window", type=int, nargs=2, required=True)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"schema": "mfu-riser-rank-audit/v1", "error": str(exc)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "error": result.get("error"),
                      "scalar_parity": result.get("scalar_parity")}))
    return 2 if "error" in result or result.get("scalar_parity") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
