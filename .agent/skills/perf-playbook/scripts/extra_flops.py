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
"""Estimate the executed-but-not-model work behind `hfu_extra_tflops_per_step`.

    extra_flops.py --spec SPEC.json [--out OUT.json] [--markdown]

The spec lists the work a training step executes that the PaLM model-FLOPs
convention does not count: Muon Newton-Schulz iterations, router fix-up
matmuls, selective recompute, or a share measured directly by profiling.
Every term must declare a `source`; a term without one is refused, because an
extra-work term with no derivation is not evidence. See
`references/optimizer-overhead.md` for the formulas and the spec fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEC_SCHEMA = "perf-playbook/extra-flops/v1"
TFLOP = 1e12


class ExtraFlopsError(Exception):
    """Raised when a spec is malformed, incomplete, or physically impossible."""


def newton_schulz_flops(rows: int, cols: int, steps: int) -> float:
    """Return the matmul FLOPs of `steps` Newton-Schulz iterations.

    Each iteration is three matmuls on the tall-thin form of the matrix
    (implementations transpose so that m <= n): ``A = X Xᵀ`` costs 2·m²·n,
    ``A A`` costs 2·m³, and ``B X`` costs 2·m²·n, with m = min(rows, cols)
    and n = max(rows, cols). Element-wise scaling and the norm are O(mn) and
    are deliberately not counted.

    Args:
        rows: First dimension of the gradient matrix.
        cols: Second dimension of the gradient matrix.
        steps: Number of Newton-Schulz iterations per optimizer step.

    Returns:
        FLOPs for one optimizer step on this matrix.

    Raises:
        ExtraFlopsError: If a dimension or the step count is not positive.
    """
    if min(rows, cols) <= 0 or steps <= 0:
        raise ExtraFlopsError(
            f"newton_schulz needs positive dims and steps, got {rows}x{cols} steps={steps}")
    short, long_side = min(rows, cols), max(rows, cols)
    per_iteration = 4 * short * short * long_side + 2 * short ** 3
    return float(per_iteration * steps)


def _label(term: dict) -> str:
    return str(term.get("name", term.get("kind", "<unnamed>")))


def _require(term: dict, key: str):
    if key not in term:
        raise ExtraFlopsError(f"term {_label(term)!r} of kind {term.get('kind')!r} needs {key!r}")
    return term[key]


def _positive_number(term: dict, key: str) -> float:
    value = _require(term, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ExtraFlopsError(f"term {_label(term)!r}: {key!r} must be a positive number, got {value!r}")
    return float(value)


def _positive_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExtraFlopsError(f"{where} must be a positive integer, got {value!r}")
    return value


def _shape_list(raw, where: str) -> list[tuple[int, int]]:
    if isinstance(raw, dict):
        raw = raw.get("shapes")
    if not isinstance(raw, list) or not raw:
        raise ExtraFlopsError(f"{where}: expected a non-empty list of 2-D shapes")
    shapes = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ExtraFlopsError(f"{where}: {entry!r} is not a 2-D shape")
        rows = _positive_int(entry[0], f"{where}: shape {entry!r} first dim")
        cols = _positive_int(entry[1], f"{where}: shape {entry!r} second dim")
        shapes.append((rows, cols))
    return shapes


def _term_shapes(term: dict, base_dir: Path) -> list[tuple[int, int]]:
    if "shapes" in term:
        return _shape_list(term["shapes"], f"term {_label(term)!r} shapes")
    path = (base_dir / str(_require(term, "shapes_file"))).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExtraFlopsError(f"cannot read shapes_file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExtraFlopsError(f"shapes_file {path} is not valid JSON: {exc}") from exc
    return _shape_list(raw, str(path))


def _flops_newton_schulz(term: dict, base_dir: Path) -> float:
    steps = _positive_int(_require(term, "steps"), f"term {_label(term)!r}: 'steps'")
    return sum(newton_schulz_flops(rows, cols, steps) for rows, cols in _term_shapes(term, base_dir))


def _flops_matmul(term: dict, _base_dir: Path) -> float:
    dims = _require(term, "dims")
    if not isinstance(dims, list) or len(dims) != 3:
        raise ExtraFlopsError(f"term {_label(term)!r}: 'dims' must be [m, k, n], got {dims!r}")
    sizes = [_positive_int(dim, f"term {_label(term)!r}: dim {dim!r}") for dim in dims]
    count = _positive_int(term.get("count", 1), f"term {_label(term)!r}: 'count'")
    return 2.0 * sizes[0] * sizes[1] * sizes[2] * count


def _flops_forward_fraction(term: dict, _base_dir: Path) -> float:
    fraction = _positive_number(term, "fraction")
    if fraction > 1:
        raise ExtraFlopsError(f"term {_label(term)!r}: 'fraction' must be <= 1, got {fraction}")
    return fraction * _positive_number(term, "model_tflops_per_step_per_device") * TFLOP


def _flops_measured(term: dict, _base_dir: Path) -> float:
    return _positive_number(term, "tflops_per_step") * TFLOP


_HANDLERS = {
    "newton_schulz": _flops_newton_schulz,
    "matmul": _flops_matmul,
    "forward_fraction": _flops_forward_fraction,
    "measured": _flops_measured,
}


def _evaluate_term(term, base_dir: Path) -> dict:
    if not isinstance(term, dict):
        raise ExtraFlopsError(f"each term must be an object, got {term!r}")
    handler = _HANDLERS.get(term.get("kind"))
    if handler is None:
        raise ExtraFlopsError(
            f"unknown term kind {term.get('kind')!r}; known kinds: {', '.join(sorted(_HANDLERS))}")
    source = term.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ExtraFlopsError(f"term {_label(term)!r} needs a non-empty 'source': a derivation, not a guess")
    shard = _positive_int(term.get("shard_devices", 1), f"term {_label(term)!r}: 'shard_devices'")
    return {
        "name": _label(term),
        "kind": term["kind"],
        "shard_devices": shard,
        "tflops_per_step": handler(term, base_dir) / shard / TFLOP,
        "source": source.strip(),
    }


def estimate(spec: dict, base_dir: Path) -> dict:
    """Aggregate a spec's terms into one per-device, per-step TFLOPs figure.

    Args:
        spec: Parsed spec object (schema `perf-playbook/extra-flops/v1`).
        base_dir: Directory that relative `shapes_file` paths resolve against.

    Returns:
        A result object with the per-term breakdown, the total, and the
        config line to paste into `training.mfu`.

    Raises:
        ExtraFlopsError: If the spec or any term is malformed.
    """
    if not isinstance(spec, dict):
        raise ExtraFlopsError("spec must be a JSON object")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ExtraFlopsError(f"spec schema must be {SPEC_SCHEMA!r}, got {spec.get('schema')!r}")
    terms = spec.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ExtraFlopsError("spec needs a non-empty 'terms' list")
    details = [_evaluate_term(term, base_dir) for term in terms]
    total = sum(detail["tflops_per_step"] for detail in details)
    return {
        "schema": SPEC_SCHEMA,
        "terms": details,
        "total_tflops_per_step": total,
        "config_line": f"hfu_extra_tflops_per_step: {total:.6g}",
    }


def render_markdown(result: dict) -> str:
    """Render the result as a derivation block for the tuning summary."""
    lines = ["| term | kind | shard | TFLOPs/step | source |", "|---|---|---|---|---|"]
    for detail in result["terms"]:
        lines.append(
            f"| {detail['name']} | {detail['kind']} | {detail['shard_devices']} "
            f"| {detail['tflops_per_step']:.6g} | {detail['source']} |")
    lines.append("")
    lines.append(f"**Extra executed work:** {result['total_tflops_per_step']:.6g} "
                 "TFLOPs per step per device")
    lines.append("")
    lines.append("```yaml")
    lines.append(f"    {result['config_line']}")
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: read a spec, print the total, optionally write artifacts."""
    parser = argparse.ArgumentParser(description="Estimate extra executed work per step.")
    parser.add_argument("--spec", required=True, help=f"spec JSON (schema {SPEC_SCHEMA})")
    parser.add_argument("--out", help="write the result JSON here")
    parser.add_argument("--markdown", action="store_true", help="also print a derivation block")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result = estimate(spec, spec_path.parent)
    except (OSError, json.JSONDecodeError, ExtraFlopsError) as exc:
        print(f"extra_flops: {exc}")
        return 2

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"extra executed work: {result['total_tflops_per_step']:.6g} TFLOPs/step/device "
          f"over {len(result['terms'])} term(s)")
    if args.markdown:
        print()
        print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
