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
"""Shared, CPU-readable routing evidence contract for validation and profiling."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

# Torch is optional for JSON captures and log-only analysis.
try:
    import torch
except ImportError:
    torch = None


ROUTING_FIELDS = {
    "expert_assignments", "tokens_per_expert", "token_permutation", "input_expert_indices",
    "input_tokens", "top_k", "step", "layer_id",
}


def load_capture(path: str) -> Dict:
    """Load a JSON runtime capture or a tensor checkpoint on CPU."""
    if Path(path).suffix == ".json":
        with open(path, encoding="utf-8") as handle:
            capture = json.load(handle)
    else:
        if torch is None:
            raise RuntimeError("PyTorch is required to read tensor checkpoints; JSON captures need no Torch")
        capture = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(capture, Mapping):
        raise ValueError("Routing capture must be a mapping")
    return capture


def flatten(value: Any) -> List:
    """Flatten a tensor or nested sequence without silently coercing its values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in flatten(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value]
    raise ValueError("Expected a numeric tensor or sequence")


def integers(value: Any) -> List[int]:
    """Read exact nonnegative integer routing indices or counts."""
    numbers = flatten(value)
    if any(not math.isfinite(n) or n < 0 or int(n) != n for n in numbers):
        raise ValueError("Routing indices/counts must be finite nonnegative integers")
    return [int(n) for n in numbers]


def routing_records(capture: Any, source: str = "checkpoint") -> List[Dict]:
    """Collect nested captures, including flattened state-dict sibling fields."""
    records = []
    if isinstance(capture, (list, tuple)):
        for index, item in enumerate(capture):
            records.extend(routing_records(item, f"{source}[{index}]"))
    elif isinstance(capture, Mapping):
        if ROUTING_FIELDS.intersection(capture) - {"step", "layer_id", "input_tokens", "top_k"}:
            records.append({**capture, "source": source})
        grouped = {}
        for key, value in capture.items():
            if isinstance(value, Mapping) or isinstance(value, list) and value and isinstance(value[0], Mapping):
                records.extend(routing_records(value, f"{source}.{key}"))
            if "." in str(key):
                prefix, field = str(key).rsplit(".", 1)
                if field in ROUTING_FIELDS:
                    grouped.setdefault(prefix, {})[field] = value
        for prefix, fields in grouped.items():
            records.extend(routing_records(fields, f"{source}.{prefix}"))
    return records


def check_record(record: Dict) -> Dict:
    """Check expert order, slot conservation and linkage to pre-dispatch choices."""
    required = ("expert_assignments", "tokens_per_expert", "token_permutation", "input_expert_indices")
    missing = [name for name in required if record.get(name) is None]
    if missing:
        return {"status": "UNVERIFIED", "reason": "Missing routing capture fields: " + ", ".join(missing)}
    try:
        assignments, counts, permutation, original = [integers(record[name]) for name in required]
        if not counts or not assignments:
            return {"status": "UNVERIFIED", "reason": "Empty routing capture"}
        expected = [expert for expert, count in enumerate(counts) for _ in range(count)]
        valid = (assignments == expected and sorted(permutation) == list(range(len(original)))
                 and len(assignments) == len(original))
        if valid:
            valid = assignments == [original[index] for index in permutation]
        return {"status": "PASS" if valid else "FAIL", "assignments_checked": len(assignments),
                "expert_major_order": assignments == expected, "permutation_valid": valid}
    except (ValueError, TypeError, OverflowError) as exc:
        return {"status": "FAIL", "error": str(exc)}


def check_routing(capture: Dict) -> Dict:
    """Require every captured record to pass, and never pass an empty set."""
    results = [{"source": record["source"], **check_record(record)} for record in routing_records(capture)]
    statuses = [item["status"] for item in results]
    status = "FAIL" if "FAIL" in statuses else (
        "PASS" if statuses and all(item == "PASS" for item in statuses) else "UNVERIFIED")
    return {"status": status, "details": {"records_seen": len(results),
            "records_checked": sum(item == "PASS" for item in statuses), "records": results},
            "reason": "Runtime captures are required for expert ordering and token permutation"}


def model_state(capture: Dict) -> Dict:
    """Find the explicit model parameter mapping."""
    for name in ("model_state_dict", "state_dict", "model"):
        if isinstance(capture.get(name), Mapping):
            return capture[name]
    return capture


def _gradient_record(name: str, param: Any, grad: Any) -> dict:
    row = {"param_name": name, "has_grad": None, "grad_norm": None, "grad_zero_ratio": None}
    if grad is not None:
        try:
            values = flatten(grad)
            expected = flatten(param)
            shape = getattr(grad, "shape", None)
            shape_ok = shape == getattr(param, "shape", None) if shape is not None else _shape(grad) == _shape(param)
            valid = bool(values) and shape_ok and len(values) == len(expected)
            finite = all(math.isfinite(number) for number in values)
            norm = math.hypot(*values) if finite else None
            row.update(has_grad=valid and finite and math.isfinite(norm) and norm > 0, grad_norm=norm,
                       grad_zero_ratio=sum(number == 0 for number in values) / len(values) if values else None)
        except (ValueError, TypeError, OverflowError):
            row["has_grad"] = False
    return row


def gradient_evidence(capture: Dict) -> Dict:
    """Compare each expert parameter with its own saved gradient, including shape."""
    state = model_state(capture)
    parameters = {str(name): value for name, value in state.items()
                  if "expert" in str(name).lower() and str(name).endswith((".weight", ".bias", "_weight"))}
    gradients = {}
    for key in ("gradients", "grads", "gradient_state_dict", "grad_state_dict"):
        if isinstance(capture.get(key), Mapping):
            gradients.update({str(name).removesuffix(".grad"): value for name, value in capture[key].items()})
    for name, value in state.items():
        if str(name).endswith(".grad"):
            gradients[str(name)[:-5]] = value
    rows = []
    for name, param in parameters.items():
        grad = gradients.get(name)
        row = _gradient_record(name, param, grad)
        rows.append(row)
    observed = any(row["has_grad"] is not None for row in rows)
    status = "UNVERIFIED"
    if rows and observed:
        status = "PASS" if all(row["has_grad"] is True for row in rows) else "FAIL"
    return {"status": status, "parameters": rows, "details": {
        "parameters_checked": len(parameters),
        "experts_checked": len(parameters),
        "experts_with_gradients": sum(row["has_grad"] is True for row in rows),
        "missing_parameters": [row["param_name"] for row in rows if row["has_grad"] is None]},
        "reason": "Explicit saved gradients are required for every expert parameter in this checkpoint shard"}


def _shape(value: Any) -> tuple:
    if not isinstance(value, (list, tuple)):
        return ()
    shapes = {_shape(item) for item in value}
    if len(shapes) > 1:
        raise ValueError("Ragged gradient or parameter array")
    return (len(value),) + (next(iter(shapes)) if shapes else ())
