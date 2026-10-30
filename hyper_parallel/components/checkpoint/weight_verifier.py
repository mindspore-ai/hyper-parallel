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
"""Check live model shards against local HF weights before training updates.

Inverse weight conversions run symbolically; actual model values are read back
in bounded tiles. No full model, expert tensor or Host embedding is gathered.
All ranks must call ``verify_hf_weights``. Run it before the first forward to
avoid confusing router/buffer updates with weight-loading errors.

This is a check of live values under the recorded conversion rules, not a
mathematical oracle for those rules. The reference decoder reuses the shared
Transformers-based quantized reader; it does not recover pre-quantization FP32 values.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from hyper_parallel import DTensor
from hyper_parallel.components.checkpoint.load_groups import (
    build_load_targets, local_target_tensor, checkpoint_logical_shapes, checkpoint_padding_boxes, reject_partial_layout,
    resolve_checkpoint_index, target_layout,
)
from hyper_parallel.components.checkpoint.quantized_hf_storage import build_quantized_hf_reader
from hyper_parallel.components.checkpoint.region_tensor import RegionTensor
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter, WeightRenaming, get_model_conversion_mapping, revert_weight_conversion,
)
from hyper_parallel.core.distributed_checkpoint.utils import create_chunk_list_for_tensor
from hyper_parallel.models.external_state import CheckpointTensorShard, get_model_external_state
from hyper_parallel.components.checkpoint._weight_check_geometry import box_union_volume, local_block_box, tile_box

logger = logging.getLogger(__name__)


class _Reference:
    """Read selected disk tensors without running model-side conversions."""

    def __init__(self, path: str) -> None:
        """Read indexed tensor headers and optional quantization metadata."""
        location = Path(path).expanduser().resolve()
        if not location.exists():
            raise ValueError(f"Weight checking requires a local HF checkpoint: {location}")
        self.index = resolve_checkpoint_index(str(location))
        self.quantized = build_quantized_hf_reader(location)
        self.shapes = {}
        if self.quantized is not None:
            self.shapes = {
                name: tuple(entry.size)
                for name, entry in self.quantized.load_metadata().state_dict_metadata.items()
                if name in self.index.files_by_key
            }
        else:
            by_file = defaultdict(list)
            for name, filename in self.index.files_by_key.items():
                by_file[filename].append(name)
            for filename, names in by_file.items():
                with safe_open(str(filename), framework="pt", device="cpu") as reader:
                    for name in names:
                        self.shapes[name] = tuple(reader.get_slice(name).get_shape())
        if not self.shapes:
            raise ValueError("No logical HF tensors were selected for checking")

    def read(self, name: str, offsets: tuple[int, ...], lengths: tuple[int, ...]) -> torch.Tensor:
        """Decode a bounded source rectangle, keeping ordinary source dtypes."""
        if self.quantized is not None and len(lengths) == 2:
            return self.quantized.read_region(name, offsets, lengths)
        with safe_open(str(self.index.files_by_key[name]), framework="pt", device="cpu") as reader:
            if not lengths:
                return reader.get_tensor(name)
            region = tuple(slice(start, start + count) for start, count in zip(offsets, lengths))
            return reader.get_slice(name)[region]


class _InverseModel:
    """Expose export conversions without mutating the live model's metadata."""

    def __init__(self, model: nn.Module, conversions: Any) -> None:
        """Bind one inverse-conversion stage to a read-only model view."""
        self.model = model
        self.config = model.config
        self._weight_conversions = conversions

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def _model_shards(model: nn.Module) -> dict[str, CheckpointTensorShard]:
    """Describe every materialized local model shard for comparison."""
    logical_shapes = checkpoint_logical_shapes(model)
    external = get_model_external_state(model)
    external_shards = {}
    if external is not None and external.parameters:
        provider = getattr(external, "checkpoint_tensor_shards", None)
        if not callable(provider):
            raise ValueError("External state must expose checkpoint_tensor_shards() for complete weight checking")
        external_shards = provider()
        if set(external_shards) != set(external.parameters):
            raise ValueError("External checkpoint shard descriptors do not cover exactly the external parameters")
    shards = {}
    for name, target in build_load_targets(model).items():
        if name in external_shards:
            shards[name] = external_shards[name]
            continue
        layout = target_layout(target)
        view = target
        if layout is not None:
            reject_partial_layout(name, layout)
            if not isinstance(view, DTensor):
                shape = getattr(layout, "tensor_shape", None)
                view = DTensor.from_local_with_layout(target.detach(), layout, shape=shape)
        chunks = create_chunk_list_for_tensor(view)
        if len(chunks) != 1:
            raise ValueError(f"Weight checking requires one rectangular local shard: {name}")
        local = local_target_tensor(target)
        if local.is_meta or tuple(local.shape) != tuple(chunks[0].sizes):
            raise ValueError(f"Live storage does not match its checkpoint shard metadata: {name}")
        logical_shape = logical_shapes.get(name)
        if logical_shape == tuple(view.shape):
            logical_shape = None
        shards[name] = CheckpointTensorShard(
            local, tuple(view.shape), tuple(chunks[0].offsets), logical_shape,
        )
    if not shards:
        raise ValueError("The model has no checkpoint tensors to check")
    return shards


def _inverse_regions(model: nn.Module, shards: dict[str, CheckpointTensorShard]) -> dict[str, RegionTensor]:
    """Reverse recorded load conversions into symbolic HF tensor regions."""
    regions = {
        name: RegionTensor.leaf(name, shard.logical_shape or shard.global_shape, shard.tensor.dtype)
        for name, shard in shards.items()
    }
    replacements = getattr(model, "_hp_used_replacement_weight_conversions", None)
    if replacements is not None:
        stages = [replacements, getattr(model, "_hp_used_base_weight_conversions", [])]
    else:
        conversions = getattr(model, "_weight_conversions", None)
        if conversions is None:
            conversions = get_model_conversion_mapping(model, add_legacy=False)
        stages = [conversions]
    for conversions in stages:
        converters = [item for item in conversions if isinstance(item, WeightConverter)]
        renamings = [item for item in conversions if isinstance(item, WeightRenaming)]
        if len(converters) + len(renamings) != len(conversions):
            raise ValueError("Unsupported recorded weight-conversion type")
        if converters:
            regions = _reverse_stage(model, regions, converters)
        for renaming in reversed(renamings):
            regions = _reverse_stage(model, regions, [renaming])
    if any(not isinstance(value, RegionTensor) for value in regions.values()):
        raise ValueError("Inverse conversion produced non-symbolic weights; bounded verification is unsupported")
    return regions


def _reverse_stage(model: nn.Module, regions: dict[str, RegionTensor],
                    conversions: list[Any]) -> dict[str, RegionTensor]:
    """Undo one stage, respecting anchored names relative to the model root.

    Loading resolves the base-model prefix after applying its renamings. Match
    full scoped names first, then relative names; symbolic leaves retain the
    original live FQNs so that changing lookup names cannot change the data read.
    """
    prefix = getattr(model, "base_model_prefix", "")
    prefix = prefix + "." if prefix else ""
    probes = [conversion.reverse_transform() for conversion in conversions]
    normalized = {}
    for name, region in regions.items():
        lookup = name
        if (prefix and name.startswith(prefix)
                and not any(probe.rename_source_key(name)[1] is not None for probe in probes)):
            relative = name[len(prefix):]
            if any(probe.rename_source_key(relative)[1] is not None for probe in probes):
                lookup = relative
        if lookup in normalized:
            raise ValueError(f"Ambiguous inverse conversion after resolving base-model prefix: {lookup}")
        normalized[lookup] = region
    return revert_weight_conversion(_InverseModel(model, conversions), normalized)


def _model_box(block: Any, offsets: tuple[int, ...],
               lengths: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map a rectangular recovered HF region back to its live model coordinates."""
    delta = tuple(start - origin for start, origin in zip(offsets, block.offsets))
    base = tuple(
        origin + sum(step[dim] * shift for step, shift in zip(block.steps, delta))
        for dim, origin in enumerate(block.base)
    )
    model_lengths = tuple(
        1 + sum(step[dim] * (size - 1) for step, size in zip(block.steps, lengths))
        for dim in range(len(base))
    )
    return base, model_lengths


def _live_tile(block: Any, shard: CheckpointTensorShard, offsets: tuple[int, ...],
               lengths: tuple[int, ...]) -> torch.Tensor:
    """Read one transformed tile from the live model shard on CPU."""
    base, _ = _model_box(block, offsets, lengths)
    tensor = shard.tensor.detach()
    strides = tensor.stride()
    tile_strides = tuple(sum(step * stride for step, stride in zip(column, strides)) for column in block.steps)
    storage_offset = tensor.storage_offset() + sum(
        (origin - start) * stride for origin, start, stride in zip(base, shard.offsets, strides)
    )
    values = tensor.as_strided(lengths, tile_strides, storage_offset).cpu()
    for operation in block.post:
        values = operation(values)
    return values


def _difference(actual: torch.Tensor, expected: torch.Tensor, storage_dtype: torch.dtype) -> dict[str, Any]:
    """Count raw and load-dtype mismatches for one pair of tensor tiles."""
    if actual.is_complex() or expected.is_complex():
        raise ValueError("Complex weight checking is not supported")
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(f"Compared weight tiles have different shapes: {actual.shape}, {expected.shape}")
    raw_bad = actual != expected
    if actual.is_floating_point() and expected.is_floating_point():
        raw_bad |= (actual == 0) & (expected == 0) & (torch.signbit(actual) != torch.signbit(expected))
    nonfinite = ~(torch.isfinite(actual) & torch.isfinite(expected))
    raw_count = int(torch.count_nonzero(raw_bad))
    nonfinite_count = int(torch.count_nonzero(nonfinite))
    if not raw_count and not nonfinite_count:
        return {
            "elements": actual.numel(), "exact_reference_mismatches": 0, "load_dtype_mismatches": 0,
            "nonfinite_elements": 0, "max_abs_error": 0.0, "max_load_dtype_abs_error": 0.0,
            "first_difference": None,
        }
    adjusted = expected.to(storage_dtype).to(actual.dtype)
    bad = actual != adjusted
    if actual.is_floating_point() and expected.is_floating_point():
        bad |= (actual == 0) & (adjusted == 0) & (torch.signbit(actual) != torch.signbit(adjusted))
    error = (actual.double() - expected.double()).abs()
    adjusted_error = (actual.double() - adjusted.double()).abs()
    first = None
    mask = bad | nonfinite
    load_mismatch = bool(torch.any(mask))
    if not load_mismatch:
        mask = raw_bad
    if bool(torch.any(mask)):
        flat_index = int(torch.argmax(mask.reshape(-1).to(torch.int8)))
        index = []
        remainder = flat_index
        for size in reversed(actual.shape):
            index.append(remainder % size)
            remainder //= size
        first = {
            "kind": "load_dtype_mismatch" if load_mismatch else "dtype_cast",
            "tile_index": list(reversed(index)),
            "actual": _json_number(actual.reshape(-1)[flat_index]),
            "reference": _json_number(expected.reshape(-1)[flat_index]),
            "reference_in_load_dtype": _json_number(adjusted.reshape(-1)[flat_index]),
        }
    return {
        "elements": actual.numel(),
        "exact_reference_mismatches": raw_count,
        "load_dtype_mismatches": int(torch.count_nonzero(bad)),
        "nonfinite_elements": nonfinite_count,
        "max_abs_error": float(error.max()) if error.numel() and bool(torch.isfinite(error).all()) else None,
        "max_load_dtype_abs_error": (
            float(adjusted_error.max())
            if adjusted_error.numel() and bool(torch.isfinite(adjusted_error).all()) else None
        ),
        "first_difference": first,
    }


def _json_number(value: torch.Tensor) -> float | str:
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _accumulate(record: dict[str, Any], tile: dict[str, Any], offsets: tuple[int, ...]) -> None:
    """Merge tile comparison counts and retain the first mismatch."""
    for field in ("elements", "exact_reference_mismatches", "load_dtype_mismatches", "nonfinite_elements"):
        record[field] = record.get(field, 0) + tile[field]
    for field in ("max_abs_error", "max_load_dtype_abs_error"):
        error = tile[field]
        if error is not None:
            record[field] = max(record.get(field, 0.0), error)
    existing = record.get("first_difference")
    candidate = tile["first_difference"]
    if candidate is not None and (existing is None or (
            existing["kind"] == "dtype_cast" and candidate["kind"] == "load_dtype_mismatch")):
        first = dict(tile["first_difference"])
        first["hf_index"] = [start + index for start, index in zip(offsets, first.pop("tile_index"))]
        record["first_difference"] = first


def _padding_check(name: str, shard: CheckpointTensorShard, limit: int) -> dict[str, Any] | None:
    """Check that physical padding beyond the logical tensor is zero-filled."""
    logical = shard.logical_shape
    if logical is None or logical == shard.global_shape:
        return None
    if (len(logical) != len(shard.global_shape)
            or any(left > right for left, right in zip(logical, shard.global_shape))):
        raise ValueError(f"Invalid logical checkpoint shape for {name}")
    record = {"model_key": name, "kind": "padding", "elements": 0}
    for box in checkpoint_padding_boxes(logical, shard.offsets, tuple(shard.tensor.shape)):
        for offsets, sizes in tile_box(*box, limit):
            slices = tuple(slice(left, left + size) for left, size in zip(offsets, sizes))
            values = shard.tensor.detach()[slices].cpu()
            _accumulate(record, _difference(values, torch.zeros_like(values), values.dtype), offsets)
    if record.get("first_difference"):
        record["first_difference"]["local_index"] = record["first_difference"].pop("hf_index")
    return record


def _check_local(model: nn.Module, reference: _Reference, limit: int) -> dict[str, Any]:
    """Compare one rank's live storage and calculate its source coverage."""
    shards = _model_shards(model)
    recovered = _inverse_regions(model, shards)
    source_coverage = defaultdict(list)
    model_coverage = defaultdict(list)
    records = []
    unmatched = []
    prefix = getattr(model, "base_model_prefix", "") + "."
    for number, (recovered_name, region) in enumerate(sorted(recovered.items()), start=1):
        name = recovered_name
        if name not in reference.shapes and name.startswith(prefix):
            name = name[len(prefix):]
        if name not in reference.shapes:
            unmatched.append(recovered_name)
            continue
        if tuple(region.shape) != reference.shapes[name]:
            raise ValueError(f"Recovered HF shape mismatch: {name}: {tuple(region.shape)} vs {reference.shapes[name]}")
        record = {
            "hf_key": name, "shape": list(region.shape), "elements": 0,
            "model_keys": sorted(region.sources()),
        }
        if math.prod(region.shape) >= 1 << 30 or number % 200 == 0:
            logger.info("Checking HF tensor %d/%d: %s, shape=%s", number, len(recovered), name, tuple(region.shape))
        for block in region.blocks:
            shard = shards[block.source]
            box = local_block_box(block, shard.offsets, tuple(shard.tensor.shape))
            if box is None:
                continue
            source_coverage[name].append(box)
            model_coverage[block.source].append(_model_box(block, *box))
            for offsets, lengths in tile_box(*box, limit):
                actual = _live_tile(block, shard, offsets, lengths)
                expected = reference.read(name, offsets, lengths)
                difference = _difference(actual, expected, shard.tensor.dtype)
                if difference["first_difference"]:
                    difference["first_difference"].update(
                        model_key=block.source, storage_dtype=str(shard.tensor.dtype),
                        reference_dtype=str(expected.dtype),
                    )
                _accumulate(record, difference, offsets)
        if record["elements"]:
            records.append(record)
    uncovered = []
    for name, shard in shards.items():
        padding = _padding_check(name, shard, limit)
        padding_elements = 0
        if padding is not None:
            records.append(padding)
            padding_elements = padding["elements"]
        covered = box_union_volume(model_coverage[name]) + padding_elements
        if covered != shard.tensor.numel():
            uncovered.append({"model_key": name, "checked": covered, "local_elements": shard.tensor.numel()})
    return {
        "tensors": records, "source_coverage": dict(source_coverage), "unmatched_recovered_keys": unmatched,
        "uncovered_model_shards": uncovered, "model_tensor_keys": list(shards),
    }


def _gather(value: Any) -> list[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _summarize(reports: list[dict[str, Any]], shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """Combine rank reports into global mismatch and coverage results."""
    coverage = defaultdict(list)
    errors = []
    mismatches = 0
    raw_mismatches = 0
    nonfinite = 0
    for rank, report in enumerate(reports):
        if report.get("error"):
            errors.append({"rank": rank, "error": report["error"]})
        for name, boxes in report.get("source_coverage", {}).items():
            coverage[name].extend(boxes)
        if report.get("uncovered_model_shards") or report.get("unmatched_recovered_keys"):
            errors.append({"rank": rank, "error": "Model weights are not completely covered; see rank report"})
        mismatches += report.get("load_dtype_mismatches", 0)
        nonfinite += report.get("nonfinite_elements", 0)
        raw_mismatches += report.get("exact_reference_mismatches", 0)
    uncovered = {
        name: {"checked": box_union_volume(coverage[name]), "expected": math.prod(shape)}
        for name, shape in shapes.items() if box_union_volume(coverage[name]) != math.prod(shape)
    }
    return {
        "status": "FAIL" if errors or uncovered or mismatches or nonfinite else "PASS",
        "exact_reference_match": raw_mismatches == 0 and not errors and not uncovered and not nonfinite,
        "load_dtype_mismatches_including_replicas": mismatches,
        "exact_reference_mismatches_including_replicas": raw_mismatches,
        "nonfinite_elements_including_replicas": nonfinite,
        "uncovered_hf_tensors": uncovered, "errors": errors,
        "ranks": len(reports), "hf_tensors": len(shapes),
        "hf_elements": sum(math.prod(shape) for shape in shapes.values()),
        "model_tensors": len({name for report in reports for name in report.get("model_tensor_keys", [])}),
    }


def _tensor_table(reports: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-tensor rows from rank reports and global coverage."""
    rows = {}
    for report in reports:
        rank = report["rank"]
        for tensor in report.get("tensors", []):
            key = tensor.get("hf_key", "<padding>:" + tensor.get("model_key", ""))
            row = rows.setdefault(key, {"name": key, "kind": tensor.get("kind", "weight"), "ranks_with_mismatch": []})
            for field in ("elements", "exact_reference_mismatches", "load_dtype_mismatches", "nonfinite_elements"):
                row[field] = row.get(field, 0) + tensor.get(field, 0)
            for field in ("max_abs_error", "max_load_dtype_abs_error"):
                row[field] = max(row.get(field, 0.0), tensor.get(field, 0.0))
            if tensor.get("load_dtype_mismatches", 0) or tensor.get("nonfinite_elements", 0):
                row["ranks_with_mismatch"].append(rank)
            existing, candidate = row.get("first_difference"), tensor.get("first_difference")
            if candidate is not None and (existing is None or (
                    existing["kind"] == "dtype_cast" and candidate["kind"] == "load_dtype_mismatch")):
                row["first_difference"] = dict(candidate, rank=rank)
    for row in rows.values():
        if row["load_dtype_mismatches"] or row["nonfinite_elements"]:
            row["result"] = "MISMATCH"
        elif row["exact_reference_mismatches"]:
            row["result"] = "DTYPE_CAST"
        else:
            row["result"] = "EXACT"
    for name, counts in summary["uncovered_hf_tensors"].items():
        row = rows.setdefault(name, {
            "name": name, "kind": "weight", "elements": 0, "exact_reference_mismatches": 0,
            "load_dtype_mismatches": 0, "nonfinite_elements": 0, "max_abs_error": 0.0,
            "max_load_dtype_abs_error": 0.0, "ranks_with_mismatch": [],
        })
        row["coverage"] = counts
        if row.get("result") != "MISMATCH":
            row["result"] = "INCOMPLETE"
    order = {"MISMATCH": 0, "INCOMPLETE": 1, "DTYPE_CAST": 2, "EXACT": 3}
    return sorted(rows.values(), key=lambda row: (order[row["result"]], row["name"]))


def _print_table(rows: list[dict[str, Any]], summary: dict[str, Any], root: Path) -> None:
    """Log the human-readable summary table on rank zero."""
    lines = [
        f"HF weight check: {summary['status']} | ranks={summary['ranks']} | "
        f"model tensors={summary['model_tensors']} | HF tensors={summary['hf_tensors']} | "
        f"weight elements={summary['hf_elements']:,}",
        f"EXACT={summary['exact_tensors']}  DTYPE_CAST={summary['dtype_cast_tensors']}  "
        f"MISMATCH={summary['mismatched_tensors']}  INCOMPLETE={summary['incomplete_tensors']}  "
        f"padding checks={summary['padding_tensors']}",
        "Result      | HF tensor / padding table                         | Checked elems | Mismatches | Max abs error",
        "------------+---------------------------------------------------+---------------+------------+--------------",
    ]
    for row in rows[:50]:
        error = "nonfinite" if row["nonfinite_elements"] else f"{row['max_abs_error']:.6g}"
        lines.append(f"{row['result']:<11} | {row['name'][-49:]:<49} | {row['elements']:>13,} | "
                     f"{row['load_dtype_mismatches']:>10,} | {error}")
        first = row.get("first_difference")
        if row["result"] == "MISMATCH" and first:
            index = first.get("hf_index", first.get("local_index"))
            lines.append(f"  {row['name']}: rank={first['rank']} index={index} "
                         f"actual={first['actual']} reference={first['reference']} "
                         f"reference_in_load_dtype={first['reference_in_load_dtype']}")
    if len(rows) > 50:
        lines.append(f"Showing 50/{len(rows)} rows (mismatches first); complete table: {root / 'tensors.csv'}")
    lines.append("Checked elements and mismatch counts in table include replicated rank observations.")
    lines.append(f"Reports: {root}")
    logger.info("\n%s", "\n".join(lines))


def _write_table(rows: list[dict[str, Any]], root: Path) -> None:
    """Write per-tensor comparison results as a CSV report."""
    fields = [
        "name", "kind", "result", "elements", "exact_reference_mismatches", "load_dtype_mismatches",
        "nonfinite_elements", "max_abs_error", "max_load_dtype_abs_error", "ranks_with_mismatch", "first_difference",
        "coverage",
    ]
    with (root / "tensors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), (list, dict))
                             else row.get(field) for field in fields})


@torch.no_grad()
def verify_hf_weights(model: nn.Module, pretrained_path: str, output_dir: str, *,
                      chunk_elements: int = 1 << 20, phase: str = "caller_managed_before_optimizer") -> dict[str, Any]:
    """Reverse-convert live model weights and compare every element to local HF storage.

    All ranks must call together, before parameter updates. The automatic builder
    hook runs before the first forward. The checker never writes model tensors.
    Quantized source weights are decoded to BF16 using Transformers; ordinary
    source dtypes are preserved. Both raw equality and exact equality after the
    declared load-dtype cast are reported. No tolerance is used. Padding is
    checked against zero separately. Unsupported inverse conversions, missing
    model/source coverage and nonfinite values fail rather than being skipped.

    Args:
        model: Finalized, materialized and weight-loaded model.
        pretrained_path: Local checkpoint; its index defines the comparison scope.
        output_dir: Directory for per-rank reports and summary.json.
        chunk_elements: Maximum number of elements per comparison tile.
        phase: Description of the invocation point, recorded in the reports.

    Returns:
        Global summary on every rank. PASS requires exact load-dtype agreement.

    Raises:
        RuntimeError: Any rank finds mismatches, unsupported conversions or missing coverage.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    root = Path(output_dir).expanduser().resolve()
    local, shapes = _check_rank_weights(model, pretrained_path, root, rank, chunk_elements, phase)
    reports = _gather(local)
    # All ranks use rank zero's reference scope, including when a local reader failed.
    scopes = _gather(shapes if rank == 0 else None)
    summary = _summarize(reports, scopes[0] or shapes)
    summary.update(
        phase=phase, reference="HF decoded weights; load-dtype equality, no tolerance",
        reference_scope="The selected HF index; unused modalities or cropped-out layers are not included",
    )
    rows = _tensor_table(reports, summary)
    _add_result_counts(rows, summary)
    write_error = _write_summary_reports(rank, root, rows, summary)
    write_errors = _gather(write_error)
    _raise_check_failure(summary, write_errors, root)
    logger.info("HF weight check PASS on %d ranks; exact reference match=%s; reports=%s",
                summary["ranks"], summary["exact_reference_match"], root)
    return summary


def _check_rank_weights(model: nn.Module, pretrained_path: str, root: Path, rank: int,
                        chunk_elements: int, phase: str) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    """Run one rank's bounded check and capture errors for the later collective."""
    shapes = {}
    local = {"rank": rank, "phase": phase, "checkpoint": str(pretrained_path)}
    try:
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        root.mkdir(parents=True, exist_ok=True)
        reference = _Reference(pretrained_path)
        shapes = reference.shapes
        if rank == 0:
            local["source_files"] = sorted({str(path.resolve()) for path in reference.index.files_by_key.values()})
        logger.info("Starting HF weight check on rank %d: %d source tensors; phase=%s", rank, len(shapes), phase)
        local.update(_check_local(model, reference, chunk_elements))
        for field in ("exact_reference_mismatches", "load_dtype_mismatches", "nonfinite_elements"):
            local[field] = sum(record.get(field, 0) for record in local["tensors"])
        (root / f"rank_{rank:05d}.json").write_text(json.dumps(local, indent=2, allow_nan=False), encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # A local failure must reach the collective so other ranks cannot continue training.
        local["error"] = f"{type(exc).__name__}: {exc}"
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / f"rank_{rank:05d}.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Cannot write HF weight-check rank report")
    return local, shapes


def _add_result_counts(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Add tensor-result totals to the global verification summary."""
    weight_rows = [row for row in rows if row["kind"] == "weight"]
    summary.update(
        exact_tensors=sum(row["result"] == "EXACT" for row in weight_rows),
        dtype_cast_tensors=sum(row["result"] == "DTYPE_CAST" for row in weight_rows),
        mismatched_tensors=sum(row["result"] == "MISMATCH" for row in weight_rows),
        incomplete_tensors=len(summary["uncovered_hf_tensors"]),
        padding_tensors=len(rows) - len(weight_rows),
    )


def _write_summary_reports(rank: int, root: Path, rows: list[dict[str, Any]],
                           summary: dict[str, Any]) -> str | None:
    """Write rank-zero aggregate reports and return any filesystem error."""
    if rank != 0:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        _write_table(rows, root)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _print_table(rows, summary, root)
    except OSError as exc:
        return str(exc)
    return None


def _raise_check_failure(summary: dict[str, Any], write_errors: list[Any], root: Path) -> None:
    """Fail collectively when any weight or report check did not pass."""
    if summary["status"] != "PASS" or any(write_errors):
        raise RuntimeError(
            f"HF weight check failed: value mismatches={summary['load_dtype_mismatches_including_replicas']}, "
            f"nonfinite={summary['nonfinite_elements_including_replicas']}, "
            f"uncovered HF tensors={len(summary['uncovered_hf_tensors'])}; "
            f"rank errors={summary['errors'][:3]}, report write errors={write_errors}; inspect {root}"
        )
