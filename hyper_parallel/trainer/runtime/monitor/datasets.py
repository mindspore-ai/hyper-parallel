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
"""Bounded CPU batch statistics and packing-metadata diagnostics."""

from collections import Counter
from typing import Any

import torch

from hyper_parallel.data.constants import IGNORE_INDEX


class DatasetStatsRecorder:
    """Accumulate every microbatch of one sampled optimizer step."""

    def __init__(self, config: Any) -> None:
        """Initialize bounded CPU counters for the configured dataset diagnostics."""
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear owned CPU counts, never retaining input batches."""
        self.lengths, self.supervised, self.segments = Counter(), Counter(), Counter()
        self.tokens = Counter()
        self.totals = Counter()
        self.failures = []
        self.unavailable = set()
        self.top_complete = True

    def _failure(self, name: str) -> None:
        self.totals["packing_failures"] += 1
        if len(self.failures) < 16:
            self.failures.append(name)

    def record(self, batch: dict) -> None:
        """Consume CPU input geometry, leaving device tensors untouched."""
        ids = batch.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.device.type != "cpu":
            self.unavailable.add("cpu_input_ids")
            return
        if ids.ndim == 0:
            self.unavailable.add("input_shape")
            return
        ids = ids.unsqueeze(0) if ids.ndim == 1 else ids.flatten(0, -2)
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.device.type == "cpu" and mask.shape == ids.shape:
            valid = mask.bool()
        else:
            valid = torch.ones_like(ids, dtype=torch.bool)
            self.unavailable.add("padding_mask")
        self.totals["rows"] += ids.shape[0]
        self.totals["slots"] += ids.numel()
        self.totals["input_tokens"] += int(valid.sum())
        self.lengths.update(valid.sum(-1).tolist())
        labels = batch.get("shift_labels")
        loss_mask = batch.get("loss_mask")
        if labels is None:
            labels = batch.get("labels")
            if labels is not None:
                labels = labels[..., 1:]
                if loss_mask is not None:
                    loss_mask = loss_mask[..., 1:]
        if (
            isinstance(labels, torch.Tensor)
            and labels.device.type == "cpu"
            and labels.ndim > 0
            and labels.numel() == ids.shape[0] * labels.shape[-1]
        ):
            supervised = labels != IGNORE_INDEX
            if isinstance(loss_mask, torch.Tensor) and loss_mask.device.type == "cpu":
                if loss_mask.shape != labels.shape:
                    self._failure("loss_mask_shape")
                else:
                    supervised &= loss_mask.bool()
            self.supervised.update(supervised.reshape(ids.shape[0], labels.shape[-1]).sum(-1).tolist())
            self.totals["valid_tokens"] += int(supervised.sum())
            self.totals["target_slots"] += labels.numel()
        else:
            self.unavailable.add("supervision")
        if self.config.top_n_tokens and self.top_complete:
            unique, counts = torch.unique(ids[valid], return_counts=True)
            if len(self.tokens) + unique.numel() > 1048576:
                self.top_complete = False
                self.tokens.clear()
                self.unavailable.add("top_n_capacity")
            else:
                self.tokens.update(dict(zip(unique.tolist(), counts.tolist())))
        if self.config.packing_stats or self.config.packing_boundary_check:
            self._packing(batch, ids)

    def _packing(self, batch: dict, ids: torch.Tensor) -> None:
        boundaries = batch.get("cu_seq_lens")
        if boundaries is None:
            self.unavailable.add("packing_metadata")
            return
        if not isinstance(boundaries, torch.Tensor) or boundaries.device.type != "cpu":
            self.unavailable.add("cpu_packing_metadata")
            return
        values = boundaries.reshape(-1).tolist()
        self.totals["packing_checks"] += 1
        if any((not isinstance(value, int) or isinstance(value, bool)) for value in values):
            if self.config.packing_boundary_check:
                self._failure("offset_dtype")
            return
        if len(values) < 2 or values[0] != 0 or any(b <= a for a, b in zip(values, values[1:])):
            if self.config.packing_boundary_check:
                self._failure("offset_order")
            return
        if values[-1] != ids.numel():
            if self.config.packing_boundary_check:
                self._failure("offset_coverage")
            return
        if any(start // ids.shape[-1] != (end - 1) // ids.shape[-1] for start, end in zip(values, values[1:])):
            if self.config.packing_boundary_check:
                self._failure("segment_crosses_row")
        lengths = [b - a for a, b in zip(values, values[1:])]
        self.segments.update(lengths)
        self.totals["segments"] += len(lengths)
        policy = batch.get("packing_policy")
        if policy not in ("isolated", "concatenated"):
            self.unavailable.add("packing_attention_policy")
            return
        positions = batch.get("position_ids")
        if self.config.packing_boundary_check and policy == "isolated":
            if (
                batch.get("reset_position_ids", True)
                and positions is not None
                and positions.device.type == "cpu"
                and positions.numel() == ids.numel()
            ):
                flat = positions.reshape(-1)
                for start, end in zip(values, values[1:]):
                    if not torch.equal(flat[start:end], torch.arange(end - start, dtype=flat.dtype)):
                        self._failure("position_reset")
            else:
                self.unavailable.add("packing_positions")
            targets = batch.get("shift_labels")
            if targets is not None and targets.device.type == "cpu" and targets.numel() == ids.numel():
                flat = targets.reshape(-1)
                for end in values[1:-1]:
                    loss_mask = batch.get("loss_mask")
                    masked = (
                        isinstance(loss_mask, torch.Tensor)
                        and loss_mask.device.type == "cpu"
                        and loss_mask.numel() == targets.numel()
                        and not loss_mask.reshape(-1)[end - 1]
                    )
                    if flat[end - 1] != IGNORE_INDEX and not masked:
                        self._failure("cross_segment_target")
            else:
                self.unavailable.add("packing_shifted_labels")
        self.unavailable.add("attention_kernel_validation")

    @staticmethod
    def distribution(counts: Counter) -> dict:
        """Return exact length quantiles without retaining per-sample arrays."""
        size = sum(counts.values())
        if not size:
            return {}
        ordered = sorted(counts.items())
        result = {
            "min": ordered[0][0],
            "max": ordered[-1][0],
            "mean": sum(k * v for k, v in ordered) / size,
            "histogram": ordered,
        }
        for q in (50, 90, 99):
            cumulative = 0
            for length, count in ordered:
                cumulative += count
                if cumulative >= max(1, (size * q + 99) // 100):
                    result[f"p{q}"] = length
                    break
        return result

    def snapshot(self) -> dict:
        """Build a CPU-only record with explicit unavailable coverage."""
        result = {"counts": dict(self.totals), "unavailable": sorted(self.unavailable)}
        if self.config.token_distribution:
            result["input_lengths"] = self.distribution(self.lengths)
            result["supervised_lengths"] = self.distribution(self.supervised)
        if self.config.top_n_tokens and self.top_complete:
            denominator = self.totals["input_tokens"]
            result["top_tokens"] = [
                {"token_id": k, "count": v, "fraction": v / denominator if denominator else 0}
                for k, v in sorted(self.tokens.items(), key=lambda item: (-item[1], item[0]))[
                    : self.config.top_n_tokens
                ]
            ]
        if self.config.packing_stats:
            slots = self.totals["slots"]
            result["padding_fraction"] = (
                1 - self.totals["input_tokens"] / slots if slots and "padding_mask" not in self.unavailable else None
            )
            rows = self.totals["rows"]
            result["segments_per_row"] = self.totals["segments"] / rows if rows else None
            result["segment_lengths"] = self.distribution(self.segments)
        if self.config.packing_boundary_check:
            result["packing_failures"] = list(self.failures)
        return result
