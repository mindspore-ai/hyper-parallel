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
"""Deterministic step membership selection before balanced placement."""

from __future__ import annotations

from collections.abc import Sequence

from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    OversizedPolicy,
    StepSampleSelection,
)


class StepSampleSelector:
    """Freeze one step's canonical sample-ID set using streaming packing.

    Reader lookahead affects only how many candidates are available. Selection
    always consumes the canonical stream prefix and therefore never skips a
    costly sample or moves a future sample into the current step.
    """

    def __init__(
            self,
            *,
            seq_len: int,
            distributed_bin_count: int,
            oversized_policy: OversizedPolicy = "error",
    ) -> None:
        """Initialize the canonical streaming-packing boundary.

        Args:
            seq_len: Maximum token count in one non-oversized reference bin.
            distributed_bin_count: Packed sequences required for one step.
            oversized_policy: ``error`` or explicit singleton overflow.
        """
        for name, value in (("seq_len", seq_len), ("distributed_bin_count", distributed_bin_count)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        self._seq_len = seq_len
        self._distributed_bin_count = distributed_bin_count
        self._oversized_policy = oversized_policy

    def select(
            self,
            candidates: Sequence[BufferedSampleMetadata],
            *,
            end_of_stream: bool,
    ) -> StepSampleSelection | None:
        """Select exactly one step from the canonical candidate prefix.

        Args:
            candidates: Metadata currently buffered across all Dataset Readers.
            end_of_stream: Whether every Dataset Reader reached epoch end.

        Returns:
            Frozen step membership, or ``None`` when more metadata is needed or
            the final tail cannot fill all distributed bins.
        """
        if not isinstance(end_of_stream, bool):
            raise ValueError("end_of_stream must be boolean.")
        ordered = self._canonical_prefix(candidates, end_of_stream=end_of_stream)
        reference_bins: list[tuple[BufferedSampleMetadata, ...]] = []
        current_bin: list[BufferedSampleMetadata] = []
        current_tokens = 0

        for item in ordered:
            pack_tokens = item.metadata.pack_tokens
            if pack_tokens > self._seq_len:
                if self._oversized_policy == "error":
                    raise ValueError(
                        f"Sample {item.key} requires {pack_tokens} tokens, exceeding seq_len={self._seq_len}. "
                        "Set oversized_policy='single' only when the packer supports singleton overflow."
                    )
                selection = self._close_bin(reference_bins, current_bin)
                if selection is not None:
                    return selection
                current_bin = []
                current_tokens = 0
                reference_bins.append((item,))
                selection = self._freeze_if_complete(reference_bins)
                if selection is not None:
                    return selection
                continue

            if current_bin and current_tokens + pack_tokens > self._seq_len:
                reference_bins.append(tuple(current_bin))
                selection = self._freeze_if_complete(reference_bins)
                if selection is not None:
                    return selection
                current_bin = []
                current_tokens = 0

            current_bin.append(item)
            current_tokens += pack_tokens
            if current_tokens == self._seq_len:
                reference_bins.append(tuple(current_bin))
                selection = self._freeze_if_complete(reference_bins)
                if selection is not None:
                    return selection
                current_bin = []
                current_tokens = 0

        if end_of_stream and current_bin:
            reference_bins.append(tuple(current_bin))
            return self._freeze_if_complete(reference_bins)
        return None

    @staticmethod
    def _canonical_prefix(
            candidates: Sequence[BufferedSampleMetadata],
            *,
            end_of_stream: bool,
    ) -> tuple[BufferedSampleMetadata, ...]:
        if any(not isinstance(item, BufferedSampleMetadata) for item in candidates):
            raise ValueError("Every step-selection candidate must be BufferedSampleMetadata.")
        keys = [item.key for item in candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Step-selection candidates must have unique SampleKey values.")
        positions = [item.global_sample_position for item in candidates]
        if len(positions) != len(set(positions)):
            raise ValueError("Step-selection candidates must have unique global_sample_position values.")
        ordered = tuple(sorted(candidates, key=lambda item: item.global_sample_position))
        if not ordered:
            return ()

        prefix = [ordered[0]]
        expected_position = ordered[0].global_sample_position + 1
        for item in ordered[1:]:
            if item.global_sample_position != expected_position:
                if end_of_stream:
                    raise ValueError(
                        f"Canonical sample stream has a gap before position {expected_position}; "
                        f"the next buffered position is {item.global_sample_position}."
                    )
                break
            prefix.append(item)
            expected_position += 1
        return tuple(prefix)

    def _close_bin(
            self,
            reference_bins: list[tuple[BufferedSampleMetadata, ...]],
            current_bin: list[BufferedSampleMetadata],
    ) -> StepSampleSelection | None:
        if not current_bin:
            return None
        reference_bins.append(tuple(current_bin))
        return self._freeze_if_complete(reference_bins)

    def _freeze_if_complete(
            self,
            reference_bins: Sequence[tuple[BufferedSampleMetadata, ...]],
    ) -> StepSampleSelection | None:
        if len(reference_bins) != self._distributed_bin_count:
            return None
        samples = tuple(item for packing_bin in reference_bins for item in packing_bin)
        key_bins = tuple(tuple(item.key for item in packing_bin) for packing_bin in reference_bins)
        return StepSampleSelection(samples=samples, reference_bins=key_bins)


__all__ = ["StepSampleSelector"]
