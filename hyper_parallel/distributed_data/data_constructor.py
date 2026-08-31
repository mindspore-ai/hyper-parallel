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
"""Target-rank Data Constructor for plan-ordered packing and collation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hyper_parallel.distributed_data.schema import DataConstructorPlan, SampleKey


def default_pack_fn(samples: Sequence[Any], seq_len: int) -> tuple[Any, ...]:
    """Preserve one planned packing bin without assuming a sample schema.

    Args:
        samples: Raw samples assigned to one Planner-validated bin, including
            an explicitly permitted singleton overflow when configured.
        seq_len: Configured normal sequence capacity.

    Returns:
        Immutable raw samples in their deterministic planned order.
    """
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError(f"seq_len must be a positive integer, but got {seq_len!r}.")
    return tuple(samples)


def default_collate_fn(packed_sequences: Sequence[Any]) -> tuple[Any, ...]:
    """Preserve the planned bins as one immutable rank-local batch.

    Args:
        packed_sequences: Outputs produced for this rank's packing bins.

    Returns:
        Immutable bins in contiguous ``pack_index`` order.
    """
    return tuple(packed_sequences)


class PackingDataConstructor:
    """Apply configured packing per planned bin and collate one local batch.

    ``pack_fn`` never runs in a Dataset Reader. It receives raw samples only
    after sample-level planning and CPU redistribution have completed.
    """

    def __init__(
            self,
            pack_fn: Callable[[Sequence[Any], int], Any],
            collate_fn: Callable[[Sequence[Any]], Any],
            *,
            seq_len: int,
    ) -> None:
        """Initialize user callbacks.

        Args:
            pack_fn: Called once per planned sequence bin as
                ``pack_fn(raw_samples, seq_len)``.
            collate_fn: Called once per local batch with packed sequences.
            seq_len: Maximum non-oversized sequence length.
        """
        if not callable(pack_fn) or not callable(collate_fn):
            raise ValueError("pack_fn and collate_fn must be callable.")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
            raise ValueError(f"seq_len must be a positive integer, but got {seq_len!r}.")
        self._pack_fn = pack_fn
        self._collate_fn = collate_fn
        self._seq_len = seq_len

    def construct(
            self,
            plan: DataConstructorPlan,
            payloads: Mapping[SampleKey, Any],
    ) -> Any:
        """Construct one target-rank local batch.

        Args:
            plan: Ordered packing bins assigned to this Data Constructor.
            payloads: Received raw samples keyed by stable Dataset Reader and
                Dataset-index identity.

        Returns:
            User-collated rank-local batch.
        """
        if not isinstance(plan, DataConstructorPlan):
            raise ValueError(f"plan must be DataConstructorPlan, but got {type(plan)}.")
        expected_keys = plan.sample_keys
        if len(payloads) != len(expected_keys) or set(payloads) != set(expected_keys):
            missing = set(expected_keys) - set(payloads)
            unexpected = set(payloads) - set(expected_keys)
            raise ValueError(
                f"Data Constructor payload keys do not match the plan; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )

        packed_sequences = []
        for packing_bin in plan.bins:
            if packing_bin.pack_tokens > self._seq_len and not packing_bin.oversized:
                raise ValueError(
                    f"Packing bin {packing_bin.pack_index} has {packing_bin.pack_tokens} tokens, "
                    f"exceeding seq_len={self._seq_len}."
                )
            raw_samples = [payloads[sample.key] for sample in packing_bin.samples]
            packed_sequences.append(self._pack_fn(raw_samples, self._seq_len))
        return self._collate_fn(packed_sequences)


__all__ = ["PackingDataConstructor", "default_collate_fn", "default_pack_fn"]
