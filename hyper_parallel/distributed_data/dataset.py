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
"""Dataset-owned contracts for externally selected raw-sample steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from hyper_parallel.distributed_data.device_prefetch import _move_to_device
from hyper_parallel.distributed_data.schema import SampleMetadata


class DistributedDataset:
    """Bind an existing step source to its metadata, collator and field policy.

    This is an iterable data-pipeline facade, not a new disk reader or sampler.
    Each source yield is a complete local step of non-empty raw-sample bins.
    Source transforms, sampling, worker prefetch and step membership stay intact.
    The collator receives one accepted bin only after sample redistribution.
    """

    def __init__(
            self,
            source: Iterable[Sequence[Sequence[Any]]],
            *,
            metadata: Callable[[Any], SampleMetadata] | str,
            collate_fn: Callable[[Sequence[Any]], Any],
            cpu_fields: Sequence[str] = (),
            log_fields: Sequence[str] = (),
    ) -> None:
        """Store the dataset's model-facing contract without reading samples.

        Args:
            source: Existing rank-local source yielding complete raw steps.
            metadata: A sample-to-metadata callable, or a sample mapping key
                containing a precomputed SampleMetadata object. The embedded
                metadata field is removed before invoking the collator.
            collate_fn: Existing model collator, called once per raw-sample bin.
            cpu_fields: Top-level collated batch fields that must remain on CPU.
                All other tensor leaves move recursively, without dtype changes.
            log_fields: Additive numeric metadata.features fields to sum per bin.
                Omit to log only sample counts, sequence lengths and costs.
        """
        if not callable(getattr(source, "__iter__", None)):
            raise ValueError("source must yield complete local raw-sample steps.")
        if not callable(metadata) and not (isinstance(metadata, str) and metadata):
            raise ValueError("metadata must be a callable or a non-empty sample field name.")
        if not callable(collate_fn):
            raise ValueError("collate_fn must be callable.")
        for name, fields in (("cpu_fields", cpu_fields), ("log_fields", log_fields)):
            if isinstance(fields, str) or any(not isinstance(key, str) or not key for key in fields):
                raise ValueError(f"{name} must be a sequence of non-empty field names.")
        self.source = source
        self.metadata = metadata
        self.collate_fn = collate_fn
        self.cpu_fields = frozenset(cpu_fields)
        self.log_fields = tuple(log_fields)

    def __iter__(self) -> Iterator[Sequence[Sequence[Any]]]:
        """Forward step selection to the source, without another worker pool."""
        return iter(self.source)

    def __len__(self) -> int:
        """Return the source step count when the source implements length."""
        return len(self.source)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes to the source when supported.

        Args:
            epoch: Source epoch to begin.
        """
        setter = getattr(self.source, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def sample_metadata(self, sample: Any) -> SampleMetadata:
        """Read an existing metadata object or derive it from one CPU sample.

        Args:
            sample: Raw CPU sample from the source.
        """
        value = self.metadata(sample) if callable(self.metadata) else sample[self.metadata]
        if not isinstance(value, SampleMetadata):
            raise ValueError("Dataset metadata must be SampleMetadata.")
        return value

    def pack(self, samples: Sequence[Any], seq_len: int) -> Any:
        """Apply the original collator after capacity-constrained placement.

        Args:
            samples: Raw samples assigned to one bin.
            seq_len: Planner capacity; the existing collator owns construction.
        """
        del seq_len
        if isinstance(self.metadata, str):
            samples = [{key: value for key, value in sample.items() if key != self.metadata} for sample in samples]
        return self.collate_fn(list(samples))

    def move_to_device(self, batch: Any, device: Any) -> Any:
        """Move model inputs while retaining declared CPU-only fields.

        Args:
            batch: One collated microbatch.
            device: Target training device.
        """
        if not self.cpu_fields:
            return _move_to_device(batch, device)
        if not isinstance(batch, Mapping):
            raise ValueError("cpu_fields requires the collator to return a mapping.")
        return {
            key: value if key in self.cpu_fields else _move_to_device(value, device)
            for key, value in batch.items()
        }

    def summarize(self, samples: Iterable[SampleMetadata]) -> dict[str, int | float]:
        """Aggregate declared numeric features without reading sample tensors.

        Args:
            samples: Metadata entries assigned to one bin.
        """
        totals = dict.fromkeys(self.log_fields, 0)
        for sample in samples:
            for name in self.log_fields:
                value = sample.features[name]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Log feature {name!r} must be numeric.")
                totals[name] += value
        return totals


def build_distributed_dataset(
        source: Iterable[Sequence[Sequence[Any]]],
        *,
        metadata: Callable[[Any], SampleMetadata] | str,
        collate_fn: Callable[[Sequence[Any]], Any],
        cpu_fields: Sequence[str] = (),
        log_fields: Sequence[str] = (),
) -> DistributedDataset:
    """Build a data contract consumed directly by build_distributed_dataloader.

    Args:
        source: Existing CPU source yielding one local step of raw-sample bins.
            It must not apply final collation or device transfer before yielding.
        metadata: SampleMetadata callback or precomputed metadata field name.
        collate_fn: Model's existing per-bin collator, not a whole-step collator.
        cpu_fields: Top-level collated fields excluded from device transfer.
        log_fields: Additive numeric metadata.features fields for per-bin logs.

    Returns:
        A dataset that owns data interpretation, not scheduling or communication.
    """
    return DistributedDataset(
        source, metadata=metadata, collate_fn=collate_fn,
        cpu_fields=cpu_fields, log_fields=log_fields,
    )


__all__ = ["DistributedDataset", "build_distributed_dataset"]
