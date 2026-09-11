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
"""Transactional Dataset reads driven by HP's native DP BatchSampler."""

from __future__ import annotations

import copy
import hashlib
import operator
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hyper_parallel.distributed_data.schema import BufferedSampleMetadata, SampleKey, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader


def _mapping_fingerprint(index_mapping: Any) -> str | None:
    """Hash integer values, not tensor pickle storage identities or mapping insertion order."""
    if index_mapping is None:
        return None
    if isinstance(index_mapping, Mapping):
        normalized = sorted((operator.index(key), operator.index(value)) for key, value in index_mapping.items())
    else:
        tolist = getattr(index_mapping, "tolist", None)
        values = tolist() if callable(tolist) else index_mapping
        normalized = tuple(operator.index(value) for value in values)
    return hashlib.sha256(pickle.dumps(normalized, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def native_sampler_fingerprint(sampler: Any, *, data_rank: int, dp_size: int, local_batch_size: int) -> str:
    """Validate the native HP sampler contract and identify its sampling policy.

    The DP coordinate is checked locally but excluded from the shared policy
    fingerprint. Checkpoints include the policy as well as the native cursor.
    """
    for name in ("__iter__", "state_dict", "load_state_dict", "set_epoch", "enable_source_batch_resume"):
        if not callable(getattr(sampler, name, None)):
            raise ValueError(f"batch_sampler must implement HP's native sampler contract: missing {name}.")
    expected = {"dp_rank": data_rank, "dp_world_size": dp_size, "micro_batch_size": local_batch_size,
                "drop_last": True}
    for name, value in expected.items():
        if getattr(sampler, name, None) != value:
            raise ValueError(f"batch_sampler.{name} must equal {value!r}.")
    if sampler.consumed_samples != sampler.total_samples and sampler.consumed_samples % (dp_size * local_batch_size):
        raise ValueError("batch_sampler.consumed_samples must align to a synchronized forward/backward round.")
    policy = (
        type(sampler).__module__, type(sampler).__qualname__,
        sampler.total_samples, sampler.micro_batch_size, sampler.global_batch_size, sampler.dp_world_size,
        getattr(sampler, "seed", None), getattr(sampler, "data_sharding", None),
        _mapping_fingerprint(sampler.index_mapping),
    )
    return hashlib.sha256(pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def preserve_sample(samples: Sequence[Any], seq_len: int) -> Any:
    """Pass one native Dataset output unchanged to the original collator."""
    del seq_len
    if len(samples) != 1:
        raise ValueError("Native BatchSampler mode must keep one Dataset output per sequence bin.")
    return samples[0]


class BatchSamplerReader:
    """Read precisely one native local batch, with no read-ahead selection or stride.

    Only the DP Constructor owns this reader. Payload loading is deferred until
    after planning when metadata is available; otherwise it precedes metadata
    extraction. The sampler's speculative cursor is never a checkpoint cursor.
    """

    VERSION = 1

    def __init__(
            self,
            sampler: Any,
            *,
            reader_rank: int,
            policy_fingerprint: str,
            metadata: Sequence[SampleMetadata] | None,
            metadata_fn: Callable[[Any], SampleMetadata] | None,
            sample_loader: PlannedSampleLoader | None,
    ) -> None:
        """Store the validated native sampler and metadata/payload sources."""
        self._sampler = sampler
        self._reader_rank = reader_rank
        self._policy_fingerprint = policy_fingerprint
        self._metadata = metadata
        self._metadata_fn = metadata_fn
        self._sample_loader = sample_loader
        self._sampler.enable_source_batch_resume()
        self._committed_state = copy.deepcopy(sampler.state_dict())
        self._pending_state: dict[str, Any] | None = None
        self._iterator = None
        self._buffer: tuple[BufferedSampleMetadata, ...] = ()
        self._payloads: dict[SampleKey, Any] = {}
        self._exhausted = False
        self._error: str | None = None

    @property
    def exhausted(self) -> bool:
        """Return whether the native sampler has exhausted its epoch."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return the number of pending native sample occurrences."""
        return len(self._buffer)

    @property
    def batch_position(self) -> int:
        """Return the native global cursor before this pending round."""
        return self._committed_state["consumed_samples"]

    def fill(self, *, min_samples: int, min_tokens: int, max_samples: int) -> str | None:
        """Prepare exactly one sampler yield, ignoring dynamic-packing targets."""
        del min_samples, min_tokens, max_samples
        if self._buffer or self._exhausted or self._error is not None:
            return self._error
        try:
            if self._iterator is None:
                self._iterator = iter(self._sampler)
            try:
                indices = next(self._iterator)
            except StopIteration:
                self._exhausted = True
                return None
            keys = self._sample_keys(indices)
            self._pending_state = copy.deepcopy(self._sampler.state_dict())
            expected_position = self.batch_position + self._sampler.global_micro_batch_size
            if self._pending_state["consumed_samples"] != expected_position:
                raise ValueError("Native BatchSampler must advance exactly one global local-batch round per yield.")
            if self._metadata is None:
                if self._sample_loader is None or self._metadata_fn is None:
                    raise ValueError("Online BatchSampler Reader requires payload loading and metadata_fn.")
                self._payloads = self._sample_loader.fetch_keys(keys)
                metadata = [self._metadata_fn(self._payloads[key]) for key in keys]
            else:
                metadata = [self._metadata[key.dataset_index] for key in keys]
            if any(not isinstance(item, SampleMetadata) for item in metadata):
                raise ValueError("BatchSampler metadata must contain SampleMetadata entries.")
            self._buffer = tuple(
                BufferedSampleMetadata(key, item, key.global_sample_position)
                for key, item in zip(keys, metadata)
            )
        except Exception as exc:
            self._error = f"Native BatchSampler Reader failed: {type(exc).__name__}: {exc}"
        return self._error

    def _sample_keys(self, indices: Sequence[int]) -> tuple[SampleKey, ...]:
        if len(indices) != self._sampler.micro_batch_size:
            raise ValueError("Native BatchSampler must yield exactly local_batch_size indices on every DP rank.")
        # A native sampler exposes local indices, not its internal permutation
        # positions. Stable DP-major occurrence slots preserve duplicates.
        start = self.batch_position + self._sampler.dp_rank * self._sampler.micro_batch_size
        keys = []
        for offset, index in enumerate(indices):
            if isinstance(index, bool):
                raise ValueError("Native BatchSampler indices must be non-negative integers.")
            keys.append(SampleKey(self._reader_rank, operator.index(index), start + offset))
        return tuple(keys)

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return metadata for this one frozen native local batch."""
        return self._buffer

    def selected_payloads(self, selected_keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return already-read payloads without rereading their Dataset indices."""
        return tuple((item.key, self._payloads[item.key]) for item in self._buffer if item.key in selected_keys)

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Advance saved progress only after this entire round is delivered."""
        if selected_keys != {item.key for item in self._buffer} or self._pending_state is None:
            raise ValueError("Native BatchSampler commit must consume the complete pending local batch.")
        self._committed_state = self._pending_state
        self._pending_state = None
        self._buffer = ()
        self._payloads = {}

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the delivered native cursor, excluding speculative prefetch."""
        return copy.deepcopy({
            "version": self.VERSION,
            "epoch": self._committed_state["epoch"],
            "policy_fingerprint": self._policy_fingerprint,
            "sampler": self._committed_state,
            "sample_loader": self._sample_loader.state_dict() if self._sample_loader is not None else None,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the original native cursor before the next sampler yield."""
        if state.get("version") != self.VERSION or state.get("policy_fingerprint") != self._policy_fingerprint:
            raise ValueError("Native BatchSampler checkpoint sampling policy changed.")
        sampler_state = state["sampler"]
        if sampler_state.get("epoch") != state.get("epoch"):
            raise ValueError("Native BatchSampler checkpoint epoch is inconsistent.")
        if (state.get("sample_loader") is None) != (self._sample_loader is None):
            raise ValueError("Native BatchSampler checkpoint payload reader ownership changed.")
        self._sampler.load_state_dict(sampler_state)
        if self._sample_loader is not None:
            self._sample_loader.load_state_dict(state["sample_loader"])
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._reset_pending()

    def set_epoch(self, epoch: int) -> None:
        """Delegate epoch changes to the native sampler without changing its policy."""
        self._sampler.load_state_dict(self._committed_state)
        self._sampler.set_epoch(epoch)
        if self._sample_loader is not None:
            self._sample_loader.set_epoch(epoch)
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._reset_pending()

    def _reset_pending(self) -> None:
        self._iterator = None
        self._pending_state = None
        self._buffer = ()
        self._payloads = {}
        self._exhausted = False
        self._error = None
