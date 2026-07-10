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
"""``GPTDataset`` — fixed-length samples drawn from a Megatron IndexedDataset.

Behaviourally aligned with Megatron-LM's ``GPTDataset``:

1. ``document_index`` — a shuffled flat array repeating every document for
   ``num_epochs`` epochs. Documents stay whole; only their visit order is
   randomised.
2. ``sample_index`` — for sample ``i`` the entry ``[doc_id, doc_offset]``
   points to the document and within-document token offset where that
   sample starts. Built by sequentially walking the document stream and
   slicing every ``seq_length`` tokens.
3. ``shuffle_index`` — a random permutation applied on top, so sample
   ``__getitem__(i)`` actually reads sample ``shuffle_index[i]``.

A sample may span multiple documents (Megatron's concatenated-doc style);
in that case we glue the relevant slices together until we reach
``seq_length + 1`` tokens (one extra so labels can be shifted, exactly
Megatron's ``add_extra_token_to_sequence=1``). If the very last sample
runs out of documents, the tail is right-padded with ``pad_token_id``.

Pure Python+numpy. No Cython helpers — the index construction runs once
per ``(seed, prefix, num_samples, seq_length, num_epochs)`` tuple and is
cached to disk under ``<prefix>_cache/`` so subsequent runs reuse it.
"""
import functools
import hashlib
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hyper_parallel.data.megatron.indexed_dataset import IndexedDataset


logger = logging.getLogger(__name__)


def _build_document_index(num_documents: int, num_epochs: int, rng: np.random.RandomState) -> np.ndarray:
    """Return a globally-shuffled, ``num_epochs``-repeated array of document indices.

    Matches Megatron-LM's ``separate_last_epoch=False`` default: tile the
    document range across ``num_epochs`` and run a single global shuffle.
    This intentionally lets a document appear several times before another
    appears once — that randomness is part of the canonical training
    distribution and the realised epoch-boundary noise is what Megatron's
    schedulers expect.
    """
    doc_idx = np.tile(np.arange(num_documents, dtype=np.int32), num_epochs)
    rng.shuffle(doc_idx)
    return doc_idx


def _build_sample_index(
    sizes: np.ndarray,
    doc_idx: np.ndarray,
    seq_length: int,
    tokens_per_epoch: int,
    num_epochs: int,
) -> np.ndarray:
    """Construct the ``[num_samples + 1, 2]`` (doc_id, doc_offset) sample table.

    Walks the per-document token stream in ``doc_idx`` order, marking the
    start of every ``seq_length``-window. Each sample includes one extra
    token so labels can be shifted by 1; the helper keeps the
    pre-Megatron-Cython slow-path semantics (no boundary correction —
    multi-document samples are reassembled by :meth:`GPTDataset._read_sample`).
    """
    tokens_to_consume = num_epochs * tokens_per_epoch - 1  # -1 to keep ``+1`` extra token in reach
    sample_stride = seq_length
    num_samples = tokens_to_consume // sample_stride
    # int64: column 0 indexes ``doc_idx`` whose length is
    # ``num_documents * num_epochs`` — int32 would wrap past ~2.1e9 slots on
    # large corpora trained for many epochs, silently selecting the wrong
    # document.
    sample_idx = np.zeros((num_samples + 1, 2), dtype=np.int64)

    cur_doc_index = 0
    cur_doc_offset = 0
    cur_doc_remaining = int(sizes[doc_idx[cur_doc_index]])
    sample_idx[0, 0] = cur_doc_index
    sample_idx[0, 1] = cur_doc_offset
    for sample_id in range(1, num_samples + 1):
        remaining = sample_stride
        while remaining > 0:
            if cur_doc_remaining > remaining:
                cur_doc_offset += remaining
                cur_doc_remaining -= remaining
                remaining = 0
            else:
                remaining -= cur_doc_remaining
                cur_doc_index += 1
                if cur_doc_index >= doc_idx.shape[0]:
                    # Document stream exhausted — record where we landed
                    # and stop. Caller will pad the tail.
                    cur_doc_remaining = 0
                    break
                cur_doc_offset = 0
                cur_doc_remaining = int(sizes[doc_idx[cur_doc_index]])
        sample_idx[sample_id, 0] = cur_doc_index
        sample_idx[sample_id, 1] = cur_doc_offset
    return sample_idx


def _build_shuffle_index(num_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Random permutation of ``[0, num_samples)``."""
    shuffle = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(shuffle)
    return shuffle


def _layout_fingerprint(doc_sizes: np.ndarray) -> str:
    """Hash the per-document token-count array.

    ``doc_sizes`` is exactly what determines every ``sample_idx`` offset, so a
    corpus regenerated at the same path with a different document layout — even
    one preserving document count and total token count — produces a different
    fingerprint and invalidates the stale cache.
    """
    return hashlib.sha1(np.ascontiguousarray(doc_sizes, dtype=np.int64).tobytes()).hexdigest()[:16]


def _cache_key(
    prefix: str,
    seq_length: int,
    num_samples: int,
    num_epochs: int,
    seed: int,
    layout_fp: str,
) -> str:
    """Stable filename suffix for the ``(doc, sample, shuffle)`` index trio.

    ``layout_fp`` fingerprints the underlying corpus layout (see
    :func:`_layout_fingerprint`) so that regenerating the ``.bin``/``.idx``
    under the same path invalidates a stale cache instead of silently
    replaying indices computed against the old token stream.
    """
    h = hashlib.sha1(
        f"{prefix}|{seq_length}|{num_samples}|{num_epochs}|{seed}|{layout_fp}".encode("utf-8"),
    ).hexdigest()[:16]
    return h


class GPTDataset(Dataset):
    """Megatron-style GPT dataset on top of an :class:`IndexedDataset`.

    Args:
        indexed_dataset: Underlying token-stream reader.
        num_samples: How many fixed-length samples to expose. Typically
            ``max_steps * global_batch_size`` so the dataloader never
            runs dry mid-training.
        seq_length: Tokens per ``input_ids`` (excluding the extra label token).
        seed: RNG seed for the document / shuffle permutations.
        pad_token_id: Token used to right-pad the very last sample when
            the document stream is exhausted before ``seq_length + 1``
            tokens are available.
        eod_mask_loss: When ``True``, the EOD token's label is replaced by
            ``-100`` so it does not contribute to the loss.
        eod_token_id: EOD token id; ignored when ``eod_mask_loss`` is False.
        cache_dir: Where to cache the (doc, sample, shuffle) index arrays.
            Defaults to ``<indexed_dataset.path_prefix>_cache``.

    Returns from ``__getitem__``:
        ``{"input_ids": Tensor[seq_length], "labels": Tensor[seq_length]}``
        — labels is a copy of input_ids (the model handles next-token
        shifting internally, matching the rest of the trainer's
        dataset contract).
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        num_samples: int,
        seq_length: int,
        seed: int = 1234,
        *,
        pad_token_id: int = 0,
        eod_mask_loss: bool = False,
        eod_token_id: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if seq_length <= 0:
            raise ValueError(f"seq_length must be > 0, got {seq_length}")

        self.indexed_dataset = indexed_dataset
        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.pad_token_id = int(pad_token_id)
        self.eod_mask_loss = bool(eod_mask_loss)
        self.eod_token_id = eod_token_id

        self.document_index, self.sample_index, self.shuffle_index = self._build_indices(cache_dir)
        # Adjacent samples often touch the same document — cache the concat
        # result for the last few requested doc ids so straddling reads
        # don't re-walk the IndexedDataset.
        self._cached_doc = functools.lru_cache(maxsize=128)(self._read_doc)

    def _build_indices(self, cache_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (or load from cache) the three Megatron-style indices."""
        sizes = self.indexed_dataset.sequence_lengths
        num_docs = self.indexed_dataset.num_documents
        if num_docs == 0:
            raise ValueError("IndexedDataset has no documents to sample from")
        # Build a per-document length array using the existing
        # ``document_indices`` boundaries.
        doc_boundaries = self.indexed_dataset.document_indices
        doc_sizes = np.zeros(num_docs, dtype=np.int64)
        for d in range(num_docs):
            start = int(doc_boundaries[d])
            end = int(doc_boundaries[d + 1])
            doc_sizes[d] = int(sizes[start:end].sum())

        tokens_per_epoch = int(doc_sizes.sum())
        if tokens_per_epoch == 0:
            raise ValueError("IndexedDataset has zero total tokens")
        # ``+1`` because each sample reads ``seq_length + 1`` tokens (input
        # plus the next-token label). Ceil so we always cover ``num_samples``.
        tokens_required = self.num_samples * self.seq_length + 1
        num_epochs = max(1, int(np.ceil(tokens_required / tokens_per_epoch)))

        cache_dir = self._resolve_cache_dir(cache_dir)
        key = _cache_key(
            self.indexed_dataset.path_prefix, self.seq_length,
            self.num_samples, num_epochs, self.seed,
            _layout_fingerprint(doc_sizes),
        )
        doc_path = os.path.join(cache_dir, f"doc_idx_{key}.npy")
        sample_path = os.path.join(cache_dir, f"sample_idx_{key}.npy")
        shuffle_path = os.path.join(cache_dir, f"shuffle_idx_{key}.npy")

        if os.path.isfile(doc_path) and os.path.isfile(sample_path) and os.path.isfile(shuffle_path):
            return np.load(doc_path), np.load(sample_path), np.load(shuffle_path)

        rng = np.random.RandomState(self.seed)
        doc_idx = _build_document_index(num_docs, num_epochs, rng)
        sample_idx = _build_sample_index(
            doc_sizes, doc_idx, self.seq_length, tokens_per_epoch, num_epochs,
        )
        # ``sample_idx`` has at most ``num_samples + 1`` rows; clip in case
        # we built slightly more (rounded up by tokens_per_epoch / seq_len).
        usable_samples = max(sample_idx.shape[0] - 1, 1)
        shuffle_idx = _build_shuffle_index(min(self.num_samples, usable_samples), rng)
        np.save(doc_path, doc_idx)
        np.save(sample_path, sample_idx)
        np.save(shuffle_path, shuffle_idx)
        return doc_idx, sample_idx, shuffle_idx

    def _resolve_cache_dir(self, user_cache_dir: Optional[str]) -> str:
        """Return a writable cache directory, falling back to ``$TMPDIR``.

        Defaulting to ``<prefix>_cache`` next to the corpus is convenient,
        but production datasets often sit on read-only shared mounts where
        ``os.makedirs`` would raise ``EROFS``. Detect and silently relocate.
        """
        candidate = user_cache_dir or (self.indexed_dataset.path_prefix + "_cache")
        try:
            os.makedirs(candidate, exist_ok=True)
            # Probe writability — ``os.makedirs`` succeeds on directories
            # that exist read-only.
            probe = os.path.join(candidate, ".hp_cache_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("")
            os.remove(probe)
            return candidate
        except OSError as exc:
            fallback = os.path.join(tempfile.gettempdir(), "hp_megatron_cache")
            os.makedirs(fallback, exist_ok=True)
            logger.warning(
                "GPTDataset cache dir %s not writable (%s); using fallback %s",
                candidate, exc, fallback,
            )
            return fallback

    def __len__(self) -> int:
        return min(self.num_samples, int(self.shuffle_index.size))

    def _read_sample(self, sample_pos: int) -> np.ndarray:
        """Concatenate the ``seq_length + 1`` tokens for sample ``sample_pos``."""
        start_doc_idx, start_offset = self.sample_index[sample_pos]
        end_doc_idx, end_offset = self.sample_index[sample_pos + 1]
        parts = []
        if start_doc_idx == end_doc_idx:
            doc_tokens = self._cached_doc(int(self.document_index[start_doc_idx]))
            parts.append(doc_tokens[start_offset:end_offset + 1])
        else:
            doc_tokens = self._cached_doc(int(self.document_index[start_doc_idx]))
            parts.append(doc_tokens[start_offset:])
            for doc_id in range(start_doc_idx + 1, end_doc_idx):
                parts.append(self._cached_doc(int(self.document_index[doc_id])))
            tail_doc_tokens = self._cached_doc(int(self.document_index[end_doc_idx]))
            parts.append(tail_doc_tokens[:end_offset + 1])
        out = np.concatenate(parts) if len(parts) > 1 else parts[0]
        needed = self.seq_length + 1
        if out.size >= needed:
            return out[:needed]
        # Pad short tail (only happens for the very last sample).
        pad = np.full(needed - out.size, self.pad_token_id, dtype=out.dtype)
        return np.concatenate([out, pad])

    def _read_doc(self, doc_id: int) -> np.ndarray:
        """All tokens of ``doc_id`` as a single 1-D array (memoised)."""
        return self.indexed_dataset.get_document(doc_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual = int(self.shuffle_index[idx])
        tokens = self._read_sample(actual)
        input_ids = torch.from_numpy(tokens[: self.seq_length].astype(np.int64))
        labels = input_ids.clone()
        if self.eod_mask_loss and self.eod_token_id is not None:
            labels[input_ids == int(self.eod_token_id)] = -100
        return {"input_ids": input_ids, "labels": labels}
