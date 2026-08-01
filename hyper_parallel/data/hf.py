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
"""HuggingFace ``datasets`` builders (hub / local JSON).

Two ``data.type`` aliases land here:

- ``hf_datasets``: ``load_dataset(train_path, split="train")`` (hub or
  local arrow directory). Tokenises with the spec-provided transform.
- ``json_file``: ``load_dataset("json", data_files=train_path, split="train")``
  — covers Alpaca-style ``instruction / input / output`` files.

The wrapped :class:`TokenizedDataset` returns plain tensor rows so the
trainer's standard padding collator can stack them. Streaming mode uses
HuggingFace's iterable dataset path plus rank-aware self-sharding, so the
trainer can bypass ``DistributedSampler`` and still preserve DP-only data
partitioning.
"""
import logging
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)

_HF_TENSOR_DTYPES = {
    "input_ids": torch.long,
    "labels": torch.long,
    "attention_mask": torch.long,
    "position_ids": torch.long,
    "image_grid_thw": torch.long,
    "video_grid_thw": torch.long,
    "mm_token_type_ids": torch.int32,
    "pixel_values": torch.float32,
    "pixel_values_videos": torch.float32,
}


def _tensorize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert known HF row fields to tensors while preserving scalars."""
    out: dict[str, Any] = {}
    for key, dtype in _HF_TENSOR_DTYPES.items():
        value = item.get(key)
        if value is not None:
            out[key] = torch.as_tensor(value, dtype=dtype)
    if "num_items_in_batch" in item:
        out["num_items_in_batch"] = int(item["num_items_in_batch"])
    return out


class TokenizedDataset(Dataset):
    """Lightweight ``torch.utils.data.Dataset`` view over an HF dataset.

    Each row is converted to ``torch.long`` tensors on demand — the HF
    table stays on disk / memory in arrow format until the dataloader
    actually pulls a sample.
    """

    def __init__(self, hf_ds: Any) -> None:
        self.data = hf_ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return _tensorize_item(self.data[idx])


class StreamingTokenizedDataset(IterableDataset):
    """Streaming ``IterableDataset`` wrapper over an HF iterable dataset.

    The wrapped HF dataset remains the source of truth for cursor/state
    handling. This wrapper only normalizes rows to ``torch.long`` tensors,
    applies an optional logical per-rank sample cap, and forwards
    ``state_dict`` / ``load_state_dict`` so ``StatefulDataLoader`` can resume.

    Args:
        hf_ds: HuggingFace iterable dataset after tokenization / sharding.
        logical_length: Optional per-rank sample cap for one logical epoch.
    """

    def __init__(self, hf_ds: Any, logical_length: Optional[int] = None) -> None:
        self.data = hf_ds
        self.logical_length = logical_length
        self._epoch = 0

    def __iter__(self):
        if hasattr(self.data, "set_epoch"):
            self.data.set_epoch(self._epoch)
        yielded = 0
        while self.logical_length is None or yielded < self.logical_length:
            made_progress = False
            for item in self.data:
                made_progress = True
                if self.logical_length is not None and yielded >= self.logical_length:
                    break
                yielded += 1
                yield _tensorize_item(item)
            if not made_progress:
                break

    def __len__(self) -> int:
        if self.logical_length is None:
            raise TypeError(
                "Streaming dataset has no static length. Set data.train_size "
                "to define a logical per-epoch sample budget."
            )
        return self.logical_length

    def set_epoch(self, epoch: int) -> None:
        """Record the current epoch for deterministic HF shuffle reset."""
        self._epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Return resumable iterator state for ``StatefulDataLoader``."""
        state = {"_epoch": self._epoch}
        if hasattr(self.data, "set_epoch"):
            self.data.set_epoch(self._epoch)
        if hasattr(self.data, "state_dict"):
            state["hf_state"] = self.data.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator state produced by :meth:`state_dict`."""
        self._epoch = int(state.get("_epoch", 0))
        if hasattr(self.data, "set_epoch"):
            self.data.set_epoch(self._epoch)
        hf_state = state.get("hf_state")
        if hf_state is not None and hasattr(self.data, "load_state_dict"):
            self.data.load_state_dict(hf_state)


def _load_raw(args: Any, data_type: str, *, streaming: bool = False) -> Any:
    """Run the appropriate ``load_dataset`` call for ``data_type``."""
    from datasets import load_dataset  # pylint: disable=C0415  # optional dep

    train_path = args.data.train_path
    if not train_path:
        raise ValueError(f"data.train_path is required when data.type='{data_type}'")

    if data_type == "json_file":
        return load_dataset("json", data_files=train_path, split="train", streaming=streaming)
    subset = args.data.subset
    if subset:
        return load_dataset(train_path, subset, split="train", streaming=streaming)
    return load_dataset(train_path, split="train", streaming=streaming)


def _maybe_truncate(ds: Any, args: Any) -> Any:
    """Apply ``data.train_size`` if it shrinks the dataset."""
    train_size = args.data.train_size
    if train_size and train_size < len(ds):
        ds = ds.select(range(train_size))
        logger.info("Dataset truncated to %d samples", train_size)
    return ds


def _maybe_clip_max_steps(*, base: Any, args: Any, global_sample_count: Optional[int]) -> None:
    """Clip ``base.state.max_steps`` when a finite global dataset size is known."""
    if global_sample_count is None:
        return
    steps_per_epoch = global_sample_count // max(args.train.global_batch_size, 1)
    num_epochs = max(int(args.train.num_train_epochs or 1), 1)
    base.state.max_steps = min(args.train.max_steps, num_epochs * steps_per_epoch)


def _per_rank_train_size(train_size: Optional[int], dp_rank: int, dp_size: int) -> Optional[int]:
    """Split a global logical sample budget across DP ranks."""
    if train_size is None:
        return None
    if train_size < 0:
        raise ValueError(f"data.train_size must be >= 0, but got {train_size}")
    base = train_size // max(dp_size, 1)
    remainder = train_size % max(dp_size, 1)
    return base + (1 if dp_rank < remainder else 0)


def _build_hf_streaming(
    *,
    base: Any,
    args: Any,
    data_transform: Any,
    data_type: str,
    dp_rank: int,
    dp_size: int,
    **_: Any,
) -> StreamingTokenizedDataset:
    """Build a streaming HF dataset with DP-rank self-sharding."""
    from datasets.distributed import split_dataset_by_node  # pylint: disable=C0415  # optional dep

    logger.info(
        "Loading streaming dataset: type=%s, path=%s, dp_rank=%d/%d",
        data_type, args.data.train_path, dp_rank, dp_size,
    )
    ds = _load_raw(args, data_type, streaming=True)

    if data_transform is not None:
        map_kwargs = {
            "batched": True,
            "remove_columns": ds.column_names,
            "desc": "Tokenizing",
        }
        try:
            ds = ds.map(data_transform, **map_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'desc'" not in str(exc):
                raise
            map_kwargs.pop("desc", None)
            ds = ds.map(data_transform, **map_kwargs)
    column_names = getattr(ds, "column_names", []) or []
    if "input_ids" in column_names:
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    if args.data.shuffle:
        # Keep a moderate fixed buffer until a dedicated config knob is needed.
        ds = ds.shuffle(seed=int(args.train.seed), buffer_size=10_000)
    ds = split_dataset_by_node(ds, rank=dp_rank, world_size=dp_size)
    if hasattr(ds, "with_format"):
        ds = ds.with_format("torch")

    global_train_size = args.data.train_size
    local_train_size = _per_rank_train_size(global_train_size, dp_rank, dp_size)
    _maybe_clip_max_steps(base=base, args=args, global_sample_count=global_train_size)
    if global_train_size is not None:
        logger.info(
            "Streaming dataset logical epoch budget: global=%d, local=%d",
            global_train_size, local_train_size,
        )
    else:
        logger.info(
            "Streaming dataset uses source exhaustion for epoch boundaries; "
            "set data.train_size to define a logical sample budget."
        )
    return StreamingTokenizedDataset(ds, logical_length=local_train_size)


def _build_hf(
    *,
    base: Any,
    args: Any,
    data_transform: Any,
    data_type: str,
    dp_rank: int = 0,
    dp_size: int = 1,
    **_: Any,
) -> Dataset:
    """Shared loader for ``hf_datasets`` and ``json_file``.

    Tokenises via ``data_transform`` when provided, then filters away empty
    sequences and wraps in :class:`TokenizedDataset`. Updates
    ``base.state.max_steps`` to ``min(cfg.max_steps, len/global_bs)`` so
    epoch boundaries stay consistent with the data on hand.
    """
    if args.data.streaming:
        return _build_hf_streaming(
            base=base,
            args=args,
            data_transform=data_transform,
            data_type=data_type,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    logger.info(
        "Loading dataset: type=%s, path=%s", data_type, args.data.train_path,
    )
    ds = _maybe_truncate(_load_raw(args, data_type, streaming=False), args)

    if data_transform is not None:
        ds = ds.map(
            data_transform,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )
    # Drop empty rows only when the dataset is already tokenized — without a
    # transform a raw text dataset has no ``input_ids`` column and the filter
    # predicate would raise ``KeyError`` instead of loading.
    if "input_ids" in ds.column_names:
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0)

    wrapped = TokenizedDataset(ds)
    # max_steps clipping must happen pre-dataloader so the train loop's
    # epoch count matches the data on hand; the trainer reads
    # ``base.state.max_steps`` further down the build chain. The cap is
    # the TOTAL step budget across all epochs — clipping to a single
    # epoch would silently truncate multi-epoch training.
    _maybe_clip_max_steps(base=base, args=args, global_sample_count=len(wrapped))
    logger.info(
        "Dataset ready: %d samples, max_steps=%d",
        len(wrapped), base.state.max_steps,
    )
    return wrapped


@DATASET_REGISTRY.register("hf_datasets")
def build_hf_datasets(**kwargs: Any) -> Dataset:
    """Build an HF hub / arrow dataset."""
    return _build_hf(data_type="hf_datasets", **kwargs)


@DATASET_REGISTRY.register("json_file")
def build_json_file(**kwargs: Any) -> Dataset:
    """Build a local Alpaca-style ``.json`` / ``.jsonl`` dataset."""
    return _build_hf(data_type="json_file", **kwargs)
