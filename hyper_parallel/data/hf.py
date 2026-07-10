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
trainer's standard padding collator can stack them. ``max_steps``
is auto-clipped to ``num_train_epochs * floor(len / global_batch_size)``
so the dataloader never runs dry mid-epoch.
"""
import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)


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
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def _load_raw(args: Any, data_type: str) -> Any:
    """Run the appropriate ``load_dataset`` call for ``data_type``."""
    from datasets import load_dataset  # pylint: disable=C0415  # optional dep

    train_path = args.data.train_path
    if not train_path:
        raise ValueError(f"data.train_path is required when data.type='{data_type}'")

    if data_type == "json_file":
        return load_dataset("json", data_files=train_path, split="train")
    subset = args.data.subset
    if subset:
        return load_dataset(train_path, subset, split="train")
    return load_dataset(train_path, split="train")


def _maybe_truncate(ds: Any, args: Any) -> Any:
    """Apply ``data.train_size`` if it shrinks the dataset."""
    train_size = args.data.train_size
    if train_size and train_size < len(ds):
        ds = ds.select(range(train_size))
        logger.info("Dataset truncated to %d samples", train_size)
    return ds


def _build_hf(*, base: Any, args: Any, data_transform: Any, data_type: str, **_: Any) -> TokenizedDataset:
    """Shared loader for ``hf_datasets`` and ``json_file``.

    Tokenises via ``data_transform`` when provided, then filters away empty
    sequences and wraps in :class:`TokenizedDataset`. Updates
    ``base.state.max_steps`` to ``min(cfg.max_steps, len/global_bs)`` so
    epoch boundaries stay consistent with the data on hand.
    """
    logger.info(
        "Loading dataset: type=%s, path=%s", data_type, args.data.train_path,
    )
    ds = _maybe_truncate(_load_raw(args, data_type), args)

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
    num_epochs = max(int(args.train.num_train_epochs or 1), 1)
    base.state.max_steps = min(
        args.train.max_steps,
        num_epochs * (len(wrapped) // max(args.train.global_batch_size, 1)),
    )
    logger.info(
        "Dataset ready: %d samples, max_steps=%d",
        len(wrapped), base.state.max_steps,
    )
    return wrapped


@DATASET_REGISTRY.register("hf_datasets")
def build_hf_datasets(**kwargs: Any) -> TokenizedDataset:
    """Build an HF hub / arrow dataset."""
    return _build_hf(data_type="hf_datasets", **kwargs)


@DATASET_REGISTRY.register("json_file")
def build_json_file(**kwargs: Any) -> TokenizedDataset:
    """Build a local Alpaca-style ``.json`` / ``.jsonl`` dataset."""
    return _build_hf(data_type="json_file", **kwargs)
