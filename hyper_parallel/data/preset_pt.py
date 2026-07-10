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
"""``preset_pt`` dataset for replaying pre-tokenized ``.pt`` batches.

Each entry is either a stacked batch dict ``{key: (B, S)-Tensor}`` or a list
of per-rank dicts. The loader flattens both forms into per-sample rows so the
standard ``DataLoader`` can batch them again with the trainer collator.
"""
import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)


class PresetPtDataset(Dataset):
    """Wrap a pre-expanded list of per-sample dicts."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


_OPTIONAL_2D_FIELDS = ("attention_mask", "mm_token_type_ids")

_PIXEL_PAIRS = (("pixel_values", "image_grid_thw"),
                ("pixel_values_videos", "video_grid_thw"))


def _split_pixel_block(
    b: Dict[str, Any], i: int, batch_size: int, pix_key: str, grid_key: str,
) -> Dict[str, Any]:
    """Slice the ``(pix_key, grid_key)`` rows owned by sample ``i``."""
    pv = b[pix_key]
    thw = b[grid_key]
    grids_per_sample = thw.shape[0] // batch_size if thw.dim() == 2 else 0
    if grids_per_sample == 0:
        return {}
    thw_i = thw[i * grids_per_sample:(i + 1) * grids_per_sample].clone()
    pv_count = int(thw_i.prod(dim=-1).sum().item())
    offset = sum(
        int(thw[j].prod(dim=-1).sum().item())
        for j in range(i * grids_per_sample)
    )
    return {
        pix_key: pv[offset:offset + pv_count].clone(),
        grid_key: thw_i,
    }


def _slice_position_ids(position_ids: Any, sample_idx: int, batch_size: int) -> Any:
    """Return one sample's position ids from an LF/HF stacked batch.

    Text Qwen3.5 batches may carry either ``[B, S]`` plain position ids or
    mRoPE ``[R, B, S]`` ids (``R`` is 3 or 4 in Transformers). Preserve the
    rotary-rank axis and slice only the batch axis so the trainer can rebuild
    the original stacked shape in its collate function.
    """
    if position_ids.dim() == 2 and position_ids.shape[0] == batch_size:
        return position_ids[sample_idx].clone()
    if position_ids.dim() == 3 and position_ids.shape[1] == batch_size:
        return position_ids[:, sample_idx].clone()
    if position_ids.dim() == 3 and position_ids.shape[0] == batch_size:
        return position_ids[sample_idx].clone()
    raise ValueError(
        "preset_pt position_ids must be [B, S], [R, B, S], or [B, R, S]; "
        f"got shape={tuple(position_ids.shape)} with batch_size={batch_size}."
    )


def _expand_batch(b: Dict[str, Any], *, vl: bool) -> List[Dict[str, Any]]:
    """Split a stacked LM/VL batch dict into per-sample dicts.

    LM samples carry only ``input_ids`` / ``labels`` (plus optionally
    ``attention_mask``). VL samples additionally carry ``mm_token_type_ids``
    and ``(pixel_values, image_grid_thw)`` / ``(pixel_values_videos,
    video_grid_thw)`` pairs sliced according to the per-sample grid product.
    """
    ids = b["input_ids"]
    labels = b["labels"]
    batch_size = ids.shape[0]
    out: List[Dict[str, Any]] = []
    for i in range(batch_size):
        rec: Dict[str, Any] = {
            "input_ids": ids[i].clone(),
            "labels": labels[i].clone(),
        }
        if "num_items_in_batch" in b:
            rec["num_items_in_batch"] = int((rec["labels"] != -100).sum().item())
        for k in _OPTIONAL_2D_FIELDS:
            v = b.get(k)
            if v is not None and v.dim() == 2:
                rec[k] = v[i].clone()
        position_ids = b.get("position_ids")
        if position_ids is not None:
            rec["position_ids"] = _slice_position_ids(position_ids, i, batch_size)
        if vl:
            for pix_key, grid_key in _PIXEL_PAIRS:
                if b.get(pix_key) is not None and b.get(grid_key) is not None:
                    rec.update(_split_pixel_block(b, i, batch_size, pix_key, grid_key))
        out.append(rec)
    return out


def _is_vl(batch_entry: Any) -> bool:
    """Heuristic: VL batches always carry pixel data (image or video)."""
    if isinstance(batch_entry, list):
        batch_entry = batch_entry[0] if batch_entry else None
    return isinstance(batch_entry, dict) and any(
        pix_key in batch_entry for pix_key, _ in _PIXEL_PAIRS)


@DATASET_REGISTRY.register("preset_pt")
def build_preset_pt(*, base: Any, args: Any, **_: Any) -> PresetPtDataset:
    """Build the preset replay dataset.

    Auto-detects whether the .pt holds VL batches (pixel_values present)
    or plain LM batches by inspecting the first entry, and dispatches the
    matching per-sample expansion.
    """
    train_path = args.data.train_path
    if not train_path:
        raise ValueError("data.train_path is required when data.type='preset_pt'")
    batches = torch.load(train_path, map_location="cpu", weights_only=False)
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"preset_pt expects List, got {type(batches)}")

    vl = _is_vl(batches[0])
    per_sample: List[Dict[str, Any]] = []
    for b in batches:
        if isinstance(b, list):
            for br in b:
                per_sample.extend(_expand_batch(br, vl=vl))
        else:
            per_sample.extend(_expand_batch(b, vl=vl))

    ds = PresetPtDataset(per_sample)
    if args.train.max_steps:
        base.state.max_steps = int(args.train.max_steps)
    logger.info(
        "preset_pt dataset (%s): %d samples loaded from %s",
        "vl" if vl else "lm", len(per_sample), train_path,
    )
    return ds
