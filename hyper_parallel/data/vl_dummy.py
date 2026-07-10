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
"""``vl_dummy`` — synthetic VL dataset for native Qwen3-VL smoke tests.

Each sample's pixel grid, image-token slice and trailing token stream are
seeded by ``train.seed + idx`` so 1-card vs N-card runs see identical content
after the distributed sampler interleaves them.
"""
import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)


class DummyVLDataset(Dataset):
    """Synthetic multimodal samples with stable shapes per index."""

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        row_width: int,
        image_tokens: int,
        image_token_id: int,
        base_seed: int,
        video_token_id: int = 151656,
        is_video: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.grid_t = grid_t
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.row_width = row_width
        self.image_tokens = image_tokens
        self.image_token_id = image_token_id
        self.base_seed = base_seed
        self.video_token_id = video_token_id
        self.is_video = is_video

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.base_seed + idx)
        if self.is_video:
            return self._video_item(idx, g)
        pixel_values = torch.randn(
            self.grid_t * self.grid_h * self.grid_w, self.row_width,
            generator=g, dtype=torch.float32,
        )
        input_ids = torch.full((self.seq_length,), 100, dtype=torch.long)
        input_ids[0] = 151643
        input_ids[1: 1 + self.image_tokens] = self.image_token_id
        tail = torch.arange(
            200 + idx % 17,
            200 + idx % 17 + self.seq_length - 1 - self.image_tokens,
            dtype=torch.long,
        )
        input_ids[1 + self.image_tokens:] = tail
        labels = input_ids.clone()
        mm_token_type_ids = torch.zeros(self.seq_length, dtype=torch.int32)
        mm_token_type_ids[1: 1 + self.image_tokens] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(
                [self.grid_t, self.grid_h, self.grid_w], dtype=torch.long,
            ),
        }

    def _video_item(self, idx: int, g: torch.Generator):
        """Build one deterministic video sample."""
        # A video lays each of grid_t frames as its own mm==2 run (a text
        # separator between frames) so get_rope_index consumes one per-frame
        # [1,h,w] grid per run; an image is a single contiguous run.
        tokens_per_frame = self.image_tokens // self.grid_t
        pixel_values_videos = torch.randn(
            self.grid_t * self.grid_h * self.grid_w, self.row_width,
            generator=g, dtype=torch.float32,
        )
        input_ids = torch.full((self.seq_length,), 100, dtype=torch.long)
        input_ids[0] = 151643
        mm_token_type_ids = torch.zeros(self.seq_length, dtype=torch.int32)
        pos = 1
        frame_idx = 0
        while frame_idx < self.grid_t:
            input_ids[pos] = 150  # text separator → split per-frame runs
            pos += 1
            input_ids[pos: pos + tokens_per_frame] = self.video_token_id
            mm_token_type_ids[pos: pos + tokens_per_frame] = 2
            pos += tokens_per_frame
            frame_idx += 1
        tail = torch.arange(
            200 + idx % 17, 200 + idx % 17 + self.seq_length - pos,
            dtype=torch.long,
        )
        input_ids[pos:] = tail
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": torch.tensor(
                [self.grid_t, self.grid_h, self.grid_w], dtype=torch.long,
            ),
        }


@DATASET_REGISTRY.register("vl_dummy")
def build_vl_dummy(*, base: Any, args: Any, **_: Any) -> DummyVLDataset:
    """Build the deterministic VL dummy dataset.

    Pulls ``patch_size`` / ``temporal_patch_size`` / ``in_channels`` /
    ``spatial_merge_size`` from ``model.config_overrides.vision_config`` so
    the row width matches the Qwen3-VL vision-tower expectation; falls back
    to the published defaults when those keys are absent.
    """
    max_steps = args.train.max_steps
    global_bs = args.train.global_batch_size
    total_samples = max_steps * global_bs

    data_cfg = args.data
    model_cfg = args.model
    extra = model_cfg.config_overrides or {}
    vision_extra = extra.get("vision_config", {}) if isinstance(extra, dict) else {}
    patch_size = int(vision_extra.get("patch_size", 16))
    temporal_patch_size = int(vision_extra.get("temporal_patch_size", 2))
    in_channels = int(vision_extra.get("in_channels", 3))
    spatial_merge = int(vision_extra.get("spatial_merge_size", 2))
    grid_t = int(data_cfg.vl_grid_t)
    grid_h = int(data_cfg.vl_grid_h)
    grid_w = int(data_cfg.vl_grid_w)
    image_token_id = int(data_cfg.image_token_id)
    video_token_id = int(data_cfg.video_token_id)
    is_video = bool(data_cfg.vl_video)
    image_tokens = grid_t * grid_h * grid_w // (spatial_merge ** 2)
    tokens_per_frame = grid_h * grid_w // (spatial_merge ** 2)
    row_width = in_channels * temporal_patch_size * patch_size * patch_size
    min_len = grid_t * (tokens_per_frame + 1) + 2 if is_video else image_tokens + 4
    seq_len = max(int(data_cfg.max_seq_len), min_len)
    base_seed = int(args.train.seed)

    ds = DummyVLDataset(
        num_samples=total_samples, seq_length=seq_len,
        grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, row_width=row_width,
        image_tokens=image_tokens, image_token_id=image_token_id, base_seed=base_seed,
        video_token_id=video_token_id, is_video=is_video,
    )
    base.state.max_steps = max_steps
    logger.info(
        "VL dummy dataset created: samples=%d seq_len=%d grid=(%d,%d,%d) image_tokens=%d",
        total_samples, seq_len, grid_t, grid_h, grid_w, image_tokens,
    )
    return ds
