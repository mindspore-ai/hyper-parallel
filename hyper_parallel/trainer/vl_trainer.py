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
"""VLTrainer for native Qwen3-VL multimodal training."""
import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from hyper_parallel.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class VLTrainer:
    """Trainer for multimodal Qwen3-VL training (text + image/video).

    Two data paths are supported:
    - ``data.type = "vl_dummy"``: deterministic synthetic multimodal tensors,
      useful for quick smoke tests without dataset preparation.
    - ``data.type = "preset_pt"``: replays pre-tokenized batches that already
      include ``pixel_values`` and ``image_grid_thw``.
    """

    def __init__(self, args):
        self.base = BaseTrainer(args)
        self.base._setup()
        self.base._build_model()
        self.base._freeze_model()
        self._build_model_assets()
        self._build_data_transform()
        self._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()
        self.base.on_init_end()

    def _build_model_assets(self):
        """Load processor when a real VL dataset is configured."""
        self.base.processor = None
        self.base.tokenizer = None
        data_type = getattr(self.base.args.data, "type", "vl_dummy")
        if data_type == "vl_dummy":
            return
        processor_path = (
            getattr(self.base.args.data, "processor_path", None)
            or getattr(self.base.args.model, "tokenizer_path", None)
            or getattr(self.base.args.model, "weights_path", None)
        )
        if not processor_path:
            raise ValueError("VL real-data mode requires data.processor_path or model.weights_path")
        from transformers import AutoProcessor  # pylint: disable=C0415

        self.base.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True,
        )
        self.base.tokenizer = getattr(self.base.processor, "tokenizer", None)
        logger.info("Processor loaded from %s", processor_path)

    def _build_data_transform(self):
        self.base.data_transform = None

    def _build_dataset(self):

        """Build dataset (internal)."""
        data_type = getattr(self.base.args.data, "type", "vl_dummy")
        if data_type == "preset_pt":
            self._build_preset_pt_dataset()
            return
        if data_type != "vl_dummy":
            raise NotImplementedError(
                f"VL trainer supports data.type 'vl_dummy' or 'preset_pt', got '{data_type}'"
            )

        max_steps = getattr(self.base.args.train, "max_steps", 1)
        global_bs = getattr(self.base.args.train, "global_batch_size", 1)
        total_samples = max_steps * global_bs
        data_cfg = self.base.args.data
        model_cfg = self.base.args.model
        extra = getattr(model_cfg, "config_overrides", None) or {}
        vision_extra = extra.get("vision_config", {}) if isinstance(extra, dict) else {}
        patch_size = int(vision_extra.get("patch_size", 16))
        temporal_patch_size = int(vision_extra.get("temporal_patch_size", 2))
        in_channels = int(vision_extra.get("in_channels", 3))
        spatial_merge = int(vision_extra.get("spatial_merge_size", 2))
        grid_t = int(getattr(data_cfg, "vl_grid_t", 2))
        grid_h = int(getattr(data_cfg, "vl_grid_h", 2))
        grid_w = int(getattr(data_cfg, "vl_grid_w", 2))
        image_token_id = int(getattr(data_cfg, "image_token_id", 151655))
        image_tokens = grid_t * grid_h * grid_w // (spatial_merge ** 2)
        row_width = in_channels * temporal_patch_size * patch_size * patch_size
        seq_len = max(int(getattr(data_cfg, "max_seq_len", 16)), image_tokens + 4)
        base_seed = int(getattr(self.base.args, "seed", 42))

        class DeterministicVLDataset(Dataset):
            """Deterministic synthetic VL dataset for smoke / FSDP regression."""

            def __len__(self):
                return total_samples

            def __getitem__(self, idx):
                g = torch.Generator().manual_seed(base_seed + idx)
                pixel_values = torch.randn(
                    grid_t * grid_h * grid_w, row_width, generator=g,
                    dtype=torch.float32,
                )
                input_ids = torch.full((seq_len,), 100, dtype=torch.long)
                input_ids[0] = 151643
                input_ids[1: 1 + image_tokens] = image_token_id
                tail = torch.arange(
                    200 + idx % 17,
                    200 + idx % 17 + seq_len - 1 - image_tokens,
                    dtype=torch.long,
                )
                input_ids[1 + image_tokens:] = tail
                labels = input_ids.clone()
                mm_token_type_ids = torch.zeros(seq_len, dtype=torch.int32)
                mm_token_type_ids[1: 1 + image_tokens] = 1
                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "mm_token_type_ids": mm_token_type_ids,
                    "pixel_values": pixel_values,
                    "image_grid_thw": torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long),
                }

        self.base.train_dataset = DeterministicVLDataset()
        self.base.state.max_steps = max_steps
        logger.info_rank0(
            "VL dummy dataset created: samples=%d seq_len=%d grid=(%d,%d,%d) image_tokens=%d",
            total_samples, seq_len, grid_t, grid_h, grid_w, image_tokens,
        )

    def _build_preset_pt_dataset(self):
        """Replay pre-tokenized VL batches from a .pt file.

        Each entry is a dict of tensors (input_ids/labels/attention_mask plus
        VL fields pixel_values/image_grid_thw/mm_token_type_ids). Sample
        layout follows the same convention as ``llm_trainer._build_preset_pt_dataset``.
        """
        # pylint: disable=C0415

        train_path = self.base.args.data.train_path
        if not train_path:
            raise ValueError("data.train_path is required when data.type='preset_pt'")
        batches = torch.load(train_path, map_location="cpu", weights_only=False)
        if not isinstance(batches, list) or not batches:
            raise ValueError(f"preset_pt expects List, got {type(batches)}")

        def _expand_dict(b: Dict[str, Any]) -> List[Dict[str, Any]]:
            ids = b["input_ids"]
            labels = b["labels"]
            out: List[Dict[str, Any]] = []
            for i in range(ids.shape[0]):
                rec = {"input_ids": ids[i].clone(), "labels": labels[i].clone()}
                for k in ("attention_mask", "mm_token_type_ids"):
                    v = b.get(k)
                    if v is not None and v.dim() >= 2:
                        rec[k] = v[i].clone()
                if b.get("pixel_values") is not None and b.get("image_grid_thw") is not None:
                    pv = b["pixel_values"]
                    thw = b["image_grid_thw"]
                    grids_per_sample = thw.shape[0] // ids.shape[0] if thw.dim() == 2 else 0
                    if grids_per_sample > 0:
                        thw_i = thw[i * grids_per_sample:(i + 1) * grids_per_sample].clone()
                        pv_count = int(thw_i.prod(dim=-1).sum().item())
                        offset = sum(
                            int(thw[j].prod(dim=-1).sum().item())
                            for j in range(i * grids_per_sample)
                        )
                        rec["pixel_values"] = pv[offset:offset + pv_count].clone()
                        rec["image_grid_thw"] = thw_i
                out.append(rec)
            return out

        per_sample = []
        for b in batches:
            if isinstance(b, list):
                for br in b:
                    per_sample.extend(_expand_dict(br))
            else:
                per_sample.extend(_expand_dict(b))

        class PresetPtVLDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        self.base.train_dataset = PresetPtVLDataset(per_sample)
        max_steps = getattr(self.base.args.train, "max_steps", None)
        if max_steps:
            self.base.state.max_steps = int(max_steps)
        logger.info_rank0(
            "preset_pt VL dataset: %d samples loaded from %s", len(per_sample), train_path,
        )

    def _build_collate_fn(self):

        """Build collate fn (internal)."""
        def _vl_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            out = {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
                "labels": torch.stack([x["labels"] for x in batch]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            }
            if "mm_token_type_ids" in batch[0]:
                out["mm_token_type_ids"] = torch.stack([x["mm_token_type_ids"] for x in batch])
            if "pixel_values" in batch[0] and batch[0].get("pixel_values") is not None:
                out["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
            if "image_grid_thw" in batch[0] and batch[0].get("image_grid_thw") is not None:
                grids = [x["image_grid_thw"] for x in batch]
                if grids[0].dim() == 1:
                    out["image_grid_thw"] = torch.stack(grids)
                else:
                    out["image_grid_thw"] = torch.cat(grids, dim=0)
            return out

        self.base.collate_fn = _vl_collate

    def train(self):
        """Run the full training loop by delegating to the underlying BaseTrainer."""
        return self.base.train()
