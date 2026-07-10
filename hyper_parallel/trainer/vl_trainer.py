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

from hyper_parallel.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class VLTrainer:
    """Trainer for multimodal Qwen3-VL training (text + image/video).

    Dataset construction delegates to :func:`hyper_parallel.data.build_dataset`.
    Built-in VL formats are ``vl_dummy`` (deterministic synthetic multimodal
    tensors) and ``preset_pt`` (replayed batches that already include
    ``pixel_values`` and ``image_grid_thw``).
    """

    def __init__(self, args):
        self.base = BaseTrainer(args)
        self.base._setup()
        self.base._build_model()
        self.base._freeze_model()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
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
        data_type = self.base.args.data.type
        if data_type == "vl_dummy":
            return
        processor_path = (
            getattr(self.base.args.data, "processor_path", None)
            or self.base.args.model.tokenizer_path
            or self.base.args.model.weights_path
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

    @staticmethod
    def _stack_positions(batch: List[Dict[str, Any]], key: str):
        values = [item[key] for item in batch]
        if values[0].dim() == 1:
            return torch.stack(values)
        return torch.stack(values).transpose(0, 1).contiguous()

    @staticmethod
    def _stack_or_cat_grids(batch: List[Dict[str, Any]], key: str):
        values = [item[key] for item in batch]
        if values[0].dim() == 1:
            return torch.stack(values)
        return torch.cat(values, dim=0)

    @staticmethod
    def _maybe_cat_optional(batch: List[Dict[str, Any]], key: str):
        if key in batch[0] and batch[0].get(key) is not None:
            return torch.cat([item[key] for item in batch], dim=0)
        return None

    def _vl_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate VL tensor rows into a trainer batch."""
        out = {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }
        if "num_items_in_batch" in batch[0]:
            out["num_items_in_batch"] = sum(int(item["num_items_in_batch"]) for item in batch)
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = torch.stack([item["mm_token_type_ids"] for item in batch])
        if "position_ids" in batch[0]:
            out["position_ids"] = self._stack_positions(batch, "position_ids")
        for key in ("pixel_values", "pixel_values_videos"):
            value = self._maybe_cat_optional(batch, key)
            if value is not None:
                out[key] = value
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in batch[0] and batch[0].get(key) is not None:
                out[key] = self._stack_or_cat_grids(batch, key)
        return out

    def _build_collate_fn(self):
        """Build collate fn (internal)."""
        self.base.collate_fn = self._vl_collate

    def train(self):
        """Run the full training loop by delegating to the underlying BaseTrainer."""
        return self.base.train()
