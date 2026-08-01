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


def _as_media_list(value: Any) -> List[Any]:
    """Normalize one-or-many media references to a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _iter_batched_rows(examples: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Convert an HF ``map(batched=True)`` payload into row dictionaries."""
    if not examples:
        return []
    first_key = next(iter(examples))
    size = len(examples[first_key])
    return [{key: value[idx] for key, value in examples.items()} for idx in range(size)]


def _build_vl_messages(row: Dict[str, Any], data_cfg: Any) -> List[Dict[str, Any]]:
    """Normalize one dataset row into a multimodal chat-template message list."""
    messages_key = getattr(data_cfg, "messages_key", "messages")
    image_key = getattr(data_cfg, "image_key", "image")
    text_key = getattr(data_cfg, "text_key", "text")

    image_queue = _as_media_list(
        row.get(image_key, row.get("images", row.get("image"))),
    )
    video_queue = _as_media_list(row.get("videos", row.get("video")))

    def _normalize_content(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError(
                "VL HuggingFace rows must provide list-style message content "
                f"or plain text, but got {type(content)}."
            )
        out: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                out.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                raise ValueError(
                    "VL message content items must be dict/str, "
                    f"but got {type(item)}."
                )
            item_type = item.get("type", "text")
            if item_type == "text":
                out.append({"type": "text", "text": item.get("text", item.get("content", ""))})
                continue
            if item_type == "image":
                image = item.get("image", item.get("path", item.get("url")))
                if image is None:
                    if not image_queue:
                        raise ValueError(
                            "VL message declared an image placeholder but no "
                            f"row-level media was found in '{image_key}'."
                        )
                    image = image_queue.pop(0)
                out.append({"type": "image", "image": image})
                continue
            if item_type == "video":
                video = item.get("video", item.get("path", item.get("url")))
                if video is None:
                    if not video_queue:
                        raise ValueError(
                            "VL message declared a video placeholder but no "
                            "row-level video field was found."
                        )
                    video = video_queue.pop(0)
                out.append({"type": "video", "video": video})
                continue
            raise ValueError(f"Unsupported VL content type: {item_type}")
        return out

    if row.get(messages_key) is not None:
        messages = row[messages_key]
        if not isinstance(messages, list):
            raise ValueError(
                f"data.messages_key='{messages_key}' must point to a list, "
                f"but got {type(messages)}."
            )
        return [
            {
                "role": message.get("role", "user"),
                "content": _normalize_content(message.get("content", "")),
            }
            for message in messages
        ]

    user_text = row.get(text_key)
    if user_text is None and row.get("instruction") is not None:
        user_text = str(row["instruction"])
        if row.get("input"):
            user_text += f"\n{row['input']}"
    assistant_text = row.get("output")

    user_content: List[Dict[str, Any]] = []
    for image in image_queue:
        user_content.append({"type": "image", "image": image})
    for video in video_queue:
        user_content.append({"type": "video", "video": video})
    if user_text is not None:
        user_content.append({"type": "text", "text": str(user_text)})
    if not user_content:
        raise ValueError(
            "VL HuggingFace row must provide either messages, text/instruction, "
            "or media fields."
        )

    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]
    if assistant_text is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(assistant_text)}]},
        )
    return messages


def _normalize_vl_processor_output(processed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert processor tensors to per-sample Python lists."""

    def _squeeze_sequence(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.dim() == 2 and value.shape[0] == 1:
                return value[0]
            if value.dim() == 3 and value.shape[1] == 1:
                return value[:, 0]
        return value

    def _squeeze_media(value: Any) -> Any:
        if torch.is_tensor(value) and value.dim() >= 3 and value.shape[0] == 1:
            return value[0]
        return value

    def _to_python(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str)):
            return value.tolist()
        return value

    input_ids = _squeeze_sequence(processed["input_ids"])
    attention_mask = _squeeze_sequence(processed.get("attention_mask"))
    input_ids_list = _to_python(input_ids)
    record = {
        "input_ids": input_ids_list,
        "labels": list(input_ids_list),
    }
    if attention_mask is not None:
        attention_mask_list = _to_python(attention_mask)
        record["attention_mask"] = attention_mask_list
        record["labels"] = [
            token if mask else -100
            for token, mask in zip(record["labels"], attention_mask_list)
        ]

    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "position_ids",
        "mm_token_type_ids",
    ):
        value = processed.get(key)
        if value is None:
            continue
        if key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            value = _squeeze_media(value)
        else:
            value = _squeeze_sequence(value)
        record[key] = _to_python(value)
    return record


def _build_vl_hf_transform(processor: Any, data_cfg: Any):
    """Create the per-row HuggingFace map transform for real VL datasets."""
    max_seq_len = int(getattr(data_cfg, "max_seq_len", 0) or 0)

    def _transform(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = _iter_batched_rows(examples)
        batch_out: Dict[str, List[Any]] = {}
        for row in rows:
            messages = _build_vl_messages(row, data_cfg)
            processed = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            record = _normalize_vl_processor_output(processed)
            if max_seq_len and len(record["input_ids"]) > max_seq_len:
                raise ValueError(
                    "VL HuggingFace sample length exceeds data.max_seq_len. "
                    "Automatic truncation is disabled because it can desync "
                    "vision placeholders and media features."
                )
            for key, value in record.items():
                batch_out.setdefault(key, []).append(value)
        return batch_out

    return _transform


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
        if self.base.processor is None:
            self.base.data_transform = None
            return
        self.base.data_transform = _build_vl_hf_transform(
            self.base.processor, self.base.args.data,
        )

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
