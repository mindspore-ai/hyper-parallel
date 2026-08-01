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
"""Unit tests for VL HuggingFace data preparation helpers."""
import os
import sys
import types
import unittest

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

_fake_tb_module = types.ModuleType("torch.utils.tensorboard")
_fake_tb_module.SummaryWriter = type("SummaryWriter", (), {})
sys.modules.setdefault("torch.utils.tensorboard", _fake_tb_module)

from hyper_parallel.data.hf import StreamingTokenizedDataset, build_json_file
from hyper_parallel.trainer.vl_trainer import (
    _build_vl_hf_transform,
    _build_vl_messages,
)


class _FakeProcessor:
    """Processor double that records messages and returns deterministic tensors."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    ):
        """Record messages and return deterministic multimodal tensors."""
        self.calls.append(messages)
        del tokenize, add_generation_prompt, return_dict, return_tensors
        return {
            "input_ids": torch.tensor([[101, 102, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "pixel_values": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
        }


class _FakeHFIterable:
    """Minimal iterable row source for dataset wrapper tests."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)


class _FakeHFMapDataset:
    """Minimal map-style dataset double for ``build_json_file`` tests."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def select(self, indices):
        return _FakeHFMapDataset([self.rows[idx] for idx in indices])

    def map(self, fn, batched=False, remove_columns=None, desc=None):
        """Apply a batched transform and return a mapped dataset clone."""
        del remove_columns, desc
        if not batched:
            raise AssertionError("Expected batched=True")
        batch = {key: [row[key] for row in self.rows] for key in self.column_names}
        output = fn(batch)
        keys = list(output.keys())
        mapped_rows = []
        for idx in range(len(output[keys[0]])):
            mapped_rows.append({key: output[key][idx] for key in keys})
        return _FakeHFMapDataset(mapped_rows)

    def filter(self, fn):
        return _FakeHFMapDataset([row for row in self.rows if fn(row)])


class TestVLHFHelpers(unittest.TestCase):
    """Coverage for VL HuggingFace data normalization."""

    def test_build_vl_messages_fills_row_level_image_placeholder(self):
        """Verify row-level image values fill blank image placeholders."""
        cfg = types.SimpleNamespace(
            messages_key="messages",
            image_key="image",
            text_key="text",
        )
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A cat."}],
                },
            ],
            "image": "/tmp/cat.jpg",
        }

        messages = _build_vl_messages(row, cfg)

        self.assertEqual(messages[0]["content"][0]["image"], "/tmp/cat.jpg")
        self.assertEqual(messages[1]["content"][0]["text"], "A cat.")

    def test_build_vl_messages_fallbacks_to_instruction_input_output(self):
        """Verify instruction/input/output rows fall back to chat messages."""
        cfg = types.SimpleNamespace(
            messages_key="messages",
            image_key="image",
            text_key="text",
        )
        row = {
            "instruction": "Describe the picture.",
            "input": "Focus on the main object.",
            "output": "A cat on a sofa.",
            "image": "/tmp/cat.jpg",
        }

        messages = _build_vl_messages(row, cfg)

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertIn("Describe the picture.", messages[0]["content"][1]["text"])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"][0]["text"], "A cat on a sofa.")

    def test_vl_transform_outputs_hf_ready_rows(self):
        """Verify VL transform emits HF-ready multimodal row fields."""
        cfg = types.SimpleNamespace(
            max_seq_len=16,
            messages_key="messages",
            image_key="image",
            text_key="text",
        )
        processor = _FakeProcessor()
        transform = _build_vl_hf_transform(processor, cfg)

        out = transform(
            {
                "messages": [[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe the image."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "A cat."}],
                    },
                ]],
                "image": ["/tmp/cat.jpg"],
            },
        )

        self.assertEqual(processor.calls[0][0]["content"][0]["image"], "/tmp/cat.jpg")
        self.assertEqual(out["input_ids"], [[101, 102, 0]])
        self.assertEqual(out["labels"], [[101, 102, -100]])
        self.assertEqual(out["attention_mask"], [[1, 1, 0]])
        self.assertEqual(out["image_grid_thw"], [[[1, 1, 2]]])

    def test_streaming_wrapper_preserves_multimodal_optional_fields(self):
        """Verify wrapper preserves optional multimodal tensor-like fields."""
        wrapped = StreamingTokenizedDataset(
            _FakeHFIterable(
                [{
                    "input_ids": [1, 2],
                    "labels": [1, 2],
                    "attention_mask": [1, 1],
                    "pixel_values": [[0.1, 0.2]],
                    "image_grid_thw": [[1, 1, 1]],
                }],
            ),
            logical_length=1,
        )

        item = next(iter(wrapped))

        self.assertEqual(tuple(item["input_ids"].shape), (2,))
        self.assertEqual(tuple(item["attention_mask"].shape), (2,))
        self.assertEqual(tuple(item["pixel_values"].shape), (1, 2))
        self.assertEqual(tuple(item["image_grid_thw"].shape), (1, 3))
        self.assertEqual(item["pixel_values"].dtype, torch.float32)

    def test_json_file_builder_preserves_multimodal_fields(self):
        """Verify json-file builder keeps multimodal outputs after mapping."""
        cfg = types.SimpleNamespace(
            max_seq_len=16,
            messages_key="messages",
            image_key="image",
            text_key="text",
            train_path="unused.jsonl",
            subset=None,
            train_size=None,
            streaming=False,
            shuffle=False,
        )
        args = types.SimpleNamespace(
            data=cfg,
            train=types.SimpleNamespace(
                seed=7,
                max_steps=8,
                num_train_epochs=1,
                global_batch_size=1,
            ),
        )
        base = types.SimpleNamespace(state=types.SimpleNamespace(max_steps=8))
        processor = _FakeProcessor()
        transform = _build_vl_hf_transform(processor, cfg)
        fake_rows = [{
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A cat."}],
                },
            ],
            "image": "/tmp/cat.jpg",
        }]
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = lambda *args_, **kwargs_: _FakeHFMapDataset(fake_rows)

        with unittest.mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            dataset = build_json_file(
                base=base,
                args=args,
                data_transform=transform,
                dp_rank=0,
                dp_size=1,
            )

        item = dataset[0]
        self.assertEqual(item["input_ids"].tolist(), [101, 102, 0])
        self.assertEqual(item["labels"].tolist(), [101, 102, -100])
        self.assertEqual(tuple(item["pixel_values"].shape), (2, 2))
        self.assertEqual(tuple(item["image_grid_thw"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
