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
"""Native VLM metadata, sequence configuration, and collator preservation."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.data.batching import FixedBatchDataLoader, build_dataloader
from hyper_parallel.data.vlm.collator import VLMCollator
from hyper_parallel.data.vlm.metadata import vlm_sample_metadata
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import Target
from tests.common.vlm_fixtures import build_image_corpus, vlm_loader_target


class StandaloneMesh:
    """Describe a single CPU DP rank without initializing distributed."""

    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def build_test_loader(dataset: object, **kwargs: object) -> object:
    """Exercise the Trainer builder with the native VLM metadata callback."""
    loaders, samplers = build_dataloader(
        vlm_loader_target(), datasets=(dataset, None, None), collate_fn=VLMCollator(),
        mesh_context=SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=StandaloneMesh()),
        training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=17),
        data_config={"load_balance": "native_batch_sampler", **kwargs},
        max_seq_len=64, metadata_fn=vlm_sample_metadata,
    )
    if samplers != (None, None, None):
        raise AssertionError(f"Collective loader must own sampler state, got {samplers}")
    return loaders[0]


class TestVLMMetadata(unittest.TestCase):
    """Verify full image-text sample integrity without accelerator hardware."""

    def test_cost_uses_raw_patches_and_keeps_padding_capacity(self) -> None:
        """Multi-image, single-image and text-only samples have distinct encoder cost."""
        with TemporaryDirectory() as directory:
            dataset = build_image_corpus(directory, size=4)
            samples = [dataset[index] for index in range(4)]
        for sample, patches in zip(samples, (80, 64, 4, 0)):
            original = {key: value.clone() for key, value in sample.items()}
            metadata = vlm_sample_metadata(sample)
            self.assertEqual(metadata.pack_tokens, 64)
            self.assertEqual(metadata.cost.llm, 64)
            self.assertEqual(metadata.cost.encoder, patches)
            self.assertEqual(int(sample["mm_token_type_ids"].sum()), patches // 4)
            for key in original:
                torch.testing.assert_close(sample[key], original[key], rtol=0, atol=0)

    def test_invalid_native_shapes_fail_before_planning(self) -> None:
        """Reject batched text, non-integral grids and lost/mismatched image payloads."""
        sample = {"input_ids": torch.ones(64), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                  "pixel_values": torch.zeros(4, 12)}
        invalid = (
            ("input_ids", torch.ones(1, 64), "1-D"),
            ("image_grid_thw", torch.ones(1, 2), "num_images"),
            ("image_grid_thw", torch.ones(1, 3), "integer"),
            ("image_grid_thw", torch.tensor([[1, 0, 2]]), "positive"),
            ("pixel_values", torch.zeros(3, 12), "patch count"),
        )
        for key, value, message in invalid:
            with self.subTest(key=key, message=message), self.assertRaisesRegex(ValueError, message):
                vlm_sample_metadata({**sample, key: value})

    def test_native_builder_uses_transform_length_and_original_collator(self) -> None:
        """Keep native per-sample padding, labels and pixel concatenation; do not pack."""
        with TemporaryDirectory() as directory:
            dataset = build_image_corpus(directory, size=4)
            loader = build_test_loader(dataset)
            for _ in range(2):
                actual = next(loader)
                indices = [int(token) - 100 for token in actual["input_ids"][:, 0]]
                expected = VLMCollator()([dataset[index] for index in indices])
                self.assertEqual(actual["input_ids"].shape, (2, 64))
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            with self.assertRaises(StopIteration):
                next(loader)
            with self.assertRaisesRegex(ValueError, "must match"):
                build_test_loader(dataset, seq_length=32)

    def test_base_forwards_explicit_metadata_callback(self) -> None:
        """The shared Trainer does not guess modalities from Dataset class names."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = SimpleNamespace(dataloader=vlm_loader_target(), dataset=SimpleNamespace(data_config={}),
                                      training=SimpleNamespace())
        base.data_transform = SimpleNamespace(max_seq_len=64)
        base.mesh = SimpleNamespace()
        base.collate_fn = VLMCollator()
        with patch("hyper_parallel.trainer.base.build_dataloader", return_value=((None,) * 3, (None,) * 3)) as build:
            base._build_dataloader(metadata_fn=vlm_sample_metadata)
        self.assertIs(build.call_args.kwargs["metadata_fn"], vlm_sample_metadata)
        self.assertEqual(build.call_args.kwargs["max_seq_len"], 64)

    def test_omitted_load_balance_keeps_native_loader(self) -> None:
        """Supplying the VLM callback does not implicitly enable distributed balancing."""
        target = Target(FixedBatchDataLoader, target_path="hyper_parallel.data.batching.FixedBatchDataLoader")
        with TemporaryDirectory() as directory:
            dataset = build_image_corpus(directory, size=4)
            loaders, samplers = build_dataloader(
                target, datasets=(dataset, None, None), collate_fn=VLMCollator(),
                mesh_context=SimpleNamespace(dp_rank=0, dp_size=1),
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=17),
                data_config={"source_type": "online"}, max_seq_len=64, metadata_fn=vlm_sample_metadata,
            )
            self.assertIsInstance(loaders[0], FixedBatchDataLoader)
            self.assertIsNotNone(samplers[0])
            expected = VLMCollator()([dataset[0], dataset[1]])
            actual = next(iter(loaders[0]))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
