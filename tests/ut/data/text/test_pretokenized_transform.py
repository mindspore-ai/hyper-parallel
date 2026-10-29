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
"""Prepared-record contracts shared by pretraining, SFT and JT inputs."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, load_from_disk
import numpy as np
import torch

from hyper_parallel.data.text import PreTokenizedTransform
from hyper_parallel.data.online.mapping import MappingTransformDataset


class TestPreTokenizedTransform(unittest.TestCase):
    """Validate target alignment, boundaries and source-buffer ownership."""

    def test_owner_record_is_not_shifted_or_retokenized(self):
        """Preserve the owner's next-token targets and two internal documents."""
        record = {"input_ids": [11, 12, 21, 22, 31, 32],
                  "labels": [-100, 21, 22, 2, 32, 2], "boundaries": [0, 4, 6]}
        sample = PreTokenizedTransform()(record)[0]
        self.assertEqual(sample["input_ids"].tolist(), record["input_ids"])
        self.assertEqual(sample["labels"].tolist(), record["labels"])
        self.assertEqual(sample["cu_seq_lens"].tolist(), [0, 4, 6])
        self.assertEqual(sample["input_ids"].dtype, torch.int64)
        self.assertEqual(sample["cu_seq_lens"].dtype, torch.int32)

    def test_jt_arrow_fields_mask_and_readonly_buffers(self):
        """JT field mapping and binary masks do not mutate Arrow-like buffers."""
        tokens = np.array([11, 12, 13, 14], dtype=np.uint16)
        labels = np.array([12, 13, 2, 2], dtype=np.int32)
        tokens.flags.writeable = labels.flags.writeable = False
        record = {"tokens": tokens, "targets": labels, "cu_seqlens": [0, 3, 4],
                  "loss_mask": [0, 1, 1, 0]}
        transform = PreTokenizedTransform(input_ids_key="tokens", labels_key="targets",
                                         boundaries_key="cu_seqlens")
        self.assertTrue(transform.is_valid_sample(record))
        np.testing.assert_array_equal(labels, [12, 13, 2, 2])
        sample = transform(record)[0]
        self.assertEqual(sample["labels"].tolist(), [-100, 13, 2, -100])
        sample["input_ids"].zero_()
        sample["labels"].zero_()
        np.testing.assert_array_equal(tokens, [11, 12, 13, 14])
        np.testing.assert_array_equal(labels, [12, 13, 2, 2])

    def test_unpacked_pretraining_and_mapping_dataset(self):
        """Prepared pretraining records need labels but no tokenizer or SFT fields."""
        record = {"input_ids": [11, 12, 13], "labels": [12, 13, 2]}
        dataset = MappingTransformDataset([record], PreTokenizedTransform())
        self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 3])
        self.assertEqual(dataset[0]["labels"].tolist(), [12, 13, 2])

    def test_empty_targets_are_strict_by_default_and_optionally_filtered(self):
        """Published records fail fast; JT can explicitly retain drop semantics."""
        record = {"input_ids": [1, 2], "labels": [2, 3], "loss_mask": [0, 0]}
        with self.assertRaisesRegex(ValueError, "no effective targets"):
            PreTokenizedTransform().is_valid_sample(record)
        transform = PreTokenizedTransform(empty_target_policy="drop")
        self.assertFalse(transform.is_valid_sample(record))
        self.assertEqual(transform(record), [])

    def test_configured_boundary_field_controls_document_layout(self):
        """Only the configured field determines boundaries; absence means one document."""
        record = {"input_ids": [1, 2], "labels": [2, 3], "cu_seqlens": [0, 1, 2]}
        for field, expected in (("boundaries", [0, 2]), ("cu_seqlens", [0, 1, 2])):
            with self.subTest(field=field):
                transform = PreTokenizedTransform(boundaries_key=field)
                self.assertTrue(transform.is_valid_sample(record))
                self.assertEqual(transform(record)[0]["cu_seq_lens"].tolist(), expected)

    def test_invalid_fields_and_boundaries(self):
        """Reject malformed records rather than silently casting or masking them."""
        base = {"input_ids": [1, 2], "labels": [2, 3], "boundaries": [0, 2]}
        invalid = [
            {"input_ids": [1.0, 2.0]}, {"input_ids": [-1, 2]}, {"labels": [2]},
            {"labels": [-1, 3]}, {"labels": [2**32, 3]}, {"input_ids": [True, False]},
            {"boundaries": [0, 1, 1, 2]}, {"boundaries": [1, 2]}, {"boundaries": [0, 3]},
            {"boundaries": [[0, 2]]}, {"boundaries": []}, {"boundaries": None},
            {"loss_mask": [0.5, 1]}, {"loss_mask": [1]},
        ]
        for update in invalid:
            transform = PreTokenizedTransform()
            for read in (transform, transform.is_valid_sample):
                with self.subTest(update=update, read=read), self.assertRaises(ValueError):
                    read({**base, **update})
        with self.assertRaisesRegex(KeyError, "labels"):
            PreTokenizedTransform()({"input_ids": [1, 2]})

    def test_configuration_and_vocabulary_validation(self):
        """Require an explicit supported convention and honor optional vocab bounds."""
        for options in ({"labels_alignment": "unshifted"},
                        {"empty_target_policy": "ignore"}, {"vocab_size": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PreTokenizedTransform(**options)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            PreTokenizedTransform(vocab_size=3)({"input_ids": [1, 2], "labels": [2, 3]})

    def test_jt_hf_arrow_save_load_and_filter(self):
        """Read JT-style saved Arrow records without regenerating the dataset."""
        records = [
            {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5],
             "cu_seqlens": [0, 2, 4], "loss_mask": [0, 1, 1, 1]},
            {"input_ids": [5, 6], "labels": [6, 7], "cu_seqlens": [0, 2], "loss_mask": [0, 0]},
        ]
        transform = PreTokenizedTransform(boundaries_key="cu_seqlens", empty_target_policy="drop")
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "prepared")
            Dataset.from_list(records).save_to_disk(path)
            source = load_from_disk(path)
            filtered = source.filter(transform.is_valid_sample)
            dataset = MappingTransformDataset(filtered, transform)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["labels"].tolist(), [-100, 3, 4, 5])
            self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 2, 4])
            self.assertEqual(source[0]["labels"], [2, 3, 4, 5])
            # Release Arrow mmaps before deleting files on NFS-backed test roots.
            del dataset, filtered, source
