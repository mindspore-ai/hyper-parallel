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
"""Real Parquet source tests, including embedded image bytes and worker replay."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from hyper_parallel.data.omni.build_dataset import build_online_omni_mapping_dataset
from hyper_parallel.data.omni.omni_transform import OmniDataTransform
from tests.common.mark_utils import arg_mark
from tests.ut.data.omni.test_omni_transform import image_bytes


class PreparedTransform(OmniDataTransform):
    """Return prepared messages to inspect source-to-processor inputs."""

    def __init__(self) -> None:
        """Use the normal eager lifecycle without needing a downloaded processor."""
        super().__init__(max_seq_len=4096, processor=object())

    def encode_sample(self, sample: dict) -> dict:
        """Prepare images while keeping source identifiers available for assertions."""
        return {**sample, "messages": self.prepare_messages(sample)}


class TestOmniParquet(unittest.TestCase):
    """Keep Parquet reads lazy and source indices deterministic."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_row_groups_filter_order_bytes_and_pickle(self):
        """Verify random access and worker replay.

        Feature: Parquet mapping sources.
        Description: Read filtered rows across groups and spawned workers.
        Expectation: Source order and embedded bytes remain unchanged.
        """
        records = [{"id": index, "messages": [{"role": "user", "content": "<image>"}],
                    "images": [{"bytes": image_bytes(color), "path": None}]}
                   for index, color in enumerate(("red", "green", "blue", "white"))]
        records[1]["messages"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.parquet"
            pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)
            dataset = build_online_omni_mapping_dataset(data_path=str(path), data_config={"shuffle": False, "cache_dir": str(Path(tmp) / "cache")}, transform=PreparedTransform())
            self.assertEqual(len(dataset), 3)
            for index, source_id in ((2, 3), (0, 0), (1, 2), (-1, 3)):
                row = dataset[index]
                self.assertEqual(row["id"], source_id)
                self.assertEqual(row["__online_source_path__"], str(path))
                self.assertEqual(row["images"][0]["bytes"], records[source_id]["images"][0]["bytes"])
                self.assertEqual(row["messages"][0]["content"][0]["image"].mode, "RGB")
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored[1]["id"], 2)
            loader = DataLoader(dataset, batch_size=None, num_workers=2, multiprocessing_context="spawn")
            self.assertEqual([row["id"] for row in loader], [0, 2, 3])
            with self.assertRaises(IndexError):
                _ = dataset[3]
            row = dataset[0]
            row["messages"][0]["content"][0]["image"].putpixel((0, 0), (0, 0, 0))
            self.assertEqual(dataset[0]["messages"][0]["content"][0]["image"].getpixel((0, 0)), (255, 0, 0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_parquet_split_builder_and_relative_paths(self):
        """Verify explicit split paths.

        Feature: Parquet split builder.
        Description: Build train and valid sources with relative image paths.
        Expectation: Paths resolve relative to the Parquet file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            records = [{"messages": [{"role": "user", "content": "<image>"}],
                        "images": [{"bytes": None, "path": "images/a.png"}]}]
            pq.write_table(pa.Table.from_pylist(records), path)
            train, valid, test = build_online_omni_mapping_dataset(
                data_path={"train": str(path), "valid": str(path)}, data_config={"cache_dir": str(Path(tmp) / "cache")}, transform=PreparedTransform(),
            )
            self.assertIsNone(test)
            self.assertEqual(len(train), len(valid))
            self.assertEqual(train[0]["messages"][0]["content"][0]["image"], str(Path(tmp) / "images/a.png"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_valid_rows_returns_empty_source(self):
        """Verify the empty-source contract.

        Feature: Parquet source validation.
        Description: Build a source containing only empty conversations.
        Expectation: Construction preserves the shared source empty-dataset contract.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.parquet"
            pq.write_table(pa.Table.from_pylist([{"messages": []}]), path)
            dataset = build_online_omni_mapping_dataset(
                data_path=str(path), data_config={"shuffle": False, "cache_dir": str(Path(tmp) / "cache")},
                transform=PreparedTransform(),
            )
            self.assertIsNone(dataset)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_image_decoding_is_deferred_until_getitem(self):
        """Verify lazy media decoding.

        Feature: Parquet media lifecycle.
        Description: Track image decoding during construction and item access.
        Expectation: Only item access invokes image decoding.
        """
        record = {"messages": [{"content": "<image>"}], "images": [{"bytes": image_bytes("red")}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.parquet"
            pq.write_table(pa.Table.from_pylist([record]), path)
            with patch("PIL.Image.open", side_effect=OSError("decode reached")) as decode:
                dataset = build_online_omni_mapping_dataset(data_path=str(path), data_config={"shuffle": False, "cache_dir": str(Path(tmp) / "cache")}, transform=PreparedTransform())
                decode.assert_not_called()
                with self.assertRaisesRegex(ValueError, r"images\[0\]"):
                    _ = dataset[0]
                decode.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_multiple_file_roots_survive_ratio_split(self):
        """Verify file provenance across source composition.

        Feature: Multi-file Parquet source paths.
        Description: Split files in different directories after filtering.
        Expectation: Each image resolves against its own source directory.
        """
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ("a", "b"):
                directory = Path(tmp) / name
                directory.mkdir()
                path = directory / "data.parquet"
                records = [{"messages": [{"role": "user", "content": "<image>"}],
                            "images": [{"bytes": None, "path": "image.png"}]}]
                pq.write_table(pa.Table.from_pylist(records), path)
                paths.append(str(path))
            train, valid, test = build_online_omni_mapping_dataset(
                data_path=paths,
                data_config={"shuffle": False, "split": "50,50,0", "cache_dir": str(Path(tmp) / "cache")},
                transform=PreparedTransform(),
            )
            self.assertIsNone(test)
            for dataset, name in ((train, "a"), (valid, "b")):
                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset[0]["messages"][0]["content"][0]["image"],
                                 str(Path(tmp) / name / "image.png"))
