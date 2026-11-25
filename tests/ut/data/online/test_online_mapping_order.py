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
"""Verify single-source sampling does not depend on the training horizon."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.data.online.mapping import build_online_mapping_source
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler


class TestOnlineMappingOrder(unittest.TestCase):
    """Keep source ordering separate from epoch sampling."""

    def test_clamped_split_sizes_never_repeat_source_samples(self):
        """Feature: Finite evaluation ranges.

        Description: Request more validation samples than the file contains.
        Expectation: Clamped mode caps at the source length; training mode keeps the target.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(4)))
            clamped = build_online_mapping_source(
                data_path=str(path), data_config={"random_seed": 42},
                train_valid_test_num_samples=(10, 0, 0), clamp_split_sizes=True,
            )
            self.assertEqual(len(clamped), 4)
            self.assertEqual({clamped[i]["id"] for i in range(len(clamped))}, set(range(4)))
            training = build_online_mapping_source(
                data_path=str(path), data_config={"random_seed": 42},
                train_valid_test_num_samples=(10, 0, 0),
            )
            self.assertEqual(len(training), 10)

    def test_finite_weighted_sources_keep_unique_stable_prefixes(self):
        """Exhausted sources yield to remaining records without changing shorter prefixes."""
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            expected = set()
            for name, length, weight in (("a", 2, 9), ("b", 8, 1)):
                path = Path(directory) / f"{name}.jsonl"
                records = [{"id": f"{name}{index}"} for index in range(length)]
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
                sources.append({"data_path": {"valid": str(path)}, "weight": weight})
                expected.update(record["id"] for record in records)
            for sampler in ("single", "cyclic"):
                config = {"sources": sources, "random_seed": 42}
                training = build_online_mapping_source(data_config=config, sampler_type=sampler)[1]
                training_ids = [training[i]["id"] for i in range(len(training))]
                self.assertEqual(sum(sample_id.startswith("a") for sample_id in training_ids), 9)
                self.assertEqual(sum(sample_id.startswith("b") for sample_id in training_ids), 1)
                natural = build_online_mapping_source(
                    data_config=config, sampler_type=sampler, clamp_split_sizes=True,
                )[1]
                full_ids = [natural[i]["id"] for i in range(len(natural))]
                self.assertEqual(len(full_ids), len(expected))
                self.assertEqual(set(full_ids), expected)
                for limit in (0, 1, 2, 5, 10, 20):
                    with self.subTest(sampler=sampler, limit=limit):
                        splits = build_online_mapping_source(
                            data_config=config, sampler_type=sampler,
                            train_valid_test_num_samples=(0, limit, 0), clamp_split_sizes=True,
                        )
                        self.assertIsNone(splits[0])
                        self.assertIsNone(splits[2])
                        source = splits[1]
                        if limit == 0:
                            self.assertIsNone(source)
                            continue
                        ids = [source[i]["id"] for i in range(len(source))]
                        self.assertEqual(ids, full_ids[:limit])
                        self.assertEqual(len(ids), len(set(ids)))

    def test_clamped_single_cyclic_source_honors_the_limit(self):
        """An explicit evaluation bound also applies to an unshuffled single source."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(4)), encoding="utf-8")
            source = build_online_mapping_source(
                data_path=str(path), data_config={}, sampler_type="cyclic",
                train_valid_test_num_samples=(2, 0, 0), clamp_split_sizes=True,
            )
            self.assertEqual([source[i]["id"] for i in range(len(source))], [0, 1])

    def test_training_horizon_preserves_cyclic_prefix(self):
        """Different training lengths must share the same initial samples.

        Feature: Single-source online sampling.
        Description: Build short and long training views of a real JSONL file.
        Expectation: Both preserve full source order and the legacy cyclic permutation.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(32)))
            expected = list(range(32))
            random.Random(42).shuffle(expected)
            for target in (16, 24, 64):
                with self.subTest(target=target):
                    source = build_online_mapping_source(
                        data_path=str(path), data_config={"random_seed": 42}, sampler_type="cyclic",
                        train_valid_test_num_samples=(target, 0, 0),
                    )
                    self.assertEqual(len(source), 32)
                    self.assertEqual([source[i]["id"] for i in range(32)], list(range(32)))
                    sampler = build_dataset_batch_sampler(
                        total_samples=len(source), micro_batch_size=2, global_batch_size=8,
                        dp_rank=0, dp_world_size=1, sampler_type="cyclic", seed=42,
                    )
                    actual = [source[i]["id"] for batch in sampler for i in batch]
                    self.assertEqual(actual, expected)

    def test_cyclic_blend_preserves_source_order_and_weights(self):
        """Feature: Cyclic source ordering.

        Description: Blend two files at 1:3 across different horizons and seeds.
        Expectation: Source order is sequential, weights are retained, and the epoch is stable.
        """
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{name}.jsonl" for name in ("a", "b")]
            for path in paths:
                path.write_text("".join(json.dumps({"source": path.stem, "id": i}) + "\n" for i in range(8)))
            for seed in (42, 99):
                for target in (4, 64):
                    with self.subTest(seed=seed, target=target):
                        source = build_online_mapping_source(
                            data_config={"random_seed": seed, "sources": [
                                {"data_path": str(paths[0]), "weight": 1},
                                {"data_path": str(paths[1]), "weight": 3}]},
                            sampler_type="cyclic", train_valid_test_num_samples=(target, 0, 0),
                        )
                        records = [source[i] for i in range(len(source))]
                        self.assertEqual(len(records), 16)
                        self.assertEqual([r["id"] for r in records if r["source"] == "a"], list(range(4)))
                        self.assertEqual([r["id"] for r in records if r["source"] == "b"],
                                         list(range(8)) + list(range(4)))

    def test_single_keeps_target_sizing_and_cyclic_keeps_explicit_source(self):
        """Feature: Sampler-specific source policy.

        Description: Build ordinary and explicitly configured sources in both modes.
        Expectation: Single honors target size; cyclic preserves the complete source and empty splits.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(8)))
            for explicit in (False, True):
                kwargs = ({"data_config": {"sources": [{"data_path": str(path), "weight": 1}]}}
                          if explicit else {"data_path": str(path), "data_config": {}})
                source = build_online_mapping_source(**kwargs, sampler_type="cyclic",
                                                     train_valid_test_num_samples=(2, 0, 0))
                self.assertEqual([source[i]["id"] for i in range(len(source))], list(range(8)))
                single = build_online_mapping_source(**kwargs, sampler_type="single",
                                                     train_valid_test_num_samples=(2, 0, 0))
                self.assertEqual(len(single), 2)
            splits = build_online_mapping_source(data_path={"train": str(path)}, data_config={},
                                                  sampler_type="cyclic")
            self.assertEqual(len(splits[0]), 8)
            self.assertEqual(splits[1:], (None, None))
            empty = build_online_mapping_source(data_path=str(path), data_config={},
                                                 sampler_type="cyclic", sample_filter=lambda _: False)
            self.assertIsNone(empty)
