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
"""Tests for the op records (shared decision S1): their file, their expressions and their checks.

How to run this:
    pytest tests/ut/auto_parallel/test_op_records.py -v
"""
import os
import tempfile
import unittest
from dataclasses import fields

import yaml

from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_records import PARTS, SWITCHES, load_op_records

_OPS = [field.name for field in fields(OpCounts)]


def _records(data):
    """The records a file holding *data* states."""
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "op_records.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return load_op_records(path)


def _minimal(**slots):
    """A records file stating every op and the *slots* given."""
    return {"ops": {op: {} for op in _OPS}, "slots": slots}


class TestTheRecords(unittest.TestCase):
    """The op records the memory model prices a layer with."""

    def test_every_op_has_a_record(self):
        """
        Feature: load_op_records.
        Description: The records the package ships.
        Expectation: Every op of the op vector has one; each switch drops
            what the op or the part it names keeps, attUp the part of
            attMM's saves an MLA layer's up-projections build, and gather is
            communication; every slot counts toward a part of the layer.
        """
        records = load_op_records()
        self.assertEqual(sorted(records.ops), sorted(_OPS))
        named = [record.switch for record in {**records.ops, **records.parts}.values() if record.switch]
        self.assertEqual(sorted(named), sorted(SWITCHES))
        self.assertEqual(records.parts["attUp"].part_of, "attMM")
        self.assertEqual(sorted(records.keeps("attUp")), ["qkv"])
        self.assertIn("qkv", records.keeps("attMM"))
        self.assertEqual([name for name, op in records.ops.items() if op.comm], ["gather"])
        self.assertTrue(all(slot.part in PARTS for slot in records.slots.values()))
        self.assertEqual(sorted(records.keeps("dropout")), ["proj", "score"])

    def test_a_record_is_checked(self):
        """
        Feature: load_op_records' checks.
        Description: Files missing an op or holding an unknown key, and slots
            whose expressions read an attribute or call anything but max and min.
        Expectation: Each is refused, naming what is wrong.
        """
        missing = {"ops": {op: {} for op in _OPS[1:]}, "slots": {}}
        cases = [
            (missing, "missing"),
            ({**_minimal(), "extra": 1}, "unknown keys"),
            (_minimal(qkv={"part": "attention", "formula": "total * ccfg.h", "ops": {"attMM": "h"}}), "Attribute"),
            (_minimal(qkv={"part": "attention", "formula": "total * open(h)", "ops": {"attMM": "h"}}), "max or min"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ModelSpecError, message):
                _records(data)

    def test_a_slot_reads_only_its_branch(self):
        """
        Feature: OpRecords.evaluate.
        Description: A slot whose op keeps one field's elements or another's,
            as a flag says, evaluated where the other field does not exist;
            then with the op dropped, and priced alone.
        Expectation: The branch the flag takes is the only one read; max and
            min are calls, not fields; a dropped op keeps nothing, and one
            priced alone keeps its elements whatever the setting.
        """
        records = _records(_minimal(qkv={
            "part": "attention", "formula": "total * max(t, 1) / min(t, 2)",
            "ops": {"attMM": "h if flag else missing", "attBMM": "3"}}))
        fields_ = {"h": 10, "flag": 1, "t": 4}
        value = fields_.__getitem__
        self.assertEqual(records.evaluate("qkv", value, lambda op: 1), (10 + 3) * 4 / 2)
        self.assertEqual(records.evaluate("qkv", value, lambda op: int(op != "attBMM")), 10 * 4 / 2)
        self.assertEqual(records.evaluate("qkv", value, lambda op: 0, only="attBMM"), 3 * 4 / 2)

    def test_the_heads_of_an_mla_layer_are_a_part_of_its_projections(self):
        """
        Feature: The attUp part of the qkv slot.
        Description: The qkv slot of an MLA layer with a queries' latent, of
            one without, and of a layer that compresses nothing, evaluated
            with every op kept, with attUp dropped, and attMM alone.
        Expectation: attMM priced alone, its part with it, keeps what the
            slot kept as one expression; attUp holds the query heads where
            the queries have a latent and the key and value heads, and
            nothing where the layer compresses nothing.
        """
        records = load_op_records()
        base = {"n_attMM": 4, "n_attParamCast": 0, "a": 128, "dh": 128, "dhr": 64, "n_kv": 128, "dc_kv": 512,
                "h": 7168, "cp": 2, "kv_shards": 1, "n_attBMM": 2}
        formula = {"micro_factor": 1, "s": 4096, "b": 1, "bytes_compute": 2, "t": 1}
        for dc_q in (1536, 0):
            names = {**base, **formula, "dc_q": dc_q}
            whole = (dc_q + 2 * 128 * (128 + 64)) + (64 + 128 * (2 * 128 + 64) + 128 * 128 + 512) * 2 / 1
            heads = (2 * 128 * (128 + 64) if dc_q else 0) + 128 * (2 * 128 + 64) * 2 / 1
            scale = 4096 * 2 / (1 * 2)
            with self.subTest(dc_q=dc_q):
                self.assertEqual(records.evaluate("qkv", names.__getitem__, lambda op: 0, only="attMM"),
                                 whole * scale)
                self.assertEqual(records.evaluate("qkv", names.__getitem__, lambda op: 0, only="attUp"),
                                 heads * scale)
                kept = records.evaluate("qkv", names.__getitem__, lambda op: 1)
                dropped = records.evaluate("qkv", names.__getitem__, lambda op: int(op != "attUp"))
                self.assertEqual(kept - dropped, heads * scale)
        names = {**base, **formula, "dc_kv": 0, "dc_q": 0}
        self.assertEqual(records.evaluate("qkv", names.__getitem__, lambda op: 0, only="attUp"), 0)


if __name__ == "__main__":
    unittest.main()
