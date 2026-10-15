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
"""Tests for the layer stack: how a config's layers get their kinds, as data."""
import copy
import unittest

from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._layer_stack import (
    LinearAttentionDims,
    derive_layers,
    resolve_layers,
    tower_layer_stack,
)
from hyper_parallel.auto_parallel._model_spec import LayerGroup, ModelSpecError, OpCounts, VisionSpec
from hyper_parallel.auto_parallel._op_profiles import LayerKind, OpProfile, load_op_profile

_COUNTS = {
    "attMM": 4, "attBMM": 2, "ffMM": 3, "softmax": 1, "dropout": 0,
    "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1, "linrec": 0,
}
_LINEAR = {
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def _hybrid_profile() -> OpProfile:
    """A two-kind hybrid profile shaped like Qwen3.5's, full attention by default."""
    full = LayerKind("full_attention", OpCounts.from_dict(_COUNTS))
    linear = LayerKind(
        "linear_attention",
        OpCounts.from_dict(dict(_COUNTS, attBMM=0, softmax=0, headCast=0, linrec=1)),
        attention="linear",
    )
    return OpProfile("hybrid", {kind.name: kind for kind in (full, linear)}, "full_attention")


def _groups(*groups) -> tuple:
    """Groups from ``(kind, count)`` or ``(kind, count, mtp)`` tuples."""
    return tuple(LayerGroup(*group) for group in groups)


class TestDerivedFromLayerTypes(unittest.TestCase):
    """A config that names each layer's kind gets exactly that stack."""

    def test_interleaved_kinds_group_by_run(self):
        """
        Feature: layer_types.
        Description: Three linear layers, one full, two linear, one full.
        Expectation: One group per run, in model order.
        """
        types = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2 + ["full_attention"]
        layers = derive_layers(_hybrid_profile(), 7, layer_types=types)
        self.assertEqual(layers, _groups(
            ("linear_attention", 3), ("full_attention", 1), ("linear_attention", 2), ("full_attention", 1),
        ))

    def test_a_longer_list_is_cut_to_the_body(self):
        """
        Feature: layer_types.
        Description: A cropped model keeps the checkpoint's full layer_types.
        Expectation: Only the first num_hidden_layers entries count.
        """
        types = ["linear_attention", "full_attention"] * 4
        layers = derive_layers(_hybrid_profile(), 3, layer_types=types)
        self.assertEqual(sum(group.count for group in layers), 3)
        self.assertEqual(layers[-1], LayerGroup("linear_attention", 1))

    def test_kinds_the_profile_lacks_keep_the_default_and_say_so(self):
        """
        Feature: layer_types of another family.
        Description: Sliding-window attention, which no profile declares.
        Expectation: Every layer takes the default kind, with a warning.
        """
        types = ["sliding_attention", "full_attention"] * 2
        with self.assertLogs("hyper_parallel.auto_parallel._layer_stack", "WARNING"):
            layers = derive_layers(load_op_profile("qwen"), 4, layer_types=types)
        self.assertEqual(layers, _groups(("decoder", 4)))


class TestDerivedFromTheProfile(unittest.TestCase):
    """Without layer_types, the profile's kinds imply the stack."""

    def test_dense_prefix_then_moe(self):
        """
        Feature: first_k_dense_replace.
        Description: DeepSeek-V3: 61 layers, the first 3 dense, one MTP layer.
        Expectation: 3 dense, 58 MoE, then one MoE MTP layer.
        """
        layers = derive_layers(load_op_profile("deepseek"), 61, mtp_depth=1, first_k_dense=3)
        self.assertEqual(layers, _groups(("dense", 3), ("moe", 58), ("moe", 1, True)))


class TestResolve(unittest.TestCase):
    """A stack is checked against its profile before anything prices it."""

    def test_a_stack_that_cannot_be_priced_is_refused(self):
        """
        Feature: the layer stack's checks.
        Description: layer_types naming 3 layers of 4, a t5 of 23 layers, a
            stated stack with a kind its family lacks, linear dimensions
            without the convolution's, and a tower naming a language model's kind.
        Expectation: Each refused while the spec is resolved, not in the
            estimator, naming what is wrong.
        """
        overrides = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 32000,
                     "arch": "qwen", "layers": [{"kind": "moe", "count": 4}]}
        partial = {key: value for key, value in _LINEAR.items() if key != "linear_conv_kernel_dim"}
        tower = VisionSpec(hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
                           layers=(LayerGroup("decoder", 6),))
        cases = (
            ("lists 3 layers", lambda: derive_layers(_hybrid_profile(), 4, layer_types=["full_attention"] * 3)),
            ("even layer count", lambda: derive_layers(load_op_profile("t5"), 23)),
            ("'moe'", lambda: resolve_hf_model_spec({"name": "unit", "config_overrides": overrides})),
            ("linear_conv_kernel_dim", lambda: LinearAttentionDims.from_fields(partial)),
            ("encoder", lambda: tower_layer_stack(tower)),
        )
        for named, refused in cases:
            with self.subTest(named=named), self.assertRaisesRegex(ModelSpecError, named):
                refused()

    def test_a_deep_copy_is_the_stack_itself(self):
        """
        Feature: deep copies of the frozen records.
        Description: Deep-copy a resolved stack, each of its groups and kinds, a kind's op counts
            and a set of linear-attention dimensions, as a search deep-copies the config holding
            them twice a candidate.
        Expectation: Each copy is the record itself, inside a copied container too.
        """
        stack = resolve_layers("deepseek", _groups(("dense", 1), ("moe", 2)))
        kinds = stack.distinct_kinds()
        records = (stack, *stack.groups, *kinds, kinds[0].ops, LinearAttentionDims.from_fields(_LINEAR))
        for record in records:
            self.assertIs(copy.deepcopy(record), record, type(record).__name__)
        self.assertIs(copy.deepcopy({"stack": stack})["stack"], stack)


if __name__ == "__main__":
    unittest.main()
