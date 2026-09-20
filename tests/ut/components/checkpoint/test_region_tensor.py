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
"""UT for :mod:`hyper_parallel.components.checkpoint.region_tensor`."""
import math
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.conversion_ops import (
    AddScalar,
    ConcatenateWithSections,
    DeinterleaveGateQKV,
    DeinterleaveQKV,
    InterleaveGateQKV,
    InterleaveQKV,
    Split,
)
from hyper_parallel.components.checkpoint.region_tensor import (
    RegionTensor,
    UnsupportedRegionOp,
    materialize,
)


def _pair(name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> tuple[RegionTensor, torch.Tensor]:
    """A checkpoint tensor of distinct values, and the region reading it whole."""
    real = (torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) * 0.37 - 5).to(dtype)
    return RegionTensor.leaf(name, shape, dtype), real


class TestRegionTensor(unittest.TestCase):
    """Regions have to describe exactly what the same operations compute on real tensors."""

    def _assert_matches(self, region: Any, real: torch.Tensor, sources: dict[str, torch.Tensor], label: str) -> None:
        """Assert that ``region`` computes ``real`` out of ``sources``, in shape, dtype and values."""
        self.assertIsInstance(region, RegionTensor, f"{label}: expected a RegionTensor, got {type(region)}")
        self.assertEqual(tuple(region.shape), tuple(real.shape), f"{label}: shape mismatch")
        self.assertEqual(region.dtype, real.dtype, f"{label}: dtype mismatch")
        values = materialize(region, sources)
        self.assertTrue(torch.equal(values, real), f"{label}: values mismatch: expected={real}, got={values}")

    def _check_all(self, cases: dict[str, Callable[..., Any]], inputs: dict[str, tuple[RegionTensor, torch.Tensor]]):
        """Run every case on the regions and on the real tensors, and compare."""
        sources = {name: real for name, (_, real) in inputs.items()}
        regions = [region for region, _ in inputs.values()]
        reals = list(sources.values())
        for label, case in cases.items():
            with self.subTest(label):
                self._assert_matches(case(*regions), case(*reals), sources, label)

    def test_shape_operations_match_torch(self):
        """
        Feature: RegionTensor reshaping, reordering, slicing and splitting.
        Description: Apply reshapes that merge and split dimensions, including after a transpose, and
            transposes, permutes, indexing, select, narrow, squeeze, unsqueeze, split, chunk and unbind,
            as methods and as torch functions, to a 2x3x4 tensor.
        Expectation: Every result matches torch in shape, dtype and values.
        """
        cases = {
            "reshape": lambda t: t.reshape(3, 8),
            "reshape -1": lambda t: t.reshape(-1),
            "view": lambda t: t.view(2, 3, 2, 2),
            "transpose": lambda t: t.transpose(0, 2),
            "torch.transpose": lambda t: torch.transpose(t, dim0=-2, dim1=-1),
            "permute": lambda t: t.permute(2, 0, 1),
            "merge after transpose": lambda t: t.transpose(1, 2).reshape(6, 4),
            "flatten all after transpose": lambda t: t.transpose(0, 1).flatten(),
            "flatten": lambda t: torch.flatten(t, 0, 1),
            "unflatten": lambda t: t.unflatten(2, (2, 2)),
            "split then merge": lambda t: t.transpose(0, 2).reshape(4, 3, 2).transpose(0, 1).reshape(3, 8),
            "partial rows": lambda t: t.reshape(2, 12)[:, 5:].reshape(-1),
            "index": lambda t: t[1],
            "slice": lambda t: t[:, 1:3],
            "stepped slice": lambda t: t[..., ::2],
            "mixed index": lambda t: t[None, 0, :, 1],
            "select": lambda t: t.select(1, -1),
            "narrow": lambda t: torch.narrow(t, 2, 1, 2),
            "squeeze": lambda t: t.unsqueeze(1).squeeze(),
            "split": lambda t: torch.split(t, [1, 2], dim=1)[1],
            "chunk": lambda t: t.chunk(2, dim=0)[1],
            "unbind": lambda t: t.unbind(2)[3],
            "contiguous": lambda t: t.transpose(0, 1).contiguous(),
        }
        self._check_all(cases, {"w": _pair("w", (2, 3, 4))})

    def test_concatenation_and_stacking_match_torch(self):
        """
        Feature: RegionTensor concatenation and stacking of several checkpoint tensors.
        Description: Concatenate a 4x3 and a 2x3 tensor along rows, and their transposes along columns;
            stack slices of them; and concatenate expert tensors stacked the way MergeModulelist does.
        Expectation: Every result matches torch.
        """
        cases = {
            "cat": lambda q, k: torch.cat([q, k]),
            "concat columns": lambda q, k: torch.concat([q.t(), k.t()], dim=1),
            "stack": lambda q, k: torch.stack([q[:2], k], dim=1),
            "experts": lambda q, k: torch.cat([torch.stack([q[:2], q[2:]]), torch.stack([k, k])], dim=1),
        }
        self._check_all(cases, {"q": _pair("q", (4, 3)), "k": _pair("k", (2, 3))})

    def test_arithmetic_and_casts_match_torch(self):
        """
        Feature: RegionTensor elementwise arithmetic with Python numbers, and dtype conversions.
        Description: Add, subtract, multiply and divide a bfloat16 tensor by numbers from either side,
            negate it, divide with rounding, and convert it.
        Expectation: Every result matches torch, values computed in bfloat16 before any conversion.
        """
        cases = {
            "add": lambda t: t + 1.0,
            "radd": lambda t: 1 + t,
            "rsub": lambda t: 1 - t,
            "rmul": lambda t: 2 * t,
            "div": lambda t: t / 3,
            "neg": lambda t: -t,
            "torch.add": lambda t: torch.add(t, 0.5),
            "floor div": lambda t: torch.div(t, 2, rounding_mode="floor"),
            "float": lambda t: t.float(),
            "add then to": lambda t: (t[1:] + 0.1).to(torch.float32),
        }
        self._check_all(cases, {"w": _pair("w", (3, 4), torch.bfloat16)})

    def test_mixed_dtypes_are_converted_as_torch_converts_them(self):
        """
        Feature: RegionTensor concatenation of tensors in different dtypes.
        Description: Concatenate a float16 and a bfloat16 tensor, then add a number.
        Expectation: The result is float32 like torch.cat's, its values match torch, and the blocks
            convert each checkpoint tensor to float32 before the addition.
        """
        half, half_real = _pair("h", (2, 3), torch.float16)
        brain, brain_real = _pair("b", (2, 3), torch.bfloat16)
        sources = {"h": half_real, "b": brain_real}

        region = torch.cat([half, brain]) + 0.1
        real = torch.cat([half_real, brain_real]) + 0.1

        self._assert_matches(region, real, sources, "mixed cat")
        post_counts = [len(block.post) for block in region.remap_blocks()]
        self.assertEqual(post_counts, [2, 2], f"expected a conversion and an addition per block, got {post_counts}")

    def test_hyper_conversion_ops_match_real_conversion(self):
        """
        Feature: RegionTensor through the conversion operations of hyper_parallel replacement modules.
        Description: Run InterleaveQKV on separate and fused weights and on biases, InterleaveGateQKV, the
            two Deinterleave operations, ConcatenateWithSections, Split and AddScalar on regions and on
            real tensors.
        Expectation: Every output of every operation matches the real conversion.
        """
        heads = {"num_key_value_heads": 2, "num_key_value_groups": 2, "query_head_dim": 3, "value_head_dim": 3}
        cases = [
            ("qkv", InterleaveQKV(**heads, source_is_fused=False), {"q": (12, 5), "k": (6, 5), "v": (6, 5)}, ["o"]),
            ("fused qkv", InterleaveQKV(**heads, source_is_fused=True), {"qkv": (24, 5)}, ["o"]),
            ("qkv bias", InterleaveQKV(**heads, source_is_fused=False), {"q": (12,), "k": (6,), "v": (6,)}, ["o"]),
            ("gate qkv", InterleaveGateQKV(**heads), {"qg": (24, 5), "k": (6, 5), "v": (6, 5)}, ["o"]),
            ("deinterleave", DeinterleaveQKV(**heads, target_is_fused=False), {"g": (24, 5)}, ["q", "k", "v"]),
            ("deinterleave gate", DeinterleaveGateQKV(**heads), {"g": (36, 5)}, ["qg", "k", "v"]),
            ("sections", ConcatenateWithSections((4, 2)), {"a": (4, 3), "b": (2, 3)}, ["o"]),
            ("split", Split((4, 2)), {"a": (6, 3)}, ["x", "y"]),
            ("add scalar", AddScalar(1.0), {"a": (6,)}, ["o"]),
        ]
        for label, operation, shapes, targets in cases:
            with self.subTest(label):
                inputs = {name: _pair(name, shape) for name, shape in shapes.items()}
                sources = {name: real for name, (_, real) in inputs.items()}
                patterns = list(shapes)
                traced = operation.convert({n: [r] for n, (r, _) in inputs.items()}, patterns, targets)
                expected = operation.convert({n: [t] for n, (_, t) in inputs.items()}, patterns, targets)
                self.assertEqual(traced.keys(), expected.keys(), f"{label}: outputs mismatch")
                for name, value in expected.items():
                    self._assert_matches(traced[name], value, sources, f"{label}.{name}")

    def test_transformers_style_conversions_match_torch(self):
        """
        Feature: RegionTensor through the tensor calls Transformers conversion operations make.
        Description: Merge per-expert gate and up projections as MergeModulelist and Concatenate do,
            split experts back as SplitModulelist does, permute for RoPE as PermuteForRope does, and
            transpose expert weights as Transpose does.
        Expectation: Every result matches torch, and the RoPE permutation takes two blocks per head.
        """
        experts = {f"e{index}.{proj}": _pair(f"e{index}.{proj}", (6, 4)) for index in range(3) for proj in "gu"}
        sources = {name: real for name, (_, real) in experts.items()}

        def merge(values: dict[str, Any]) -> Any:
            gate = torch.stack([values[f"e{index}.g"] for index in range(3)], dim=0)
            up = torch.stack([values[f"e{index}.u"] for index in range(3)], dim=0)
            return torch.cat([gate, up], dim=1)

        merged = merge({name: region for name, (region, _) in experts.items()})
        merged_real = merge(sources)
        self._assert_matches(merged, merged_real, sources, "merge experts")
        self._assert_matches(torch.transpose(merged, dim0=-2, dim1=-1).contiguous(),
                             torch.transpose(merged_real, dim0=-2, dim1=-1).contiguous(), sources, "transpose")
        self._assert_matches(torch.chunk(merged, 3, dim=0)[1].squeeze(),
                             torch.chunk(merged_real, 3, dim=0)[1].squeeze(), sources, "split experts")

        heads, half, width = 4, 3, 5
        rope, rope_real = _pair("q", (heads * half * 2, width))

        def permute(tensor: Any) -> Any:
            return tensor.view(heads, half, 2, width).transpose(1, 2).reshape(heads * half * 2, width)

        permuted = permute(rope)
        self._assert_matches(permuted, permute(rope_real), {"q": rope_real}, "rope permute")
        self.assertEqual(len(permuted.blocks), heads * 2, f"expected {heads * 2} blocks, got {len(permuted.blocks)}")

    def test_remap_blocks_describe_where_elements_come_from(self):
        """
        Feature: RegionTensor.remap_blocks.
        Description: Concatenate the transposes of a 4x3 "q" and a 2x3 "k" along columns.
        Expectation: Two blocks, each reading its whole checkpoint tensor with rows and columns swapped.
        """
        q, _ = _pair("q", (4, 3))
        k, _ = _pair("k", (2, 3))

        blocks = sorted(torch.cat([q.t(), k.t()], dim=1).remap_blocks(), key=lambda block: block.offsets)

        described = [(b.offsets, b.lengths, b.source, b.base, b.coeff, b.post) for b in blocks]
        expected = [
            ((0, 0), (3, 4), "q", (0, 0), ((0, 1), (1, 0)), ()),
            ((0, 4), (3, 2), "k", (0, 0), ((0, 1), (1, 0)), ()),
        ]
        self.assertEqual(described, expected, f"blocks mismatch: expected={expected}, got={described}")

    def test_operations_a_region_cannot_describe_raise(self):
        """
        Feature: RegionTensor refusals.
        Description: Multiply two regions, index with a tensor, write in place, compare, read a value,
            call a torch function regions do not support, and view as another dtype.
        Expectation: UnsupportedRegionOp every time.
        """
        region, _ = _pair("w", (2, 3))
        cases = {
            "tensor product": lambda: region * region,
            "tensor index": lambda: region[torch.tensor([0])],
            "in place add": lambda: region.add_(1),
            "setitem": lambda: region.__setitem__(0, 1.0),
            "compare": lambda: region == 0,
            "bool": lambda: bool(region),
            "item": region.item,
            "matmul": lambda: torch.matmul(region, region),
            "view dtype": lambda: region.view(torch.int32),
            "cat with real": lambda: torch.cat([region, torch.zeros(2, 3)]),
        }
        for label, case in cases.items():
            with self.subTest(label), self.assertRaises(UnsupportedRegionOp):
                case()

    def test_region_cut_into_too_many_blocks_is_refused(self):
        """
        Feature: RegionTensor block limit.
        Description: With the limit lowered to four blocks, permute a tensor for RoPE, which takes eight.
        Expectation: UnsupportedRegionOp, so the conversion runs on real tensors instead.
        """
        region, _ = _pair("q", (24, 5))
        with patch("hyper_parallel.components.checkpoint.region_tensor.MAX_BLOCKS", 4):
            with self.assertRaises(UnsupportedRegionOp):
                region.view(4, 3, 2, 5).transpose(1, 2).reshape(24, 5)


if __name__ == "__main__":
    unittest.main()
