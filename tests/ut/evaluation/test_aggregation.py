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
"""Numerical regression tests for native and compensated evaluation totals."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.evaluation.aggregation import (
    ReducedStatistics,
    accumulate_statistics,
    accumulation_dtype,
    reduce_scalar,
    reduce_statistics,
)
from hyper_parallel.evaluation.types import LossStatistics, ReductionContext


def _statistics(value: float, denominator: float = 1.0) -> dict:
    """Build FP32 source values, matching losses produced on an accelerator."""
    return {"foundation": LossStatistics(torch.tensor(value), torch.tensor(denominator), is_lm_nll=True)}


def _replicate_schema(outputs: list, value: object, **kwargs: object) -> None:
    """Stand in for two ranks with identical control records."""
    del kwargs
    outputs[:] = [value, value]


class TestNllDeclarations(unittest.TestCase):
    """Only explicit booleans permit PPL and their meaning must stay consistent."""

    def test_statistics_without_a_declaration_are_not_nll(self):
        """A standalone statistic needs an explicit opt-in before producing PPL."""
        statistics = {"foundation": LossStatistics(torch.tensor(6.0), torch.tensor(3))}
        reduced = reduce_statistics([statistics], ReductionContext(), torch.device("cpu"))
        self.assertIs(statistics["foundation"].is_lm_nll, False)
        self.assertIs(reduced.is_lm_nll("foundation"), False)
        self.assertEqual(reduced.mean("foundation"), 2.0)

    def test_nonboolean_declarations_are_rejected(self):
        """Strings and integer flags cannot accidentally opt a statistic into PPL."""
        for invalid in ("false", "true", "lm_nll", 0, 1, None, torch.tensor(True)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "is_lm_nll must be bool"):
                    LossStatistics(torch.tensor(1.0), torch.tensor(1), is_lm_nll=invalid)
                with self.assertRaisesRegex(ValueError, "is_lm_nll must be bool"):
                    ReducedStatistics({"foundation": 1.0}, 1.0, {"foundation": invalid}, {"foundation": 1.0})

    def test_reduction_preserves_each_statistics_nll_declaration(self):
        """Reducing a plain NLL and an auxiliary objective keeps their flags separate."""
        statistics = {
            "foundation": LossStatistics(torch.tensor(6.0), torch.tensor(3), is_lm_nll=True),
            "aux": LossStatistics(torch.tensor(4.0), torch.tensor(2), is_lm_nll=False),
        }
        reduced = reduce_statistics([statistics, statistics], ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.nll_flags, {"foundation": True, "aux": False})
        self.assertIs(reduced.is_lm_nll("foundation"), True)
        self.assertIs(reduced.is_lm_nll("aux"), False)
        self.assertIs(reduced.is_lm_nll("missing"), False)
        self.assertEqual(reduced.numerators, {"foundation": 12.0, "aux": 8.0})
        self.assertEqual(reduced.denominators, {"foundation": 6.0, "aux": 4.0})

    def test_conflicting_rank_declarations_fail_before_numeric_reduction(self):
        """All ranks inspect the full schema before combining differently defined losses."""
        def gather(outputs: list, value: object, **kwargs: object) -> None:
            """Supply a second rank that disagrees with the local NLL declaration."""
            del kwargs
            outputs[:] = [value, {"foundation": not value["foundation"]}]

        for flag in (True, False):
            with self.subTest(is_lm_nll=flag):
                statistics = {"foundation": LossStatistics(torch.tensor(1.0), torch.tensor(1), is_lm_nll=flag)}
                with (patch("hyper_parallel.evaluation.aggregation.dist.all_gather_object", side_effect=gather),
                      patch("hyper_parallel.evaluation.aggregation.dist.all_reduce") as collective):
                    with self.assertRaisesRegex(ValueError, "inconsistent is_lm_nll across ranks"):
                        reduce_statistics([statistics], ReductionContext(object(), 2), torch.device("cpu"))
                collective.assert_not_called()


class TestCompensatedStatistics(unittest.TestCase):
    """Exercise the NPU accumulation arithmetic on CPU without a device dependency."""

    def test_device_dtype_selection(self):
        """Only NPU requires the FP32 accumulation fallback."""
        for device_type, expected in (("npu", torch.float32), ("cpu", torch.float64), ("cuda", torch.float64)):
            with self.subTest(device_type=device_type):
                self.assertEqual(accumulation_dtype(SimpleNamespace(type=device_type)), expected)

    def test_small_increments_survive_above_fp32_integer_limit(self):
        """A long pass must retain additions that ordinary FP32 summing discards."""
        totals = {}
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            accumulate_statistics(totals, _statistics(2**24, 2**24))
            with (patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-batch CPU transfer")),
                  patch.object(torch.Tensor, "item", side_effect=AssertionError("per-batch scalar read"))):
                for _ in range(65):
                    accumulate_statistics(totals, _statistics(1.0))
            reduced = reduce_statistics([totals], ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.numerators["foundation"], 2**24 + 65)
        self.assertEqual(reduced.num_tokens, 2**24 + 65)
        self.assertEqual(len(totals), 1)
        for value in vars(totals["foundation"]).values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.numel(), 1)

    def test_merging_accumulated_contributions_retains_each_residual(self):
        """Re-accumulation must not erase the residual retained by either input."""
        contributions = []
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            for count in (32, 33):
                totals = {}
                accumulate_statistics(totals, _statistics(2**24, 2**24))
                for _ in range(count):
                    accumulate_statistics(totals, _statistics(1.0))
                contributions.append(totals)
            reduced = reduce_statistics(contributions, ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.numerators["foundation"], 2**25 + 65)
        self.assertEqual(reduced.num_tokens, 2**25 + 65)

    def test_signed_cancellation_retains_a_small_objective(self):
        """Magnitude-aware compensation also handles signed objective statistics."""
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            reduced = reduce_statistics(
                [_statistics(2**24), _statistics(1.0), _statistics(-(2**24))],
                ReductionContext(), torch.device("cpu"),
            )
        self.assertEqual(reduced.numerators["foundation"], 1.0)

    def test_distributed_reconstruction_keeps_residual_and_integer_count(self):
        """One FP32 collective transports both compensated values and odd counts."""
        def reduce_pair(value: torch.Tensor, **kwargs: object) -> None:
            """Model an identical second rank without rounding the CPU reconstruction."""
            del kwargs
            self.assertEqual(value.dtype, torch.float32)
            value.mul_(2)

        totals = {}
        with (patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32),
              patch("hyper_parallel.evaluation.aggregation.dist.all_gather_object", side_effect=_replicate_schema),
              patch("hyper_parallel.evaluation.aggregation.dist.all_reduce", side_effect=reduce_pair) as collective):
            accumulate_statistics(totals, _statistics(2**24, 2**24))
            accumulate_statistics(totals, _statistics(1.0))
            reduced = reduce_statistics(
                [totals], ReductionContext(object(), 2), torch.device("cpu"),
                sample_count=torch.tensor(2**24 + 1, dtype=torch.int64),
            )
        collective.assert_called_once()
        self.assertEqual(reduced.numerators["foundation"], 2**25 + 2)
        self.assertEqual(reduced.num_tokens, 2**25 + 2)
        self.assertEqual(reduced.num_samples, 2**25 + 2)

    def test_local_integer_count_does_not_pass_through_fp32(self):
        """A local report preserves an odd count even when loss totals use FP32."""
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            reduced = reduce_statistics([], ReductionContext(), torch.device("cpu"),
                                        sample_count=torch.tensor(2**24 + 1, dtype=torch.int64))
        self.assertEqual(reduced.num_samples, 2**24 + 1)
        self.assertEqual(reduced.numerators, {})

    def test_large_integer_denominator_preserves_its_cast_residual(self):
        """One large token count must survive the initial integer-to-FP32 conversion."""
        statistics = {"foundation": LossStatistics(torch.tensor(1.0), torch.tensor(2**24 + 1))}
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            reduced = reduce_statistics([statistics], ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.num_tokens, 2**24 + 1)

    def test_empty_and_zero_weight_statistics_stay_empty(self):
        """Dummy batches must not invent a loss or tokens in compensated mode."""
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            for contributions in ([], [{}, _statistics(0.0, 0.0)]):
                with self.subTest(contributions=contributions):
                    reduced = reduce_statistics(contributions, ReductionContext(), torch.device("cpu"))
                    self.assertIsNone(reduced.mean("foundation"))
                    self.assertEqual(reduced.num_tokens, 0.0)

    def test_nonfinite_or_negative_denominator_is_rejected(self):
        """Compensation must not hide invalid contributions from validation."""
        with patch("hyper_parallel.evaluation.aggregation.accumulation_dtype", return_value=torch.float32):
            for numerator, denominator in ((float("nan"), 1), (float("inf"), 1), (1, -1)):
                with self.subTest(numerator=numerator, denominator=denominator):
                    with self.assertRaisesRegex(ValueError, "negative denominator or non-finite"):
                        reduce_statistics([_statistics(numerator, denominator)],
                                          ReductionContext(), torch.device("cpu"))

    def test_native_fp64_local_values_keep_their_precision(self):
        """The NPU fallback must not lower native CPU/CUDA accumulation precision."""
        exact = 1.0 + 2**-50
        statistics = {"foundation": LossStatistics(torch.tensor(exact, dtype=torch.float64), torch.tensor(1.0))}
        reduced = reduce_statistics([statistics], ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.numerators["foundation"], exact)

    def test_native_fp64_local_values_keep_their_range(self):
        """A finite local FP64 objective does not need to fit the FP32 wire format."""
        statistics = {"foundation": LossStatistics(torch.tensor(1e100, dtype=torch.float64), torch.tensor(1.0))}
        reduced = reduce_statistics([statistics], ReductionContext(), torch.device("cpu"))
        self.assertEqual(reduced.numerators["foundation"], 1e100)

    def test_native_fp64_small_negative_denominator_is_rejected(self):
        """A negative weight cannot escape validation by underflowing on an FP32 wire."""
        statistics = {"foundation": LossStatistics(torch.tensor(1.0), torch.tensor(-1e-50, dtype=torch.float64))}
        with self.assertRaisesRegex(ValueError, "negative denominator"):
            reduce_statistics([statistics], ReductionContext(), torch.device("cpu"))

    def test_distributed_small_negative_denominator_is_rejected_before_wire_cast(self):
        """All contributors reject a negative weight before FP32 underflow can hide it."""
        statistics = {"foundation": LossStatistics(torch.tensor(1.0), torch.tensor(-1e-50, dtype=torch.float64))}
        with (patch("hyper_parallel.evaluation.aggregation.dist.all_gather_object", side_effect=_replicate_schema),
              patch("hyper_parallel.evaluation.aggregation.dist.all_reduce") as collective):
            with self.assertRaisesRegex(ValueError, "negative denominator"):
                reduce_statistics([statistics], ReductionContext(object(), 2), torch.device("cpu"))
        collective.assert_not_called()

    def test_separate_scalar_reduction_keeps_integer_count(self):
        """Custom metrics still get an accurate separately reduced sample count."""
        def reduce_pair(value: torch.Tensor, **kwargs: object) -> None:
            """Model two identical integer counts on the FP32 wire."""
            del kwargs
            value.mul_(2)

        with patch("hyper_parallel.evaluation.aggregation.dist.all_reduce", side_effect=reduce_pair):
            reduced = reduce_scalar(torch.tensor(2**24 + 1, dtype=torch.int64), ReductionContext(object(), 2))
        self.assertEqual(reduced.dtype, torch.float64)
        self.assertEqual(reduced.item(), 2**25 + 2)
