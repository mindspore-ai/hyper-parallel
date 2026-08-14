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
"""Unit tests for shape-only mega-kernel memory estimation."""

import unittest
from unittest import mock

import hyper_parallel.core.multicore as multicore_api
from hyper_parallel.core.multicore import (
    MemoryCategory,
    MemoryComponent,
    MegaKernelMemoryEstimate,
    MegaMoeGradMemorySpec,
    MegaMoeMemorySpec,
    estimate_mega_kernel_peak_memory,
    estimate_mega_moe_grad_peak_memory,
    estimate_mega_moe_peak_memory,
    register_mega_kernel_memory_estimator,
)


class TestMegaMoeMemoryEstimation(unittest.TestCase):
    """Test forward MegaMoe memory accounting without device execution."""

    def test_estimate_matches_forward_tensor_shapes(self) -> None:
        """Test public subtotals against the TP2/EP2 forward specification."""
        spec = MegaMoeMemorySpec(
            tp=2, ep=2, seq_size=1024, expert_num=16, top_k=8,
            hidden_size=5120, intermediate_size=2048,
            gmm_workspace_bytes=32 * 1024 * 1024,
        )

        estimate = estimate_mega_moe_peak_memory(spec)

        token_num = 4096
        local_expert_num = 8
        activation = token_num * 5120 * 2
        expected_inputs = (
            activation
            + local_expert_num * 5120 * 4096 * 2
            + local_expert_num * 2048 * 5120 * 2
            + 16 * (4 * 8 + 2 * 4)
            + local_expert_num * 2 * 8
        )
        expected_outputs = activation * 3 + token_num * 4096 * 2 + token_num * 2048 * 2
        expected_workspace = 32 * 1024 * 1024
        expected_runtime = 15_077_472 + 2 * 2016 * 24 + 3936 + 4096
        expected_cann_workspace = 95_421_440
        expected_reservation_overhead = 1024 * 1024 * 1024 - activation * 2 - 4096
        expected_peak = (
            expected_inputs + expected_outputs + expected_workspace
            + expected_runtime + expected_cann_workspace + expected_reservation_overhead
        )

        self.assertEqual(estimate.kernel_name, "mega_moe")
        self.assertEqual(estimate.bytes_for(MemoryCategory.INPUT), expected_inputs)
        self.assertEqual(estimate.bytes_for(MemoryCategory.OUTPUT), expected_outputs)
        self.assertEqual(estimate.bytes_for(MemoryCategory.EXPLICIT_WORKSPACE), expected_workspace)
        self.assertEqual(estimate.bytes_for(MemoryCategory.RUNTIME), expected_runtime)
        self.assertEqual(
            estimate.bytes_for(MemoryCategory.IMPLICIT_WORKSPACE), expected_cann_workspace
        )
        self.assertEqual(
            estimate.bytes_for(MemoryCategory.RESERVATION_OVERHEAD), expected_reservation_overhead
        )
        self.assertEqual(estimate.footprint_bytes, expected_peak - expected_reservation_overhead)
        self.assertEqual(estimate.peak_bytes, expected_peak)
        self.assertAlmostEqual(estimate.peak_mib, expected_peak / (1024 * 1024))

    def test_parallel_degrees_change_local_shapes(self) -> None:
        """Test TP shrinks token buffers and EP shrinks per-rank weights."""
        base_spec = MegaMoeMemorySpec(
            tp=1, ep=1, seq_size=128, expert_num=8, top_k=2,
            hidden_size=64, intermediate_size=32, gmm_workspace_bytes=0,
        )
        sharded_spec = MegaMoeMemorySpec(
            tp=2, ep=2, seq_size=128, expert_num=8, top_k=2,
            hidden_size=64, intermediate_size=32, gmm_workspace_bytes=0,
        )

        base = estimate_mega_moe_peak_memory(base_spec)
        sharded = estimate_mega_moe_peak_memory(sharded_spec)

        self.assertEqual(sharded_spec.per_rank_token_num, base_spec.per_rank_token_num // 2)
        self.assertEqual(sharded_spec.local_expert_num, base_spec.local_expert_num // 2)
        self.assertLess(sharded.bytes_for(MemoryCategory.INPUT), base.bytes_for(MemoryCategory.INPUT))
        self.assertLess(sharded.bytes_for(MemoryCategory.OUTPUT), base.bytes_for(MemoryCategory.OUTPUT))

    def test_invalid_specs_raise_descriptive_errors(self) -> None:
        """Test invalid divisibility and capacity constraints are rejected."""
        valid_args = {
            "tp": 2, "ep": 2, "seq_size": 128, "expert_num": 8,
            "top_k": 2, "hidden_size": 64, "intermediate_size": 32,
        }

        with self.assertRaisesRegex(ValueError, "expert_num must be divisible by ep"):
            MegaMoeMemorySpec(**{**valid_args, "expert_num": 7})
        with self.assertRaisesRegex(ValueError, "seq_size \\* top_k must be divisible by tp"):
            MegaMoeMemorySpec(**{**valid_args, "tp": 3})
        with self.assertRaisesRegex(ValueError, "gmm_workspace_bytes must be a non-negative integer"):
            MegaMoeMemorySpec(**valid_args, gmm_workspace_bytes=-1)
        with self.assertRaisesRegex(ValueError, "smaller than required symmetric tensors"):
            estimate_mega_moe_peak_memory(MegaMoeMemorySpec(**valid_args, symmetric_heap_bytes=1))

    def test_estimation_does_not_initialize_platform_backend(self) -> None:
        """Test shape-only estimation has no platform initialization side effect."""
        spec = MegaMoeMemorySpec(
            tp=1, ep=1, seq_size=8, expert_num=2, top_k=1,
            hidden_size=16, intermediate_size=8,
        )

        with mock.patch.object(multicore_api, "get_platform") as get_platform:
            estimate_mega_moe_peak_memory(spec)

        get_platform.assert_not_called()

    def test_generic_api_rejects_wrong_spec_type(self) -> None:
        """Test the registered MegaMoe estimator enforces its specification."""
        with self.assertRaisesRegex(TypeError, "requires MegaMoeMemorySpec"):
            estimate_mega_kernel_peak_memory("mega_moe", object())


class TestMegaMoeGradMemoryEstimation(unittest.TestCase):
    """Test backward MegaMoe memory accounting without device execution."""

    def test_estimate_matches_backward_tensor_shapes(self) -> None:
        """Test backward categories against the TP2/EP2 specification."""
        spec = MegaMoeGradMemorySpec(
            tp=2, ep=2, seq_size=512, expert_num=16, top_k=8,
            hidden_size=5120, intermediate_size=2048,
            gmm_workspace_bytes=32 * 1024 * 1024, num_cube_cores=20,
        )

        estimate = estimate_mega_moe_grad_peak_memory(spec)

        token_num = 2048
        local_expert_num = 8
        activation = token_num * 5120 * 2
        intermediate = token_num * 2048 * 2
        gate = token_num * 4096 * 2
        w1 = local_expert_num * 5120 * 4096 * 2
        w2 = local_expert_num * 2048 * 5120 * 2
        routing = 16 * (4 * 8 + 2 * 4) + local_expert_num * 8
        expected_inputs = activation * 2 + intermediate + gate + w1 + w2 + routing
        expected_outputs = activation * 3 + intermediate + gate + w1 + w2
        expected_explicit = 48 * 1024 * 1024
        expected_runtime = 15_077_472 + 4 * 2016 * 20 + 3936 + 4096
        expected_implicit = 99_615_744
        expected_reservation = 1024 * 1024 * 1024 - activation * 2 - 4096

        self.assertEqual(estimate.kernel_name, "mega_moe_grad")
        self.assertEqual(estimate.bytes_for(MemoryCategory.INPUT), expected_inputs)
        self.assertEqual(estimate.bytes_for(MemoryCategory.OUTPUT), expected_outputs)
        self.assertEqual(
            estimate.bytes_for(MemoryCategory.EXPLICIT_WORKSPACE), expected_explicit
        )
        self.assertEqual(estimate.bytes_for(MemoryCategory.RUNTIME), expected_runtime)
        self.assertEqual(
            estimate.bytes_for(MemoryCategory.IMPLICIT_WORKSPACE), expected_implicit
        )
        self.assertEqual(
            estimate.bytes_for(MemoryCategory.RESERVATION_OVERHEAD), expected_reservation
        )

    def test_invalid_swiglu_workspace_is_rejected(self) -> None:
        """Test the backward-only workspace capacity is validated."""
        with self.assertRaisesRegex(ValueError, "swiglu_grad_workspace_bytes"):
            MegaMoeGradMemorySpec(
                tp=1, ep=1, seq_size=8, expert_num=2, top_k=1,
                hidden_size=16, intermediate_size=8,
                swiglu_grad_workspace_bytes=-1,
            )

    def test_generic_api_requires_backward_spec(self) -> None:
        """Test backward registration rejects the forward specification type."""
        spec = MegaMoeMemorySpec(
            tp=1, ep=1, seq_size=8, expert_num=2, top_k=1,
            hidden_size=16, intermediate_size=8,
        )
        with self.assertRaisesRegex(TypeError, "requires MegaMoeGradMemorySpec"):
            estimate_mega_kernel_peak_memory("mega_moe_grad", spec)


class TestMegaKernelMemoryEstimatorRegistry(unittest.TestCase):
    """Test extension points used by future mega kernels."""

    def test_custom_estimator_can_be_registered_and_called(self) -> None:
        """Test future kernels can plug in without changing generic dispatch."""
        kernel_name = "unit_test_mega_kernel"

        def estimator(size_bytes):
            return MegaKernelMemoryEstimate(
                kernel_name=kernel_name,
                components=(MemoryComponent("buffer", MemoryCategory.INPUT, size_bytes),),
            )

        register_mega_kernel_memory_estimator(kernel_name, estimator, replace=True)
        estimate = estimate_mega_kernel_peak_memory(kernel_name, 1234)

        self.assertEqual(estimate.peak_bytes, 1234)
        self.assertEqual(estimate.bytes_for(MemoryCategory.INPUT), 1234)

    def test_unknown_kernel_reports_available_estimators(self) -> None:
        """Test unknown names fail before any platform or kernel interaction."""
        with self.assertRaisesRegex(ValueError, "no memory estimator registered"):
            estimate_mega_kernel_peak_memory("missing_kernel", object())

    def test_duplicate_registration_requires_replace(self) -> None:
        """Test accidental estimator overrides are rejected."""
        kernel_name = "unit_test_duplicate_kernel"

        def estimator(_spec):
            return MegaKernelMemoryEstimate(kernel_name=kernel_name, components=())

        register_mega_kernel_memory_estimator(kernel_name, estimator, replace=True)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_mega_kernel_memory_estimator(kernel_name, estimator)
