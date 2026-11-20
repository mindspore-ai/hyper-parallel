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
"""Guard the numerical gate itself and audit native/backend fixture coverage."""

from dataclasses import replace
import unittest

import torch

from hyper_parallel.components.ops import ops
from tests.common.op_precision import assert_precision, check_case
from tests.common.op_precision_suite import FAMILY_BUILDERS, precision_cases, registered_backends


class TestPrecisionChecks(unittest.TestCase):
    """Ensure precision tests reject wrong metadata, gradients and diluted error."""

    def test_every_native_variant_has_an_independent_oracle(self) -> None:
        """Registration changes must explicitly extend the precision inventory."""
        expected = {(family, variant) for family, variant, backend, _ in registered_backends() if backend == "native"}
        actual = {(case.family, case.variant) for family in FAMILY_BUILDERS
                  for case in precision_cases(family, torch.float32)}
        self.assertEqual(actual, expected)

    def test_every_backend_variant_has_a_hardware_fixture(self) -> None:
        """Every registered accelerator variant receives a real parity case."""
        expected = {(family, variant) for family, variant, backend, _ in registered_backends() if backend != "native"}
        actual = {(case.family, case.variant) for family in FAMILY_BUILDERS
                  for case in precision_cases(family, torch.bfloat16, hardware=True)}
        self.assertEqual(actual, expected)

    def test_pointwise_success_does_not_bypass_relative_l2(self) -> None:
        """A small systematic drift can pass allclose while exceeding the L2 budget."""
        reference = torch.ones(128)
        actual = reference + 5e-5
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
        with self.assertRaisesRegex(AssertionError, "relative_l2=.*limit="):
            assert_precision(actual, reference, dtype=torch.float32)

    def test_relative_l2_success_does_not_bypass_pointwise(self) -> None:
        """A large tensor must not conceal one incorrect near-zero element."""
        reference = torch.ones(2048)
        reference[0] = 0.0
        actual = reference.clone()
        actual[0] = 1e-4
        with self.assertRaisesRegex(AssertionError, "worst_flat_index=0"):
            assert_precision(actual, reference, dtype=torch.float32)

    def test_zero_reference_requires_exact_zero(self) -> None:
        """Masked rows and empty-expert gradients cannot pass using absolute tolerance."""
        with self.assertRaisesRegex(AssertionError, "structural zero"):
            assert_precision(torch.full((4,), 1e-9), torch.zeros(4), dtype=torch.float32)

    def test_dtype_and_discrete_values_are_contracts(self) -> None:
        """Neither casting outputs nor numerical tolerances may conceal metadata errors."""
        with self.assertRaisesRegex(AssertionError, "expected="):
            assert_precision(torch.ones(2, dtype=torch.float16), torch.ones(2), dtype=torch.float16)
        with self.assertRaisesRegex(AssertionError, "discrete"):
            assert_precision(torch.tensor([-1, 1]), torch.tensor([-1, 0]), dtype=torch.float32)
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            assert_precision(torch.tensor([float("nan")]), torch.zeros(1), dtype=torch.float32)

    def test_equal_forward_with_broken_backward_fails_and_restores_config(self) -> None:
        """The gate must detect detach/zero gradients even if all output values agree."""
        case = precision_cases("rms_norm", torch.float32)[0]

        def detached_reference(x, weight, eps):
            return case.reference(x, weight, eps).detach() + x * 0.0

        previous = ops.get_backends()
        self.addCleanup(ops.set_backends, previous)
        ops.set_backends({"swiglu": "native"})
        with self.assertRaisesRegex(AssertionError, "gradient"):
            check_case(replace(case, reference=detached_reference))
        self.assertEqual(ops.get_backends(), {"swiglu": "native"})

    def test_case_does_not_change_global_random_state(self) -> None:
        """Fixtures and VJPs must be reproducible independently of test ordering."""
        state = torch.random.get_rng_state()
        check_case(precision_cases("rms_norm", torch.float32)[0])
        torch.testing.assert_close(torch.random.get_rng_state(), state, atol=0, rtol=0)
