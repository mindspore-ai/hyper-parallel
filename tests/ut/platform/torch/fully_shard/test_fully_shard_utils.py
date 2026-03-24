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
"""Unit tests for fully_shard utils (no NPU required).

Covers MixedPrecisionPolicy, OffloadPolicy, and CPUOffloadPolicy from
hyper_parallel.core.fully_shard.utils. All tests run on CPU with mocked platform.
"""
import os
import unittest

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    OffloadPolicy,
    CPUOffloadPolicy,
)


class TestMixedPrecisionPolicy(unittest.TestCase):
    """Unit tests for MixedPrecisionPolicy (param/reduce/output dtypes, cast flags)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_default_values(self):
        """Default policy has None dtypes and cast_forward_inputs=True.

        description: Instantiate MixedPrecisionPolicy with no args.
        expectation: param_dtype, reduce_dtype, output_dtype are None; cast_forward_inputs True.
        feature: MixedPrecisionPolicy default configuration.
        """
        # Act
        policy = MixedPrecisionPolicy()
        # Assert
        self.assertIsNone(policy.param_dtype)
        self.assertIsNone(policy.reduce_dtype)
        self.assertIsNone(policy.output_dtype)
        self.assertTrue(policy.cast_forward_inputs)
        self.assertFalse(policy.apply_grad_on_fp32_main_grad)

    def test_custom_param_dtype(self):
        """Custom param_dtype is stored correctly.

        description: Create policy with param_dtype=torch.float16.
        expectation: policy.param_dtype equals torch.float16.
        feature: MixedPrecisionPolicy param_dtype.
        """
        # Act
        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        # Assert
        self.assertEqual(policy.param_dtype, torch.float16)

    def test_custom_reduce_dtype(self):
        """Custom reduce_dtype is stored correctly.

        description: Create policy with reduce_dtype=torch.bfloat16.
        expectation: policy.reduce_dtype equals torch.bfloat16.
        feature: MixedPrecisionPolicy reduce_dtype.
        """
        # Act
        policy = MixedPrecisionPolicy(reduce_dtype=torch.bfloat16)
        # Assert
        self.assertEqual(policy.reduce_dtype, torch.bfloat16)

    def test_custom_output_dtype(self):
        """Custom output_dtype is stored correctly.

        description: Create policy with output_dtype=torch.float16.
        expectation: policy.output_dtype equals torch.float16.
        feature: MixedPrecisionPolicy output_dtype.
        """
        # Act
        policy = MixedPrecisionPolicy(output_dtype=torch.float16)
        # Assert
        self.assertEqual(policy.output_dtype, torch.float16)

    def test_cast_forward_inputs_false(self):
        """cast_forward_inputs can be set to False.

        description: Create policy with cast_forward_inputs=False.
        expectation: policy.cast_forward_inputs is False.
        feature: MixedPrecisionPolicy cast_forward_inputs flag.
        """
        # Act
        policy = MixedPrecisionPolicy(cast_forward_inputs=False)
        # Assert
        self.assertFalse(policy.cast_forward_inputs)

    def test_full_mixed_precision_config(self):
        """Full mixed precision config is stored correctly.

        description: Create policy with param_dtype, reduce_dtype, output_dtype, cast_forward_inputs.
        expectation: All set values are stored and readable.
        feature: MixedPrecisionPolicy full configuration.
        """
        # Act
        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float16,
            cast_forward_inputs=True,
        )
        # Assert
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertEqual(policy.output_dtype, torch.float16)


class TestOffloadPolicy(unittest.TestCase):
    """Unit tests for OffloadPolicy (base offload policy, no device-specific options)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_offload_policy_instantiation(self):
        """OffloadPolicy can be instantiated (base class).

        description: Instantiate OffloadPolicy with no args.
        expectation: Instance is of type OffloadPolicy.
        feature: OffloadPolicy base class.
        """
        # Act
        policy = OffloadPolicy()
        # Assert
        self.assertIsInstance(policy, OffloadPolicy)


class TestCPUOffloadPolicy(unittest.TestCase):
    """Unit tests for CPUOffloadPolicy (pin_memory and inheritance from OffloadPolicy)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_default_pin_memory_true(self):
        """Default CPUOffloadPolicy has pin_memory=True.

        description: Instantiate CPUOffloadPolicy with no args.
        expectation: pin_memory is True and instance is OffloadPolicy.
        feature: CPUOffloadPolicy default pin_memory.
        """
        # Act
        policy = CPUOffloadPolicy()
        # Assert
        self.assertTrue(policy.pin_memory)
        self.assertIsInstance(policy, OffloadPolicy)

    def test_pin_memory_false(self):
        """CPUOffloadPolicy can set pin_memory=False.

        description: Instantiate CPUOffloadPolicy(pin_memory=False).
        expectation: policy.pin_memory is False.
        feature: CPUOffloadPolicy pin_memory option.
        """
        # Act
        policy = CPUOffloadPolicy(pin_memory=False)
        # Assert
        self.assertFalse(policy.pin_memory)

    def test_inherits_from_offload_policy(self):
        """CPUOffloadPolicy inherits from OffloadPolicy.

        description: Create CPUOffloadPolicy and check type.
        expectation: isinstance(policy, OffloadPolicy) is True.
        feature: CPUOffloadPolicy inheritance.
        """
        # Act
        policy = CPUOffloadPolicy()
        # Assert
        self.assertIsInstance(policy, OffloadPolicy)


if __name__ == "__main__":
    unittest.main()
