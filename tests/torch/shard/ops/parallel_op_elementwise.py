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
"""test torch dtensor with distributed linear"""

import numpy as np
import torch
from torch import nn
from torch import optim
from hyper_parallel import DTensor, Layout, SkipDTensorDispatch, hsdp
from tests.torch.utils import init_dist
from tests.torch.shard.utils import global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
standalone_x_np = np.random.randn(16, 8).astype(np.float32)

class SimpleModel(nn.Module):
    """simple model with elementwise operators"""
    def __init__(self, dist=False, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 8).npu())   # shape: (out_features, in_features)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(4).npu())    # shape: (out_features)
        else:
            self.bias = None
        self.dist = dist

    def _add_operations(self, tensor):
        """Test various add operations, preserving original tensor if needed"""
        # Save original tensor for potential combination
        original = tensor.clone()

        # 1. Add scalar value
        result = tensor + 2.5

        # 2. Add tensor to itself (x + x)
        result = result + result

        # 3. Add with broadcast
        broadcast_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).npu()
        result = result + broadcast_tensor

        # 4. In-place add
        result += 1.0

        # 5. Add with negative value
        result = result + (-0.5)

        # 6. Add with zero (identity operation)
        result = result + 0.0

        # 7. Chain add operations
        result = result + 0.1 + 0.2

        # 8. Add with different data types (int to float)
        result = result + 3

        # 9. Add with complex expression
        result = result + (2.0 * 3.0 - 1.0)

        # Combine with original if needed to maintain range
        result = 0.7 * result + 0.3 * original

        return result

    def _and_operations(self, tensor):
        """Test __and__ (bitwise AND) operations while preserving float type and range"""
        # Save original tensor
        original = tensor.clone()

        # Convert to int for bitwise operations
        int_tensor = tensor.int()

        # Test with scalar
        int_tensor = int_tensor & 3

        # Test with tensor
        mask = torch.tensor([1, 2, 3, 4], dtype=torch.int32).npu()
        int_tensor = int_tensor & mask

        # Convert back to float and combine with original
        result = int_tensor.float()

        # Combine with original to maintain reasonable range
        # Weighted combination: 30% bitwise result + 70% original
        return 0.3 * result + 0.7 * original

    def _eq_operations(self, tensor):
        """Test __eq__ (equality) operations while preserving float type"""
        # Save original tensor
        original = tensor.clone()

        # Test with scalar
        eq_result = tensor == 0.5

        # Test with tensor
        eq_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0]).npu()
        eq_result = tensor == eq_tensor

        # Use equality result for masking
        result = tensor.masked_fill(eq_result, 0.0)

        # Combine with original to avoid all zeros
        return 0.6 * result + 0.4 * original

    def _floordiv_operations(self, tensor):
        """Test __floordiv__ (floor division) operations"""
        # Save original tensor
        original = tensor.clone()

        # Avoid division by zero by ensuring positive divisor
        # Add small positive value to avoid zero or negative
        tensor_positive = tensor + tensor.abs() + 1.0

        with torch.no_grad():
            # Test with scalar
            result_scalar = tensor_positive // 2.0

            # Test with tensor
            divisor_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0]).npu()
            result_tensor = result_scalar // divisor_tensor

        result = result_tensor + 0.0 * original
        # Combine with original to maintain range
        return 0.8 * result + 0.2 * original

    def _invert_operations(self, tensor):
        """Test __invert__ (bitwise NOT) operations while preserving float type"""
        # Save original tensor
        original = tensor.clone()

        # Convert to int, apply invert, convert back
        int_tensor = tensor.int()
        int_tensor = ~int_tensor

        # Convert back to float and combine with original
        result = int_tensor.float()

        # Combine with original to maintain reasonable range
        return 0.3 * result + 0.7 * original

    def _or_operations(self, tensor):
        """Test __or__ (bitwise OR) operations while preserving float type"""
        # Save original tensor
        original = tensor.clone()

        # Convert to int
        int_tensor = tensor.int()

        # Test with scalar
        int_tensor = int_tensor | 5

        # Test with tensor
        or_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
        int_tensor = int_tensor | or_mask

        # Convert back to float and combine with original
        result = int_tensor.float()

        # Combine with original to maintain reasonable range
        return 0.3 * result + 0.7 * original

    def _rsub_operations(self, tensor):
        """Test __rsub__ (reverse subtraction) operations"""
        # Save original tensor
        original = tensor.clone()

        # Test with scalar
        result = 5.0 - tensor  # Use reasonable constant

        # Test with tensor
        sub_from = torch.tensor([5.0, 10.0, 15.0, 20.0]).npu()
        result = sub_from - result

        # Combine with original to maintain reasonable range
        return 0.7 * result + 0.3 * original

    def _clone_operations(self, tensor):
        """Test clone operations"""
        # Basic clone
        cloned = tensor.clone()

        # Clone and modify
        cloned = cloned + 0.5  # Small modification

        # Use clone in computation
        result = tensor + cloned

        return result

    def _gelu_operations(self, tensor):
        """Test gelu operations"""
        # Basic GELU
        result = torch.nn.functional.gelu(tensor)

        # GELU with approximate (if available)
        result = torch.nn.functional.gelu(result, approximate='tanh')

        return result

    def _gt_operations(self, tensor):
        """Test gt (greater than) operations"""
        # Save original tensor
        original = tensor.clone()

        # Test with scalar
        gt_mask = tensor > 0.0  # Compare with 0 to get reasonable mask

        # Test with tensor
        threshold = torch.tensor([0.0, 0.5, 1.0, 1.5]).npu()
        gt_mask = tensor > threshold

        # Use gt result - replace values > threshold with 1.0
        result = tensor.masked_fill(gt_mask, 1.0)

        # Combine with original to avoid losing all information
        return 0.5 * result + 0.5 * original

    def _le_operations(self, tensor):
        """Test le (less than or equal) operations"""
        # Save original tensor
        original = tensor.clone()

        # Test with scalar
        le_mask = tensor <= 1.0

        # Test with tensor
        le_threshold = torch.tensor([0.5, 1.0, 1.5, 2.0]).npu()
        le_mask = tensor <= le_threshold

        # Use le result
        result = tensor.masked_fill(le_mask, 0.5)

        # Combine with original
        return 0.6 * result + 0.4 * original

    def _long_operations(self, tensor):
        """Test long (convert to long) operations"""
        # Convert to long and back
        long_tensor = tensor.long()

        # Perform operations on long tensor
        long_tensor = long_tensor + 1

        # Convert back to float
        result = long_tensor.float()

        # Combine with original to preserve information
        return 0.5 * result + 0.5 * tensor

    def _lt_operations(self, tensor):
        """Test lt (less than) operations"""
        # Save original tensor
        original = tensor.clone()

        # Test with scalar
        lt_mask = tensor < 1.0

        # Test with tensor
        lt_threshold = torch.tensor([1.0, 2.0, 3.0, 4.0]).npu()
        lt_mask = tensor < lt_threshold

        # Use lt result
        result = tensor.masked_fill(lt_mask, 0.8)

        # Combine with original
        return 0.6 * result + 0.4 * original

    def _masked_fill_operations(self, tensor):
        """Test masked_fill operations"""
        # Create different masks
        mask1 = tensor > 1.0
        mask2 = tensor < 0.0
        mask3 = (tensor >= 0.0) & (tensor <= 1.0)

        # Test with scalar value
        result = tensor.clone()
        result = result.masked_fill(mask1, 1.0)  # Cap large values at 1.0
        result = result.masked_fill(mask2, 0.0)  # Set negative values to 0.0
        result = result.masked_fill(mask3, 0.5)  # Set mid-range to 0.5

        # Test with tensor value
        fill_values = torch.tensor(0.1).npu()
        result = result.masked_fill(mask1, fill_values)

        return result

    def _masked_fill_inplace_operations(self, tensor):
        """Test masked_fill_ (in-place) operations"""
        # Create a copy for in-place operations
        result = tensor.clone()

        # Create masks
        mask1 = result > 1.0
        mask2 = result < 0.0

        # In-place with scalar
        result.masked_fill_(mask1, 1.0)  # Cap at 1.0

        # In-place with tensor
        fill_values = torch.tensor(0.1).npu()
        result.masked_fill_(mask2, fill_values)

        return result

    def _pow_operations(self, tensor):
        """Test pow (power) operations"""
        # Ensure positive values for pow
        positive_tensor = tensor.abs() + 0.1

        # Test with scalar exponent
        result = positive_tensor.pow(0.5)  # Use sqrt instead of square to avoid large values

        # Test with tensor exponent
        exponents = torch.tensor([0.5, 1.0, 1.5, 2.0]).npu()
        result = result.pow(exponents)

        return result

    def _rsqrt_operations(self, tensor):
        """Test rsqrt (reciprocal square root) operations"""
        # Add small epsilon to avoid division by zero
        positive_tensor = tensor.abs() + 0.1

        # Test rsqrt
        result = positive_tensor.rsqrt()

        # Chain with other operations
        result = result.rsqrt()

        return result

    def _sub_inplace_operations(self, tensor):
        """Test sub_ (in-place subtraction) operations"""
        # Create a copy for in-place operations
        result = tensor.clone()

        # Test with scalar (small subtraction)
        result.sub_(0.1)

        # Test with tensor (small values)
        sub_values = torch.tensor([0.01, 0.02, 0.03, 0.04]).npu()
        result.sub_(sub_values)

        # Chain sub_ with small values
        result.sub_(0.05).sub_(0.025)

        return result

    def _unsqueeze_operations(self, tensor):
        """Test unsqueeze operations"""
        # Test different dimensions
        result = tensor.unsqueeze(0)      # Add dimension at beginning
        result = result.unsqueeze(-1)     # Add dimension at end
        result = result.unsqueeze(1)      # Add dimension in middle

        # Chain unsqueeze operations
        result = result.unsqueeze(0).unsqueeze(-1)

        # Use unsqueeze for broadcasting
        broadcast_target = torch.ones(1, 4, 1, 1).npu() * 0.1  # Small random values
        result = result + broadcast_target

        # Squeeze back to original shape
        return result.squeeze()

    def forward(self, x):
        """
        x: shape (batch, in_features)
        weight: (out_features, in_features) - DTensor
        bias:   (out_features) - DTensor or local
        """
        # Linear operation
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # ================== Before reduce_partial ==================
        # Test add operations (support partial state)
        out = self._add_operations(out)

        # ================== reduce_partial ==================
        if self.dist and isinstance(out, DTensor):
            out = out.reduce_partial()

        # ================== After reduce_partial ==================
        # Test all interfaces with various usages

        # Test add operations again after reduce_partial
        out = self._add_operations(out)

        # Test all other operations
        out = self._and_operations(out)
        out = self._eq_operations(out)
        out = self._floordiv_operations(out)
        out = self._invert_operations(out)
        out = self._or_operations(out)
        out = self._rsub_operations(out)
        out = self._clone_operations(out)
        out = self._gelu_operations(out)
        out = self._gt_operations(out)
        out = self._le_operations(out)
        out = self._long_operations(out)
        out = self._lt_operations(out)
        out = self._masked_fill_operations(out)
        out = self._masked_fill_inplace_operations(out)
        out = self._pow_operations(out)
        out = self._rsqrt_operations(out)
        out = self._sub_inplace_operations(out)
        out = self._unsqueeze_operations(out)

        # Final operations with safe ranges
        out = torch.relu(out)
        out = torch.sum(out)
        return out


def test_distributed_linear_with_elementwise_ops():
    """
    Feature: dtensor + torch.nn.functional.linear with bias
    Description:
        - Train a simple model: Linear → ReLU → Sum
        - Compare loss between single-machine and distributed training
    Expectation: Run successfully.
    """
    init_dist()
    step = 10

    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    standalone_x = torch.from_numpy(standalone_x_np).npu()

    for _ in range(step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    dist_model = SimpleModel(dist=True).npu()

    # weight:  (8,4)  -> split on second dim
    local_w = torch.ones(4, 4).npu()
    # bias:   (4,)   -> broadcast (no split)
    local_b = torch.zeros(4).npu()

    layout = Layout((4, 2), ("dp", "tp"))
    w_layout = layout("None", "tp")
    b_layout = layout("None",)
    x_layout = layout("dp", "tp")

    dist_model.register_parameter(
        "weight",
        nn.Parameter(DTensor.from_local(local_w, w_layout.mesh, w_layout.placements)),
    )
    dist_model.register_parameter(
        "bias",
        nn.Parameter(DTensor.from_local(local_b, b_layout.mesh, b_layout.placements)),
    )

    dist_x = global_to_local(standalone_x, x_layout)

    dist_model = hsdp(dist_model)
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range(step):
        dist_loss = dist_model(dist_x)
        dist_loss = dist_loss.reduce_partial()

        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num)
        dist_loss.backward(backward_input)

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    # loss
    assert np.allclose(
        standalone_loss.cpu().detach().numpy(),
        dist_loss.to_local().cpu().detach().numpy(),
        atol=1e-3,
        rtol=1e-3,
    ), "loss mismatch between standalone and distributed"
