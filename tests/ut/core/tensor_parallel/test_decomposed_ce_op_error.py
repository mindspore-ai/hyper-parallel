# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Tests for decomposed CE op error handling in loss_parallel context."""

import pytest

from hyper_parallel.core.tensor_parallel import loss_parallel
from hyper_parallel.core.tensor_parallel._ce_op_registry import (
    is_loss_parallel_op,
    is_decomposed_ce_op,
    LOSS_PARALLEL_OP_NAMES,
    DECOMPOSED_CE_OP_NAMES,
)

_SKIP_REASON = "loss_parallel CE dispatch disabled on r1.0.0"


class TestDecomposedCEOpRegistry:
    """Test the separation of CE entry points and decomposed ops."""

    def test_cross_entropy_is_entry_point(self):
        """cross_entropy should be in LOSS_PARALLEL_OP_NAMES."""
        assert "cross_entropy" in LOSS_PARALLEL_OP_NAMES()
        assert is_loss_parallel_op("cross_entropy") is True

    def test_torch_cross_entropy_is_entry_point(self):
        """torch_cross_entropy should be in LOSS_PARALLEL_OP_NAMES."""
        assert "torch_cross_entropy" in LOSS_PARALLEL_OP_NAMES()
        assert is_loss_parallel_op("torch_cross_entropy") is True

    def test_log_softmax_is_decomposed(self):
        """log_softmax should be in DECOMPOSED_CE_OP_NAMES."""
        assert "log_softmax" in DECOMPOSED_CE_OP_NAMES()
        assert is_decomposed_ce_op("log_softmax") is True
        assert is_loss_parallel_op("log_softmax") is False

    def test_nll_loss_is_decomposed(self):
        """nll_loss should be in DECOMPOSED_CE_OP_NAMES."""
        assert "nll_loss" in DECOMPOSED_CE_OP_NAMES()
        assert is_decomposed_ce_op("nll_loss") is True
        assert is_loss_parallel_op("nll_loss") is False

    def test_softmax_is_decomposed(self):
        """softmax should be in DECOMPOSED_CE_OP_NAMES."""
        assert "softmax" in DECOMPOSED_CE_OP_NAMES()
        assert is_decomposed_ce_op("softmax") is True
        assert is_loss_parallel_op("softmax") is False

    def test_backward_ops_are_decomposed(self):
        """Backward ops should be in DECOMPOSED_CE_OP_NAMES."""
        decomposed_backward_ops = [
            "nll_loss_forward",
            "nll_loss_backward",
            "log_softmax_backward",
            "softmax_backward",
        ]
        for op_name in decomposed_backward_ops:
            assert op_name in DECOMPOSED_CE_OP_NAMES(), f"{op_name} should be decomposed"
            assert is_decomposed_ce_op(op_name) is True
            assert is_loss_parallel_op(op_name) is False

    def test_entry_and_decomposed_sets_are_disjoint(self):
        """ENTRY and DECOMPOSED sets should not overlap."""
        entry_set = LOSS_PARALLEL_OP_NAMES()
        decomposed_set = DECOMPOSED_CE_OP_NAMES()
        overlap = entry_set & decomposed_set
        assert len(overlap) == 0, f"Sets should be disjoint, but overlap: {overlap}"


class TestDecomposedCEOpDispatchError:
    """Test that decomposed ops raise ValueError in loss_parallel context."""

    def test_dispatch_check_function_exists(self):
        """Test that the check function is called in dispatch flow."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        dispatcher = OpDispatcher()

        assert hasattr(dispatcher, '_check_decomposed_ce_op_in_loss_parallel')
        assert callable(dispatcher._check_decomposed_ce_op_in_loss_parallel)

    def test_check_does_not_raise_outside_context(self):
        """Check should not raise outside loss_parallel context."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        dispatcher = OpDispatcher()

        dispatcher._check_decomposed_ce_op_in_loss_parallel("nll_loss", (), {})
        dispatcher._check_decomposed_ce_op_in_loss_parallel("log_softmax", (), {})

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_check_raises_with_sharded_dtensor_in_context(self):
        """Check should raise ValueError with Shard(-1) DTensor in context."""
        import os
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel import init_device_mesh
        import torch

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 8)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(-1)])

        dispatcher = OpDispatcher()

        with loss_parallel():
            with pytest.raises(ValueError) as exc_info:
                dispatcher._check_decomposed_ce_op_in_loss_parallel(
                    "nll_loss", (dtensor,), {}
                )

            assert "decomposed component of cross_entropy" in str(exc_info.value)
            assert "Use F.cross_entropy" in str(exc_info.value)

    def test_decomposed_op_no_error_outside_context(self):
        """Decomposed ops should work fine outside loss_parallel context."""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(4, 8, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.randint(0, 8, (4,))
        loss = F.nll_loss(log_probs, targets)

        assert loss.item() is not None
        loss.backward()
        assert logits.grad is not None

    def test_check_does_not_raise_with_replicated_dtensor(self):
        """Check should not raise with Replicate DTensor in context."""
        import os
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Replicate
        from hyper_parallel import init_device_mesh
        import torch

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 8)
        dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])

        dispatcher = OpDispatcher()

        with loss_parallel():
            dispatcher._check_decomposed_ce_op_in_loss_parallel(
                "log_softmax", (dtensor,), {}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
