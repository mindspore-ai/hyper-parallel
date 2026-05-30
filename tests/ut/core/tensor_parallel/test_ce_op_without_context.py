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
"""Tests for CE ops with Shard(-1) logits without loss_parallel context."""

import os

import pytest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"


class TestCEOpWithoutLossParallelContext:
    """Test that CE ops with Shard(-1) logits raise error outside loss_parallel context."""

    def test_cross_entropy_with_sharded_logits_without_context_raises_error(self):
        """cross_entropy with Shard(-1) logits should raise error outside loss_parallel."""
        import torch
        import torch.nn.functional as F
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard

        mesh = init_device_mesh("cpu", (1,))
        logits_local = torch.randn(4, 16, requires_grad=True)
        logits_dtensor = DTensor.from_local(logits_local, mesh, [Shard(-1)])
        targets = torch.randint(0, 16, (4,))

        with pytest.raises(ValueError) as exc_info:
            F.cross_entropy(logits_dtensor, targets)

        assert "requires loss_parallel context" in str(exc_info.value)
        assert "Shard(-1)" in str(exc_info.value)
        assert "with loss_parallel():" in str(exc_info.value)

    def test_cross_entropy_with_sharded_logits_in_context_works(self):
        """cross_entropy with Shard(-1) logits should work in loss_parallel context."""
        import torch
        import torch.nn.functional as F
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel.core.tensor_parallel import loss_parallel

        mesh = init_device_mesh("cpu", (1,))
        logits_local = torch.randn(4, 16, requires_grad=True)
        logits_dtensor = DTensor.from_local(logits_local, mesh, [Shard(-1)])

        # Skip the actual execution since it requires distributed backend
        # Just verify the check doesn't raise
        with loss_parallel(mesh=mesh):
            # The check should not raise
            from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
            _OP_DISPATCHER._check_ce_op_without_loss_parallel_context("cross_entropy", (logits_dtensor,))

    def test_cross_entropy_with_replicated_logits_no_error(self):
        """cross_entropy with Replicate logits should not raise error."""
        import torch
        import torch.nn.functional as F
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Replicate

        mesh = init_device_mesh("cpu", (1,))
        logits_local = torch.randn(4, 16, requires_grad=True)
        logits_dtensor = DTensor.from_local(logits_local, mesh, [Replicate()])
        targets = torch.randint(0, 16, (4,))

        loss = F.cross_entropy(logits_dtensor, targets)

        assert loss is not None
        assert loss.item() >= 0

    def test_cross_entropy_with_regular_tensor_no_error(self):
        """cross_entropy with regular tensor should not raise error."""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(4, 16, requires_grad=True)
        targets = torch.randint(0, 16, (4,))

        loss = F.cross_entropy(logits, targets)

        assert loss is not None
        assert loss.item() >= 0

    def test_error_message_contains_backward_hint(self):
        """Error message should mention that backward also needs to be in context."""
        import torch
        import torch.nn.functional as F
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard

        mesh = init_device_mesh("cpu", (1,))
        logits_local = torch.randn(4, 16, requires_grad=True)
        logits_dtensor = DTensor.from_local(logits_local, mesh, [Shard(-1)])
        targets = torch.randint(0, 16, (4,))

        with pytest.raises(ValueError) as exc_info:
            F.cross_entropy(logits_dtensor, targets)

        error_msg = str(exc_info.value)
        assert "loss.backward()" in error_msg

    def test_gather_not_called_on_error(self):
        """Verify that full_tensor() is not called when error is raised."""
        import torch
        import torch.nn.functional as F
        from unittest.mock import patch, MagicMock
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard

        mesh = init_device_mesh("cpu", (1,))
        logits_local = torch.randn(4, 16, requires_grad=True)
        logits_dtensor = DTensor.from_local(logits_local, mesh, [Shard(-1)])
        targets = torch.randint(0, 16, (4,))

        with patch.object(DTensor, 'full_tensor') as mock_full_tensor:
            mock_full_tensor.return_value = logits_local

            with pytest.raises(ValueError):
                F.cross_entropy(logits_dtensor, targets)

            mock_full_tensor.assert_not_called()

    def test_dispatch_check_function_exists(self):
        """Verify the check function exists."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        dispatcher = OpDispatcher()
        assert hasattr(dispatcher, '_check_ce_op_without_loss_parallel_context')
        assert callable(dispatcher._check_ce_op_without_loss_parallel_context)


class TestCEOpCheckLogic:
    """Test the internal check logic."""

    def test_check_raises_for_cross_entropy_with_shard_minus1(self):
        """Check should raise for cross_entropy with Shard(-1)."""
        import torch
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 16)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(-1)])

        dispatcher = OpDispatcher()

        with pytest.raises(ValueError) as exc_info:
            dispatcher._check_ce_op_without_loss_parallel_context("cross_entropy", (dtensor,))

        assert "requires loss_parallel context" in str(exc_info.value)

    def test_check_no_raise_for_torch_cross_entropy_with_shard_minus1_in_context(self):
        """Check should not raise in loss_parallel context."""
        import torch
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel.core.tensor_parallel import loss_parallel

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 16)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(-1)])

        dispatcher = OpDispatcher()

        with loss_parallel():
            dispatcher._check_ce_op_without_loss_parallel_context("cross_entropy", (dtensor,))

    def test_check_no_raise_for_replicated_tensor(self):
        """Check should not raise for Replicate tensor."""
        import torch
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Replicate

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 16)
        dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])

        dispatcher = OpDispatcher()
        dispatcher._check_ce_op_without_loss_parallel_context("cross_entropy", (dtensor,))

    def test_check_no_raise_for_non_ce_op(self):
        """Check should not raise for non-CE op."""
        import torch
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel import init_device_mesh
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Shard

        mesh = init_device_mesh("cpu", (1,))
        local_tensor = torch.randn(4, 16)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(-1)])

        dispatcher = OpDispatcher()
        dispatcher._check_ce_op_without_loss_parallel_context("matmul", (dtensor, dtensor))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
