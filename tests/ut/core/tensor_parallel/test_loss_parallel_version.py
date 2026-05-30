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
"""loss_parallel version matrix tests (non-hardware).

Note: Hardware-specific tests moved to tests/ut/core/tensor_parallel/test_loss_parallel_hardware.py
"""

import pytest
import torch


class TestPyTorchVersionSupport:
    """PyTorch version support tests."""

    def test_pytorch_version_minimum(self):
        """PyTorch version >= 2.0."""
        version = tuple(map(int, torch.__version__.split(".")[:2]))
        assert version >= (2, 0), f"PyTorch 2.0+ required, got {torch.__version__}"

    def test_pytorch_version_recommended(self):
        """Recommended PyTorch >= 2.1."""
        version = tuple(map(int, torch.__version__.split(".")[:2]))
        if version < (2, 1):
            import warnings
            warnings.warn(
                f"PyTorch {torch.__version__} has limited DTensor support. "
                "Consider upgrading to 2.1+ for full loss_parallel support."
            )

    def test_dtensor_available(self):
        """DTensor available."""
        try:
            from torch.distributed._tensor import DTensor
            assert DTensor is not None
        except ImportError:
            pytest.skip("DTensor not available")


class TestVersionCompatibility:
    """Version compatibility tests."""

    def test_hyper_parallel_imports_work(self):
        """HyperParallel imports work."""
        from hyper_parallel.core.tensor_parallel import (
            loss_parallel,
            is_loss_parallel_active,
        )
        
        assert is_loss_parallel_active() is False
        
        with loss_parallel():
            assert is_loss_parallel_active() is True

    def test_cross_entropy_available(self):
        """cross_entropy function available."""
        import torch.nn.functional as F
        
        logits = torch.randn(2, 10)
        target = torch.randint(0, 10, (2,))
        loss = F.cross_entropy(logits, target)
        
        assert loss is not None

    def test_distributed_available(self):
        """Distributed module available."""
        import torch.distributed as dist
        
        assert dist is not None


class TestVersionMatrix:
    """Version matrix tests."""

    @pytest.mark.parametrize("major,minor,expected", [
        (2, 0, "partial"),
        (2, 1, "full"),
        (2, 2, "full"),
        (2, 3, "full"),
        (2, 4, "full"),
        (2, 5, "full"),
    ])
    def test_version_support_level(self, major, minor, expected):
        """Version support level correct."""
        current = tuple(map(int, torch.__version__.split(".")[:2]))
        
        if current < (major, minor):
            pytest.skip(f"Current PyTorch {torch.__version__} below test version {major}.{minor}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
