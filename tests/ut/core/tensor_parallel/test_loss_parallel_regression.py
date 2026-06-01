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
"""loss_parallel regression tests (non-hardware).

Note: Hardware-specific tests moved to tests/ut/core/tensor_parallel/test_loss_parallel_hardware.py
"""

import pytest
import torch
import torch.nn.functional as F

from hyper_parallel.core.tensor_parallel import (
    loss_parallel,
    is_loss_parallel_active,
)


class TestBackwardCompatibility:
    """Backward compatibility tests."""

    def test_import_paths_stable(self):
        """Import paths stable."""
        # Reimport to test import path stability
        from hyper_parallel.core.tensor_parallel import (  # pylint: disable=W0404,W0621
            loss_parallel,
            is_loss_parallel_active,
        )

    def test_api_signature_stable(self):
        """API signature stable."""
        import inspect
        from hyper_parallel.core.tensor_parallel import loss_parallel  # pylint: disable=W0404,W0621
        
        sig = inspect.signature(loss_parallel)
        params = list(sig.parameters.keys())
        
        assert "mesh" in params
        assert "strict" in params

    def test_default_behavior_unchanged(self):
        """Default behavior unchanged."""
        from hyper_parallel.core.tensor_parallel import (  # pylint: disable=W0404,W0621
            loss_parallel,
            is_loss_parallel_active,
        )
        
        with loss_parallel():
            assert is_loss_parallel_active() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
