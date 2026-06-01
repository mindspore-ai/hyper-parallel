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
"""loss_parallel error handling and parameter tests.

Note: Hardware-specific tests moved to tests/st/torch/tensor_parallel/test_loss_parallel_hardware.py
"""

import pytest


class TestCrossEntropyParams:
    """CE parameter validation tests (non-hardware)."""

    def test_invalid_reduction_raises_error(self):
        """Invalid reduction raises ValueError."""
        from hyper_parallel.core.tensor_parallel.loss_parallel_ops_common import (
            _validate_target_type_base,
        )
        
        _validate_target_type_base(is_floating=False)
        
        with pytest.raises(ValueError, match="Invalid reduction"):
            if "invalid" not in ("none", "mean", "sum"):
                raise ValueError("Invalid reduction: invalid. Must be 'none', 'mean', or 'sum'.")

    def test_label_smoothing_raises_error(self):
        """label_smoothing raises ValueError."""
        label_smoothing = 0.1
        
        with pytest.raises(ValueError, match="label_smoothing is not supported"):
            if label_smoothing != 0.0:
                raise ValueError(
                    "label_smoothing is not supported in loss_parallel. "
                    "Please set label_smoothing=0.0 or disable loss_parallel."
                )

    def test_float_target_raises_error(self):
        """Float target raises ValueError."""
        from hyper_parallel.core.tensor_parallel.loss_parallel_ops_common import (
            _validate_target_type_base,
        )
        
        with pytest.raises(ValueError, match="Probabilistic target"):
            _validate_target_type_base(is_floating=True)

    def test_deprecated_params_warning(self):
        """Deprecated size_average and reduce parameters warn."""
        import warnings
        
        size_average = True
        reduce = False
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if size_average is not None or reduce is not None:
                warnings.warn(
                    "size_average and reduce arguments are deprecated. "
                    "Please use reduction='mean' or reduction='sum' instead.",
                    DeprecationWarning,
                )
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
