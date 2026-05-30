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
"""loss_parallel context and cache tests."""

import pytest

from hyper_parallel.core.tensor_parallel import (
    loss_parallel,
    is_loss_parallel_active,
)
from hyper_parallel.core.tensor_parallel.loss_parallel import (
    get_loss_parallel_count,
    _get_loss_parallel_token,
)


class TestLossParallelContext:
    """Context manager tests."""

    def test_context_not_active_by_default(self):
        """Context is not active by default."""
        assert is_loss_parallel_active() is False

    def test_context_activation(self):
        """Context is activated after entering."""
        assert is_loss_parallel_active() is False
        with loss_parallel():
            assert is_loss_parallel_active() is True
        assert is_loss_parallel_active() is False

    def test_nested_context(self):
        """Nested contexts are handled correctly."""
        assert is_loss_parallel_active() is False
        
        with loss_parallel():
            assert is_loss_parallel_active() is True
            
            with loss_parallel():
                assert is_loss_parallel_active() is True
            
            assert is_loss_parallel_active() is True
        
        assert is_loss_parallel_active() is False

    def test_context_count(self):
        """Nested count is correct."""
        assert get_loss_parallel_count() == 0
        
        with loss_parallel():
            assert get_loss_parallel_count() == 1
            
            with loss_parallel():
                assert get_loss_parallel_count() == 2
                
                with loss_parallel():
                    assert get_loss_parallel_count() == 3
                
                assert get_loss_parallel_count() == 2
            
            assert get_loss_parallel_count() == 1
        
        assert get_loss_parallel_count() == 0

    def test_context_exception_handling(self):
        """Context exits correctly on exception."""
        assert is_loss_parallel_active() is False
        
        try:
            with loss_parallel():
                assert is_loss_parallel_active() is True
                raise ValueError("test exception")
        except ValueError:
            pass
        
        assert is_loss_parallel_active() is False


class TestLossParallelToken:
    """Cache key token tests."""

    def test_token_outside_context(self):
        """Token is None outside context."""
        token = _get_loss_parallel_token()
        assert token is None

    def test_token_inside_context(self):
        """Token is not None inside context."""
        with loss_parallel():
            token = _get_loss_parallel_token()
            assert token is not None

    def test_token_changes_between_contexts(self):
        """Different contexts have different tokens."""
        with loss_parallel():
            token1 = _get_loss_parallel_token()
        
        with loss_parallel():
            token2 = _get_loss_parallel_token()
        
        assert token1 != token2

    def test_token_same_within_nested_context(self):
        """Token is the same within nested contexts."""
        with loss_parallel():
            token1 = _get_loss_parallel_token()
            
            with loss_parallel():
                token2 = _get_loss_parallel_token()
            
            token3 = _get_loss_parallel_token()
        
        assert token1 == token2 == token3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])