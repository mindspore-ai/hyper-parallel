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
"""Test that cross_entropy still works after narrowing LOSS_PARALLEL_OP_NAMES."""

import pytest

_SKIP_REASON = "loss_parallel CE dispatch disabled on r1.0.0"


class TestCrossEntropyStillWorks:
    """Verify cross_entropy is still dispatched correctly after registry changes."""

    def test_cross_entropy_in_loss_parallel_op_names(self):
        """cross_entropy should still be in LOSS_PARALLEL_OP_NAMES."""
        from hyper_parallel.core.tensor_parallel._ce_op_registry import (
            LOSS_PARALLEL_OP_NAMES,
            is_loss_parallel_op,
        )

        assert "cross_entropy" in LOSS_PARALLEL_OP_NAMES()
        assert is_loss_parallel_op("cross_entropy") is True

    def test_torch_cross_entropy_in_loss_parallel_op_names(self):
        """torch_cross_entropy should still be in LOSS_PARALLEL_OP_NAMES."""
        from hyper_parallel.core.tensor_parallel._ce_op_registry import (
            LOSS_PARALLEL_OP_NAMES,
            is_loss_parallel_op,
        )

        assert "torch_cross_entropy" in LOSS_PARALLEL_OP_NAMES()
        assert is_loss_parallel_op("torch_cross_entropy") is True

    def test_mint_cross_entropy_in_loss_parallel_op_names(self):
        """mint_nn_functional_cross_entropy should still be in LOSS_PARALLEL_OP_NAMES."""
        from hyper_parallel.core.tensor_parallel._ce_op_registry import (
            LOSS_PARALLEL_OP_NAMES,
            is_loss_parallel_op,
        )

        assert "mint_nn_functional_cross_entropy" in LOSS_PARALLEL_OP_NAMES()
        assert is_loss_parallel_op("mint_nn_functional_cross_entropy") is True

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_cross_entropy_dispatch_path_unchanged(self):
        """cross_entropy should still dispatch through loss_parallel path."""
        import os
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        from hyper_parallel.core.tensor_parallel import loss_parallel

        dispatcher = OpDispatcher()

        with loss_parallel():
            assert dispatcher._should_dispatch_loss_parallel("cross_entropy") is True
            assert dispatcher._should_dispatch_loss_parallel("torch_cross_entropy") is True

    def test_decomposed_ops_not_in_loss_parallel_op_names(self):
        """Decomposed ops should NOT be in LOSS_PARALLEL_OP_NAMES."""
        from hyper_parallel.core.tensor_parallel._ce_op_registry import (
            LOSS_PARALLEL_OP_NAMES,
            DECOMPOSED_CE_OP_NAMES,
            is_loss_parallel_op,
            is_decomposed_ce_op,
        )

        decomposed_ops = ["nll_loss", "log_softmax", "softmax"]
        for op in decomposed_ops:
            assert op not in LOSS_PARALLEL_OP_NAMES(), f"{op} should NOT be in LOSS_PARALLEL_OP_NAMES"
            assert op in DECOMPOSED_CE_OP_NAMES(), f"{op} should be in DECOMPOSED_CE_OP_NAMES"
            assert is_loss_parallel_op(op) is False, f"{op} should not be a loss_parallel op"
            assert is_decomposed_ce_op(op) is True, f"{op} should be a decomposed op"

    def test_registry_sets_are_disjoint(self):
        """LOSS_PARALLEL_OP_NAMES and DECOMPOSED_CE_OP_NAMES should be disjoint."""
        from hyper_parallel.core.tensor_parallel._ce_op_registry import (
            LOSS_PARALLEL_OP_NAMES,
            DECOMPOSED_CE_OP_NAMES,
        )

        entry_set = LOSS_PARALLEL_OP_NAMES()
        decomposed_set = DECOMPOSED_CE_OP_NAMES()
        overlap = entry_set & decomposed_set
        assert len(overlap) == 0, f"Sets should be disjoint, but found overlap: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
