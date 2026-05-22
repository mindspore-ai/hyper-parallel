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
"""Unit tests for hyper_parallel.custom_ops.experimental.experimental_ops.

Each function is a thin delegation wrapper around ``_platform.custom_ops.<method>``.
Tests mock the module-level ``_platform`` in ``experimental_ops`` to verify:

- The correct ``custom_ops`` method is invoked.
- All positional and keyword arguments are forwarded unchanged.
- The return value from the platform is propagated to the caller.

No NPU hardware or real MindSpore runtime is required.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.custom_ops.experimental.experimental_ops import (  # noqa: E402
    _MAX_INT64,
    npu_dense_lightning_indexer_grad_kl_loss,
    npu_dense_lightning_indexer_softmax_lse,
    npu_mhc_post,
    npu_mhc_pre_sinkhorn,
    npu_sparse_lightning_indexer_grad_kl_loss,
)

_PATCH_TARGET = "hyper_parallel.custom_ops.experimental.experimental_ops._platform"


class TestNpuDenseLightningIndexerSoftmaxLse(unittest.TestCase):
    """
    Feature: npu_dense_lightning_indexer_softmax_lse wrapper.
    Description: Verify all arguments are forwarded to _platform.custom_ops and
                 the return value is propagated.
    Expectation: Exactly one call with the correct arguments; return value passed through.
    """

    @patch(_PATCH_TARGET)
    def test_required_args_only_uses_defaults(self, mock_platform):
        """Only required positional args — keyword args must default correctly."""
        expected = (MagicMock(), MagicMock())
        mock_platform.custom_ops.npu_dense_lightning_indexer_softmax_lse.return_value = expected

        result = npu_dense_lightning_indexer_softmax_lse("q_idx", "k_idx", "weights")

        mock_platform.custom_ops.npu_dense_lightning_indexer_softmax_lse.assert_called_once_with(
            "q_idx", "k_idx", "weights",
            None, None,
            "BSND", 3, _MAX_INT64, _MAX_INT64,
        )
        self.assertIs(result, expected)

    @patch(_PATCH_TARGET)
    def test_all_kwargs_forwarded(self, mock_platform):
        """All optional keyword arguments are forwarded to the platform."""
        mock_platform.custom_ops.npu_dense_lightning_indexer_softmax_lse.return_value = ("max", "sum")
        seq_q = MagicMock()
        seq_k = MagicMock()

        result = npu_dense_lightning_indexer_softmax_lse(
            "q", "k", "w",
            actual_seq_qlen=seq_q,
            actual_seq_klen=seq_k,
            layout="TND",
            sparse_mode=0,
            pre_tokens=512,
            next_tokens=0,
        )

        mock_platform.custom_ops.npu_dense_lightning_indexer_softmax_lse.assert_called_once_with(
            "q", "k", "w", seq_q, seq_k, "TND", 0, 512, 0,
        )
        self.assertEqual(result, ("max", "sum"))


class TestNpuDenseLightningIndexerGradKlLoss(unittest.TestCase):
    """
    Feature: npu_dense_lightning_indexer_grad_kl_loss wrapper.
    Description: All positional and optional keyword arguments forwarded to platform.
    Expectation: Platform method called once with all args; return propagated.
    """

    @patch(_PATCH_TARGET)
    def test_required_args_only(self, mock_platform):
        """Required positional args only — rope and seq-len defaults are None."""
        expected = (MagicMock(),) * 4
        mock_platform.custom_ops.npu_dense_lightning_indexer_grad_kl_loss.return_value = expected

        result = npu_dense_lightning_indexer_grad_kl_loss(
            "query", "key", "q_idx", "k_idx", "weights",
            "softmax_max", "softmax_sum", "softmax_max_idx", "softmax_sum_idx",
            0.125,
        )

        mock_platform.custom_ops.npu_dense_lightning_indexer_grad_kl_loss.assert_called_once_with(
            "query", "key", "q_idx", "k_idx", "weights",
            "softmax_max", "softmax_sum", "softmax_max_idx", "softmax_sum_idx",
            0.125,
            None, None,
            None, None,
            "BSND", 3, _MAX_INT64, _MAX_INT64,
        )
        self.assertIs(result, expected)

    @patch(_PATCH_TARGET)
    def test_optional_kwargs_forwarded(self, mock_platform):
        """Optional rope and seq-len kwargs are forwarded when supplied."""
        mock_platform.custom_ops.npu_dense_lightning_indexer_grad_kl_loss.return_value = None
        q_rope, k_rope, seq_q, seq_k = MagicMock(), MagicMock(), MagicMock(), MagicMock()

        npu_dense_lightning_indexer_grad_kl_loss(
            "q", "k", "qi", "ki", "w",
            "sm_max", "sm_sum", "smi_max", "smi_sum",
            1.0,
            query_rope=q_rope, key_rope=k_rope,
            actual_seq_qlen=seq_q, actual_seq_klen=seq_k,
            layout="TND", sparse_mode=0,
            pre_tokens=128, next_tokens=64,
        )

        mock_platform.custom_ops.npu_dense_lightning_indexer_grad_kl_loss.assert_called_once_with(
            "q", "k", "qi", "ki", "w",
            "sm_max", "sm_sum", "smi_max", "smi_sum",
            1.0,
            q_rope, k_rope,
            seq_q, seq_k,
            "TND", 0, 128, 64,
        )


class TestNpuSparseLightningIndexerGradKlLoss(unittest.TestCase):
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss wrapper.
    Description: sparse_indices extra arg plus optional rope/seq-len forwarded correctly.
    Expectation: Platform method called once with all args; return propagated.
    """

    @patch(_PATCH_TARGET)
    def test_required_args_only(self, mock_platform):
        """Required args forwarded; optional args default to None."""
        expected = (MagicMock(),) * 4
        mock_platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss.return_value = expected

        result = npu_sparse_lightning_indexer_grad_kl_loss(
            "query", "key", "q_idx", "k_idx", "weights",
            "sparse_indices",
            "softmax_max", "softmax_sum",
            0.5,
        )

        mock_platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss.assert_called_once_with(
            "query", "key", "q_idx", "k_idx", "weights",
            "sparse_indices",
            "softmax_max", "softmax_sum",
            0.5,
            None, None,
            None, None,
            "BSND", 3, _MAX_INT64, _MAX_INT64,
        )
        self.assertIs(result, expected)

    @patch(_PATCH_TARGET)
    def test_all_kwargs_forwarded(self, mock_platform):
        """All optional kwargs including rope and layout forwarded."""
        mock_platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss.return_value = None
        q_rope, k_rope = MagicMock(), MagicMock()

        npu_sparse_lightning_indexer_grad_kl_loss(
            "q", "k", "qi", "ki", "w", "si",
            "sm_max", "sm_sum", 2.0,
            query_rope=q_rope, key_rope=k_rope,
            layout="TND", sparse_mode=1,
            pre_tokens=256, next_tokens=32,
        )

        mock_platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss.assert_called_once_with(
            "q", "k", "qi", "ki", "w", "si",
            "sm_max", "sm_sum", 2.0,
            q_rope, k_rope,
            None, None,
            "TND", 1, 256, 32,
        )


class TestNpuMhcPost(unittest.TestCase):
    """
    Feature: npu_mhc_post wrapper.
    Description: Four positional args forwarded to platform; return propagated.
    Expectation: Platform method called once with x, h_res, h_out, h_post.
    """

    @patch(_PATCH_TARGET)
    def test_delegates_four_args(self, mock_platform):
        """All four positional arguments forwarded; return value passed through."""
        expected = MagicMock()
        mock_platform.custom_ops.npu_mhc_post.return_value = expected

        result = npu_mhc_post("x", "h_res", "h_out", "h_post")

        mock_platform.custom_ops.npu_mhc_post.assert_called_once_with(
            "x", "h_res", "h_out", "h_post"
        )
        self.assertIs(result, expected)


class TestNpuMhcPreSinkhorn(unittest.TestCase):
    """
    Feature: npu_mhc_pre_sinkhorn wrapper.
    Description: Required args and optional keyword args forwarded to platform.
    Expectation: Platform method called once with the correct argument set.
    """

    @patch(_PATCH_TARGET)
    def test_required_args_use_defaults(self, mock_platform):
        """Required args only — optional kwargs default to their declared values."""
        expected = tuple(MagicMock() for _ in range(8))
        mock_platform.custom_ops.npu_mhc_pre_sinkhorn.return_value = expected

        result = npu_mhc_pre_sinkhorn("x", "phi", "alpha", "bias")

        mock_platform.custom_ops.npu_mhc_pre_sinkhorn.assert_called_once_with(
            "x", "phi", "alpha", "bias",
            4, 20, 1e-6, 1e-6, True,
        )
        self.assertIs(result, expected)

    @patch(_PATCH_TARGET)
    def test_custom_kwargs_forwarded(self, mock_platform):
        """Custom keyword argument values are forwarded to the platform."""
        mock_platform.custom_ops.npu_mhc_pre_sinkhorn.return_value = None

        npu_mhc_pre_sinkhorn(
            "x", "phi", "alpha", "bias",
            hc_mult=8,
            num_iters=10,
            hc_eps=1e-4,
            norm_eps=1e-5,
            out_flag=False,
        )

        mock_platform.custom_ops.npu_mhc_pre_sinkhorn.assert_called_once_with(
            "x", "phi", "alpha", "bias",
            8, 10, 1e-4, 1e-5, False,
        )


if __name__ == "__main__":
    unittest.main()
