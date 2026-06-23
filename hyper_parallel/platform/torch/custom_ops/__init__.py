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
"""Torch platform custom operations — experimental ops are not supported.

.. warning::
    This is an experimental API that subject to change or deletion.

Custom operators are currently MindSpore-only.  Calling them from PyTorch
raises ``NotImplementedError``.
"""


class TorchCustomOps:
    """Torch-side custom ops — raises NotImplementedError for all operators."""

    @staticmethod
    def npu_dense_lightning_indexer_softmax_lse(*args, **kwargs):
        """NPU dense lightning indexer softmax log-sum-exp operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_dense_lightning_indexer_softmax_lse is not supported "
            "on the PyTorch platform."
        )

    @staticmethod
    def npu_dense_lightning_indexer_grad_kl_loss(*args, **kwargs):
        """NPU dense lightning indexer gradient KL loss operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_dense_lightning_indexer_grad_kl_loss is not supported "
            "on the PyTorch platform."
        )

    @staticmethod
    def npu_sparse_lightning_indexer_grad_kl_loss(*args, **kwargs):
        """NPU sparse lightning indexer gradient KL loss operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_sparse_lightning_indexer_grad_kl_loss is not supported "
            "on the PyTorch platform."
        )

    @staticmethod
    def npu_mhc_post(*args, **kwargs):
        """NPU MHC post-processing operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_mhc_post is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_mhc_pre_sinkhorn(*args, **kwargs):
        """NPU MHC pre-Sinkhorn operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_mhc_pre_sinkhorn is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_mhc_pre_clamp_sinkhorn(*args, **kwargs):
        """Clamped NPU MHC pre-Sinkhorn operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_mhc_pre_clamp_sinkhorn is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_lightning_indexer(*args, **kwargs):
        """NPU lightning indexer operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_lightning_indexer is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_sparse_flash_mla(*args, **kwargs):
        """NPU sparse flash MLA operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_sparse_flash_mla is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_sparse_flash_mla_grad(*args, **kwargs):
        """NPU sparse flash MLA backward kernel; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_sparse_flash_mla_grad is not supported on the PyTorch platform."
        )

    @staticmethod
    def npu_sparse_lightning_indexer_kl_loss_grad(*args, **kwargs):
        """NPU sparse lightning indexer KL loss grad operator; not supported on PyTorch."""
        raise NotImplementedError(
            "npu_sparse_lightning_indexer_kl_loss_grad is not supported "
            "on the PyTorch platform."
        )
