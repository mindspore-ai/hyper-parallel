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
"""Experimental custom operators for HyperParallel.

.. warning::
    This is an experimental API that subject to change or deletion.

These operators delegate to the platform-specific ``custom_ops`` interface.
On MindSpore they wrap Ascend NPU custom C++ kernels through the
``DFunction`` distributed dispatch framework.  On PyTorch they raise
``NotImplementedError``.

Usage::

    from hyper_parallel.custom_ops.experimental import npu_dense_lightning_indexer_softmax_lse
    softmax_max, softmax_sum = npu_dense_lightning_indexer_softmax_lse(
        query_index, key_index, weights, layout='BSND')

When inputs are ``DTensor`` objects, the call is automatically routed
through the registered ``DistributedOp`` for layout inference and
re-distribution.
"""
from hyper_parallel.custom_ops.experimental.experimental_ops import (
    npu_dense_lightning_indexer_grad_kl_loss,
    npu_dense_lightning_indexer_softmax_lse,
    npu_mhc_post,
    npu_mhc_pre_clamp_sinkhorn,
    npu_mhc_pre_sinkhorn,
    npu_sparse_lightning_indexer_grad_kl_loss,
)

__all__ = [
    "npu_dense_lightning_indexer_grad_kl_loss",
    "npu_dense_lightning_indexer_softmax_lse",
    "npu_mhc_post",
    "npu_mhc_pre_clamp_sinkhorn",
    "npu_mhc_pre_sinkhorn",
    "npu_sparse_lightning_indexer_grad_kl_loss",
]
