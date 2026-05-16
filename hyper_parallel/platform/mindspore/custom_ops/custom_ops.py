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
"""MindSpore platform custom ops delegation to DFunction wrappers.

.. warning::
    This is an experimental API that subject to change or deletion.
"""

from hyper_parallel.platform.mindspore.custom_ops.custom_op_impl import (
    NpuDenseLightningIndexerGradKlLossDFunction,
    NpuDenseLightningIndexerSoftmaxLseDFunction,
    NpuMhcPostDFunction,
    NpuMhcPreSinkhornDFunction,
    NpuSparseLightningIndexerGradKlLossDFunction,
)


class MindSporeCustomOps:
    """Delegates custom operator calls to DFunction-based implementations."""

    @staticmethod
    def npu_dense_lightning_indexer_softmax_lse(*args, **kwargs):
        return NpuDenseLightningIndexerSoftmaxLseDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_dense_lightning_indexer_grad_kl_loss(*args, **kwargs):
        return NpuDenseLightningIndexerGradKlLossDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_sparse_lightning_indexer_grad_kl_loss(*args, **kwargs):
        return NpuSparseLightningIndexerGradKlLossDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_post(*args, **kwargs):
        return NpuMhcPostDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_pre_sinkhorn(*args, **kwargs):
        return NpuMhcPreSinkhornDFunction.apply(*args, **kwargs)
