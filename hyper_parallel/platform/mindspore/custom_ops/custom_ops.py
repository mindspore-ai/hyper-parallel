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
from typing import Any

from hyper_parallel.platform.mindspore.custom_ops.chunk_kda_stages import run_chunk_kda_stage
from hyper_parallel.platform.mindspore.custom_ops.custom_op_impl import (
    NpuChunkKdaDFunction,
    NpuFlashAttentionVarLenV4DFunction,
    NpuDenseLightningIndexerGradKlLossDFunction,
    NpuDenseLightningIndexerSoftmaxLseDFunction,
    NpuLightningIndexerDFunction,
    NpuLightningIndexerEnhanceDFunction,
    NpuMhcPostDFunction,
    NpuMhcPreClampSinkhornDFunction,
    NpuMhcPreCmhcDFunction,
    NpuMhcPreSinkhornDFunction,
    NpuSituGluDFunction,
    NpuSparseFlashMlaDFunction,
    NpuSparseLightningIndexerGradKlLossDFunction,
    NpuSparseLightningIndexerKlLossGradDFunction,
    npu_sparse_flash_mla_grad as _npu_sparse_flash_mla_grad,
)


class MindSporeCustomOps:
    """Delegates custom operator calls to DFunction-based implementations."""

    @staticmethod
    def npu_flash_attention_varlen_v4(*args, **kwargs):
        """TND varlen FlashAttention via aclnn V4 (differentiable)."""
        return NpuFlashAttentionVarLenV4DFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_dense_lightning_indexer_softmax_lse(*args, **kwargs):
        """Compute dense lightning indexer softmax log-sum-exp via custom NPU operator."""
        return NpuDenseLightningIndexerSoftmaxLseDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_dense_lightning_indexer_grad_kl_loss(*args, **kwargs):
        """Compute dense lightning indexer KL-loss gradient via custom NPU operator."""
        return NpuDenseLightningIndexerGradKlLossDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_sparse_lightning_indexer_grad_kl_loss(*args, **kwargs):
        """Compute sparse lightning indexer KL-loss gradient via custom NPU operator."""
        return NpuSparseLightningIndexerGradKlLossDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_post(*args, **kwargs):
        """Apply the NPU MHC post-processing custom operator."""
        return NpuMhcPostDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_pre_sinkhorn(*args, **kwargs):
        """Apply the NPU MHC pre-Sinkhorn custom operator."""
        return NpuMhcPreSinkhornDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_pre_clamp_sinkhorn(*args, **kwargs):
        """Apply the NPU MHC pre-clamp-Sinkhorn custom operator."""
        return NpuMhcPreClampSinkhornDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_mhc_pre_cmhc(*args: Any, **kwargs: Any) -> Any:
        """Apply the NPU MHC pre-CMHC fused custom operator."""
        return NpuMhcPreCmhcDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_situ_glu(*args: Any, **kwargs: Any) -> Any:
        """Apply the NPU SiTU-GLU fused activation custom operator."""
        return NpuSituGluDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_lightning_indexer(*args, **kwargs):
        """Sparse-attention preprocessing (top-K key selection) via custom NPU operator."""
        return NpuLightningIndexerDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_lightning_indexer_enhance(*args, **kwargs):
        """Sparse-attention preprocessing via the enhanced top-K custom NPU operator."""
        return NpuLightningIndexerEnhanceDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_sparse_flash_mla(*args, **kwargs):
        """MLA sparse attention (forward + grad + metadata) via custom NPU operator."""
        return NpuSparseFlashMlaDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_sparse_flash_mla_grad(*args, **kwargs):
        """Run the raw MLA sparse-attention backward kernel and return all six outputs."""
        return _npu_sparse_flash_mla_grad(*args, **kwargs)

    @staticmethod
    def npu_sparse_lightning_indexer_kl_loss_grad(*args, **kwargs):
        """Sparse lightning indexer KL-loss gradient via custom NPU operator."""
        return NpuSparseLightningIndexerKlLossGradDFunction.apply(*args, **kwargs)

    @staticmethod
    def npu_chunk_kda(*args, **kwargs):
        """Run differentiable Chunk KDA with automatic backward dispatch."""
        initial_state, disable_recompute, state_v_first = args[21:] if len(args) > 21 else (None, True, False)
        # Reuse the state-capable stage chain for its paired safe-gate ABI.
        # The legacy fused backward produces nonfinite gradients for the MF
        # Ulysses workload; the stage chain also covers ordinary no-state loss.
        paired_stages = args[9] and args[11] and args[20] and args[12] is not None
        if paired_stages or initial_state is not None or not disable_recompute or state_v_first:
            from hyper_parallel.platform.mindspore.custom_ops.chunk_kda_autograd import run_chunk_kda_stages
            names = ("layout", "output_final_state", "safe_gate", "lower_bound", "use_gate_in_kernel",
                     "a_log", "dt_bias", "cu_seqlens", "chunk_indices", "epsilon", "use_qk_l2norm_in_kernel",
                     "use_beta_sigmoid_in_kernel", "allow_neg_eigval", "use_exp2")
            options = dict(zip(names, args[7:21]))
            options.update(initial_state=initial_state, disable_recompute=disable_recompute,
                           state_v_first=state_v_first)
            return run_chunk_kda_stages(*args[:7], **options)
        return NpuChunkKdaDFunction.apply(*args[:21], **kwargs)

    @staticmethod
    def npu_chunk_kda_fwd_prepare(*args: Any) -> Any:
        """Prepare local head-major KDA caches without automatic backward."""
        return run_chunk_kda_stage("npu_chunk_kda_fwd_prepare", *args)

    @staticmethod
    def npu_chunk_kda_pre_process_fwd(*args: Any) -> Any:
        """Build local forward affine summaries."""
        return run_chunk_kda_stage("npu_chunk_kda_pre_process_fwd", *args)

    @staticmethod
    def npu_chunk_kda_fwd_h(*args: Any) -> Any:
        """Propagate local states from caller-supplied boundary states."""
        return run_chunk_kda_stage("npu_chunk_kda_fwd_h", *args)

    @staticmethod
    def npu_chunk_kda_fwd_finalize(*args: Any) -> Any:
        """Finalize local attention output."""
        return run_chunk_kda_stage("npu_chunk_kda_fwd_finalize", *args)

    @staticmethod
    def npu_chunk_kda_bwd_prepare(*args: Any) -> Any:
        """Prepare local block gradients."""
        return run_chunk_kda_stage("npu_chunk_kda_bwd_prepare", *args)

    @staticmethod
    def npu_chunk_kda_pre_process_bwd(*args: Any) -> Any:
        """Build local backward affine summaries."""
        return run_chunk_kda_stage("npu_chunk_kda_pre_process_bwd", *args)

    @staticmethod
    def npu_chunk_kda_bwd_dhu(*args: Any) -> Any:
        """Propagate state gradients from caller-supplied dht."""
        return run_chunk_kda_stage("npu_chunk_kda_bwd_dhu", *args)

    @staticmethod
    def npu_chunk_kda_bwd_finalize(*args: Any) -> Any:
        """Finalize local input and parameter gradients."""
        return run_chunk_kda_stage("npu_chunk_kda_bwd_finalize", *args)

    @staticmethod
    def npu_chunk_kda_merge_fwd_bwd(*args: Any) -> Any:
        """Merge already gathered summaries into a local state buffer."""
        return run_chunk_kda_stage("npu_chunk_kda_merge_fwd_bwd", *args)
