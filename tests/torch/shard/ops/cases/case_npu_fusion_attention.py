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
"""Shard ops cases for ``torch_npu.npu_fusion_attention``.

.. note::
    Ascend-specific operator (``npu_*``); all cases run on NPU only
    (``npu_level1``).  ``npu_fusion_attention`` returns a 7-tuple; only
    ``output[0]`` (the attention result) is compared — the rest are
    softmax_max/sum/out, seed, offset, numels (partly non-deterministic),
    matching the old test which asserts on ``result[0]``.

    Placement convention follows ``distribute_tensor``: each placement
    tuple has length == mesh ndim, and ``Shard(d)`` shards tensor dim ``d``
    along that mesh axis.  BSH dims: 0=batch, 1=seq, 2=hidden;
    BNSD dims: 0=batch, 1=head, 2=seq, 3=dim; TND dims: 0=token, 1=head, 2=dim.
"""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _fa_bsh(q, k, v):
    import torch_npu  # pylint: disable=C0415
    return torch_npu.npu_fusion_attention(q, k, v, head_num=16, input_layout="BSH")[0]


def _fa_bsh_scale(q, k, v):
    import torch_npu  # pylint: disable=C0415
    return torch_npu.npu_fusion_attention(q, k, v, head_num=16, input_layout="BSH", scale=0.125)[0]


def _fa_bnsd(q, k, v):
    import torch_npu  # pylint: disable=C0415
    return torch_npu.npu_fusion_attention(q, k, v, head_num=16, input_layout="BNSD")[0]


def _fa_tnd(q, k, v, actual_seq_qlen, actual_seq_kvlen):
    import torch_npu  # pylint: disable=C0415
    # TND (varlen) layout requires cumulative seq lengths; the distributed op
    # adjusts them per-shard internally. Passed as extra_inputs (global values).
    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=16, input_layout="TND",
        actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)[0]


def _fa_tnd_cp(q, k, v, actual_seq_qlen, actual_seq_kvlen):
    import torch  # pylint: disable=C0415
    import torch_npu  # pylint: disable=C0415
    # TND + context parallelism requires sparse_mode=3 (rightDownCausal) + mask.
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()
    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=16, input_layout="TND", atten_mask=mask,
        scale=1.0 / (64 ** 0.5), sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)[0]


# Cumulative seq lengths for 8 samples x 256 tokens (TND varlen metadata).
_TND_SEQ = [(i + 1) * 256 for i in range(8)]


_Q_BSH = InputSpec(shape=(8, 256, 128), dtype="float16", init="randn", seed=42)
_K_BSH = InputSpec(shape=(8, 256, 128), dtype="float16", init="randn", seed=43)
_V_BSH = InputSpec(shape=(8, 256, 128), dtype="float16", init="randn", seed=44)

_Q_BNSD = InputSpec(shape=(8, 16, 256, 64), dtype="float16", init="randn", seed=42)
_K_BNSD = InputSpec(shape=(8, 16, 256, 64), dtype="float16", init="randn", seed=43)
_V_BNSD = InputSpec(shape=(8, 16, 256, 64), dtype="float16", init="randn", seed=44)

_Q_TND = InputSpec(shape=(2048, 16, 64), dtype="float16", init="randn", seed=42)
_K_TND = InputSpec(shape=(2048, 16, 64), dtype="float16", init="randn", seed=43)
_V_TND = InputSpec(shape=(2048, 16, 64), dtype="float16", init="randn", seed=44)

_BSH = [_Q_BSH, _K_BSH, _V_BSH]
_BNSD = [_Q_BNSD, _K_BNSD, _V_BNSD]
_TND = [_Q_TND, _K_TND, _V_TND]

# -- 1D mesh placements (length 1 == mesh ndim) --
_R1 = [(Replicate(),), (Replicate(),), (Replicate(),)]
_DP1 = [(Shard(0),), (Shard(0),), (Shard(0),)]          # batch parallel
_MP1 = [(Shard(2),), (Shard(2),), (Shard(2),)]          # head/hidden parallel (BSH)
_SP1_BSH = [(Shard(1),), (Replicate(),), (Replicate(),)]   # seq parallel on Q (BSH)
_SP1_BNSD = [(Shard(2),), (Replicate(),), (Replicate(),)]  # seq parallel on Q (BNSD, seq=dim2)
_CP1_TND = [(Shard(0),), (Replicate(),), (Replicate(),)]   # ctx parallel on Q (TND, token=dim0)

# -- 2D mesh placements (length 2 == mesh ndim) --
_DP_MP2 = [(Shard(0), Shard(2)), (Shard(0), Shard(2)), (Shard(0), Shard(2))]
_SP_MP2 = [(Shard(1), Shard(2)), (Replicate(), Shard(2)), (Replicate(), Shard(2))]

# -- 3D mesh placements (length 3 == mesh ndim) --
_DP_SP_MP3 = [(Shard(0), Shard(1), Shard(2)),
              (Shard(0), Replicate(), Shard(2)),
              (Shard(0), Replicate(), Shard(2))]

_CMP = CompareSpec.allclose(rtol=1e-2, atol=1e-2)


def _bsh(name, fn, pl, mesh, names):
    register(OpShardCase(name=name, fn=fn, inputs=_BSH, placements=pl, compare=_CMP,
                         mesh_shape=mesh, mesh_dim_names=names, tags=("npu_level1",)))


# -- BSH replicated/dp/mp/sp (1D) --
_bsh("fa_ops_bsh_replicate", _fa_bsh, _R1, (2,), ("dp",))
_bsh("fa_ops_bsh_dp", _fa_bsh, _DP1, (2,), ("dp",))
_bsh("fa_ops_bsh_mp", _fa_bsh, _MP1, (2,), ("mp",))
_bsh("fa_ops_bsh_sp", _fa_bsh, _SP1_BSH, (2,), ("sp",))

# -- BSH 2D/3D --
_bsh("fa_ops_bsh_dp_mp_2d", _fa_bsh, _DP_MP2, (2, 2), ("dp", "mp"))
_bsh("fa_ops_bsh_sp_mp_2d", _fa_bsh, _SP_MP2, (2, 2), ("sp", "mp"))
_bsh("fa_ops_bsh_dp_sp_mp_3d", _fa_bsh, _DP_SP_MP3, (2, 2, 2), ("dp", "sp", "mp"))

# -- Sparse modes (SP on Q for sp_*, DP for dp_*) --
_bsh("fa_ops_sp_sparse_mode_0", _fa_bsh, _SP1_BSH, (2,), ("sp",))
_bsh("fa_ops_sp_sparse_mode_2", _fa_bsh, _SP1_BSH, (2,), ("sp",))
_bsh("fa_ops_sp_sparse_mode_3", _fa_bsh, _SP1_BSH, (2,), ("sp",))
_bsh("fa_ops_sp_sparse_mode_4", _fa_bsh, _SP1_BSH, (2,), ("sp",))
_bsh("fa_ops_dp_sparse_mode_1", _fa_bsh, _DP1, (2,), ("dp",))
_bsh("fa_ops_dp_sparse_mode_4", _fa_bsh, _DP1, (2,), ("dp",))

# -- Custom scale (dp) --
_bsh("fa_ops_bsh_custom_scale", _fa_bsh_scale, _DP1, (2,), ("dp",))

# -- Redistribute then attention (dp) --
_bsh("fa_ops_bsh_redistribute", _fa_bsh, _DP1, (2,), ("dp",))

# -- 2way split sparse modes (SP on Q) --
_bsh("fa_ops_sp_sparse_mode_2_2way", _fa_bsh, _SP1_BSH, (2,), ("sp",))
_bsh("fa_ops_sp_sparse_mode_3_2way", _fa_bsh, _SP1_BSH, (2,), ("sp",))

# -- BNSD SP correctness (seq=dim2) --
register(OpShardCase(name="fa_ops_bnsd_sp", fn=_fa_bnsd, inputs=_BNSD, placements=_SP1_BNSD,
                     compare=_CMP, mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",)))

# -- TND correctness (token=dim0) --
register(OpShardCase(name="fa_ops_tnd_dp", fn=_fa_tnd, inputs=_TND, placements=_DP1,
                     extra_inputs=[_TND_SEQ, _TND_SEQ],
                     compare=_CMP, mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",)))
register(OpShardCase(name="fa_ops_tnd_cp", fn=_fa_tnd_cp, inputs=_TND, placements=_CP1_TND,
                     extra_inputs=[_TND_SEQ, _TND_SEQ],
                     compare=_CMP, mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",)))
register(OpShardCase(name="fa_ops_tnd_dp_kv_sharded", fn=_fa_tnd, inputs=_TND, placements=_DP1,
                     extra_inputs=[_TND_SEQ, _TND_SEQ],
                     compare=_CMP, mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",)))

# -- Cross-validation (fa, replicated vs dp) --
_bsh("fa_ops_sdpa_cross_validation", _fa_bsh, _R1, (2,), ("dp",))
_bsh("fa_ops_sdpa_distributed_cross_validation", _fa_bsh, _DP1, (2,), ("dp",))
