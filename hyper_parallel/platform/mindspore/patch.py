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
"""MindSpore runtime patch (torch-mainline + MS overrides).

The torch codebase is the default implementation; on ``HYPER_PARALLEL_PLATFORM=
mindspore`` ``hyper_parallel/__init__.py`` runs :func:`enable` before importing
any core module so that every symbol here is rebound before core code uses it.
"""
# pylint: disable=import-outside-toplevel

_MS_PATCHED = False


def _ms_get_rank(group=None):
    """torch ``dist.get_rank(group)`` surfaced over MindSpore's ``get_rank``."""
    # pylint: disable=import-outside-toplevel
    from mindspore.communication import get_rank as get_rank_id

    if group is None:
        return get_rank_id()
    return get_rank_id(group)


def _ms_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):
    """MindSpore implementation of ``comm.differentiable_all_gather_concat``."""
    import mindspore as ms
    from mindspore.common.tensor import Tensor
    from mindspore.ops.function import comm_func

    data = _ms_ensure_contiguous(data)
    # rank_list is accepted for torch parity; MindSpore keeps the existing group order.
    output, _ = comm_func.all_gather_into_tensor(None, data, group=group)
    if concat_dim == 0:
        return output
    output_tensors = ms.ops.Split(output_num=concat_size)(output)
    return ms.mint.concat(output_tensors, concat_dim)


def _ms_ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous."""
    if not x.is_contiguous() or x.storage_offset() != 0:
        x = x.contiguous()
    return x


def _ms_p2p_exchange(tensor, peer_rank: int, group=None):
    """MindSpore has no symmetric P2P exchange; fail loudly at the call site."""
    raise NotImplementedError(
        "p2p_exchange is not yet supported on the MindSpore platform."
    )


def _ms_construct_strided_slice(x, begin, end, stride):
    """MindSpore implementation of ``comm.construct_strided_slice``."""
    import mindspore as ms

    return ms.ops.strided_slice(x, begin, end, stride)


def enable() -> None:
    """Apply the MindSpore runtime patch. Idempotent."""
    global _MS_PATCHED
    if _MS_PATCHED:
        return

    # Reuse the existing torch-like backward/autograd compatibility patch.
    from hyper_parallel.platform.mindspore.autograd_compat import (
        enable_mindspore_backward_compat,
    )
    enable_mindspore_backward_compat()

    # 情形 1: compatible one-liner -> patch torch.distributed directly.
    import torch.distributed as dist
    dist.get_rank = _ms_get_rank

    # 情形 2/3/4: translated or MS-only helpers -> rebind the torch-first gateway.
    import hyper_parallel.comm as hp_comm
    hp_comm.differentiable_all_gather_concat = _ms_all_gather_concat
    hp_comm.p2p_exchange = _ms_p2p_exchange
    hp_comm.construct_strided_slice = _ms_construct_strided_slice

    # Tier C: DTensorBase must already be the MS class before `class DTensor(DTensorBase)`
    # is defined, so rebind it in the dedicated module (loaded before dtensor.py).
    import hyper_parallel.core.dtensor._dtensor_base as _base
    from hyper_parallel.platform.mindspore.dtensor import DTensorBase as _MSDTensorBase
    _base.DTensorBase = _MSDTensorBase

    _MS_PATCHED = True
