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
"""Distributed-aware gradient clipping utilities."""
from typing import Optional

from hyper_parallel.platform import get_platform

__all__: list[str] = ["clip_grad_norm_"]


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
):
    r"""Distributed-aware gradient norm clipping.

    Drop-in replacement for the standard ``clip_grad_norm_`` that
    correctly handles sharded parameters by deriving communication from
    each parameter's DTensor spec (``device_mesh`` + ``placements``).

    Supports FSDP, HSDP, TP+FSDP, and any parallelism expressible via
    DTensor placements.  Plain (non-DTensor) parameters are treated as
    replicated and require no communication.

    Args:
        parameters: An ``nn.Module``, a single ``Tensor``, or an iterable
            of ``Tensor`` s whose gradients to clip.  When an ``nn.Module``
            is given, ``module.parameters()`` is used.
        max_norm (float): max norm of the gradients.
        norm_type (float): type of the used p-norm. Can be ``'inf'``
            for infinity norm. Default: 2.0.
        error_if_nonfinite (bool): if ``True``, an error is thrown if
            the total norm is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``.
        foreach (bool, optional): Unused, accepted for API compatibility
            with the standard ``clip_grad_norm_``.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    platform = get_platform()
    return platform.clip_grad_norm_(
        parameters, max_norm, norm_type,
        error_if_nonfinite=error_if_nonfinite, foreach=foreach,
    )
