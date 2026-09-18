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
"""Distributed implementation for npu_situ_glu operator."""
from typing import Any, Dict, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp

platform = get_platform()


def _create_output_layout(mesh: Any, tensor_map: tuple) -> Layout:
    """Create an output layout with placements derived from ``tensor_map``."""
    output_layout = Layout.from_device_mesh(mesh)
    output_layout.set_tensor_map(tensor_map)
    output_layout.tensor_map_to_placement()
    return output_layout


def _normalize_situ_glu_args(x, dim=-1, beta=4.0, linear_beta=25.0, activate_left=True):
    """Normalize npu_situ_glu arguments.

    ``activate_left`` is pinned to True by the experimental wrapper to match
    the mindformers SiTUGLU Cell's ``chunk(x, 2, dim)`` gate=front semantics;
    it is still accepted positionally so the DistributedOp dispatch path
    (which receives args forwarded verbatim from the wrapper) normalizes
    consistently regardless of calling style.
    """
    return (x, dim, beta, linear_beta, activate_left), {}


class NpuSituGluDistributedOp(DistributedOp):
    """DistributedOp for npu_situ_glu operator.

    Single tensor input ``x`` (sharded on batch dims), single tensor output
    ``y`` whose shape equals ``x`` with the ``dim`` axis halved. The output
    reuses ``x``'s tensor_map: halving a dimension is a shape-inference concern
    handled by the kernel/infershape, not by the layout — the shard axis stays
    the same.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_situ_glu_args(*args, **kwargs)
        dtensor_x = norm_args[0]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                dtensor_x.to_local(),
                norm_args[1],
                norm_args[2],
                norm_args[3],
                norm_args[4],
            )
            local_kwargs = {}
        else:
            local_args = (dtensor_x.to_local(),)
            local_kwargs = {
                'dim': norm_args[1],
                'beta': norm_args[2],
                'linear_beta': norm_args[3],
                'activate_left': norm_args[4],
            }

        cache_values = [dtensor_x.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        x_layout, = cache_values
        self._check_partial_inputs([x_layout])
        mesh = x_layout.mesh
        x_tm = x_layout.tensor_map
        # y shares x's tensor_map: the dim axis is halved in shape (kernel
        # concern), but the shard placement is identical.
        return (_create_output_layout(mesh, x_tm),), None
