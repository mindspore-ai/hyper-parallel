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
"""CPU Level0 lifecycle for the eight-rank HSDP reordered-mesh regression."""

import torch.distributed as dist

from hyper_parallel import destroy_process_group
from tests.torch.fully_shard._test_hsdp_param import (
    test_hsdp_param_v2_reordered_mesh_remaps_dp_dims_for_dtensor as _check_reordered_mesh,
)
from tests.torch.utils import init_dist_gloo


def test_reordered_mesh_gloo_lifecycle() -> None:
    """Run the full DP2/FSDP2/TP2 check and release Gloo before interpreter shutdown."""
    init_dist_gloo()
    try:
        assert dist.get_world_size() == 8, "The reordered mesh requires DP2, FSDP2 and TP2"
        _check_reordered_mesh()
        dist.barrier()
    finally:
        destroy_process_group()
    assert not dist.is_initialized(), "The CPU Level0 worker retained its default process group"
