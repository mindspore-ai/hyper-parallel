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
"""Teardown for distributed FSDP worker tests."""
import pytest


@pytest.fixture(autouse=True)
def cleanup_hsdp_worker_process_group(request):
    """Release HSDP worker process groups before interpreter exit."""
    yield
    if request.path.name != "_test_hsdp_param.py":
        return
    # This conftest is also loaded by the lightweight launcher process.
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache  # pylint: disable=import-outside-toplevel

    if dist.is_initialized():
        dist.destroy_process_group()
    clear_device_mesh_cache()
