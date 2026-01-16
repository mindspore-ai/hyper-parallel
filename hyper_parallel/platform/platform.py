# Copyright 2025 Huawei Technologies Co., Ltd
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
"""framework platform api"""
import os
from enum import auto, Enum
# Environment variable name used to specify the AI framework platform to use
HYPER_PARALLEL_PLATFORM = "HYPER_PARALLEL_PLATFORM"

# Identifier for the MindSpore framework
HYPER_PARALLEL_PLATFORM_MINDSPORE = "mindspore"

# Identifier for the PyTorch framework
HYPER_PARALLEL_PLATFORM_TORCH = "torch"


class PlatformType(Enum):
    """Enumeration class for AI framework platform types.

    Used to identify different deep learning framework platform types.
    """
    MINDSPORE = auto()
    PYTORCH = auto()


# Global platform instance, used to cache the created platform object
platform = None


def get_mindspore_platform():
    """Create mindspore platform"""
    # pylint: disable=C0415
    from hyper_parallel.platform.mindspore.platform import MindSporePlatform
    global platform
    platform = MindSporePlatform()
    return platform


def get_torch_platform():
    """Create torch platform"""
    # pylint: disable=C0415
    from hyper_parallel.platform.torch.platform import TorchPlatform
    global platform
    platform = TorchPlatform()
    return platform


def get_platform():
    """Obtain a framework platform instance.

    Returns the appropriate AI framework platform instance based on environment variables or a default priority order.
    The lookup priority is as follows:
    1. Platform specified by environment variable
    2. MindSpore platform (default preferred choice)
    3. PyTorch platform (fallback option)

    Returns:
        Platform: An instance of the framework platform

    Raises:
        ImportError: Raised when none of the supported frameworks are available
    """
    if platform is not None:
        return platform
    platform_type = os.environ.get(HYPER_PARALLEL_PLATFORM)
    if platform_type is not None and isinstance(platform_type, str):
        platform_type = platform_type.lower()
        if platform_type == HYPER_PARALLEL_PLATFORM_MINDSPORE:
            return get_mindspore_platform()
        if platform_type == HYPER_PARALLEL_PLATFORM_TORCH:
            return get_torch_platform()
    try:
        return get_mindspore_platform()
    except ImportError:
        return get_torch_platform()


EXISTING_COMM_GROUPS = {}


class Platform:
    """Platform api"""
    current_grad_handle = None
    post_grad_handle_process = None
    grad_sync_stream = None

    @staticmethod
    def get_rank():
        raise NotImplementedError("Platform subclasses must implement get_rank")

    @staticmethod
    def get_global_rank(group, group_rank):
        raise NotImplementedError("Platform subclasses must implement get_global_rank")

    @staticmethod
    def get_world_size():
        raise NotImplementedError("Platform subclasses must implement get_world_size")

    @staticmethod
    def get_op_name(func):
        raise NotImplementedError("Platform subclasses must implement get_op_name")

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        raise NotImplementedError("Platform subclasses must implement differentiable_all_gather_concat")

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        raise NotImplementedError("Platform subclasses must implement chunk")

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        raise NotImplementedError("Platform subclasses must implement differentiable_all_to_all")

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        raise NotImplementedError("Platform subclasses must implement differentiable_all_reduce")

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        raise NotImplementedError("Platform subclasses must implement differentiable_reduce_scatter")

    @staticmethod
    def init_parameters(module, stage_index):
        """platform ms need init parameter interface"""
        if module is None:
            raise ValueError("input module must not be none.")
        if stage_index < 0:
            raise ValueError("input stage_index must be positive.")

    @staticmethod
    def get_cell_construct(cell):
        raise NotImplementedError("Platform subclasses must implement get_cell_construct")

    @staticmethod
    def get_cells_and_names(cell):
        raise NotImplementedError("Platform subclasses must implement get_cells_and_names")

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        raise NotImplementedError("Platform subclasses must implement search_parameter_by_name")

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        raise NotImplementedError("Platform subclasses must implement update_parameter_by_name")

    @staticmethod
    def set_layout_into_parameter(param, layout):
        raise NotImplementedError("Platform subclasses must implement set_layout_into_parameter")

    @staticmethod
    def get_param_local_shape(param):
        raise NotImplementedError("Platform subclasses must implement get_param_local_shape")

    @staticmethod
    def get_param_local_data(param):
        raise NotImplementedError("Platform subclasses must implement get_param_local_data")

    @staticmethod
    def update_param_data(param, data):
        raise NotImplementedError("Platform subclasses must implement update_param_data")

    @staticmethod
    def get_param_type_size(param):
        raise NotImplementedError("Platform subclasses must implement get_param_type_size")

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        raise NotImplementedError("Platform subclasses must implement new_zero_parameter")

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        raise NotImplementedError("Platform subclasses must implement new_tensor")

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        raise NotImplementedError("Platform subclasses must implement full_like")

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        raise NotImplementedError("Platform subclasses must implement set_tensor_requires_grad")

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement all_gather_into_tensor")

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement all_reduce")

    @staticmethod
    def broadcast(data, src, group, async_op=False):
        raise NotImplementedError("Platform subclasses must implement broadcast")

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        raise NotImplementedError("Platform subclasses must implement isend")

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        raise NotImplementedError("Platform subclasses must implement irecv")

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        raise NotImplementedError("Platform subclasses must implement send_object_list")

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        raise NotImplementedError("Platform subclasses must implement send_object_list")

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement reduce_scatter_tensor")

    @staticmethod
    def parameters_dict(cell):
        raise NotImplementedError("Platform subclasses must implement parameters_dict")

    @staticmethod
    def save_checkpoint(cell, file_path: str) -> None:
        raise NotImplementedError("Platform subclasses must implement save_checkpoint")

    @staticmethod
    def load_checkpoint(file_path: str) -> dict:
        raise NotImplementedError("Platform subclasses must implement load_checkpoint")

    def _create_group(self, rank_list, group_name=None):
        raise NotImplementedError("Platform subclasses must implement _create_group")

    def new_stream(self):
        raise NotImplementedError("Platform subclasses must implement new_stream")

    def get_stream_context(self):
        raise NotImplementedError("Platform subclasses must implement get_stream_context")

    @staticmethod
    def get_tensor_transform():
        raise NotImplementedError("Platform subclasses must implement get_tensor_transform")

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        raise NotImplementedError("Platform subclasses must implement construct_strided_slice")

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        raise NotImplementedError("Platform subclasses must implement micro_batch")

    def create_group(self, rank_list, group_name=None):
        """create comm group with rank list"""
        if group_name is None:
            group_key = hash(tuple(rank_list))
        else:
            group_key = group_name
        if group_key in EXISTING_COMM_GROUPS:
            return EXISTING_COMM_GROUPS[group_key]

        group = self._create_group(rank_list, group_name)
        EXISTING_COMM_GROUPS[group_key] = group
        return group

    def _process_current_handle(self):
        """wait current handle"""
        if Platform.current_grad_handle is None:
            return

        Platform.current_grad_handle.wait()
        if Platform.post_grad_handle_process is None:
            return
        # pylint: disable=E1102
        Platform.post_grad_handle_process()

    def set_grad_reduce_handle(self, handle, post_process=None):
        """wait current handle and set new handle"""
        if Platform.grad_sync_stream is None:
            Platform.grad_sync_stream = self.new_stream()
        stream_context = self.get_stream_context()
        with stream_context(Platform.grad_sync_stream):
            self._process_current_handle()
        Platform.current_grad_handle = handle
        Platform.post_grad_handle_process = post_process

    def wait_grad_handle(self):
        """wait grad handle"""
        if Platform.current_grad_handle is None:
            return
        if Platform.grad_sync_stream is None:
            Platform.grad_sync_stream = self.new_stream()
        stream_context = self.get_stream_context()
        with stream_context(Platform.grad_sync_stream):
            self._process_current_handle()
            sync_event = Platform.grad_sync_stream.record_event()
        sync_event.wait()
        Platform.current_grad_handle = None
        Platform.post_grad_handle_process = None
