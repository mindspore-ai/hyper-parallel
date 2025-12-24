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
from enum import auto, Enum


class PlatformType(Enum):
    """
    PlatformType
    """
    MINDSPORE = auto()
    PYTORCH = auto()


platform = None


# pylint: disable=C0415
def get_platform():
    """get framework platform"""
    global platform
    if platform is not None:
        return platform
    try:
        from hyper_parallel.platform.mindspore.platform import MindSporePlatform
        platform = MindSporePlatform()
        return platform
    except ImportError:
        from hyper_parallel.platform.torch.platform import TorchPlatform
        platform = TorchPlatform()
        return platform


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
    def register_forward_pre_hook(cell, hook):
        return cell.register_forward_pre_hook(hook)

    @staticmethod
    def register_forward_hook(cell, hook):
        return cell.register_forward_hook(hook)

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
    def register_backward_pre_hook(cell, hook):
        return cell.register_backward_pre_hook(hook)

    @staticmethod
    def register_backward_hook(cell, hook):
        return cell.register_backward_hook(hook)

    @staticmethod
    def get_param_local_shape(param):
        raise NotImplementedError("Platform subclasses must implement get_param_local_shape")

    @staticmethod
    def get_param_local_data(param):
        raise NotImplementedError("Platform subclasses must implement get_param_local_data")

    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        param.data = data

    @staticmethod
    def get_param_type_size(param):
        raise NotImplementedError("Platform subclasses must implement get_param_type_size")

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad):
        raise NotImplementedError("Platform subclasses must implement new_zero_parameter")

    @staticmethod
    def new_tensor(tensor_shape, tensor_type):
        raise NotImplementedError("Platform subclasses must implement new_tensor")

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement all_gather_into_tensor")

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement all_reduce")

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        raise NotImplementedError("Platform subclasses must implement reduce_scatter_tensor")

    def _create_group(self, rank_list, group_name=None):
        raise NotImplementedError("Platform subclasses must implement _create_group")

    def new_stream(self):
        raise NotImplementedError("Platform subclasses must implement new_stream")

    def get_stream_context(self):
        raise NotImplementedError("Platform subclasses must implement get_stream_context")

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
