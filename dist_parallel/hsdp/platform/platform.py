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
from mindspore.parallel import DTensor
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.communication import create_group
import mindspore.communication.comm_func as comm_func

EXISTING_COMM_GROUPS = dict()


class Platform:
    """Platform api"""
    current_grad_handle = None
    post_grad_handle_process = None
    grad_sync_stream = None

    @staticmethod
    def register_forward_pre_hook(cell, hook):
        return cell.register_forward_pre_hook(hook)

    @staticmethod
    def register_forward_hook(cell, hook):
        return cell.register_forward_hook(hook)

    @staticmethod
    def register_backward_pre_hook(cell, hook):
        return cell.register_backward_pre_hook(hook)

    @staticmethod
    def register_backward_hook(cell, hook):
        return cell.register_backward_hook(hook)

    @staticmethod
    def get_param_local_shape(param):
        """get param local shape"""
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """get param local shape"""
        return param

    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        param.data = data

    @staticmethod
    def get_param_type_size(param):
        pass

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad):
        pass

    @staticmethod
    def new_tensor(tensor_shape, tensor_type):
        pass

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        pass

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        pass

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        pass

    def _create_group(self, group_name, rank_list):
        pass

    def new_stream(self):
        pass

    def get_stream_context(self):
        pass

    def create_group(self, group_name, rank_list):
        if group_name in EXISTING_COMM_GROUPS:
            return EXISTING_COMM_GROUPS[group_name]

        group = _create_group(group_name, rank_list)
        EXISTING_COMM_GROUPS[group_name] = group
        return group

    def _process_current_handle(self):
        """wait current handle"""
        if Platform.current_grad_handle is None:
            return

        Platform.current_grad_handle.wait()
        if Platform.post_grad_handle_process is None:
            return
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
