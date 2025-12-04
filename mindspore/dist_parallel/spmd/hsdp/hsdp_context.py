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
"""HSDP context"""


class HSDPContext:
    """
    HSDP context for global hsdp state.
    """
    def __init__(self):
        """init  handle and handle process"""
        self.current_grad_handle = None
        self.post_grad_handle_process = None
        self.grad_sync_stream = None

    def _process_current_handle(self):
        """wait current handle"""
        if self.current_grad_handle is None:
            return

        self.current_grad_handle.wait()
        if self.post_grad_handle_process is None:
            return
        self.post_grad_handle_process()

    def set_grad_reduce_handle(self, handle, post_process=None):
        """wait current handle and set new handle"""
        import mindspore as ms
        sync_stream = self.get_grad_sync_stream()
        with ms.runtime.StreamCtx(sync_stream):
            self._process_current_handle()
        self.current_grad_handle = handle
        self.post_grad_handle_process = post_process

    def wait_grad_handle(self):
        """wait current handle"""
        if self.current_grad_handle is None:
            return
        import mindspore as ms
        sync_stream = self.get_grad_sync_stream()
        with ms.runtime.StreamCtx(sync_stream):
            self._process_current_handle()
            sync_event = sync_stream.record_event()
        sync_event.wait()
        self.current_grad_handle = None
        self.post_grad_handle_process = None

    def get_grad_sync_stream(self):
        import mindspore as ms
        if self.grad_sync_stream is None:
            self.grad_sync_stream = ms.runtime.Stream()
        return self.grad_sync_stream


hsdp_context = HSDPContext()
