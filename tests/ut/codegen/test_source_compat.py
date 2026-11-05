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
"""Tests for model-owned Kimi remote-source compatibility."""

from transformers.utils import generic

from hyper_parallel.models.kimi_k3.adapter.codegen import prepare_model_import


def test_kimi_output_recorder_compat_installs_only_when_missing():
    """Install the Kimi descriptor only when Transformers does not provide it."""
    sentinel = object()
    original = getattr(generic, "OutputRecorder", sentinel)
    if original is not sentinel:
        delattr(generic, "OutputRecorder")
    try:
        prepare_model_import()
        installed = generic.OutputRecorder
        recorder = installed(dict, index=2)

        prepare_model_import()

        assert generic.OutputRecorder is installed
        assert recorder.target_class is dict
        assert recorder.index == 2
    finally:
        if original is sentinel:
            delattr(generic, "OutputRecorder")
        else:
            generic.OutputRecorder = original
