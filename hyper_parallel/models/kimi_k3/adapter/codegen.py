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
"""Code generation compatibility owned by the Kimi model families."""


def prepare_model_import() -> None:
    """Restore the data-only descriptor expected by Kimi remote source."""
    from transformers.utils import generic  # pylint: disable=C0415

    if hasattr(generic, "OutputRecorder"):
        return

    class OutputRecorder:  # pylint: disable=too-few-public-methods
        """Describe the output class and tuple index recorded by remote source."""

        def __init__(self, target_class: type, index: int = 0) -> None:
            self.target_class = target_class
            self.index = index

    generic.OutputRecorder = OutputRecorder


__all__ = ["prepare_model_import"]
