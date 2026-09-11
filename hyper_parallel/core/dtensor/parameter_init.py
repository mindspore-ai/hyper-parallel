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
"""Parameter init"""
# pylint: disable=C9006,C9007


def init_parameters(module, stage_index=0):
    r"""
        init parameters.

        Args:
            module: The module to init parameters.
            stage_index: stage index for init.
        Raises:
            ValueError: If the `module` is not a module.
    """
    if module is None:
        raise ValueError("input module must not be none.")
    if stage_index < 0:
        raise ValueError("input stage_index must be positive.")
