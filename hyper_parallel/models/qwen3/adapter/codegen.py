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
"""Code generation source renderers selected by the Qwen3 adapters."""

from typing import Any


def render_replacement_source(component: type, *, interface: Any) -> Any:
    """Render the fused GQA component selected by Qwen3 replacement rules."""
    from hyper_parallel.codegen.inline.component_source import (  # pylint: disable=C0415
        render_component_source,
    )

    return render_component_source(component, interface=interface)


__all__ = ["render_replacement_source"]
