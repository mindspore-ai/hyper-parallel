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
"""Regression coverage for the Stage 9 legacy-runtime removal."""

from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.check.preflight import verify_generated_source
from hyper_parallel.codegen.generated_support import to_local_if_dtensor


def test_generated_support_unwraps_dtensor_like_values():
    value = object()
    wrapped = SimpleNamespace(to_local=lambda: value)

    assert to_local_if_dtensor(wrapped) is value
    assert to_local_if_dtensor(value) is value


def test_generated_preflight_compiles_without_executing_source(tmp_path):
    modeling_path = tmp_path / "modeling_demo_gen_npu.py"
    marker_path = tmp_path / "executed"
    modeling_path.write_text(
        f"open({str(marker_path)!r}, 'w', encoding='utf-8').write('bad')\n",
        encoding="utf-8",
    )

    verify_generated_source(SimpleNamespace(modeling_path=str(modeling_path)))

    assert not marker_path.exists()


def test_generated_preflight_rejects_invalid_python(tmp_path):
    modeling_path = tmp_path / "modeling_demo_gen_npu.py"
    modeling_path.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(SyntaxError):
        verify_generated_source(SimpleNamespace(modeling_path=str(modeling_path)))
