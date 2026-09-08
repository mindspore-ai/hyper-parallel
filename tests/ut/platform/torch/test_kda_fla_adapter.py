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
"""CPU-only contract tests for the optional external FLA KDA adapter."""
from types import SimpleNamespace

import pytest
import torch

from hyper_parallel.platform.torch.custom_ops.kda import fla_adapter


@pytest.fixture(autouse=True)
def _clear_fla_runtime_cache():
    """Keep the cached optional runtime isolated between tests."""
    fla_adapter._require_fla_kda_runtime.cache_clear()
    yield
    fla_adapter._require_fla_kda_runtime.cache_clear()


def test_fla_adapter_reports_missing_optional_dependency(monkeypatch):
    """Selecting Triton without FLA fails explicitly instead of falling back."""
    original_import = fla_adapter.importlib.import_module

    def _import_without_fla(module_name):
        if module_name == "fla":
            raise ModuleNotFoundError("No module named 'fla'", name="fla")
        return original_import(module_name)

    monkeypatch.setattr(fla_adapter.importlib, "import_module", _import_without_fla)

    with pytest.raises(RuntimeError, match="optional flash-linear-attention"):
        fla_adapter._require_fla_kda_runtime()


def test_fla_adapter_rejects_an_old_fla_version(monkeypatch):
    """An installed FLA without the required KDA API is not silently accepted."""
    original_import = fla_adapter.importlib.import_module

    def _import_old_fla(module_name):
        if module_name == "fla":
            return SimpleNamespace(__version__="0.5.1")
        return original_import(module_name)

    monkeypatch.setattr(fla_adapter.importlib, "import_module", _import_old_fla)

    with pytest.raises(RuntimeError, match="requires FLA >= 0.6.0"):
        fla_adapter._require_fla_kda_runtime()


def test_run_fla_chunk_kda_preserves_the_public_contract(monkeypatch):
    """The Hyper adapter forwards the fused Kimi K3 options to FLA unchanged."""
    captured = {}
    expected_output = torch.randn(1, 4, 2, 3)
    expected_state = torch.randn(1, 2, 3, 3)

    def _fake_chunk_kda(**kwargs):
        captured.update(kwargs)
        return expected_output, expected_state

    runtime = SimpleNamespace(chunk_kda=_fake_chunk_kda)
    monkeypatch.setattr(fla_adapter, "_require_fla_kda_runtime", lambda: runtime)

    query = torch.randn(1, 4, 2, 3)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn_like(query)
    beta = torch.randn(1, 4, 2)
    a_log = torch.randn(2)
    dt_bias = torch.randn(2)
    initial_state = torch.randn(1, 2, 3, 3)

    output, final_state = fla_adapter.run_fla_chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=0.5,
        initial_state=initial_state,
        output_final_state=True,
        lower_bound=-4.0,
        chunk_size=32,
        safe_gate=False,
    )

    assert output is expected_output
    assert final_state is expected_state
    assert captured == {
        "q": query,
        "k": key,
        "v": value,
        "g": gate,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "scale": 0.5,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": False,
        "lower_bound": -4.0,
        "chunk_size": 32,
    }
