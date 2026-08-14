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
"""CPU contracts for model-scoped vLLM alignment selection."""

import pytest

from rl.roles.model import ModelRegistration
from rl.roles.rollout.vllm import VLLMGenerationEngine


def _model(hyper_model_name: str = "qwen3_5") -> ModelRegistration:
    return ModelRegistration("qwen", hyper_model_name, "/model", "/tokenizer")


def test_native_vllm_rejects_hyper_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native server must not silently ignore Hyper-only alignment."""
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")

    with pytest.raises(ValueError, match="requires model_implementation='hyper'"):
        VLLMGenerationEngine(
            _model(),
            {"vllm": {"model_implementation": "native"}},
            client=object(),
        )


def test_alignment_rejects_unsupported_hyper_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Qwen3.5 dependency patch must not activate for another model."""
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")

    with pytest.raises(ValueError, match="supports only Qwen3.5"):
        VLLMGenerationEngine(_model("qwen3"), {"vllm": {}}, client=object())


def test_alignment_value_is_canonicalized_for_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owned server receives one validated Boolean alignment value."""
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", " TRUE ")
    engine = VLLMGenerationEngine(_model(), {"vllm": {}}, client=object())

    server_environment = engine._server_environment("0")  # pylint: disable=protected-access
    server_command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert server_environment["HYPER_VLLM_ALIGNMENT"] == "true"
    assert "--no-enable-chunked-prefill" in server_command


def test_alignment_rejects_cached_or_compiled_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported cache and graph configurations fail before server startup."""
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")

    with pytest.raises(ValueError, match="does not support prefix caching"):
        VLLMGenerationEngine(
            _model(),
            {"vllm": {"enable_prefix_caching": True}},
            client=object(),
        )
    with pytest.raises(ValueError, match="requires enforce_eager=true"):
        VLLMGenerationEngine(
            _model(),
            {"vllm": {"enforce_eager": False}},
            client=object(),
        )


def test_alignment_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid values fail before a vLLM process is created."""
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "alignment")

    with pytest.raises(ValueError, match="must be true or false"):
        VLLMGenerationEngine(_model(), {"vllm": {}}, client=object())
