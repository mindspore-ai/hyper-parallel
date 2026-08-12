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
"""Tests for random Qwen3 checkpoint preparation in the Trainer demo."""

from examples.training_demo import prepare_model


class _Savable:
    """Record arguments passed to a mocked save_pretrained call."""

    def __init__(self) -> None:
        self.saved = None

    def save_pretrained(self, path, **kwargs) -> None:
        """Record the output path and keyword arguments."""
        self.saved = (path, kwargs)


def test_prepare_qwen3_checkpoint_builds_from_config_without_hub_weights(monkeypatch, tmp_path) -> None:
    """Build random BF16 weights from config and save model plus tokenizer."""
    config = object()
    model = _Savable()
    tokenizer = _Savable()
    config_calls = []
    tokenizer_calls = []
    model_calls = []

    monkeypatch.setattr(
        prepare_model.AutoConfig,
        "from_pretrained",
        lambda model_id: config_calls.append(model_id) or config,
    )
    monkeypatch.setattr(
        prepare_model.AutoTokenizer,
        "from_pretrained",
        lambda model_id: tokenizer_calls.append(model_id) or tokenizer,
    )

    def _from_config(received_config, **kwargs):
        model_calls.append((received_config, kwargs))
        return model

    monkeypatch.setattr(prepare_model.AutoModelForCausalLM, "from_config", _from_config)

    prepare_model.prepare_qwen3_checkpoint(tmp_path)

    assert config_calls == [prepare_model.QWEN3_MODEL_ID]
    assert tokenizer_calls == [prepare_model.QWEN3_MODEL_ID]
    assert model_calls == [
        (
            config,
            {
                "torch_dtype": prepare_model.torch.bfloat16,
                "attn_implementation": "sdpa",
            },
        )
    ]
    assert model.saved == (
        tmp_path,
        {"safe_serialization": True, "max_shard_size": "5GB"},
    )
    assert tokenizer.saved == (tmp_path, {})
