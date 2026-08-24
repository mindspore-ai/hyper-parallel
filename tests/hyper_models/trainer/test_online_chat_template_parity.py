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
"""Parity test between Online conversation transforms and Hugging Face rendering."""

import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from hyper_parallel.auto_models.components.datasets.llm.build_data_transform import ConversationTransform
from hyper_parallel.auto_models.components.datasets.llm.build_dataset import build_llm_dataset
from hyper_parallel.auto_models.components.datasets.llm.chat_template import build_chat_template
from hyper_parallel.auto_models.components.utils.constants import IGNORE_INDEX


def _build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a tiny local Hugging Face tokenizer with a native chat template."""
    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "user": 3,
        "assistant": 4,
        ":": 5,
        "hello": 6,
        "world": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + eos_token }}"
        "{% endfor %}"
    )
    return tokenizer


def test_online_conversation_transform_matches_huggingface_apply_chat_template(monkeypatch) -> None:
    """Compare the complete Online-to-transform result with native HF tokenization."""
    tokenizer = _build_tokenizer()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    source_dataset = [{"messages": messages}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            disable_progress_bars=lambda: None,
            enable_progress_bars=lambda: None,
            load_dataset=lambda *args, **kwargs: source_dataset,
        ),
    )
    transform = ConversationTransform(
        build_chat_template("tokenizer", tokenizer),
        max_seq_len=128,
        text_keys="messages",
    )
    transformed_dataset = build_llm_dataset(
        data_path="unused",
        data_config={
            "source_type": "online",
            "dataset_type": "mapping",
            "hf_dataset_name": "dummy/conversations",
            "namespace": "train",
        },
        transform=transform,
    )

    actual = transformed_dataset[0]
    expected = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    assert actual["input_ids"].tolist() == expected["input_ids"]
    assert actual["attention_mask"].tolist() == expected["attention_mask"]

    user_prefix = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]
    expected_labels = [IGNORE_INDEX] * len(user_prefix) + expected["input_ids"][len(user_prefix):]
    assert actual["labels"].tolist() == expected_labels
    assert actual["input_ids"].dtype == torch.long
