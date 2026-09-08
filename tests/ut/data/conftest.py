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
"""Shared fakes for ``tests/ut/data``.

Fixtures here never touch the network, the Hugging Face Hub, or real tokenizers;
they exist so text/VLM transform, template and collator tests run hermetically.
"""

import pytest


class FakeTokenizer:
    """Deterministic character-level tokenizer stand-in.

    Encoding contract (locked by the chat-template golden tests):

    * known special strings (``vocab`` keys) are matched greedily, longest
      first, and map to their vocab id;
    * every other character maps to ``ord(character)``;
    * ``add_special_tokens=True`` prepends the ``<s>`` id.

    ``apply_chat_template`` renders each message as ``<role>content</role>``
    and encodes the concatenation, which is prefix-stable by construction.
    """

    bos_token = "<s>"
    eos_token = "</s>"
    unk_token_id = 0
    chat_template = None

    def __init__(self):
        """Initialise the fixed vocabulary."""
        self.vocab = {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<image_placeholder>": 3,
            "<|return|>": 4,
            "<|end|>": 5,
        }
        self.saved_to = []

    def _specials_by_length(self):
        return sorted(self.vocab, key=len, reverse=True)

    def encode(self, text, add_special_tokens=True):
        """Encode text greedily into special-token and per-character ids."""
        ids = [self.vocab[self.bos_token]] if add_special_tokens else []
        specials = self._specials_by_length()
        index = 0
        while index < len(text):
            for special in specials:
                if text.startswith(special, index):
                    ids.append(self.vocab[special])
                    index += len(special)
                    break
            else:
                ids.append(ord(text[index]))
                index += 1
        return ids

    def convert_tokens_to_ids(self, token):
        """Return the vocab id for a special token (``unk`` id when missing)."""
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=True
    ):
        """Render messages prefix-stably and return encoded ``input_ids``."""
        del add_generation_prompt
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        ids = self.encode(text, add_special_tokens=False) if tokenize else text
        return {"input_ids": ids} if return_dict else ids

    def save_pretrained(self, output_dir):
        """Record the save target instead of writing files."""
        self.saved_to.append(output_dir)


class FailingSaveTokenizer(FakeTokenizer):
    """Tokenizer whose ``save_pretrained`` always fails."""

    def save_pretrained(self, output_dir):
        """Raise to exercise the template save-failure path."""
        raise OSError(f"cannot write to {output_dir}")


class PrefixRewritingTokenizer(FakeTokenizer):
    """Tokenizer whose chat template rewrites earlier prefixes (unsupported)."""

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=True
    ):
        """Render with the first content uppercased once a second message arrives."""
        del add_generation_prompt
        parts = []
        for index, message in enumerate(messages):
            content = message["content"]
            if index == 0 and len(messages) > 1:
                content = content.upper()
            parts.append(f"<{message['role']}>{content}</{message['role']}>")
        ids = self.encode("".join(parts), add_special_tokens=False)
        return {"input_ids": ids} if return_dict else ids


class GptOssTokenizer(FakeTokenizer):
    """Tokenizer with the GPT-OSS terminal ``<|return|>`` → ``<|end|>`` rewrite."""

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=True
    ):
        """End assistant turns with ``<|return|>``, rewritten to ``<|end|>`` once followed."""
        del add_generation_prompt
        parts = []
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                # the terminal token closes the turn: it is the last token of
                # the rendered prefix, as in the real GPT-OSS template
                has_later_message = index + 1 < len(messages)
                terminal = "<|end|>" if has_later_message else "<|return|>"
                parts.append(f"<assistant>{message['content']}{terminal}")
                continue
            parts.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        ids = self.encode("".join(parts), add_special_tokens=False)
        return {"input_ids": ids} if return_dict else ids


class FakeProcessor:
    """VLM processor stand-in: records calls, returns deterministic batches.

    Combines a :class:`FakeTokenizer` with a trivial image path: each image
    becomes one ``pixel_values`` row of ones. No hub or network access.
    """

    image_token = "<image_placeholder>"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def __call__(self, text=None, images=None, **kwargs):
        """Record the call and return a minimal multimodal batch dict."""
        self.calls.append({"text": text, "images": images, "kwargs": kwargs})
        result = {}
        if text is not None:
            result["input_ids"] = self.tokenizer.encode(text)
        if images is not None:
            result["pixel_values"] = [[1.0, 1.0, 1.0] for _ in images]
        return result

    def apply_chat_template(self, messages, **kwargs):
        """Delegate to the wrapped tokenizer's prefix-stable template."""
        return self.tokenizer.apply_chat_template(messages, **kwargs)


@pytest.fixture
def fake_tokenizer():
    """Provide a fresh deterministic fake tokenizer."""
    return FakeTokenizer()
