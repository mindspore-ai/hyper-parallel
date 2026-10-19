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
"""Golden cases for the merged chat template implementation.

``hyper_parallel.data.text.chat_template`` is the canonical merge (05
§11.3) of the former ``components/data/chat_template.py`` and
``components/datasets/llm/chat_template.py`` duplicates, which the
pre-merge golden cases proved behaviorally identical. These cases pin the
golden encodings the merge had to preserve; the final test pins the merged
module surface (single registry, shared IGNORE_INDEX, DatasetLogger).
"""
# pylint: disable=wrong-import-position

import json
import os
import unittest
from typing import Any


from hyper_parallel.data.dataset_logging import DatasetLogger
from hyper_parallel.data.text import chat_template
from hyper_parallel.data.text.build_data_transform import TextConversationTransform, _normalize_conversation

from tests.common.mark_utils import arg_mark
from tests.ut.data.conftest import (
    FailingSaveTokenizer,
    FakeTokenizer,
    GptOssTokenizer,
    PrefixRewritingTokenizer,
)

_MESSAGES = [
    {"role": "system", "content": "Be helpful.", "loss_mask": 0},
    {"role": "user", "content": "Hi there", "loss_mask": 0},
    {"role": "assistant", "content": "Hello!", "loss_mask": 1},
]


def _ids(text, add_special_tokens=False):
    """Golden token ids under the FakeTokenizer contract."""
    return FakeTokenizer().encode(text, add_special_tokens=add_special_tokens)


class MultiTurnThinkingTokenizer(FakeTokenizer):
    """Expose generation boundaries for reasoning and tool-turn masking tests."""

    def __init__(self):
        super().__init__()
        self.generation_histories = []

    def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False, return_dict=True, **kwargs,
    ):
        """Render each assistant's reasoning independently of later turns."""
        del kwargs
        parts = []
        for message in messages:
            body = message["content"]
            if message["role"] == "assistant":
                body = "<think>\n" + message.get("reasoning_content", "") + "\n</think>\n\n" + body
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            self.generation_histories.append(len(messages))
            parts.append("<|im_start|>assistant\n<think>\n")
        text = "".join(parts)
        result = self.encode(text, add_special_tokens=False) if tokenize else text
        return {"input_ids": result} if return_dict else result


class ToolOnlyThinkingTokenizer(MultiTurnThinkingTokenizer):
    """Render calls without content and merge adjacent newlines like the real tokenizer."""

    def __init__(self):
        super().__init__()
        self.vocab.update({"\n\n": 6, "\n": 7})

    def apply_chat_template(self, messages, **kwargs):
        """Keep tool-call serialization in the native template, not the training encoder."""
        rendered_messages = []
        for message in messages:
            rendered = dict(message)
            rendered["content"] = message.get("content") or ""
            rendered["reasoning_content"] = message.get("reasoning_content") or ""
            for call in message.get("tool_calls", []):
                rendered["content"] += "<tool_call>" + json.dumps(call, ensure_ascii=False) + "</tool_call>"
            rendered_messages.append(rendered)
        return super().apply_chat_template(rendered_messages, **kwargs)


class TestChatTemplateGoldens(unittest.TestCase):
    """Golden encodings preserved by the merged chat template module."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_template_golden(self):
        """default: ``Role: content<eos>\n`` per message; loss_mask selects labels."""
        expected_ids = (
            _ids("System: Be helpful.</s>\n")
            + _ids("User: Hi there</s>\n")
            + _ids("Assistant: Hello!</s>\n")
        )
        result = chat_template.build_chat_template(
            "default", FakeTokenizer()
        ).encode_messages(_MESSAGES)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(result["attention_mask"], [1] * len(expected_ids))
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_ids) - len(_ids("Assistant: Hello!</s>\n")))
            + _ids("Assistant: Hello!</s>\n"),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_template_requires_loss_mask(self):
        """default: a message without ``loss_mask`` raises KeyError."""
        messages = [{"role": "user", "content": "Hi"}]
        template = chat_template.build_chat_template("default", FakeTokenizer())
        with self.assertRaises(KeyError):
            template.encode_messages(messages)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_logs_first_rendering_once(self):
        """tokenizer: optionally log one raw conversation and its native rendering."""
        template = chat_template.build_chat_template(
            "tokenizer",
            FakeTokenizer(),
            log_first_rendered_template=True,
        )
        messages = [{"role": "user", "content": "Hi"}]

        with self.assertLogs("hyper_parallel.data.text.chat_template", level="INFO") as captured:
            template.encode_messages(messages)
            template.encode_messages(messages)

        output = "\n".join(captured.output)
        self.assertEqual(output.count("First chat-template input messages"), 1)
        self.assertEqual(output.count("First rendered chat template"), 1)
        self.assertIn("<user>Hi</user>", output)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_forwards_rendering_kwargs(self):
        """tokenizer: use identical native-template kwargs for every incremental rendering."""
        class RecordingTokenizer(FakeTokenizer):
            """Record keyword arguments forwarded to the native template."""

            def __init__(self):
                super().__init__()
                self.template_kwargs = []
                self.generation_prompts = []

            def apply_chat_template(
                    self,
                    messages: Any,
                    tokenize: bool = True,
                    add_generation_prompt: bool = False,
                    return_dict: bool = True,
                    **kwargs: Any,
            ) -> Any:
                """Record native-template options before delegating to the fake tokenizer."""
                self.template_kwargs.append(kwargs)
                self.generation_prompts.append(add_generation_prompt)
                return super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    return_dict=return_dict,
                )

        tokenizer = RecordingTokenizer()
        template = chat_template.build_chat_template(
            "tokenizer",
            tokenizer,
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": True},
        )
        template.encode_messages([
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ])

        self.assertEqual(tokenizer.template_kwargs, [
            {"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": True},
            {"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": True},
            {"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": True},
        ])
        self.assertEqual(tokenizer.generation_prompts, [False, False, True])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_forwards_sample_tools(self):
        """tokenizer: forward tool definitions together with static rendering options."""
        class RecordingTokenizer(FakeTokenizer):
            """Record keyword arguments forwarded to the native template."""

            def __init__(self):
                super().__init__()
                self.template_kwargs = []
                self.generation_prompts = []

            def apply_chat_template(
                    self,
                    messages: Any,
                    tokenize: bool = True,
                    add_generation_prompt: bool = False,
                    return_dict: bool = True,
                    **kwargs: Any,
            ) -> Any:
                """Record native-template options before delegating to the fake tokenizer."""
                self.template_kwargs.append(kwargs)
                self.generation_prompts.append(add_generation_prompt)
                return super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    return_dict=return_dict,
                )

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        tokenizer = RecordingTokenizer()
        template = chat_template.build_chat_template(
            "tokenizer",
            tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )
        template.encode_messages(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            tools=tools,
        )

        self.assertEqual(tokenizer.template_kwargs, [
            {"enable_thinking": True, "tools": tools},
            {"enable_thinking": True, "tools": tools},
            {"enable_thinking": True, "tools": tools},
        ])
        self.assertEqual(tokenizer.generation_prompts, [False, False, True])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_masks_native_assistant_generation_prefix(self):
        """tokenizer: mask the same assistant prefix that inference pre-fills."""
        class ThinkingTokenizer(FakeTokenizer):
            """Render a Qwen-style assistant generation prefix with an opening think tag."""

            def __init__(self):
                super().__init__()
                self.vocab.update({"\n\n": 6, "\n": 7})

            def apply_chat_template(
                    self,
                    messages: Any,
                    tokenize: bool = True,
                    add_generation_prompt: bool = False,
                    return_dict: bool = True,
                    **kwargs: Any,
            ) -> Any:
                """Render a prefix-stable Qwen-style conversation."""
                del kwargs
                text = "".join(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                    for message in messages
                )
                if add_generation_prompt:
                    text += "<|im_start|>assistant\n<think>\n"
                ids = self.encode(text, add_special_tokens=False) if tokenize else text
                return {"input_ids": ids} if return_dict else ids

        tokenizer = ThinkingTokenizer()
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>\n\nReasoning</think>\nAnswer"},
        ]
        result = chat_template.build_chat_template("tokenizer", tokenizer).encode_messages(messages)
        completed_conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            return_dict=False,
        )
        generation_prefix_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
        )
        generation_prefix = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        assistant_target = tokenizer.encode(
            completed_conversation[len(generation_prefix_text):],
            add_special_tokens=False,
        )
        whole_conversation_encoding = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )["input_ids"]

        self.assertNotEqual(whole_conversation_encoding, generation_prefix + assistant_target)
        self.assertEqual(result["input_ids"], generation_prefix + assistant_target)
        self.assertEqual(result["labels"][:len(generation_prefix)], [-100] * len(generation_prefix))
        self.assertEqual(result["labels"][len(generation_prefix):], assistant_target)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_multiturn_preserves_all_assistant_targets(self):
        """Mask each new source without masking targets from preceding turns."""
        for turns in (1, 2, 3):
            with self.subTest(turns=turns):
                tokenizer = MultiTurnThinkingTokenizer()
                messages = [{"role": "system", "content": "Be helpful."}]
                expected_ids = _ids("<|im_start|>system\nBe helpful.<|im_end|>\n")
                expected_labels = [-100] * len(expected_ids)
                for turn in range(turns):
                    question, reasoning, answer = f"Question {turn}", f"Reason {turn}", f"Answer {turn}"
                    messages.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "reasoning_content": reasoning, "content": answer},
                    ])
                    source = _ids(
                        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    )
                    target = _ids(f"{reasoning}\n</think>\n\n{answer}<|im_end|>\n")
                    expected_ids.extend(source + target)
                    expected_labels.extend([-100] * len(source) + target)
                result = chat_template.build_chat_template("tokenizer", tokenizer).encode_messages(messages)
                self.assertEqual(result["input_ids"], expected_ids)
                self.assertEqual(result["labels"], expected_labels)
                self.assertEqual(tokenizer.generation_histories, list(range(2, 2 * turns + 1, 2)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_multiturn_tools_and_explicit_masks(self):
        """Tool observations stay masked unless explicitly enabled; skipped assistants stay masked."""
        for tool_loss_mask in (0, 1):
            with self.subTest(tool_loss_mask=tool_loss_mask):
                tokenizer = MultiTurnThinkingTokenizer()
                messages = [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Call tool", "reasoning_content": "Choose tool"},
                    {"role": "tool", "content": "Result", "loss_mask": tool_loss_mask},
                    {"role": "assistant", "content": "Skipped", "loss_mask": 0},
                    {"role": "user", "content": "Follow up"},
                    {"role": "assistant", "content": "Final", "reasoning_content": "Check result"},
                ]
                expected_labels = []
                for message in messages:
                    segment = tokenizer.apply_chat_template([message], tokenize=False, return_dict=False)
                    if message["role"] == "assistant" and message.get("loss_mask", 1):
                        header = "<|im_start|>assistant\n<think>\n"
                        expected_labels.extend([-100] * len(_ids(header)) + _ids(segment[len(header):]))
                    else:
                        ids = _ids(segment)
                        expected_labels.extend(ids if message.get("loss_mask", 0) else [-100] * len(ids))
                template = chat_template.build_chat_template("tokenizer", tokenizer)
                result = template.encode_messages(messages, tools=[{"type": "function", "function": {"name": "f"}}])
                self.assertEqual(result["labels"], expected_labels)
                self.assertEqual(result["input_ids"], tokenizer.apply_chat_template(messages)["input_ids"])
                self.assertEqual(tokenizer.generation_histories, [1, 5])
                truncated = template.encode_messages(messages, max_seq_len=80)
                self.assertEqual(truncated["input_ids"], result["input_ids"][-80:])
                self.assertEqual(truncated["labels"], expected_labels[-80:])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_tool_only_turns_keep_bpe_boundaries_and_masks(self):
        """Empty/missing content and reasoning must not lose tool targets when history grows."""
        calls = [{"type": "function", "function": {"name": "lookup", "arguments": {"items": ["x", "y"]}}},
                 {"type": "function", "function": {"name": "count", "arguments": {}}}]
        for content in ("", None, "missing"):
            for reasoning in ("", None, "missing"):
                for tool_role in ("user", "tool"):
                    for loss_mask in (0, 1):
                        with self.subTest(content=content, reasoning=reasoning, role=tool_role, loss_mask=loss_mask):
                            assistant = {"role": "assistant", "tool_calls": calls, "loss_mask": loss_mask}
                            if content != "missing":
                                assistant["content"] = content
                            if reasoning != "missing":
                                assistant["reasoning_content"] = reasoning
                            messages = [{"role": "user", "content": "Question"}, assistant,
                                        {"role": tool_role, "content": "Result"}, dict(assistant),
                                        {"role": tool_role, "content": "Second result"},
                                        {"role": "assistant", "content": "Done", "reasoning_content": "Check"}]
                            original = json.dumps(messages)
                            normalized = _normalize_conversation(messages)
                            self.assertEqual(json.dumps(messages), original)
                            tokenizer = ToolOnlyThinkingTokenizer()
                            template = chat_template.build_chat_template("tokenizer", tokenizer)
                            expected_ids, expected_labels = [], []
                            for end, message in enumerate(normalized, 1):
                                segment = tokenizer.apply_chat_template([message], tokenize=False, return_dict=False)
                                if message["role"] == "assistant" and message.get("loss_mask", 1):
                                    header = "<|im_start|>assistant\n<think>\n"
                                    source = tokenizer.encode(header, add_special_tokens=False)
                                    target = tokenizer.encode(segment[len(header):], add_special_tokens=False)
                                    expected_ids.extend(source + target)
                                    expected_labels.extend([-100] * len(source) + target)
                                else:
                                    ids = tokenizer.encode(segment, add_special_tokens=False)
                                    expected_ids.extend(ids)
                                    expected_labels.extend([-100] * len(ids))
                                result = template.encode_messages(normalized[:end], max_seq_len=10000)
                                self.assertEqual(result["input_ids"], expected_ids)
                                self.assertEqual(result["labels"], expected_labels)
                            if loss_mask:
                                whole_ids = tokenizer.apply_chat_template(normalized)["input_ids"]
                                self.assertNotEqual(result["input_ids"], whole_ids)
                            transform = TextConversationTransform(template, 10000, "messages", "tools")
                            sample = transform({"messages": messages, "tools": []})[0]
                            self.assertEqual(sample["input_ids"].tolist(), expected_ids[:-1])
                            self.assertEqual(sample["labels"].tolist(), expected_labels[1:])
                            truncated = template.encode_messages(normalized, max_seq_len=80)
                            self.assertEqual(truncated["labels"], expected_labels[-80:])
                            self.assertEqual(truncated["input_ids"], expected_ids[-80:])

    def test_tokenizer_tool_only_history_rewrite_is_still_rejected(self):
        """A real tool-argument rewrite is not mistaken for harmless BPE retokenization."""
        class RewritingToolTokenizer(ToolOnlyThinkingTokenizer):
            """Change a historical tool argument after a response arrives."""

            def apply_chat_template(self, messages, **kwargs):
                changed = json.loads(json.dumps(messages))
                if len(changed) >= 3:
                    changed[1]["tool_calls"][0]["function"]["arguments"] = {"query": "rewritten"}
                return super().apply_chat_template(changed, **kwargs)

        messages = [{"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "", "reasoning_content": "",
                     "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "original"}}}]},
                    {"role": "tool", "content": "Result"}]
        template = chat_template.build_chat_template("tokenizer", RewritingToolTokenizer())
        with self.assertRaisesRegex(ValueError, "prefix|shortened"):
            template.encode_messages(messages)

    def test_normalization_still_requires_content_without_tool_calls(self):
        """Only assistant messages carrying calls may omit content."""
        for message in ({"role": "assistant"}, {"role": "user", "tool_calls": [{"function": {}}]},
                        {"role": "assistant", "tool_calls": []}):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, "content"):
                _normalize_conversation([message])

    def test_tokenizer_multiturn_preserve_thinking_controls_history_stability(self) -> None:
        """Preserve supervised history when enabled; continue rejecting rewrites otherwise."""
        class HistoryRewritingTokenizer(MultiTurnThinkingTokenizer):
            """Emulate a native template that removes historical reasoning."""

            def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
                """Remove historical reasoning only when preservation is not enabled."""
                modified = [dict(message) for message in messages]
                if len(modified) >= 3 and not kwargs.get("preserve_thinking", False):
                    modified[1]["reasoning_content"] = ""
                return super().apply_chat_template(modified, **kwargs)

        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer", "reasoning_content": "Reason"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Final", "reasoning_content": "Check"},
        ]
        for kwargs in ({}, {"preserve_thinking": False}):
            with self.subTest(kwargs=kwargs):
                template = chat_template.build_chat_template(
                    "tokenizer", HistoryRewritingTokenizer(), chat_template_kwargs=kwargs,
                )
                with self.assertRaisesRegex(ValueError, "prefix|shortened"):
                    template.encode_messages(messages)

        template = chat_template.build_chat_template(
            "tokenizer", HistoryRewritingTokenizer(), chat_template_kwargs={"preserve_thinking": True},
        )
        prefix = template.encode_messages(messages[:2])
        result = template.encode_messages(messages)
        reference = chat_template.build_chat_template("tokenizer", MultiTurnThinkingTokenizer())
        expected = reference.encode_messages(messages)
        self.assertEqual(result, expected)
        self.assertEqual(result["labels"][:len(prefix["labels"])], prefix["labels"])

    def test_preserve_thinking_keeps_tool_only_targets(self) -> None:
        """The option must preserve empty-reasoning tool targets and masked tool observations."""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]},
            {"role": "tool", "content": "Result"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Final", "reasoning_content": "Check"},
        ]
        reference = chat_template.build_chat_template("tokenizer", ToolOnlyThinkingTokenizer())
        template = chat_template.build_chat_template(
            "tokenizer", ToolOnlyThinkingTokenizer(), chat_template_kwargs={"preserve_thinking": True},
        )
        for end in range(1, len(messages) + 1):
            with self.subTest(message_count=end):
                self.assertEqual(template.encode_messages(messages[:end]), reference.encode_messages(messages[:end]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_validates_rendering_kwargs(self):
        """tokenizer: reject Trainer-owned arguments and invalid known reasoning options."""
        invalid_kwargs = (
            ({"tokenize": False}, "Trainer-owned arguments"),
            ({"tools": []}, "Trainer-owned arguments"),
            ({"enable_thinking": "yes"}, "enable_thinking must be a bool"),
            ({"preserve_thinking": 1}, "preserve_thinking must be a bool"),
            ({"preserve_thinking": "true"}, "preserve_thinking must be a bool"),
            ({"reasoning_effort": "high"}, "reasoning_effort must be one of"),
        )
        for kwargs, expected_error in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, expected_error):
                chat_template.build_chat_template(
                    "tokenizer",
                    FakeTokenizer(),
                    chat_template_kwargs=kwargs,
                )

        with self.assertRaisesRegex(ValueError, "require chat_template='tokenizer'"):
            chat_template.build_chat_template(
                "chatml",
                FakeTokenizer(),
                chat_template_kwargs={"enable_thinking": True},
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_chatml_template_golden(self):
        """chatml: ``<|im_start|>role\ncontent<|im_end|>\n``; missing loss_mask defaults by role."""
        expected_segments = [
            _ids("<|im_start|>system\nBe helpful.<|im_end|>\n"),
            _ids("<|im_start|>user\nHi there<|im_end|>\n"),
            _ids("<|im_start|>assistant\nHello!<|im_end|>\n"),
        ]
        messages = [dict(message) for message in _MESSAGES]
        for message in messages:  # chatml defaults loss_mask by role
            del message["loss_mask"]
        result = chat_template.build_chat_template(
            "chatml", FakeTokenizer()
        ).encode_messages(messages)
        expected_ids = sum(expected_segments, [])
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_segments[0]) + len(expected_segments[1]))
            + expected_segments[2],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_llama2_template_golden(self):
        """llama2: ``<<SYS>>``/``[INST]`` format, tool role, unknown role raises."""
        messages = _MESSAGES + [{"role": "tool", "content": "result", "loss_mask": 0}]
        expected_ids = (
            _ids("<<SYS>>\nBe helpful.\n<</SYS>>\n\n")
            + _ids("<s>[INST] Hi there [/INST]")
            + _ids(" Hello! </s>")
            + _ids("<s>[TOOL] result [/TOOL]")
        )
        result = chat_template.build_chat_template(
            "llama2", FakeTokenizer()
        ).encode_messages(messages)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(
            result["labels"],
            [-100] * len(_ids("<<SYS>>\nBe helpful.\n<</SYS>>\n\n"))
            + [-100] * len(_ids("<s>[INST] Hi there [/INST]"))
            + _ids(" Hello! </s>")
            + [-100] * len(_ids("<s>[TOOL] result [/TOOL]")),
        )

        template = chat_template.build_chat_template("llama2", FakeTokenizer())
        with self.assertRaisesRegex(ValueError, "Unknown role"):
            template.encode_messages(
                [{"role": "narrator", "content": "x", "loss_mask": 0}]
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_janus_template_golden(self):
        """Janus: role formats, image seq/emb masks, generation task_type switch."""
        image_content = "<image_placeholder>" * 576
        messages = [
            {"role": "system", "content": "SYS", "loss_mask": 0},
            {"role": "user", "content": image_content, "loss_mask": 1},
            {"role": "assistant", "content": "A", "loss_mask": 1},
        ]
        result = chat_template.build_chat_template(
            "Janus", FakeTokenizer()
        ).encode_messages(messages)
        image_id = FakeTokenizer().vocab["<image_placeholder>"]
        # system ids include the BOS prefix; user/assistant do not.
        self.assertEqual(
            result["input_ids"],
            _ids("SYS\n\n", add_special_tokens=True)
            + _ids("User: " + image_content + "\n\n")
            + _ids("Assistant: A<｜end▁of▁sentence｜>"),
        )
        self.assertEqual(len(result["images_seq_mask"]), len(result["input_ids"]))
        self.assertEqual(result["images_emb_mask"], [[True] * 576])
        # non-generation task: only image placeholder positions keep loss in
        # the user message; the assistant message keeps full loss.
        image_positions = sum(
            1 for token in result["input_ids"] if token == image_id
        )
        self.assertEqual(image_positions, 576)
        kept = [token for token in result["labels"] if token != -100]
        self.assertEqual(
            kept,
            [image_id] * 576 + _ids("Assistant: A<｜end▁of▁sentence｜>"),
        )

        # generation task types keep full content loss instead of image-only loss
        generation = chat_template.build_chat_template(
            "Janus", FakeTokenizer()
        ).encode_messages(messages, task_type="wikihow_generation")
        user_labels = generation["labels"]
        self.assertIn(ord("U"), user_labels)  # "User: ..." prefix keeps loss

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_golden(self):
        """tokenizer: prefix-stable native template with assistant-only default loss."""
        messages = [dict(message) for message in _MESSAGES]
        for message in messages:
            del message["loss_mask"]
        result = chat_template.build_chat_template(
            "tokenizer", FakeTokenizer()
        ).encode_messages(messages)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        assistant_segment = _ids("<assistant>Hello!</assistant>")
        expected_ids = _ids(rendered)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(result["attention_mask"], [1] * len(expected_ids))
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_ids) - len(assistant_segment)) + assistant_segment,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_groups_leading_system_with_user(self):
        """tokenizer: do not render a system-only prefix when deriving loss boundaries."""
        class UserRequiredTokenizer(FakeTokenizer):
            """Reject message prefixes that do not contain a user query."""

            def __init__(self):
                super().__init__()
                self.rendered_prefixes = []

            def apply_chat_template(
                    self,
                    messages: Any,
                    tokenize: bool = True,
                    add_generation_prompt: bool = False,
                    return_dict: bool = True,
                    **kwargs: Any,
            ) -> Any:
                """Record valid prefixes and emulate a tokenizer requiring a user message."""
                del kwargs
                if not any(message["role"] == "user" for message in messages):
                    raise ValueError("No user query found in messages.")
                self.rendered_prefixes.append(list(messages))
                return super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    return_dict=return_dict,
                )

        messages = [
            {"role": "system", "content": None, "reasoning_effort": None},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        tokenizer = UserRequiredTokenizer()
        result = chat_template.build_chat_template("tokenizer", tokenizer).encode_messages(messages)

        self.assertEqual([len(prefix) for prefix in tokenizer.rendered_prefixes], [2, 3, 2])
        user_prefix = _ids("<system>None</system><user>Question</user>")
        assistant_segment = _ids("<assistant>Answer</assistant>")
        self.assertEqual(result["input_ids"], user_prefix + assistant_segment)
        self.assertEqual(result["labels"], [-100] * len(user_prefix) + assistant_segment)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_rejects_prefix_rewrite(self):
        """tokenizer: structural prefix rewrites raise ValueError."""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        template = chat_template.build_chat_template(
            "tokenizer", PrefixRewritingTokenizer()
        )
        with self.assertRaisesRegex(ValueError, "prefix"):
            template.encode_messages(messages)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gpt_oss_terminal_rewrite_golden(self):
        """gpt_oss: trailing ``<|return|>`` rewritten to ``<|end|>`` keeps its label."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = chat_template.build_chat_template(
            "gpt_oss", GptOssTokenizer()
        ).encode_messages(messages)
        end_id = GptOssTokenizer().vocab["<|end|>"]
        self.assertEqual(
            result["input_ids"],
            _ids("<user>q1</user>")
            + _ids("<assistant>a1<|end|>")
            + _ids("<user>q2</user>"),
        )
        # the rewritten terminal token keeps loss (not IGNORE_INDEX)
        rewritten_index = result["input_ids"].index(end_id)
        self.assertEqual(result["labels"][rewritten_index], end_id)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_max_seq_len_keeps_tail(self):
        """All templates truncate from the front, keeping the sequence tail."""
        long_messages = [
            {"role": "user", "content": "abcdefgh", "loss_mask": 0},
            {"role": "assistant", "content": "ijklmnop", "loss_mask": 1},
        ]
        for name in ("default", "chatml", "llama2"):
            template = chat_template.build_chat_template(name, FakeTokenizer())
            full = template.encode_messages(long_messages)
            truncated = template.encode_messages(long_messages, max_seq_len=8)
            self.assertEqual(truncated["input_ids"], full["input_ids"][-8:], name)
            self.assertEqual(truncated["labels"], full["labels"][-8:], name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_save_pretrained_failure_is_swallowed(self):
        """save_pretrained logs and swallows tokenizer save failures."""
        tokenizer = FailingSaveTokenizer()
        template = chat_template.build_chat_template("default", tokenizer)
        with self.assertLogs(level="WARNING"):
            template.save_pretrained("/nonexistent-dir")
        self.assertEqual(
            tokenizer.chat_template, template.get_jinja_template()
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_merged_module_surface(self):
        """Pin how the merge resolved the recorded non-behavioral conflicts.

        * one ``CHAT_TEMPLATE_REGISTRY`` with the union of both registries;
        * ``IGNORE_INDEX`` is the shared constants import (==-100);
        * logging goes through ``DatasetLogger`` (the datasets/llm side).
        """
        self.assertEqual(chat_template.IGNORE_INDEX, -100)
        self.assertEqual(
            sorted(chat_template.CHAT_TEMPLATE_REGISTRY),
            ["Janus", "chatml", "default", "gpt_oss", "llama2", "tokenizer"],
        )
        self.assertIsInstance(chat_template.logger, DatasetLogger)


if __name__ == "__main__":
    unittest.main()
