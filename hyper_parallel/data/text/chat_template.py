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
"""Chat-template implementations used by LLM conversation transforms.

Canonical merge (05 §11.3) of the former
``components/data/chat_template.py`` and
``components/datasets/llm/chat_template.py`` duplicates: one
``CHAT_TEMPLATE_REGISTRY``, the ``DatasetLogger`` logging path, and
``IGNORE_INDEX`` shared from the constants module.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, MutableMapping, Optional, Sequence, Type, Union
import torch

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_dataset_logger(__name__)

ROLE_SUPPORTED = ["system", "user", "assistant", "tool"]


class Registry(MutableMapping):
    """Domain registry for chat-template formatters.

    Inlined from the former ``auto_models/components/utils/registry.py`` in
    stage 7 (05 §10.3 row 565): the generic Registry was merged into the
    domain registries that consume it instead of living in a utils package.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    registry = []

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: Human-readable name of the registered category, used in
                error messages.
        """
        self._name = name
        self.registry.append(name)
        self._local_mapping = {}
        self._global_mapping = {}

    def __getitem__(self, key: str) -> Union[Type, Callable]:
        """Look up a registered class or function by key.

        Args:
            key: Registration key.

        Returns:
            The locally overridden value when present, otherwise the globally
            registered value.

        Raises:
            ValueError: If ``key`` is not registered.
        """
        # First check if instance has a local override
        if key not in self.valid_keys():
            raise ValueError(f"Unknown {self._name} name: {key}. No {self._name} registered for this source.")
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key: str, value: Union[Type, Callable]) -> None:
        """Set a local override for ``key`` without affecting other instances."""
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key: str) -> None:
        """Delete the local override for ``key``."""
        del self._local_mapping[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all valid keys, local overrides taking precedence."""
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self) -> int:
        """Return the number of distinct registered keys."""
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    def register(
        self,
        key: str,
        cls_or_func: Optional[Union[Type, Callable]] = None,
    ) -> Union[Type, Callable]:
        """Register a class or function under ``key``.

        Can be used either as a plain call or as a decorator::

            registry.register("name", MyClass)

            @registry.register("name")
            class MyClass: ...

        Args:
            key: Registration key.
            cls_or_func: The class or function to register. When ``None``, a
                decorator is returned instead.

        Returns:
            The registered class/function, or a decorator when ``cls_or_func``
            is ``None``.

        Raises:
            ValueError: If ``key`` is already registered (decorator form).
        """
        if cls_or_func is not None:
            self._global_mapping[key] = cls_or_func
            return cls_or_func

        def decorator(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
            """Register the decorated class or function under ``key``."""
            if key in self._global_mapping:
                raise ValueError(
                    f"{self._name} for '{key}' is already registered. Cannot register duplicate {self._name}."
                )
            self._global_mapping.update({key: cls_or_func})
            return cls_or_func

        return decorator

    def valid_keys(self) -> List[str]:
        """Return the list of all registered keys."""
        return list(self.keys())

CHAT_TEMPLATE_REGISTRY = Registry("ChatTemplate")


def _format_janus_message(
        message: Dict[str, str], index: int, message_count: int, task_type: str, assistant_count: int
) -> tuple[str, int]:
    """Format one Janus message and update the generation assistant count."""
    role = message["role"]
    content = message["content"]
    separators = ["\n\n", "<｜end▁of▁sentence｜>"]
    if content == "":
        return role + ":", assistant_count
    if "assistant" in role and (
            "wikihow_generation" in task_type or "interleave_generation" in task_type
    ):
        prefix = "Assistant: " if assistant_count == 0 else ""
        suffix = separators[1] if index + 1 == message_count else separators[0]
        return prefix + content.strip() + suffix, assistant_count + 1
    if "assistant" in role:
        return "Assistant: " + content.strip() + separators[1], assistant_count
    if "user" in role:
        return "User: " + content.strip() + separators[0], assistant_count
    if "system" in role and "wikihow_generation" in task_type:
        instruction = "Please generate a step-by-step tutorial with images for the following question."
        return content.strip() + separators[0] + instruction + separators[0], assistant_count
    if "system" in role:
        return content.strip() + separators[0], assistant_count
    raise ValueError(f"Unknown role {role}, should be one of {{system, user, assistant}}.")


def _janus_labels(content_ids: List[int], image_token_id: int, loss_mask: int, task_type: str) -> List[int]:
    """Build labels for one encoded Janus message."""
    if loss_mask != 1:
        return [IGNORE_INDEX] * len(content_ids)
    if (
            image_token_id in content_ids
            and "wikihow_generation" not in task_type
            and "interleave_generation" not in task_type
    ):
        return [image_token_id if token == image_token_id else IGNORE_INDEX for token in content_ids]
    return content_ids


def build_chat_template(template_name: str, tokenizer: "PreTrainedTokenizer") -> "ChatTemplate":
    """Build the registered chat template for a tokenizer.

    Args:
        template_name: Registry key of the chat template implementation.
        tokenizer: Hugging Face tokenizer used for encoding and special tokens.

    Returns:
        The configured chat template instance.
    """
    return CHAT_TEMPLATE_REGISTRY[template_name](tokenizer)


class ChatTemplate(ABC):
    """
    Abstract class for chat template.
    """

    def __init__(self, tokenizer: Any) -> None:
        """Store the tokenizer used for encoding and special tokens."""
        self.tokenizer = tokenizer

    def save_pretrained(self, output_dir: str) -> None:
        """Attach the jinja template to the tokenizer and save it.

        Args:
            output_dir: Directory the tokenizer assets are written to.
        """
        self.tokenizer.chat_template = self.get_jinja_template()
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            logger.warning("Failed to save tokenizer.")

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, and labels.
        """

    @abstractmethod
    def get_jinja_template(self) -> str:
        """
        Gets the jinja template for the chat template.
        """


@CHAT_TEMPLATE_REGISTRY.register("default")
class DefaultTemplate(ChatTemplate):
    """Render each message as ``Role: content`` followed by the EOS token."""

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """Encode messages into input_ids, attention_mask, and loss-masked labels.

        Args:
            messages: Conversation records with ``role``, ``content``, and ``loss_mask`` fields.
            max_seq_len: Maximum sequence length kept from the end of the encoding.

        Returns:
            Model inputs with input_ids, attention_mask, and labels.
        """
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["role"].title() + ": " + message["content"].strip() + self.tokenizer.eos_token + "\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        """Return the jinja template matching the default ``Role: content`` rendering."""
        return (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{ message['role'].title() + ': ' + message['content'] | trim + eos_token + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("tokenizer")
class TokenizerTemplate(ChatTemplate):
    """Use the tokenizer's native chat template and train only selected messages."""

    def _update_prefix_labels(
            self,
            previous_ids: List[int],
            current_ids: List[int],
            labels: List[int],
    ) -> None:
        """Ensure appending a message does not rewrite the existing token prefix."""
        # ``labels`` is only adjusted by overrides that permit a terminal rewrite.
        del labels
        previous_length = len(previous_ids)
        if current_ids[:previous_length] != previous_ids:
            raise ValueError(
                "The tokenizer chat template structurally rewrote an earlier conversation prefix; "
                "the generic tokenizer template requires prefix-stable rendering."
            )

    def encode_messages(
            self,
            messages: Sequence[Dict[str, str]],
            max_seq_len: int = 8192,
    ) -> Dict[str, List[int]]:
        """Encode messages with the tokenizer chat template and mask non-assistant loss.

        Each message is appended incrementally so token boundaries follow the
        tokenizer's own prefix-stable rendering.

        Args:
            messages: Conversation records with ``role``, ``content``, and optional ``loss_mask``.
            max_seq_len: Maximum sequence length kept from the end of the encoding.

        Returns:
            Model inputs with input_ids, attention_mask, and labels.

        Raises:
            ValueError: If the template rewrites or shortens an earlier conversation prefix.
        """
        input_ids: List[int] = []
        labels: List[int] = []
        previous_length = 0

        for end, message in enumerate(messages, start=1):
            encoded = self.tokenizer.apply_chat_template(
                messages[:end],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            current_ids = encoded["input_ids"]
            current_length = len(current_ids)
            if current_length < previous_length:
                raise ValueError(
                    "The tokenizer chat template shortened the conversation after adding a message; "
                    "assistant-only loss masking requires monotonic message boundaries."
                )

            self._update_prefix_labels(input_ids, current_ids, labels)
            loss_mask = message.get("loss_mask", 1 if message["role"] == "assistant" else 0)
            new_ids = current_ids[previous_length:]
            labels.extend(new_ids if loss_mask == 1 else [IGNORE_INDEX] * len(new_ids))
            input_ids = current_ids
            previous_length = current_length

        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def get_jinja_template(self) -> str:
        """Return the tokenizer's native jinja chat template.

        Raises:
            ValueError: If the tokenizer does not define a chat template.
        """
        if not self.tokenizer.chat_template:
            raise ValueError("The tokenizer does not define a native chat template.")
        return self.tokenizer.chat_template


@CHAT_TEMPLATE_REGISTRY.register("gpt_oss")
class GptOssTokenizerTemplate(TokenizerTemplate):
    """Support GPT-OSS terminal rewrites from <|return|> to <|end|>."""

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        """Resolve the GPT-OSS terminal token IDs required for prefix rewrites.

        Args:
            tokenizer: Hugging Face tokenizer that must define <|return|> and <|end|>.

        Raises:
            ValueError: If either terminal token is missing from the vocabulary.
        """
        super().__init__(tokenizer)
        self.return_token_id = tokenizer.convert_tokens_to_ids("<|return|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if tokenizer.unk_token_id in (self.return_token_id, self.end_token_id):
            raise ValueError("The GPT-OSS chat template requires <|return|> and <|end|> tokenizer tokens.")

    def _update_prefix_labels(
            self,
            previous_ids: List[int],
            current_ids: List[int],
            labels: List[int],
    ) -> None:
        """Accept only the terminal <|return|>-to-<|end|> rewrite and fix its label.

        Args:
            previous_ids: Token IDs rendered before the latest message was appended.
            current_ids: Token IDs rendered after the latest message was appended.
            labels: Loss labels aligned with ``previous_ids``, updated in place.

        Raises:
            ValueError: If the rewrite is not the supported terminal substitution.
        """
        previous_length = len(previous_ids)
        rewritten_positions = [
            index
            for index in range(previous_length)
            if previous_ids[index] != current_ids[index]
        ]
        if not rewritten_positions:
            return

        is_terminal_rewrite = (
            rewritten_positions == [previous_length - 1]
            and len(current_ids) > previous_length
            and previous_ids[-1] == self.return_token_id
            and current_ids[previous_length - 1] == self.end_token_id
            and self.return_token_id not in current_ids[previous_length:]
        )
        if not is_terminal_rewrite:
            raise ValueError(
                "The GPT-OSS tokenizer chat template structurally rewrote an earlier conversation prefix; "
                "only the terminal <|return|>-to-<|end|> substitution is supported."
            )

        if labels[-1] != IGNORE_INDEX:
            labels[-1] = self.end_token_id


@CHAT_TEMPLATE_REGISTRY.register("llama2")
class Llama2Template(ChatTemplate):
    """Render messages with the Llama-2 [INST]/<<SYS>> prompt format."""

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """Encode messages into input_ids, attention_mask, and loss-masked labels.

        Args:
            messages: Conversation records with ``role``, ``content``, and ``loss_mask`` fields.
            max_seq_len: Maximum sequence length kept from the end of the encoding.

        Returns:
            Model inputs with input_ids, attention_mask, and labels.

        Raises:
            ValueError: If a message role is not one of system, user, assistant, or tool.
        """
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            if message["role"] == "system":
                content_str = "<<SYS>>\n" + message["content"].strip() + "\n<</SYS>>\n\n"
            elif message["role"] == "user":
                content_str = self.tokenizer.bos_token + "[INST] " + message["content"].strip() + " [/INST]"
            elif message["role"] == "assistant":
                content_str = " " + message["content"].strip() + " " + self.tokenizer.eos_token
            elif message["role"] == "tool":
                content_str = self.tokenizer.bos_token + "[TOOL] " + message["content"].strip() + " [/TOOL]"
            else:
                raise ValueError(
                    f"Unknown role {message['role']}, should be one of {{system, user, assistant, tool}}."
                )

            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        """Return the jinja template matching the Llama-2 prompt format."""
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
            "{% elif message['role'] == 'tool' %}"
            "{{ bos_token + '[TOOL] ' + content | trim + ' [/TOOL]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content | trim + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("Janus")
class JanusTemplate(ChatTemplate):
    """Render Janus multimodal conversations with image placeholder masks."""

    def encode_messages(
            self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192, task_type: str = ""
    ) -> Dict[str, List[int]]:
        """Encode messages into token, label, and image mask fields.

        Args:
            messages: Conversation records with ``role``, ``content``, and ``loss_mask`` fields.
            max_seq_len: Maximum sequence length kept from the end of the encoding.
            task_type: Optional task selector enabling generation-specific separators.

        Returns:
            Model inputs with input_ids, attention_mask, labels, images_seq_mask,
            and images_emb_mask.

        Raises:
            ValueError: If a message role is not one of system, user, or assistant.
        """
        input_ids, attention_mask, labels = [], [], []
        images_seq_mask, images_emb_mask = [], []
        assistant_count = 0
        for idx, message in enumerate(messages):
            content_str, assistant_count = _format_janus_message(
                message, idx, len(messages), task_type, assistant_count
            )
            if "system" in message["role"]:
                content_ids = self.tokenizer.encode(content_str)
            else:
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            image_token_id = self.tokenizer.vocab.get("<image_placeholder>")
            content_ids_tensor = torch.tensor(content_ids)
            images_seq_mask += (content_ids_tensor == image_token_id).tolist()
            image_token_id = self.tokenizer.vocab.get("<image_placeholder>")
            num_image_tokens = torch.sum(content_ids_tensor == image_token_id).item()
            n_image = num_image_tokens // 576
            if n_image > 0:
                images_emb_mask.append([True] * num_image_tokens)

            labels += _janus_labels(content_ids, image_token_id, message["loss_mask"], task_type)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": images_emb_mask,
        }
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        """Return the jinja template matching the Janus ChatML-style rendering."""
        return (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("chatml")
class ChatmlTemplate(ChatTemplate):
    """Render messages with the ChatML <|im_start|>/<|im_end|> format."""

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """Encode messages into input_ids, attention_mask, and loss-masked labels.

        Args:
            messages: Conversation records with ``role``, ``content``, and optional ``loss_mask``.
            max_seq_len: Maximum sequence length kept from the end of the encoding.

        Returns:
            Model inputs with input_ids, attention_mask, and labels.
        """
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)

            if "loss_mask" in message:
                loss_mask = message["loss_mask"]
            else:
                loss_mask = 1 if message["role"] == "assistant" else 0
            if loss_mask == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        """Return the jinja template matching the ChatML rendering."""
        return (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
