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
"""Token-exact tokenizer chat-template encoding for agent observations."""

from typing import Any, Mapping, Optional, Sequence

from rl.agentic.envs.base import Action, Observation

from hyper_parallel import get_platform

platform = get_platform()

CHAT_TEMPLATE_MESSAGES = "chat_template_messages"


class TokenizerChatTemplateEncoder:
    """Append native chat-template continuations without replacing rollout tokens.

    Generated token IDs remain authoritative because decoding and re-tokenizing
    model output is not guaranteed to round-trip through a BPE tokenizer. The
    encoder therefore compares multiple placeholder assistant messages to
    derive only their shared closing delimiter, next observation, and following
    generation marker.
    """

    _ACTION_PLACEHOLDERS = ("a", "0", "Z")

    def __init__(self, tokenizer: Any, device: Optional[Any] = None) -> None:
        """Initialize one stateful encoder for exactly one agent episode."""
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ValueError(
                "agentic.apply_chat_template requires a tokenizer with apply_chat_template()"
            )
        self.tokenizer = tokenizer
        self.device = device
        self._messages: list[dict[str, Any]] = []
        self._started = False
        self._awaiting_observation = False

    @staticmethod
    def _template_role(role: str) -> str:
        """Map the generic environment role to a tokenizer-supported role."""
        return "user" if role == "environment" else role

    @staticmethod
    def _normalize_initial_messages(messages: Any) -> list[dict[str, Any]]:
        """Validate structured initial messages supplied by an environment."""
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{CHAT_TEMPLATE_MESSAGES} must be a sequence of mappings")
        normalized = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError(f"Every {CHAT_TEMPLATE_MESSAGES} entry must be a mapping")
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"Every {CHAT_TEMPLATE_MESSAGES} entry must define a role")
            normalized.append(dict(message))
        if not normalized:
            raise ValueError(f"{CHAT_TEMPLATE_MESSAGES} must not be empty")
        return normalized

    def _render(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """Render structured messages plus the next assistant generation marker."""
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        if input_ids is None:
            raise ValueError("Tokenizer chat template did not return input_ids")
        if hasattr(input_ids, "detach"):
            input_ids = input_ids.detach().cpu().tolist()
        elif hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise ValueError("Tokenizer chat template must return one token sequence")
            input_ids = input_ids[0]
        if not isinstance(input_ids, list) or not all(isinstance(token_id, int) for token_id in input_ids):
            raise ValueError("Tokenizer chat template input_ids must be a flat integer sequence")
        return input_ids

    def _render_observation_delta(self, content: str, role: str) -> list[int]:
        """Derive the template suffix after an already generated assistant span."""
        if not self._awaiting_observation or not self._messages:
            raise RuntimeError("Cannot encode incremental feedback before recording an action")
        if self._messages[-1].get("role") != "assistant":
            raise RuntimeError("Recorded chat history does not end with an assistant action")
        observation_message = {
            "role": self._template_role(role),
            "content": content,
        }
        messages_before_action = self._messages[:-1]
        rendered_continuations = [
            self._render(
                (
                    *messages_before_action,
                    {"role": "assistant", "content": placeholder},
                    observation_message,
                )
            )
            for placeholder in self._ACTION_PLACEHOLDERS
        ]
        if len({tuple(rendered) for rendered in rendered_continuations}) < 2:
            raise ValueError(
                "Tokenizer chat template ignored every assistant action placeholder"
            )
        suffix_length = 0
        maximum_suffix_length = min(map(len, rendered_continuations))
        while suffix_length < maximum_suffix_length:
            token_ids = {
                rendered[-suffix_length - 1]
                for rendered in rendered_continuations
            }
            if len(token_ids) != 1:
                break
            suffix_length += 1
        delta_ids = (
            []
            if suffix_length == 0
            else rendered_continuations[0][-suffix_length:]
        )
        if not delta_ids:
            raise ValueError(
                "Tokenizer chat template cannot derive an incremental observation suffix"
        )
        self._messages.append(observation_message)
        self._awaiting_observation = False
        return delta_ids

    def __call__(
        self,
        content: str,
        role: str,
        metadata: dict[str, Any],
    ) -> Observation:
        """Encode one initial or incremental observation as a template suffix."""
        observation_metadata = dict(metadata)
        initial_messages = observation_metadata.pop(CHAT_TEMPLATE_MESSAGES, None)
        if not self._started:
            self._messages = (
                self._normalize_initial_messages(initial_messages)
                if initial_messages is not None
                else [{"role": self._template_role(role), "content": content}]
            )
            self._started = True
            delta_ids = self._render(self._messages)
            if not delta_ids:
                raise ValueError("Tokenizer chat template produced an empty initial observation")
        else:
            if initial_messages is not None:
                raise ValueError(f"{CHAT_TEMPLATE_MESSAGES} is valid only for the initial observation")
            delta_ids = self._render_observation_delta(content, role)
        observation_metadata["role"] = role
        return Observation(
            content=content,
            token_ids=platform.tensor(
                delta_ids,
                dtype=platform.tensor_dtype.long,
                device=self.device,
            ),
            metadata=observation_metadata,
        )

    def record_action(self, action: Action) -> None:
        """Add one exact generated assistant span before encoding feedback."""
        if not self._started:
            raise RuntimeError("Cannot record an action before the initial chat observation")
        if self._awaiting_observation:
            raise RuntimeError("Cannot record another action before encoding its observation")
        action_token_ids = action.token_ids.detach().cpu().tolist()
        if not isinstance(action_token_ids, list) or not all(
            isinstance(token_id, int) for token_id in action_token_ids
        ):
            raise ValueError("Action token_ids must be a flat integer sequence")
        self._messages.append({"role": "assistant", "content": action.content})
        self._awaiting_observation = True
