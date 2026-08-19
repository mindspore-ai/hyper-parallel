# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the tokenizer used by the private LLM data stages."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AutoTokenizer:
    """
    Auto tokenizer class that dispatches to appropriate tokenizer implementations.

    Similar to HuggingFace's AutoTokenizer, but with a custom registry for specialized
    tokenizer implementations.

    The dispatch logic is:
    1. If a custom tokenizer is registered for the model type, use it
    2. Otherwise, fall back to NeMoAutoTokenizerWithBosEosEnforced

    Example:
        >>> # Will use MistralCommonBackend if available for Mistral models
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> # Force using HF AutoTokenizer with BOS/EOS enforcement
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("gpt2", force_default=True)
    """

    # Make registry accessible at class level
    _registry = None

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            *args,
            force_default: bool = False,
            force_hf: bool = True,
            trust_remote_code: bool = False,
            **kwargs,
    ) -> Any:
        """
        Load a tokenizer from a pretrained model.

        Args:
            pretrained_model_name_or_path: Model identifier or path
            force_default: If True, always use NeMoAutoTokenizerWithBosEosEnforced
            force_hf: If True, return the raw HF AutoTokenizer without any wrapping
            trust_remote_code: Whether to trust remote code when loading config
            **kwargs: Additional arguments passed to the tokenizer's from_pretrained

        Returns:
            A tokenizer instance appropriate for the model type
        """
        # If force_hf, just use the base HF AutoTokenizer
        if force_hf:
            # Transformers is an optional dependency for tokenizer construction.
            from transformers import AutoTokenizer as HFAutoTokenizer  # pylint: disable=C0415

            tokenizer = HFAutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
            )
            if not hasattr(tokenizer, "eod"):
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is None:
                    raise ValueError("The tokenizer must define either 'eod' or 'eos_token_id'")
                tokenizer.eod = eos_token_id
            return tokenizer

        raise ValueError("Only the Hugging Face tokenizer backend is currently supported")


def build_tokenizer(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Build the configured LLM tokenizer.

    Args:
        pretrained_model_name_or_path: Model identifier or local tokenizer path.
        **kwargs: Options forwarded to ``AutoTokenizer.from_pretrained``.

    Returns:
        The constructed tokenizer.
    """
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = [
    "AutoTokenizer",
    "build_tokenizer",
]


def __dir__():
    return sorted(__all__)
