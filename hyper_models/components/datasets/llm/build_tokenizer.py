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

import json
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class _DatasetTokenizer:
    """Provide the stable tokenizer identity required by Dataset caches."""

    def __init__(
            self,
            *tokenizer_paths: str,
            vocab_size: int | None = None,
            eod_token_id: int | None = None,
            **tokenizer_options: Any,
    ) -> None:
        if (vocab_size is None) != (eod_token_id is None):
            raise ValueError("vocab_size and eod_token_id must be provided together")
        if vocab_size is not None and eod_token_id is not None:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be positive")
            if not 0 <= eod_token_id < vocab_size:
                raise ValueError("eod_token_id must be within the tokenizer vocabulary")
            tokenizer_options["vocab_size"] = vocab_size
            tokenizer_options["eod_token_id"] = eod_token_id

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
        for option, value in tokenizer_options.items():
            self.unique_identifiers[option] = str(value)
        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        if vocab_size is not None and eod_token_id is not None:
            self.vocab_size = vocab_size
            self.eod = eod_token_id
            self.eos_token_id = eod_token_id
            self.pad_token_id = eod_token_id

    def __len__(self) -> int:
        """Return the configured tokenizer vocabulary size."""
        return self.vocab_size


class _HFAutoTokenizer(_DatasetTokenizer):
    """Wrap a Hugging Face tokenizer with the Dataset tokenizer contract."""

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            *args: Any,
            trust_remote_code: bool,
            **kwargs: Any,
    ) -> None:
        tokenizer_options = dict(kwargs)
        tokenizer_options["trust_remote_code"] = trust_remote_code
        if args:
            tokenizer_options["inputs"] = args
        super().__init__(pretrained_model_name_or_path, **tokenizer_options)

        # Transformers is optional and only required when constructing an HF tokenizer.
        from transformers import AutoTokenizer as HFAutoTokenizer  # pylint: disable=C0415

        self.tokenizer = HFAutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
        )
        if not hasattr(self.tokenizer, "eod"):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("The tokenizer must define either 'eod' or 'eos_token_id'")
            self.tokenizer.eod = eos_token_id

    def __getattr__(self, name: str) -> Any:
        """Delegate tokenizer attributes to the wrapped Hugging Face tokenizer."""
        return getattr(self.tokenizer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate runtime tokenizer options after the wrapped tokenizer exists."""
        tokenizer = self.__dict__.get("tokenizer")
        if tokenizer is None or name in {"tokenizer", "unique_identifiers", "unique_description"}:
            object.__setattr__(self, name, value)
            return
        setattr(tokenizer, name, value)

    def __len__(self) -> int:
        """Return the wrapped tokenizer vocabulary size."""
        return len(self.tokenizer)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward tokenization calls to the wrapped tokenizer."""
        return self.tokenizer(*args, **kwargs)


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
            pretrained_model_name_or_path: str | None = None,
            *args,
            force_default: bool = False,
            force_hf: bool = True,
            trust_remote_code: bool = False,
            tokenizer_type: str | None = None,
            vocab_size: int | None = None,
            eod_token_id: int | None = None,
            **kwargs,
    ) -> Any:
        """
        Load a tokenizer from a pretrained model.

        Args:
            pretrained_model_name_or_path: Tokenizer path used for loading or stable Dataset cache identity.
            force_default: If True, always use NeMoAutoTokenizerWithBosEosEnforced
            force_hf: If True, build a wrapped Hugging Face tokenizer.
            trust_remote_code: Whether to trust remote code when loading config
            tokenizer_type: Use ``pretokenized`` for an already-tokenized corpus.
            vocab_size: Pretokenized-corpus vocabulary size.
            eod_token_id: Pretokenized-corpus end-of-document token ID.
            **kwargs: Additional arguments passed to the tokenizer's from_pretrained

        Returns:
            A tokenizer instance appropriate for the model type
        """
        if tokenizer_type == "pretokenized":
            if pretrained_model_name_or_path is None:
                raise ValueError("pretrained_model_name_or_path is required for a pretokenized Dataset")
            return _DatasetTokenizer(
                pretrained_model_name_or_path,
                vocab_size=vocab_size,
                eod_token_id=eod_token_id,
            )
        if tokenizer_type is not None:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type!r}")

        if vocab_size is not None or eod_token_id is not None:
            raise ValueError("vocab_size and eod_token_id are only valid for tokenizer_type='pretokenized'")

        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path is required for a Hugging Face tokenizer")

        # If force_hf, use the Hugging Face tokenizer through the Dataset identity adapter.
        if force_hf:
            tokenizer = _HFAutoTokenizer(
                pretrained_model_name_or_path,
                *args,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
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
