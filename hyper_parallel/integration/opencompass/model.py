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
"""OpenCompass model adapter for HyperParallel causal language models."""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from opencompass.models.base import BaseModel
    from opencompass.registry import MODELS
except ImportError as exc:
    raise ModuleNotFoundError(
        "HyperOpenCompassModel requires OpenCompass 0.5.4; install the "
        "'opencompass' HyperParallel extra"
    ) from exc

from hyper_parallel.data.text import AutoTokenizer
from hyper_parallel.integration.opencompass.scoring import causal_lm_ppl_scores
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM


logger = logging.getLogger(__name__)


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits from mapping, model-output, or tuple return values."""
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (list, tuple)) and outputs:
        logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model forward output must expose Tensor logits")
    return logits


def _model_device(model: Any) -> torch.device:
    """Resolve the device that receives tokenized inputs."""
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("cannot resolve a materialized model device")


@MODELS.register_module()
class HyperOpenCompassModel(BaseModel):
    """OpenCompass PPL adapter backed by ``HyperAutoModelForCausalLM``.

    The first supported contract is single-process MMLU PPL evaluation with
    OpenCompass 0.5.4. Distributed result ownership and vocabulary-parallel
    scoring are deliberately rejected until their parity matrix is complete.

    Args:
        path: HyperParallel/HuggingFace checkpoint identifier or local path.
        tokenizer_path: Tokenizer identifier or path. Defaults to ``path``.
        max_seq_len: Maximum tokenized input length.
        tokenizer_only: Load only the tokenizer for length estimation.
        tokenizer_kwargs: Keyword arguments forwarded to the tokenizer loader.
        model_kwargs: Keyword arguments forwarded to HyperParallel model loading.
        hyper_kwargs: Additional HyperParallel model-construction arguments.
        batch_padding: Score multi-item calls as one padded batch.
        pad_token_id: Explicit padding token ID when the tokenizer lacks one.
        meta_template: Optional OpenCompass prompt meta-template.

    Raises:
        NotImplementedError: If a multi-process evaluation is requested.
        ValueError: If batch scoring has no usable padding token.
    """

    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 2048,
        tokenizer_only: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
        batch_padding: bool = True,
        pad_token_id: Optional[int] = None,
        meta_template: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                raise NotImplementedError(
                    "HyperOpenCompassModel currently supports single-process MMLU PPL only; "
                    f"got world_size={world_size}"
                )
        self.batch_padding = batch_padding
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or path,
            **dict(tokenizer_kwargs or {}),
        )
        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        if self.tokenizer.pad_token_id is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token
        if self.batch_padding and self.tokenizer.pad_token_id is None:
            raise ValueError("batch_padding requires tokenizer.pad_token_id or an explicit pad_token_id")

        self.model = None
        if not tokenizer_only:
            load_kwargs = dict(model_kwargs or {})
            duplicate = set(load_kwargs) & set(hyper_kwargs or {})
            if duplicate:
                raise ValueError(f"model_kwargs and hyper_kwargs contain duplicate keys: {sorted(duplicate)}")
            load_kwargs.update(hyper_kwargs or {})
            self.model = HyperAutoModelForCausalLM.from_pretrained(path, **load_kwargs)
            self.model.eval()
            logger.info("Loaded HyperParallel OpenCompass model from %s", path)

    def _tokenize(self, inputs: Sequence[str], padding: bool) -> Dict[str, torch.Tensor]:
        """Tokenize one score batch and place it on the model device."""
        if self.model is None:
            raise RuntimeError("tokenizer_only=True does not support PPL scoring")
        encoded = self.tokenizer(
            list(inputs),
            padding=padding,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        device = _model_device(self.model)
        return {
            name: value.to(device)
            for name, value in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }

    def _score_batch(
        self,
        inputs: Sequence[str],
        mask_length: Optional[Sequence[int]],
        padding: bool,
    ) -> np.ndarray:
        """Score one tokenized batch and return CPU float values."""
        tokens = self._tokenize(inputs, padding=padding)
        with torch.no_grad():
            outputs = self.model(**tokens)
            scores = causal_lm_ppl_scores(
                _extract_logits(outputs),
                tokens["input_ids"],
                self.tokenizer.pad_token_id,
                mask_length=mask_length,
            )
        return scores.detach().cpu().numpy()

    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Return MMLU candidate NLL scores in input order.

        Args:
            inputs: Complete OpenCompass candidate prompt strings.
            mask_length: Optional leading-token exclusion count per input.

        Returns:
            NumPy array of normalized NLL scores; lower is better.
        """
        if not inputs:
            return np.array([], dtype=np.float32)
        if mask_length is not None and len(mask_length) != len(inputs):
            raise ValueError("mask_length must contain one value per input")
        if self.batch_padding and len(inputs) > 1:
            return self._score_batch(inputs, mask_length, padding=True)

        scores = []
        for index, text in enumerate(inputs):
            item_mask = None if mask_length is None else [mask_length[index]]
            scores.append(self._score_batch([text], item_mask, padding=False))
        return np.concatenate(scores)

    def get_token_len(self, prompt: str) -> int:
        """Return the untruncated tokenizer length used by OpenCompass."""
        return len(self.tokenizer.encode(prompt))

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode one prompt for optional OpenCompass tooling."""
        return torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs for optional OpenCompass tooling."""
        return self.tokenizer.decode(tokens.tolist())

    def generate(self, inputs: List[str], max_out_len: int, **kwargs: Any) -> List[str]:
        """Reject generation until a separate generation parity gate exists."""
        del inputs, max_out_len, kwargs
        raise NotImplementedError("HyperOpenCompassModel currently supports PPL evaluation only")

    def get_ppl_tokenwise(self, *args: Any, **kwargs: Any) -> List[float]:
        """Reject tokenwise PPL until its OpenCompass contract is validated."""
        del args, kwargs
        raise NotImplementedError("tokenwise PPL is not implemented")
