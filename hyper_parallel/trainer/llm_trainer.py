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
"""LLMTrainer — Language Model pretraining and SFT.

Holds a ``BaseTrainer`` instance and calls its ``_build_*`` methods
selectively. Overrides ``_build_model_assets``, ``_build_data_transform``
and ``_build_collate_fn``; dataset construction is delegated to the
shared :func:`hyper_parallel.data.build_dataset` registry.
"""
import logging
from typing import Any, Dict, List

import torch

from hyper_parallel.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class LLMTrainer:
    """Trainer for LM pretraining and SFT.

    Composition pattern — calls BaseTrainer's _build_* methods in order,
    overriding data pipeline steps for real tokenized data.

    Supports every ``data.type`` registered with
    :data:`hyper_parallel.data.DATASET_REGISTRY` — built-in formats are
    ``dummy`` (random tokens), ``hf_datasets`` / ``json_file`` (HF +
    Alpaca), ``preset_pt`` (replayed batches), and ``megatron`` (Megatron
    ``.bin``/``.idx``).

    Args:
        args: Training configuration parsed from YAML.
    """

    def __init__(self, args):
        self.base = BaseTrainer(args)

        # 13 steps — call base's methods, override where needed
        self.base._setup()
        self.base._build_model()
        self.base._freeze_model()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()
        # Fire one-shot ``on_init_end`` AFTER every ``_build_*`` — this is
        # the canonical "trainer is fully built" lifecycle hook.
        self.base.on_init_end()

    # ------------------------------------------------------------------
    # Overridden _build_* methods
    # ------------------------------------------------------------------

    def _build_model_assets(self):
        """Build tokenizer for data processing.

        Pre-tokenized formats (``dummy`` random tokens, ``megatron`` .bin/.idx,
        ``preset_pt`` replayed tensors) carry no raw text, so no tokenizer is
        built for them. Text formats (``hf_datasets`` / ``json_file``) load an
        HF AutoTokenizer from ``model.tokenizer_path`` or ``model.weights_path``.
        """
        data_type = self.base.args.data.type
        if data_type in ('dummy', 'megatron', 'preset_pt'):
            self.base.tokenizer = None
            return

        # Try tokenizer_path first, fall back to weights_path
        model_cfg = self.base.args.model
        tokenizer_path = model_cfg.tokenizer_path
        if not tokenizer_path:
            tokenizer_path = model_cfg.weights_path

        if not tokenizer_path:
            raise ValueError(
                "data.type='hf_datasets' requires model.tokenizer_path or "
                "model.weights_path to load tokenizer."
            )

        from transformers import AutoTokenizer  # pylint: disable=C0415  # optional dep
        self.base.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        # Ensure pad token exists
        if self.base.tokenizer.pad_token is None:
            self.base.tokenizer.pad_token = self.base.tokenizer.eos_token
        logger.info("Tokenizer loaded: %s (vocab=%d)",
                     tokenizer_path, len(self.base.tokenizer))

    def _build_data_transform(self):
        """Build tokenization transform.

        Creates a function that tokenizes raw text into input_ids + labels.
        Labels are a copy of input_ids (causal LM: predict next token).
        Prompt tokens can be masked with -100 for SFT.
        """
        if self.base.tokenizer is None:
            self.base.data_transform = None
            return

        max_seq_len = self.base.args.data.max_seq_len
        tokenizer = self.base.tokenizer
        text_key = self.base.args.data.text_key
        data_type = self.base.args.data.type
        template = self.base.args.data.template

        def _tokenize_fn(examples):
            """Tokenize text and create causal LM labels.

            Supports:
            - Plain text (text_key field)
            - Alpaca format (instruction/input/output)
            """
            # SFT label masking: prompt tokens → IGNORE_INDEX, response
            # tokens kept. Truncation prioritises the response side.
            ignore_index = -100

            def _infer_seqlen(s_len, t_len, cutoff):
                if t_len * 2 < cutoff:
                    max_t = cutoff
                elif s_len * 2 < cutoff:
                    max_t = cutoff - s_len
                else:
                    max_t = int(cutoff * (t_len / (s_len + t_len)))
                new_t = min(max_t, t_len)
                max_s = max(cutoff - new_t, 0)
                new_s = min(max_s, s_len)
                return new_s, new_t

            if "instruction" in examples and data_type == "json_file" and template == "empty":
                instructions = examples["instruction"]
                inputs = examples.get("input", [""] * len(instructions))
                outputs = examples["output"]
                result = {"input_ids": [], "labels": []}
                for inst, inp, out in zip(instructions, inputs, outputs):
                    prompt_text = inst + (("\n" + inp) if inp else "")
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    response_ids = tokenizer(out, add_special_tokens=False)["input_ids"]
                    s_len, t_len = _infer_seqlen(len(prompt_ids), len(response_ids), max_seq_len)
                    prompt_ids = prompt_ids[:s_len]
                    response_ids = response_ids[:t_len]
                    ids = prompt_ids + response_ids
                    labels = [ignore_index] * len(prompt_ids) + list(response_ids)
                    if len(ids) > 0:
                        result["input_ids"].append(ids)
                        result["labels"].append(labels)
                return result

            if "instruction" in examples and data_type == "json_file":
                # Alpaca format with chat-style template (legacy default)
                instructions = examples["instruction"]
                inputs = examples.get("input", [""] * len(instructions))
                outputs = examples["output"]
                texts = []
                for inst, inp, out in zip(instructions, inputs, outputs):
                    if inp:
                        texts.append(f"Human: {inst}\n{inp}\n\nAssistant: {out}")
                    else:
                        texts.append(f"Human: {inst}\n\nAssistant: {out}")
            else:
                # Plain text format
                texts = examples[text_key]
                if isinstance(texts, str):
                    texts = [texts]

            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_attention_mask=False,
            )

            result = {"input_ids": [], "labels": []}
            for ids in tokenized["input_ids"]:
                if len(ids) > 0:
                    result["input_ids"].append(ids)
                    result["labels"].append(ids.copy())

            return result

        self.base.data_transform = _tokenize_fn
        logger.info("Data transform: tokenize max_seq_len=%d, format=%s",
                     max_seq_len, "alpaca" if data_type == "json_file" else text_key)

    def _build_collate_fn(self):
        """Build collator with proper padding.

        Pads input_ids with pad_token_id (or 0) and labels with -100.
        SequenceParallel TP and context parallel slice the sequence dim, so
        variable-length batches additionally pad up to a multiple of
        ``cp * tp``; the pad rides label ``-100`` and is masked out of the CE.
        """

        pad_id = 0
        if self.base.tokenizer and self.base.tokenizer.pad_token_id is not None:
            pad_id = self.base.tokenizer.pad_token_id
        seq_divisor = self.base.parallel_dims.seq_divisor

        def _lm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Pad sequences to max length in batch."""
            max_len = max(item["input_ids"].size(0) for item in batch)
            if seq_divisor > 1 and max_len % seq_divisor:
                max_len += seq_divisor - max_len % seq_divisor
            input_ids_list = []
            labels_list = []

            for item in batch:
                seq_len = item["input_ids"].size(0)
                pad_len = max_len - seq_len

                if pad_len > 0:
                    input_ids_list.append(
                        torch.cat([item["input_ids"],
                                   torch.full((pad_len,), pad_id, dtype=torch.long)])
                    )
                    labels_list.append(
                        torch.cat([item["labels"],
                                   torch.full((pad_len,), -100, dtype=torch.long)])
                    )
                else:
                    input_ids_list.append(item["input_ids"])
                    labels_list.append(item["labels"])

            out = {
                "input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list),
            }
            if "num_items_in_batch" in batch[0]:
                out["num_items_in_batch"] = sum(
                    int(item["num_items_in_batch"]) for item in batch
                )
            if "attention_mask" in batch[0]:
                masks = []
                for item in batch:
                    pad_len = max_len - item["attention_mask"].size(0)
                    masks.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
                out["attention_mask"] = torch.stack(masks)
            if "position_ids" in batch[0]:
                positions = []
                for item in batch:
                    pos = item["position_ids"]
                    pad_len = max_len - pos.shape[-1]
                    positions.append(torch.nn.functional.pad(pos, (0, pad_len), value=0))
                if positions[0].dim() == 1:
                    out["position_ids"] = torch.stack(positions)
                else:
                    out["position_ids"] = torch.stack(positions).transpose(0, 1).contiguous()
            return out

        self.base.collate_fn = _lm_collate

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def train(self):
        """Delegate to BaseTrainer.train()."""
        self.base.train()

    def train_step(self, data_iterator):
        """Delegate to BaseTrainer.train_step()."""
        return self.base.train_step(data_iterator)
