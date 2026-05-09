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

holds a ``BaseTrainer`` instance and calls
its ``_build_*`` methods selectively. Overrides ``_build_model_assets``,
``_build_data_transform``, ``_build_dataset``, and ``_build_collate_fn``
for complete real-data training pipeline.

"""
import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from hyper_parallel.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)

class LLMTrainer:
    """Trainer for LM pretraining and SFT.

    Composition pattern — calls BaseTrainer's _build_* methods in order,
    overriding data pipeline steps for real tokenized data.

    Supports:
    - ``data.type = "dummy"``: random tokens for quick FSDP validation
    - ``data.type = "hf_datasets"``: real HuggingFace datasets with tokenization

    Args:
        args: Training configuration parsed from YAML.
    """

    def __init__(self, args):
        self.base = BaseTrainer(args)

        # 13 steps — call base's methods, override where needed
        self.base._setup()
        self.base._build_model()
        self.base._freeze_model()
        self._build_model_assets()           # 覆盖: 加载 tokenizer
        self._build_data_transform()         # 覆盖: tokenize 函数
        self._build_dataset()                # 覆盖: 真实数据集加载 + tokenize
        self._build_collate_fn()             # 覆盖: 支持变长 padding
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

        For dummy data, tokenizer is not needed.
        For real data, loads HF AutoTokenizer from ``model.weights_path``
        or ``model.tokenizer_path``.
        """
        data_type = getattr(self.base.args.data, 'type', 'dummy')
        if data_type == 'dummy':
            self.base.tokenizer = None
            return

        # Try tokenizer_path first, fall back to weights_path
        model_cfg = self.base.args.model
        tokenizer_path = getattr(model_cfg, 'tokenizer_path', None)
        if not tokenizer_path:
            tokenizer_path = getattr(model_cfg, 'weights_path', None)

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

        max_seq_len = getattr(self.base.args.data, 'max_seq_len', 2048)
        tokenizer = self.base.tokenizer
        text_key = getattr(self.base.args.data, 'text_key', 'text')
        data_type = getattr(self.base.args.data, 'type', 'dummy')
        template = getattr(self.base.args.data, 'template', 'empty')

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

    def _build_dataset(self):
        """Build training dataset with full tokenization pipeline.

        For dummy data, delegates to BaseTrainer.
        For hf_datasets, loads + tokenizes + filters empty examples.
        """
        data_type = getattr(self.base.args.data, 'type', 'dummy')

        if data_type == 'dummy':
            self.base._build_dataset()  # pylint: disable=protected-access
            return

        if data_type == 'preset_pt':
            self._build_preset_pt_dataset()
            return

        if data_type not in ('hf_datasets', 'json_file'):
            raise ValueError(
                f"LLMTrainer supports data.type 'dummy', 'hf_datasets', 'json_file', or 'preset_pt', "
                f"got '{data_type}'"
            )

        # pylint: disable=C0415
        from datasets import load_dataset  # pylint: disable=C0415  # optional dep

        train_path = self.base.args.data.train_path
        data_subset = getattr(self.base.args.data, 'subset', None)

        logger.info("Loading dataset: type=%s, path=%s", data_type, train_path)

        if data_type == 'json_file':
            # Load local JSON file (alpaca format: instruction/input/output)
            ds = load_dataset("json", data_files=train_path, split="train")
        elif data_subset:
            ds = load_dataset(train_path, data_subset, split="train")
        else:
            ds = load_dataset(train_path, split="train")

        # Limit dataset size if specified
        train_size = getattr(self.base.args.data, 'train_size', None)
        if train_size and train_size < len(ds):
            ds = ds.select(range(train_size))
            logger.info("Dataset truncated to %d samples", train_size)

        # Tokenize
        if self.base.data_transform:
            ds = ds.map(
                self.base.data_transform,
                batched=True,
                remove_columns=ds.column_names,
                desc="Tokenizing",
            )

        # Filter empty sequences
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0)

        # Convert to torch tensors
        class TokenizedDataset(torch.utils.data.Dataset):
            """Wrap HF dataset for torch DataLoader."""
            def __init__(self, hf_ds):
                self.data = hf_ds
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                return {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "labels": torch.tensor(item["labels"], dtype=torch.long),
                }

        self.base.train_dataset = TokenizedDataset(ds)
        self.base.state.max_steps = min(
            self.base.args.train.max_steps,
            len(self.base.train_dataset) // max(
                self.base.args.train.global_batch_size, 1
            ),
        )
        logger.info("Dataset ready: %d samples, max_steps=%d",
                     len(self.base.train_dataset), self.base.state.max_steps)

    def _build_preset_pt_dataset(self):
        """Load pre-tokenized batches from a .pt file (List[Dict[str, Tensor]]).

        ``data.train_path`` is the .pt file. Each entry is a dict of tensors
        with shape ``(global_batch, seq_len)``. The dataset returns a flat
        sequence of per-sample dicts so the standard DataLoader can batch them.
        Use this when the dataset has already been tokenized offline and the
        token stream should be replayed deterministically.
        """
        # pylint: disable=C0415

        train_path = self.base.args.data.train_path
        if not train_path:
            raise ValueError("data.train_path is required when data.type='preset_pt'")
        batches = torch.load(train_path, map_location="cpu", weights_only=False)
        if not isinstance(batches, list) or not batches:
            raise ValueError(f"preset_pt expects List, got {type(batches)}")
        per_sample = []
        for b in batches:
            # Two formats: stacked dict ``{input_ids: (B,S), labels: (B,S)}``,
            # or list of per-rank dicts (preserves per-rank dynamic seq_len).
            if isinstance(b, list):
                for br in b:
                    ids = br["input_ids"]
                    labels = br["labels"]
                    attn = br.get("attention_mask")
                    for i in range(ids.shape[0]):
                        rec = {
                            "input_ids": ids[i].clone(),
                            "labels": labels[i].clone(),
                        }
                        if attn is not None and attn.dim() == 2:
                            rec["attention_mask"] = attn[i].clone()
                        per_sample.append(rec)
            else:
                ids = b["input_ids"]
                labels = b["labels"]
                attn = b.get("attention_mask")
                for i in range(ids.shape[0]):
                    rec = {
                        "input_ids": ids[i].clone(),
                        "labels": labels[i].clone(),
                    }
                    if attn is not None and attn.dim() == 2:
                        rec["attention_mask"] = attn[i].clone()
                    per_sample.append(rec)

        class PresetPtDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        self.base.train_dataset = PresetPtDataset(per_sample)
        max_steps = getattr(self.base.args.train, "max_steps", None)
        if max_steps:
            self.base.state.max_steps = int(max_steps)
        logger.info("preset_pt dataset: %d samples loaded from %s", len(per_sample), train_path)

    def _build_collate_fn(self):
        """Build collator with proper padding.

        Pads input_ids with pad_token_id (or 0) and labels with -100.
        """

        pad_id = 0
        if self.base.tokenizer and self.base.tokenizer.pad_token_id is not None:
            pad_id = self.base.tokenizer.pad_token_id

        def _lm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Pad sequences to max length in batch."""
            max_len = max(item["input_ids"].size(0) for item in batch)
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

            return {
                "input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list),
            }

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
