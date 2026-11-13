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
"""Worker: run ``run_mode: eval`` end to end and compare with a manual reference.

The worker runs under ``torchrun`` + ``pytest`` (see ``test_eval_e2e.py``). It
writes its own tiny tokenizer and corpus into a temporary directory, so no
external asset is needed.

Environment (optional):
    HP_E2E_BACKEND: process-group backend; defaults to ``hccl`` on NPU hosts and
        ``gloo`` otherwise.
"""

import json
import math
import os
import random
import tempfile
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.runtime.evaluation import LossEvalStep
from hyper_parallel.trainer.runtime.run_mode import validate_run_config
from hyper_parallel.trainer.text_trainer import TextTrainer

_WORDS = "the quick brown fox jumps over lazy dog model train eval loss token data".split()
_SEQ = 16
_STATE = {"dir": "", "restore": None}


def _backend() -> str:
    """Pick the process-group backend for this host."""
    explicit = os.environ.get("HP_E2E_BACKEND")
    if explicit:
        return explicit
    return "hccl" if hasattr(torch, "npu") and torch.npu.is_available() else "gloo"


def _prepare_assets(root: str) -> None:
    """Write a deterministic corpus and a word-level tokenizer under ``root``."""
    from tokenizers import Tokenizer, models, pre_tokenizers  # pylint: disable=import-outside-toplevel
    from transformers import PreTrainedTokenizerFast  # pylint: disable=import-outside-toplevel

    rng = random.Random(0)
    with open(os.path.join(root, "data.jsonl"), "w", encoding="utf-8") as handle:
        for _ in range(64):
            handle.write(json.dumps({"text": " ".join(rng.choice(_WORDS) for _ in range(40))}) + "\n")
    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2}
    for word in _WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
    ).save_pretrained(os.path.join(root, "tok"))


def build_tiny_qwen3(distributed_setup: Any = None, **_: Any) -> torch.nn.Module:
    """Create a small randomly initialized dense Qwen3 on the Trainer's setup."""
    from transformers import Qwen3Config  # pylint: disable=import-outside-toplevel
    from hyper_parallel.models._transformers import HyperAutoModelForCausalLM  # pylint: disable=import-outside-toplevel

    # Hugging Face configs build their __init__ dynamically, so pylint cannot see the keywords.
    # pylint: disable-next=unexpected-keyword-arg
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        use_cache=False, tie_word_embeddings=False,
    )
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, torch_dtype="float32", attn_implementation="sdpa",
    )


def _raw_config(max_samples: int) -> dict:
    """Return the YAML root for a one-card (or DP) evaluation run."""
    return {
        "model": {"_target_": f"{__name__}.build_tiny_qwen3"},
        "run_mode": "eval",
        "training": {"backend": _backend(), "seed": 7, "init_device": "meta", "eval_iters": 0},
        "model_assets": {"tokenizer": {
            "_target_": "hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained",
            "pretrained_model_name_or_path": _STATE["dir"] + "/tok", "local_files_only": True}},
        "dataset": {
            "_target_": "hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset",
            "data_path": _STATE["dir"] + "/data.jsonl",
            "data_config": {"labels_are_shifted": True, "split": "50, 50, 0"},
            "data_transform": {"_target_": "hyper_parallel.data.text.text_transform.build_text_transform",
                               "data_type": "plaintext", "text_keys": "text", "max_seq_len": _SEQ},
        },
        "dataloader": {
            "_target_": "hyper_parallel.data.batching.TokenBatchLoader", "min_buffered_samples": 4,
            "collate_fn": {"_target_": "hyper_parallel.data.batching.build_online_text_collate_fn"},
            "get_batch": {"_target_": "hyper_parallel.data.batching.TextParallelBatch",
                          "source_type": "online", "attention_mode": "dense"},
            "sampler_type": "single", "drop_last": True, "num_workers": 0,
        },
        "checkpoint": {"save_ckpt": False, "restore_from": _STATE["restore"]},
        "evaluation": {
            "split": "valid", "batch_size": 2, "max_samples": max_samples, "max_batches": None,
            "metrics": [{"_target_": "hyper_parallel.evaluation.metrics.LossMetric"},
                        {"_target_": "hyper_parallel.evaluation.metrics.PerplexityMetric"}],
        },
    }


def run_eval(max_samples: int = 16) -> dict:
    """Run the full Trainer eval path and return its report as a dict."""
    config = resolve_config(_raw_config(max_samples))
    validate_run_config(config)
    trainer = TextTrainer(config)
    report = trainer.evaluate()
    return {"report": report.to_dict(), "model": trainer.base.model, "trainer": trainer}


def reference_loss(trainer: Any) -> tuple:
    """Recompute the mean loss over the same split with a plain, independent loop.

    The batches come from the same ``get_batch`` runtime the evaluator uses, so
    both sides consume identical data; only the loss arithmetic is independent
    (one cross-entropy pass in FP32, summed over valid tokens).
    """
    base = trainer.base
    total, count, samples = 0.0, 0, 0
    base.model.eval()
    # Rewind the shared cursor exactly as the evaluator does for its own pass.
    LossEvalStep(base, split_name="valid").reset()
    iterator = iter(base.valid_dataloader)
    with torch.no_grad():
        while True:
            try:
                model_inputs, loss_inputs = base.get_batch(iterator)
            except StopIteration:
                break
            # The Online transform pre-shifts targets (``labels_are_shifted``), so
            # ``shift_labels`` already aligns one target with every position and
            # the logits must not be shifted a second time.
            targets = loss_inputs["shift_labels"]
            logits = base.model(**model_inputs, use_cache=False).logits.float()
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100, reduction="none")
            valid = targets.reshape(-1) != -100
            total += float(per_token[valid].sum())
            count += int(valid.sum())
            samples += int((targets != -100).any(dim=-1).sum())
    return total / count, count, samples


def _run_and_check(restore: Any = None) -> dict:
    """Evaluate once through the real Trainer and verify it against the reference."""
    with tempfile.TemporaryDirectory() as root:
        # Rank 0 owns the asset directory; every rank reads the same files.
        shared = [root]
        if dist.is_initialized():
            dist.broadcast_object_list(shared, src=0)
        _STATE["dir"], _STATE["restore"] = shared[0], restore
        if not dist.is_initialized() or dist.get_rank() == 0:
            _prepare_assets(_STATE["dir"])
        if dist.is_initialized():
            dist.barrier()
        result = run_eval()
        report, trainer = result["report"], result["trainer"]
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world == 1:
            expected, tokens, samples = reference_loss(trainer)
            assert math.isclose(report["metrics"]["loss"], expected, rel_tol=1e-4), (report, expected)
            assert math.isclose(report["metrics"]["perplexity"], math.exp(expected), rel_tol=1e-3), report
            assert int(report["num_tokens"]) == tokens, (report["num_tokens"], tokens)
            assert report["num_samples"] == samples, (report["num_samples"], samples)
        assert report["status"] == "ok", report
        if dist.is_initialized():
            gathered = [None] * world
            dist.all_gather_object(gathered, report["metrics"])
            assert all(item == gathered[0] for item in gathered), gathered
        if dist.is_initialized():
            dist.barrier()
        return report


def test_eval_matches_reference_single_card() -> None:
    """Verify standalone loss/perplexity against an independent loop.

    Feature: run_mode eval, single card.
    Description: Run TextTrainer.evaluate on a tiny random Qwen3 and recompute the
        token-weighted loss over the same batches in a plain cross-entropy loop.
    Expectation: Loss, perplexity, token and sample counts agree.
    """
    _run_and_check()


def test_eval_dp_ranks_report_identical_global_metrics() -> None:
    """Verify every DP rank reports the same global metrics.

    Feature: run_mode eval, data parallel.
    Description: Run the evaluation on two or more ranks.
    Expectation: All ranks return identical metrics and counts.
    """
    _run_and_check()
