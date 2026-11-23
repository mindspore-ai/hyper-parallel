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
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import get_model_state_dict
from hyper_parallel.evaluation import EvaluationReport, Evaluator, LossMetric, PerplexityMetric
from hyper_parallel.evaluation.aggregation import accumulate_statistics, reduce_statistics
from hyper_parallel.evaluation.types import KIND_LM_NLL, EvalStepResult, LossStatistics, ReductionContext
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.runtime.eval_step import LossEvalStep
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
    rng = random.Random(0)
    with open(os.path.join(root, "data.jsonl"), "w", encoding="utf-8") as handle:
        for index in range(64):
            handle.write(json.dumps({"id": index, "text": " ".join(rng.choice(_WORDS) for _ in range(40))}) + "\n")
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
    config = Qwen3Config.from_dict({
        "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 64,
        "use_cache": False, "tie_word_embeddings": False,
    })
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, torch_dtype="float32", attn_implementation="sdpa",
    )


def _raw_config(max_samples: Optional[int]) -> dict:
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
                raw_batch = next(iterator)
                model_inputs, loss_inputs = base.get_batch(iter([raw_batch]))
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
            boundaries = raw_batch.get("cu_seq_lens")
            if boundaries is None:
                samples += int((targets != -100).any(dim=-1).sum())
            else:
                for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                    samples += int(targets.reshape(-1)[start:end].ne(-100).any())
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


def build_native_eval_qwen3(distributed_setup: Any = None, **kwargs: Any) -> torch.nn.Module:
    """Build a native model for dense-DP acceptance without accelerator-only FSDP."""
    del kwargs
    config = Qwen3Config.from_dict({"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                                   "num_hidden_layers": 2, "num_attention_heads": 4,
                                   "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
                                   "use_cache": False, "tie_word_embeddings": False})
    model = Qwen3ForCausalLM(config)
    device_type = distributed_setup.mesh_context.device_mesh.device_type
    return model.to(torch.device(device_type, int(os.environ.get("LOCAL_RANK", "0"))))


def _check_native_evaluation(trainer: TextTrainer, expected_count: int) -> None:
    """Check record uniqueness and global metrics independently of dataset sizing."""
    report = trainer.evaluate()
    repeated = trainer.evaluate()
    _assert_repeatable_reports(report, repeated)
    sampler = trainer.base.valid_batch_sampler
    lane_indices = [None] * dist.get_world_size()
    dist.all_gather_object(lane_indices, sampler.indices)
    selected = [index for indices in lane_indices for index in indices]
    assert sorted(selected) == list(range(expected_count))
    source = trainer.base.valid_dataset.source_dataset
    record_ids = [source[index]["id"] for index in selected]
    assert len(set(record_ids)) == expected_count
    if sampler.has_samples:
        mean, tokens, samples = reference_loss(trainer)
        local = torch.tensor([mean * tokens, tokens, samples], device=trainer.base.device, dtype=torch.float32)
    else:
        local = torch.zeros(3, device=trainer.base.device, dtype=torch.float32)
    dist.all_reduce(local)
    expected_loss = float(local[0] / local[1])
    for result in (report, repeated):
        assert math.isclose(result.metrics["loss"], expected_loss, rel_tol=1e-5)
        assert math.isclose(result.metrics["perplexity"], math.exp(expected_loss), rel_tol=1e-5)
        assert result.num_tokens == int(local[1])
        assert result.num_samples == int(local[2])


def test_native_eval_range_and_empty_lanes() -> None:
    """Verify finite ranges and empty lanes through the real task host and native model."""
    torch.set_num_threads(1)
    backend = _backend()
    if backend == "hccl":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend)
    rank = dist.get_rank()
    shared = [tempfile.mkdtemp(prefix="hp-eval-native-") if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0)
    _STATE["dir"], _STATE["restore"] = shared[0], None
    if rank == 0:
        _prepare_assets(shared[0])
        records = (Path(shared[0]) / "data.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        for name, rows in (("small", records[:2]), ("large", records[2:10])):
            (Path(shared[0]) / f"{name}.jsonl").write_text("".join(rows), encoding="utf-8")
    dist.barrier()
    try:
        for weighted, natural_size in ((False, 32), (True, 10)):
            for maximum in (5, 1, None, 100):
                raw = _raw_config(maximum)
                raw["model"] = {"_target_": f"{__name__}.build_native_eval_qwen3"}
                raw["checkpoint"]["checkpoint_dir"] = str(Path(shared[0]) / "checkpoint")
                if weighted:
                    raw["dataset"].pop("data_path")
                    raw["dataset"]["data_config"] = {"labels_are_shifted": True, "sources": [
                        {"data_path": {"valid": str(Path(shared[0]) / f"{name}.jsonl")}, "weight": weight}
                        for name, weight in (("small", 9), ("large", 1))
                    ]}
                trainer = TextTrainer(resolve_config(raw))
                expected_count = natural_size if maximum is None else min(maximum, natural_size)
                _check_native_evaluation(trainer, expected_count)
    finally:
        dist.destroy_process_group()


def _assert_repeatable_reports(report: EvaluationReport, repeated: EvaluationReport) -> None:
    """Keep coverage exact while allowing device-kernel rounding between forwards."""
    assert report.status == repeated.status == "ok"
    assert report.num_tokens == repeated.num_tokens
    assert report.num_samples == repeated.num_samples
    assert report.num_batches == repeated.num_batches
    assert report.metrics.keys() == repeated.metrics.keys()
    for name, value in report.metrics.items():
        assert math.isclose(value, repeated.metrics[name], rel_tol=1e-6), (name, value, repeated.metrics[name])


def _native_from_sharded_model(trainer: TextTrainer) -> torch.nn.Module:
    """Gather actual FSDP weights on every lane for a collective-free reference."""
    base = trainer.base
    assert base.mesh.dp_shard_size == 2
    sharded = [parameter for parameter in base.model.parameters()
               if isinstance(parameter, DTensor) and any(placement.is_shard() for placement in parameter.placements)]
    assert sharded, "The acceptance model must contain real sharded parameters"
    for parameter in sharded:
        assert parameter.to_local().numel() * 2 == parameter.numel(), (
            parameter.shape, parameter.to_local().shape, parameter.placements)
    full_state = get_model_state_dict(
        base.model, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
    config = Qwen3Config.from_dict(base.model.config.to_dict())
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).to(base.device).eval()
    reference.load_state_dict(full_state, strict=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, full_state[name], atol=0, rtol=0)
    assert not any(isinstance(parameter, DTensor) for parameter in reference.parameters())
    return reference


def _native_reference_totals(trainer: TextTrainer, model: torch.nn.Module) -> tuple[float, int, int]:
    """Compute local CE totals without any FSDP forward or statistics reduction."""
    base = trainer.base
    if not base.valid_batch_sampler.has_samples:
        return 0.0, 0, 0
    LossEvalStep(base, split_name="valid").reset()
    numerators, token_count, sample_count = [], 0, 0
    with torch.no_grad():
        for raw_batch in base.valid_dataloader:
            model_inputs, loss_inputs = base.get_batch(iter([raw_batch]))
            model_inputs = {key: value for key, value in model_inputs.items() if key not in ("labels", "shift_labels")}
            targets = loss_inputs["shift_labels"]
            logits = model(**model_inputs, use_cache=False).logits.float()
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="none")
            valid = targets.ne(-100)
            numerators.append(float(per_token[valid.reshape(-1)].cpu().double().sum()))
            token_count += int(valid.sum())
            boundaries = raw_batch.get("cu_seq_lens")
            if boundaries is None:
                sample_count += int(valid.any(dim=-1).sum())
            else:
                for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                    sample_count += int(valid.reshape(-1)[start:end].any())
    return math.fsum(numerators), token_count, sample_count


def _check_fsdp_evaluation(trainer: TextTrainer, expected_records: int) -> None:
    """Compare repeated sharded evaluation with an independently reduced native reference."""
    reference = _native_from_sharded_model(trainer)
    report = trainer.evaluate()
    repeated = trainer.evaluate()
    _assert_repeatable_reports(report, repeated)
    actual = (report.status, report.metrics, report.num_tokens, report.num_samples)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, actual)
    assert all(item == actual for item in gathered), gathered
    sampler = trainer.base.valid_batch_sampler
    dist.all_gather_object(gathered, list(sampler.indices))
    indices = [index for lane in gathered for index in lane]
    assert sorted(indices) == list(range(expected_records)), indices
    source = trainer.base.valid_dataset.source_dataset
    assert len({source[index]["id"] for index in indices}) == expected_records
    dist.all_gather_object(gathered, _native_reference_totals(trainer, reference))
    tokens = sum(item[1] for item in gathered)
    samples = sum(item[2] for item in gathered)
    expected_loss = math.fsum(item[0] for item in gathered) / tokens
    for result in (report, repeated):
        assert result.num_tokens == tokens == expected_records * 40
        assert result.num_samples == samples == expected_records * 3
        assert math.isclose(result.metrics["loss"], expected_loss, rel_tol=1e-4), (result, expected_loss)
        assert math.isclose(result.metrics["perplexity"], math.exp(expected_loss), rel_tol=1e-3), result


def test_fsdp_eval_range_and_empty_lanes() -> None:
    """Verify true two-way parameter sharding with uneven, empty and full data lanes.

    Feature: standalone evaluation with FSDP parameter all-gathers.
    Description: Evaluate five records, one record and the natural validation split twice.
    Expectation: All ranks match native-model CE, perplexity and exact token/sample totals.
    """
    torch.set_num_threads(1)
    backend = _backend()
    assert backend == "hccl", "This acceptance test requires the NPU/HCCL runtime"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend)
    try:
        assert dist.get_world_size() == 2
        with tempfile.TemporaryDirectory(prefix="hp-eval-fsdp-") as root:
            shared = [root]
            dist.broadcast_object_list(shared, src=0)
            _STATE["dir"], _STATE["restore"] = shared[0], None
            if dist.get_rank() == 0:
                _prepare_assets(shared[0])
            dist.barrier()
            for maximum in (5, 1, None):
                raw = _raw_config(maximum)
                raw["fsdp_config"] = {"dp_shard_size": 2}
                raw["checkpoint"]["checkpoint_dir"] = str(Path(shared[0]) / "checkpoint")
                trainer = TextTrainer(resolve_config(raw))
                _check_fsdp_evaluation(trainer, 32 if maximum is None else maximum)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_hccl_evaluation_preserves_small_increments() -> None:
    """Verify accumulation residuals and one shared metric reduction on real NPUs.

    Feature: high-precision evaluation statistics through HCCL.
    Description: Add unit increments after 2**24 and reduce a non-FP32-exact global total.
    Expectation: Numerator, denominator and samples are exact; loss/PPL share one all-reduce.
    """
    assert _backend() == "hccl", "This acceptance test requires the NPU/HCCL runtime"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        context = ReductionContext(dist.group.WORLD, dist.get_world_size())
        increments = 32 + dist.get_rank()

        def step(index: int) -> Optional[EvalStepResult]:
            """Produce FP32 device inputs whose accumulated total needs residual bits."""
            if index > increments:
                return None
            value = 2**24 if index == 0 else 1
            scalar = torch.tensor(value, device=device, dtype=torch.float32)
            assert scalar.device.type == "npu"
            return EvalStepResult(
                {"foundation": LossStatistics(scalar, scalar.clone(), KIND_LM_NLL)},
                sample_count=torch.tensor(value, device=device, dtype=torch.int64),
            )

        totals = {}
        sample_count = torch.zeros((), device=device, dtype=torch.int64)
        for index in range(increments + 1):
            contribution = step(index)
            accumulate_statistics(totals, contribution.statistics)
            sample_count += contribution.sample_count
        reduced = reduce_statistics([totals], context, device, sample_count=sample_count)
        expected = sum(2**24 + 32 + rank for rank in range(dist.get_world_size()))
        assert reduced.numerators["foundation"] == expected
        assert reduced.denominators["foundation"] == expected
        assert reduced.num_samples == expected
        with patch("torch.distributed.all_reduce", wraps=dist.all_reduce) as all_reduce:
            report = Evaluator(step, [LossMetric(), PerplexityMetric()], context, device).run()
        assert all_reduce.call_count == 1
        assert report.num_tokens == report.num_samples == expected
        assert report.metrics["loss"] == 1.0
        assert math.isclose(report.metrics["perplexity"], math.e, rel_tol=1e-15)
    finally:
        dist.destroy_process_group()
