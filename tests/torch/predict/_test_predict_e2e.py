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
"""Real Qwen3/Qwen3-MoE generation and independent DP acceptance workers."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from hyper_parallel.components.checkpoint import build_checkpointer
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.infer import GenerationOptions, GenerationRuntime, generate
from hyper_parallel.predict import PredictionContext, Predictor


def _device():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(rank)
        return torch.device(f"npu:{rank}"), "hccl"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}"), "nccl"
    return torch.device("cpu"), "gloo"


def _model(device, moe=False):
    torch.manual_seed(7)
    kwargs = {"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
              "eos_token_id": None, "pad_token_id": 0, "tie_word_embeddings": False}
    if moe:
        config = Qwen3MoeConfig.from_dict({**kwargs, "num_experts": 4, "num_experts_per_tok": 2,
                                          "moe_intermediate_size": 16})
        model = Qwen3MoeForCausalLM(config)
    else:
        model = Qwen3ForCausalLM(Qwen3Config.from_dict(kwargs))
    model.config._attn_implementation = "eager"
    return model.to(device).eval()


def _tokenizer():
    vocabulary = {"<pad>": 0, "<unk>": 1, **{f"w{i}": i for i in range(2, 32)}}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")


def test_greedy_matches_huggingface() -> None:
    """Compare actual model tokens with HF for cache/no-cache and left padding."""
    device, _ = _device()
    torch.set_num_threads(1)
    ids = torch.tensor([[0, 0, 4, 5], [2, 3, 4, 7]], device=device)
    mask = ids.ne(0).long()
    for moe in (False, True):
        model = _model(device, moe)
        for use_cache in (False, True):
            expected = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5,
                                      do_sample=False, use_cache=use_cache, pad_token_id=0)
            actual = generate(model, ids, attention_mask=mask, return_dict=True,
                              options=GenerationOptions(max_new_tokens=5, use_cache=use_cache, pad_token_id=0))
            for row, prompt_length in enumerate(actual.prompt_lengths.tolist()):
                torch.testing.assert_close(actual.sequences[row, prompt_length:prompt_length + 5],
                                           expected[row, ids.size(1):])
            assert actual.generated_lengths.tolist() == [5, 5]
            assert actual.finish_reasons == ["length", "length"]
            assert all(parameter.grad is None for parameter in model.parameters())


def _assert_hf_completions(output: Path, prompts: list[str], model: Any, tokenizer: Any, device: torch.device) -> None:
    """Verify ordered records and every completion against native HF generation."""
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == [f"q-{i}" for i in range(len(prompts))]
    for record, prompt in zip(records, prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        expected = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=0)
        completion = expected[0, inputs.input_ids.size(1):]
        assert record["generated_text"] == tokenizer.decode(completion, skip_special_tokens=True)


def test_dp_prediction_is_complete_ordered_and_failure_safe() -> None:
    """Run non-divisible inputs, an empty lane, and a recoverable lane failure."""
    device, backend = _device()
    torch.set_num_threads(1)
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    context = PredictionContext(rank, world, dist.group.WORLD)
    root = [tempfile.mkdtemp(prefix="hp-predict-") if rank == 0 else None]
    dist.broadcast_object_list(root, src=0)
    root = Path(root[0])
    model, tokenizer = _model(device), _tokenizer()
    try:
        for count in (5, 1):
            source, output = root / f"input-{count}.jsonl", root / f"output-{count}.jsonl"
            prompts = [" ".join(f"w{index + 2}" for index in range(length)) for length in range(1, count + 1)]
            if rank == 0:
                source.write_text("".join(json.dumps({"id": f"q-{i}", "prompt": prompt}) + "\n"
                                          for i, prompt in enumerate(prompts)), encoding="utf-8")
            dist.barrier()
            config = SimpleNamespace(input=str(source), output_file=str(output), batch_size=2,
                                     max_batch_tokens=10, bucket_by_length=True, overwrite=False)
            predictor = Predictor(model, tokenizer, GenerationOptions(max_new_tokens=4, pad_token_id=0),
                                  GenerationRuntime(device=device), config, context=context)
            summary = predictor.run_file()
            assert summary.num_requests == summary.num_samples == count
            if rank == 0:
                _assert_hf_completions(output, prompts, model, tokenizer, device)
            dist.barrier()
        config.output_file = str(root / "failed.jsonl")
        config.input = str(root / "input-5.jsonl")
        saved_forward = model.forward
        if rank == 1:
            model.forward = _failed_forward
        try:
            Predictor(model, tokenizer, GenerationOptions(max_new_tokens=1, pad_token_id=0),
                      GenerationRuntime(device=device), config, context=context).run_file()
        except RuntimeError as error:
            assert "injected model failure" in str(error)
        else:
            raise AssertionError("every lane must observe the failure")
        finally:
            model.forward = saved_forward
        assert not Path(config.output_file).exists()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _failed_forward(**kwargs):
    del kwargs
    raise RuntimeError("injected model failure")


def build_host_model(distributed_setup: Any = None, **kwargs: Any) -> torch.nn.Module:
    """Exercise the production AutoModel builder with the predict assembly plan.

    Args:
        distributed_setup: Optional distributed setup forwarded to the production builder.
        **kwargs: Extra task-host arguments ignored by this fixed tiny-model fixture.

    Returns:
        Tiny Qwen3 causal language model using float32 parameters and eager attention.
    """
    del kwargs
    config = Qwen3Config.from_dict({"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                                   "num_hidden_layers": 2, "num_attention_heads": 4,
                                   "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
                                   "eos_token_id": None, "pad_token_id": 0, "tie_word_embeddings": False})
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, torch_dtype="float32", attn_implementation="eager",
    )


def test_text_trainer_predict_restores_model_only() -> None:
    """Run the real task host with a DCP checkpoint and no dataset/optimizer."""
    device, backend = _device()
    torch.set_num_threads(1)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    shared = [tempfile.mkdtemp(prefix="hp-predict-host-") if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0)
    root = Path(shared[0])
    tokenizer = _tokenizer()
    model = _model(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.5)
    if rank == 0:
        tokenizer.save_pretrained(root / "tokenizer")
        (root / "input.txt").write_text("w2 w3\nw4\nw5 w6 w7\n", encoding="utf-8")
    dist.barrier()
    checkpoint = root / "checkpoints" / "global_step_7"
    build_checkpointer().save(str(checkpoint), {"model": model.state_dict()}, global_step=7)
    raw = {
        "run_mode": "predict", "model": {"_target_": f"{__name__}.build_host_model"},
        "training": {"backend": backend, "seed": 9},
        "checkpoint": {"restore_from": "LATEST", "checkpoint_dir": str(root / "checkpoints"), "save_ckpt": False},
        "model_assets": {"tokenizer": {"_target_": "transformers.AutoTokenizer.from_pretrained",
                                       "pretrained_model_name_or_path": str(root / "tokenizer"),
                                       "local_files_only": True}},
        "prediction": {"input": str(root / "input.txt"), "output_file": str(root / "output.jsonl"),
                       "batch_size": 2, "generation": {"max_new_tokens": 3}},
    }
    try:
        trainer = TextTrainer(resolve_config(raw))
        assert trainer.base.optimizer is None
        assert trainer.base.lr_scheduler is None
        assert trainer.base.hsdp_model_parts == []
        for name, value in trainer.base.model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        summary = trainer.predict()
        assert summary.num_requests == 3
        if rank == 0:
            records = [json.loads(line) for line in (root / "output.jsonl").read_text(encoding="utf-8").splitlines()]
            for record, prompt in zip(records, ("w2 w3", "w4", "w5 w6 w7")):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                expected = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=0)
                assert record["generated_text"] == tokenizer.decode(expected[0, inputs.input_ids.size(1):],
                                                                     skip_special_tokens=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
