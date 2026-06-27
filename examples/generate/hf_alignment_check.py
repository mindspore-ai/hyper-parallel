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
"""Compare hyper generate output with HuggingFace greedy generate."""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hyper_parallel.infer import GenerationConfig, generate


class HFGenerateAdapter(torch.nn.Module):
    """Adapter exposing HuggingFace causal LM output to hyper generate."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
    ):
        del attention_mask
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
        }


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compare_cache_no_cache_logits(model, input_ids, generated_ids, max_steps: int):
    steps = min(max_steps, generated_ids.size(1))
    if steps <= 0:
        return []

    values = []
    context_ids = input_ids
    cache_outputs = model(input_ids=context_ids, use_cache=True)
    past_key_values = cache_outputs.past_key_values
    cache_logits = cache_outputs.logits[:, -1, :]

    for step in range(steps):
        no_cache_outputs = model(input_ids=context_ids, use_cache=False)
        no_cache_logits = no_cache_outputs.logits[:, -1, :]
        similarity = torch.nn.functional.cosine_similarity(
            cache_logits.float(),
            no_cache_logits.float(),
            dim=-1,
        )
        values.extend(similarity.detach().cpu().tolist())

        next_token = generated_ids[:, step:step + 1]
        context_ids = torch.cat([context_ids, next_token], dim=-1)
        if step == steps - 1:
            break
        cache_outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = cache_outputs.past_key_values
        cache_logits = cache_outputs.logits[:, -1, :]
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF model path or model id")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--logits-compare-steps", type=int, default=16)
    parser.add_argument("--logits-cosine-threshold", type=float, default=0.999)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _validate_args(args) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.logits_compare_steps < 0:
        raise ValueError("logits-compare-steps must be >= 0")


def _load_model_and_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    return model, tokenizer, HFGenerateAdapter(model)


def _run_alignment(args, model, adapter, tokenizer, input_ids, pad_token_id, eos_token_id):
    hf_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    hyper_cache_output = generate(
        adapter,
        input_ids,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        ),
    )
    hyper_no_cache_output = generate(
        adapter,
        input_ids,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=False,
        ),
    )
    generated_ids = hyper_cache_output[:, input_ids.shape[1]:]
    logits_cosine = _compare_cache_no_cache_logits(
        model,
        input_ids,
        generated_ids,
        args.logits_compare_steps,
    )
    return hf_output, hyper_cache_output, hyper_no_cache_output, logits_cosine


def _build_result(
    args,
    tokenizer,
    input_ids,
    outputs,
    logits_cosine,
    device,
):
    hf_output, hyper_cache_output, hyper_no_cache_output = outputs
    generated_ids = hyper_cache_output[:, input_ids.shape[1]:]
    logits_cosine_min = min(logits_cosine) if logits_cosine else None
    logits_cosine_pass = (
        logits_cosine_min is None
        or logits_cosine_min >= args.logits_cosine_threshold
    )
    return {
        "model": args.model,
        "prompt": args.prompt,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "logits_compare_steps": min(args.logits_compare_steps, generated_ids.size(1)),
        "logits_cosine_similarity": logits_cosine,
        "logits_cosine_min": logits_cosine_min,
        "logits_cosine_threshold": args.logits_cosine_threshold,
        "logits_cosine_pass": logits_cosine_pass,
        "hf_text": tokenizer.decode(hf_output[0], skip_special_tokens=True),
        "hyper_cache_text": tokenizer.decode(hyper_cache_output[0], skip_special_tokens=True),
        "hyper_no_cache_text": tokenizer.decode(hyper_no_cache_output[0], skip_special_tokens=True),
        "hf_new_text": tokenizer.decode(
            hf_output[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        ),
        "generated_new_tokens": int(hf_output.shape[1] - input_ids.shape[1]),
        "hf_vs_hyper_cache_ids_match": torch.equal(hf_output.cpu(), hyper_cache_output.cpu()),
        "hyper_cache_vs_no_cache_ids_match": torch.equal(
            hyper_cache_output.cpu(),
            hyper_no_cache_output.cpu(),
        ),
        "hf_ids": hf_output[0].detach().cpu().tolist(),
        "hyper_cache_ids": hyper_cache_output[0].detach().cpu().tolist(),
        "hyper_no_cache_ids": hyper_no_cache_output[0].detach().cpu().tolist(),
    }


def _check_result(result) -> None:
    if result["generated_new_tokens"] <= 0:
        raise AssertionError("model did not generate new tokens")
    if not result["hf_vs_hyper_cache_ids_match"]:
        raise AssertionError("hyper generate cache output differs from HuggingFace generate")
    if not result["hyper_cache_vs_no_cache_ids_match"]:
        raise AssertionError("hyper generate cache and no-cache outputs differ")
    if not result["logits_cosine_pass"]:
        raise AssertionError("cache/no-cache logits cosine similarity is below threshold")


def main():
    args = parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    model, tokenizer, adapter = _load_model_and_tokenizer(args, device)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = _run_alignment(
            args, model, adapter, tokenizer, input_ids, pad_token_id, eos_token_id,
        )
    result = _build_result(args, tokenizer, input_ids, outputs[:3], outputs[3], device)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    _check_result(result)


if __name__ == "__main__":
    main()
