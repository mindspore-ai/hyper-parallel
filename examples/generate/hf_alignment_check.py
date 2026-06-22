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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF model path or model id")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    adapter = HFGenerateAdapter(model)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
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

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
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
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if result["generated_new_tokens"] <= 0:
        raise AssertionError("model did not generate new tokens")
    if not result["hf_vs_hyper_cache_ids_match"]:
        raise AssertionError("hyper generate cache output differs from HuggingFace generate")
    if not result["hyper_cache_vs_no_cache_ids_match"]:
        raise AssertionError("hyper generate cache and no-cache outputs differ")


if __name__ == "__main__":
    main()
