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
"""Generate micro-benchmark."""
import argparse
import importlib
import json
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_infer = importlib.import_module("hyper_parallel.infer")
GenerationConfig = _infer.GenerationConfig
build_causal_mask = _infer.build_causal_mask
build_position_ids = _infer.build_position_ids
generate = _infer.generate


class CacheLengthLM(nn.Module):
    """Small deterministic LM for measuring generate loop overhead."""

    def __init__(self, vocab_size: int = 32000, use_cache: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.calls = []

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        """Run deterministic forward with optional cache output."""
        del position_ids, attention_mask, kwargs
        batch_size, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[-2]
        total_len = past_len + seq_len
        token_id = total_len % self.vocab_size
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -1000.0,
            device=input_ids.device,
            dtype=torch.float32,
        )
        logits[:, -1, token_id] = 1000.0
        self.calls.append((seq_len, past_len))
        past = None
        if self.use_cache and use_cache:
            key = torch.zeros(batch_size, 1, total_len, 4, device=input_ids.device)
            value = torch.zeros(batch_size, 1, total_len, 4, device=input_ids.device)
            past = [(key, value)]
        return {"logits": logits, "past_key_values": past}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _run_case(args, device: torch.device, use_cache: bool) -> dict:
    """Measure one full generate case."""
    model = CacheLengthLM(vocab_size=args.vocab_size, use_cache=use_cache).to(device)
    input_ids = (
        torch.arange(args.prompt_len, device=device, dtype=torch.long)
        .view(1, -1)
        .expand(args.batch_size, -1)
        .contiguous()
        % args.vocab_size
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=None,
        use_cache=use_cache,
    )

    for _ in range(args.warmup):
        generate(model, input_ids, config)
    _synchronize(device)

    elapsed = []
    last_output = None
    for _ in range(args.repeat):
        model.calls.clear()
        start = time.perf_counter()
        last_output = generate(model, input_ids, config)
        _synchronize(device)
        elapsed.append(time.perf_counter() - start)

    total_generated = args.batch_size * args.max_new_tokens
    mean_seconds = mean(elapsed)
    return {
        "mean_seconds": mean_seconds,
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "tokens_per_second": total_generated / mean_seconds,
        "output_shape": list(last_output.shape),
        "calls": len(model.calls),
        "first_call": list(model.calls[0]),
        "last_call": list(model.calls[-1]),
    }


def _measure_prefill(args, device: torch.device) -> dict:
    """Measure the standalone prefill forward path."""
    model = CacheLengthLM(vocab_size=args.vocab_size, use_cache=True).to(device)
    input_ids = (
        torch.arange(args.prompt_len, device=device, dtype=torch.long)
        .view(1, -1)
        .expand(args.batch_size, -1)
        .contiguous()
        % args.vocab_size
    )
    position_ids = build_position_ids(input_ids)
    attention_mask = build_causal_mask(input_ids)

    for _ in range(args.warmup):
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
    _synchronize(device)

    elapsed = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        _synchronize(device)
        elapsed.append(time.perf_counter() - start)

    mean_seconds = mean(elapsed)
    return {
        "mean_seconds": mean_seconds,
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "tokens_per_second": (args.batch_size * args.prompt_len) / mean_seconds,
    }


def parse_args():
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _validate_args(args) -> None:
    """Validate benchmark arguments."""
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.prompt_len <= 0:
        raise ValueError("prompt-len must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.warmup == 0:
        warnings.warn(
            "warmup is 0; benchmark results may include first-run overhead",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")


def main():
    """Run the generate benchmark and optionally write JSON output."""
    args = parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "vocab_size": args.vocab_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "prefill": _measure_prefill(args, device),
        "with_cache": _run_case(args, device, use_cache=True),
        "no_cache": _run_case(args, device, use_cache=False),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
