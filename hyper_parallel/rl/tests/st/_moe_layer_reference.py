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
"""Capture native first-layer intermediates for an exact-token MoE comparison."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


class CaptureComplete(Exception):
    """Signal that the requested diagnostic layer has completed."""


def main() -> None:
    """Record only the native first layer without evaluating the remaining decoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    trace = json.loads(Path(args.trace).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.weights, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    hidden = {}

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                hidden[name] = value.detach().float().cpu()
            if name == "model.layers.0":
                raise CaptureComplete()
        return hook

    for name, module in model.named_modules():
        if name == "model.embed_tokens" or name == "model.layers.0" or name.startswith("model.layers.0."):
            module.register_forward_hook(capture(name))
    sequences = torch.tensor([trace["prompt_ids"] + trace["response_ids"]], dtype=torch.long)
    try:
        with torch.no_grad():
            model(input_ids=sequences, attention_mask=torch.ones_like(sequences), use_cache=False)
    except CaptureComplete:
        torch.save(hidden, args.output)
        print(json.dumps({name: list(value.shape) for name, value in hidden.items()}), flush=True)
    else:
        raise RuntimeError("Requested first-layer hook was not invoked")


if __name__ == "__main__":
    main()
