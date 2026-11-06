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
"""Compare distributed Actor replay with a native CPU Transformers reference."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    """Replay exact token IDs and compare decoder-layer outputs without distributed rewrites."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    trace = json.loads(Path(args.trace).read_text())
    actor = json.loads((Path(args.actor) / "rank_0.json").read_text())
    snapshots = torch.load(Path(args.actor) / "hidden.pt", map_location="cpu", weights_only=True)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.weights, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    print("Native CPU model loaded", flush=True)
    sequences = torch.tensor([trace["prompt_ids"] + trace["response_ids"]], dtype=torch.long)
    hidden = {}
    comparisons = []

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            value = value.detach()[:, snapshots["positions"]].float()
            hidden[name] = value
            if name in snapshots["hidden"]:
                reference = snapshots["hidden"][name]
                delta = (value - reference).abs()
                record = {"module": name, "shape": list(value.shape),
                          "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                          "native_rms": float(value.square().mean().sqrt()),
                          "actor_rms": float(reference.square().mean().sqrt())}
                comparisons.append(record)
                print(json.dumps(record), flush=True)
        return hook

    for name, module in model.named_modules():
        if name in snapshots["hidden"]:
            module.register_forward_hook(capture(name))
    with torch.no_grad():
        output = model(input_ids=sequences, attention_mask=torch.ones_like(sequences),
                       use_cache=False, logits_to_keep=len(trace["response_ids"]) + 1)
        logits = output.logits[:, :-1].float()
        targets = sequences[:, -len(trace["response_ids"]):]
        actual = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze().cpu()
    sampled = torch.tensor(trace["log_probs"])
    actor_probs = torch.tensor(actor["actor_log_probs"])
    result = {"native_vs_sampled_mean_abs": float((actual - sampled).abs().mean()),
              "native_vs_actor_mean_abs": float((actual - actor_probs).abs().mean()),
              "native_log_probs": actual.tolist(), "layer_comparisons": comparisons}
    (destination / "result.json").write_text(json.dumps(result, indent=2))
    torch.save(hidden, destination / "hidden.pt")
    print(json.dumps({key: value for key, value in result.items() if key != "layer_comparisons"}), flush=True)


if __name__ == "__main__":
    main()
