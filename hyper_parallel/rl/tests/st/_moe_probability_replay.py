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
"""Replay exact sampled token IDs through the distributed Actor without updating weights."""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from rl.trainer.build import build_trainer


def main() -> None:
    """Capture per-token probabilities and intermediate decoder states for one fixed trace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(args.config).read_text())
    trace = json.loads(Path(args.trace).read_text())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config)
    try:
        trainer.residency.move(offload=False)
        trainer.actor.eval()
        model = trainer.actor.actor_model
        attention = model.model.layers[0].self_attn
        def capture_position(_module, _args, kwargs):
            if dist.get_rank() == 0:
                values = kwargs.get("position_embeddings")
                if values is not None:
                    torch.save([getattr(value, "_local_tensor", value).detach().cpu() for value in values],
                               destination / "position_embeddings.pt")

        attention.register_forward_pre_hook(capture_position, with_kwargs=True)
        original_attention = attention.attention_interface

        def capture_attention(module, query, key, value, attention_mask, **kwargs):
            result = original_attention(module, query, key, value, attention_mask, **kwargs)
            if dist.get_rank() in (0, 1):
                captured = {}
                for name, tensor in (("query", query), ("key", key), ("value", value),
                                     ("mask", attention_mask), ("output", result[0])):
                    captured[name] = (None if tensor is None else
                                      getattr(tensor, "_local_tensor", tensor).detach().cpu())
                captured["scaling"] = kwargs.get("scaling")
                captured["rotary_interleaved"] = module.rotary_interleaved
                torch.save(captured, destination / f"attention_rank_{dist.get_rank()}.pt")
            return result

        attention.attention_interface = capture_attention
        prompt_length = len(trace["prompt_ids"])
        ids = trace["prompt_ids"] + trace["response_ids"]
        sequences = torch.tensor([ids], dtype=torch.long, device=trainer.device)
        mask = torch.ones_like(sequences)
        positions = [0, prompt_length - 1, prompt_length, len(ids) - 2]
        hidden = {}

        def capture(name):
            def hook(_module, _inputs, output):
                if dist.get_rank() not in (0, 1):
                    return
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor):
                    return
                value = getattr(value, "_local_tensor", value).detach()
                if name.startswith("model.layers.0."):
                    hidden[name] = value.float().cpu()
                elif value.ndim == 3 and value.shape[1] == len(ids):
                    hidden[name] = value[:, positions].float().cpu()
            return hook

        handles = []
        for name, module in model.named_modules():
            if name.startswith("model.layers.0.") or name in ("model.embed_tokens", "model.norm") or (
                    name.startswith("model.layers.") and name.count(".") == 2):
                handles.append(module.register_forward_hook(capture(name)))
        with torch.no_grad():
            log_probs = trainer.actor.sequence_log_probs(sequences, mask)
        for handle in handles:
            handle.remove()
        actual = log_probs[0, prompt_length - 1:len(ids) - 1].float().cpu()
        expected = torch.tensor(trace["log_probs"], dtype=torch.float32)
        payload = {"rank": dist.get_rank(), "prompt_length": prompt_length,
                   "response_length": len(trace["response_ids"]),
                   "mean_abs_difference": float((actual - expected).abs().mean()),
                   "max_abs_difference": float((actual - expected).abs().max()),
                   "actor_log_probs": actual.tolist(), "sampled_log_probs": expected.tolist(),
                   "model_type": type(model).__name__,
                   "attention_implementation": str(getattr(model.config, "_attn_implementation", None))}
        (destination / f"rank_{dist.get_rank()}.json").write_text(json.dumps(payload, indent=2))
        if dist.get_rank() in (0, 1):
            torch.save({"positions": positions, "hidden": hidden}, destination / f"hidden_rank_{dist.get_rank()}.pt")
            print(json.dumps({key: value for key, value in payload.items()
                              if key not in ("actor_log_probs", "sampled_log_probs")}), flush=True)
        dist.barrier()
        trainer.lifecycle.close()
    except BaseException as error:
        trainer.lifecycle.fail(error, "fixed-token-probability-replay")
        raise


if __name__ == "__main__":
    main()
