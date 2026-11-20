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
"""Same-process S2 parity through the construction path training uses.

    python -m examples.training_demo.hf_text_bringup.reproduce.parity_check \
        --model-dir /path/to/hf-model [--num-hidden-layers N] \
        [--dtype bfloat16] [--image] [--output parity.json]

Builds the model twice in one process on one device: once with plain
Transformers, and once through the entry a training run uses -- the kit's
``build_model.py`` with ``load_pretrained`` for a causal LM, the VLM facade's
``from_pretrained`` for an image-text model, never a convenience flag that
hands loading back to Transformers. It then compares three things bitwise:
every parameter, every buffer, and the logits of one forward pass. Buffers are
compared on their own because non-persistent ones (RoPE ``inv_freq`` and the
like) are in no checkpoint and no ``state_dict()``: a parity that compares
weights only cannot see them. Exit 0 when all three match, 1 otherwise; the
JSON report names every mismatch.
"""
# A device-side check compares tensors directly.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from examples.training_demo.hf_text_bringup.build_model import HfCausalLmOptions, build_hf_causal_lm
from hyper_parallel.models._transformers import HyperAutoModelForImageTextToText

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# Multimodal placeholder ids: a text-only probe must stay below them, or the
# forward expects pixel inputs it was not given.
_MEDIA_TOKEN_KEYS = ("image_token_id", "video_token_id", "vision_start_token_id",
                     "vision_end_token_id", "vision_token_id")


def is_image_text(config: Any) -> bool:
    """Say whether the directory holds an image-text model."""
    return getattr(config, "vision_config", None) is not None


def build_reference(model_dir: str, dtype: str, image_text: bool,
                    num_hidden_layers: int | None) -> torch.nn.Module:
    """Plain Transformers build at the target dtype (never a post-hoc cast)."""
    model_class = AutoModelForImageTextToText if image_text else AutoModelForCausalLM
    extra = {} if num_hidden_layers is None else {"num_hidden_layers": num_hidden_layers}
    return model_class.from_pretrained(model_dir, dtype=_DTYPES[dtype], attn_implementation="eager",
                                       local_files_only=True, **extra).eval()


def build_repository(model_dir: str, dtype: str, image_text: bool,
                     num_hidden_layers: int | None) -> torch.nn.Module:
    """Build through the same entry and options a training run uses."""
    if image_text:
        if num_hidden_layers is not None:
            raise ValueError("--num-hidden-layers is supported for causal LMs only")
        return HyperAutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="eager", local_files_only=True).eval()
    return build_hf_causal_lm(model_dir, options=HfCausalLmOptions(
        load_pretrained=True, num_hidden_layers=num_hidden_layers, torch_dtype=dtype,
        attn_implementation="eager")).eval()


def text_inputs(config: Any, seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    """Seeded token ids that avoid multimodal placeholders."""
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    high = int(text_config.vocab_size)
    for key in _MEDIA_TOKEN_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, int) and 0 < value < high:
            high = value
    generator = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(0, high, (1, seq_len), generator=generator)}


def image_text_inputs(model_dir: str, seed: int) -> dict[str, torch.Tensor]:
    """One seeded synthetic image plus a prompt, through the model's processor."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel

    pixels = np.random.default_rng(seed).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "Describe the image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[Image.fromarray(pixels)], return_tensors="pt")
    return {name: value for name, value in batch.items() if isinstance(value, torch.Tensor)}


def compare_tensors(ours: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Name-by-name bitwise comparison, dtype and shape included."""
    shared = sorted(set(ours) & set(reference))
    different = []
    for name in shared:
        mine, theirs = ours[name], reference[name]
        if (mine.dtype != theirs.dtype or mine.shape != theirs.shape
                or not torch.equal(mine.detach().cpu(), theirs.detach().cpu())):
            different.append(name)
    return {
        "compared": len(shared),
        "only_reference": sorted(set(reference) - set(ours)),
        "only_repository": sorted(set(ours) - set(reference)),
        "different": different,
    }


def _clean(section: dict[str, Any]) -> bool:
    return not (section["only_reference"] or section["only_repository"] or section["different"])


@torch.no_grad()
def run_parity(model_dir: str, dtype: str = "bfloat16", num_hidden_layers: int | None = None,
               seq_len: int = 64, seed: int = 0, image: bool = False) -> dict[str, Any]:
    """Build both sides, compare parameters, buffers and logits; return the report."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    image_text = is_image_text(config)
    if image and not image_text:
        raise ValueError("--image needs an image-text model directory")
    repository = build_repository(model_dir, dtype, image_text, num_hidden_layers)
    device = next(repository.parameters()).device
    reference = build_reference(model_dir, dtype, image_text, num_hidden_layers).to(device)

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "construction": ("HyperAutoModelForImageTextToText.from_pretrained" if image_text
                         else "build_model.build_hf_causal_lm(load_pretrained=True)"),
        "dtype": dtype,
        "device": str(device),
        "num_hidden_layers": num_hidden_layers,
        "parameters": compare_tensors(dict(repository.named_parameters()),
                                      dict(reference.named_parameters())),
        "buffers": compare_tensors(dict(repository.named_buffers()), dict(reference.named_buffers())),
    }
    inputs = image_text_inputs(model_dir, seed) if image else text_inputs(config, seq_len, seed)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    ours = repository(**inputs, use_cache=False).logits
    theirs = reference(**inputs, use_cache=False).logits
    report["inputs"] = {name: list(value.shape) for name, value in inputs.items()}
    report["logits"] = {
        "equal": bool(torch.equal(ours, theirs)),
        "max_abs_diff": float((ours.float() - theirs.float()).abs().max().item()),
        "finite": bool(torch.isfinite(ours).all().item()),
    }
    passed = (_clean(report["parameters"]) and _clean(report["buffers"])
              and report["logits"]["equal"] and report["logits"]["finite"])
    report["status"] = "PASS" if passed else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    parser.add_argument("--num-hidden-layers", type=int, default=None,
                        help="crop both sides to the first N layers (single-die fit)")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", action="store_true",
                        help="drive an image-text model with one synthetic image")
    parser.add_argument("--output", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    report = run_parity(args.model_dir, dtype=args.dtype, num_hidden_layers=args.num_hidden_layers,
                        seq_len=args.seq_len, seed=args.seed, image=args.image)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for section in ("parameters", "buffers"):
        part = report[section]
        print(f"{section}: compared={part['compared']} different={len(part['different'])} "
              f"only_reference={len(part['only_reference'])} only_repository={len(part['only_repository'])}"
              + (f" first={part['different'][:3]}" if part["different"] else ""))
    print(f"logits: equal={report['logits']['equal']} max_abs_diff={report['logits']['max_abs_diff']:.3e} "
          f"finite={report['logits']['finite']}")
    print(f"PARITY {report['status']} ({report['construction']}, {report['dtype']}, {report['device']})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
