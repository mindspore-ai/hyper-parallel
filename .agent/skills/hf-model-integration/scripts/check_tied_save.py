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
"""Predict whether a model's checkpoint save will hit the shared-tensor wall.

The repository's sharded save writes through ``safetensors``, which refuses a
state dict whose entries share storage. Weight tying produces exactly that,
so a tied model fails to checkpoint while every other gate stays green — a
failure a long-run alone never reveals, because the long-run does not save.

This answers the question before any device is booked: it meta-builds the
architecture, finds parameters that alias one another, and reports them.
No weights are loaded and no accelerator is touched.

    python check_tied_save.py --model-dir /path/to/hf-model

Exit 0: no aliased parameters, the safetensors save path is unobstructed.
Exit 2: aliased parameters found; save will raise unless the writer untangles
        them. Name them in the run's S5 record (save/restore result, or an
        explicit out-of-scope statement).
Exit 1: usage or input problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def aliased_parameters(model) -> dict[int, list[str]]:
    """Group parameter names by the storage they share (groups of 2+ only)."""
    by_storage: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = parameter.untyped_storage().data_ptr() if parameter.is_meta is False \
            else id(parameter)
        by_storage.setdefault(key, []).append(name)
    return {key: names for key, names in by_storage.items() if len(names) > 1}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Report parameters that share storage (tied weights).")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        print(f"[tied-save] no config.json under {model_dir}", flush=True)
        return 1

    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoConfig  # pylint: disable=import-outside-toplevel

    # Reuse the layout pin's class resolution so composite families build the
    # class their checkpoint declares rather than a convenient subset.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_released_layout import _meta_build  # pylint: disable=import-outside-toplevel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True,
                                        trust_remote_code=False)
    config.use_cache = False
    model = _meta_build(config, torch)

    groups = aliased_parameters(model)
    tie_flag = bool(getattr(config, "tie_word_embeddings", False)) or bool(
        getattr(getattr(config, "text_config", None), "tie_word_embeddings", False))
    print(f"[tied-save] config tie_word_embeddings={tie_flag}", flush=True)
    if not groups:
        print("[tied-save] PASS: no aliased parameters; safetensors save is "
              "unobstructed", flush=True)
        return 0
    for names in list(groups.values())[:5]:
        print(f"[tied-save] aliased: {' == '.join(sorted(names))}", flush=True)
    print(f"[tied-save] FAIL: {len(groups)} aliased group(s); the sharded "
          "safetensors save will raise on shared storage. Record a "
          "save/restore result or an explicit out-of-scope statement in S5.",
          flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
