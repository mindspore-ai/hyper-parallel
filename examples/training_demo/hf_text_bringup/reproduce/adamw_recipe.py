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
"""Write the kit recipe with AdamW alone in place of Muon+AdamW.

    python -m examples.training_demo.hf_text_bringup.reproduce.adamw_recipe \
        --output /path/to/train_adamw.yaml

The S3 attribution cell compares the same cross-layout pair under two
optimizers. The trainer refuses to change an optimizer ``_target_`` through a
CLI override, so the variant is a recipe of its own: identical to
``train_hf_text.yaml`` except for the ``optimizer`` block, which carries the
kit's own AdamW settings. Pass the result to the launchers as ``RECIPE``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

KIT_RECIPE = Path(__file__).resolve().parents[1] / "train_hf_text.yaml"
ADAMW_OPTIMIZER = {
    "_target_": "hyper_parallel.components.optim.AdamW",
    "fp32_main_params": True,
    "no_decay_params": ["bias", "norm", "ln_"],
    "adamw_config": {
        "adamw_lr": 1.0e-5,
        "adamw_weight_decay": 0.01,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1.0e-8,
    },
}


def adamw_recipe(source: Path = KIT_RECIPE) -> dict:
    """Return the source recipe with only its optimizer block replaced."""
    recipe = yaml.safe_load(source.read_text(encoding="utf-8"))
    if "optimizer" not in recipe:
        raise ValueError(f"{source} has no optimizer block to replace")
    recipe["optimizer"] = ADAMW_OPTIMIZER
    return recipe


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(KIT_RECIPE))
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(adamw_recipe(Path(args.source)), sort_keys=False),
                      encoding="utf-8")
    print(f"[recipe] AdamW-only variant written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
