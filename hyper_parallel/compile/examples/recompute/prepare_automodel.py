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
"""Prepare deterministic indexed tokens and AutoModel graph validation YAMLs.

This fixture uses random model initialization from the supplied model config;
it does not validate pretrained weight loading or language-model convergence.
"""

# pylint: disable=forbidden-backend-import
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from hyper_parallel.data.tools.io import IndexedDatasetBuilder


def main() -> None:
    """Derive small correctness runs from the upstream TP2 + FSDP2 example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assets = args.model_assets.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_config = json.loads((assets / "config.json").read_text(encoding="utf-8"))
    template = Path(__file__).resolve().parents[1] / "automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["model"]["pretrained_model_name_or_path"] = str(assets)
    config["model"]["load_base_model"] = False
    config["model_init_dtype"] = "float32"
    prefix = output / "tokens"
    builder = IndexedDatasetBuilder(str(prefix) + ".bin")
    generator = torch.Generator().manual_seed(20261012)
    for _ in range(128):
        builder.add_item(torch.randint(100, min(10000, model_config["vocab_size"]), (257,), generator=generator))
        builder.end_document()
    builder.finalize(str(prefix) + ".idx")
    config["dataset"]["data_path"] = str(prefix)
    tokenizer = config["dataset"]["model_assets"]["tokenizer"]
    tokenizer["pretrained_model_name_or_path"] = str(assets)
    tokenizer["vocab_size"] = model_config["vocab_size"]
    eos = model_config["eos_token_id"]
    tokenizer["eod_token_id"] = eos[0] if isinstance(eos, list) else eos
    for mode in ("none", "full", "selective", "save_compute"):
        variant = copy.deepcopy(config)
        variant["compile"]["recompute"] = {
            "mode": "selective" if mode == "save_compute" else mode,
            "module_patterns": ["model.layers.*"],
            "selective_policy": "save_compute" if mode == "save_compute" else "eager",
        }
        (output / f"{mode}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
    print(f"Prepared fixtures in {output}")


if __name__ == "__main__":
    main()
