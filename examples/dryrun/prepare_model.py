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
"""Generate the config-only Llama models referenced by the dry-run examples."""

import argparse
from pathlib import Path

from transformers import LlamaConfig


def prepare_example_configs(output_dir: Path) -> None:
    """Write small model configurations without downloading weights.

    Args:
        output_dir: Parent directory for the smoke and pipeline models.
    """
    models = {
        "tiny_llama": dict(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        "pp_llama": dict(
            vocab_size=32768,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        ),
    }
    for name, shape in models.items():
        destination = output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        LlamaConfig(max_position_embeddings=8192, **shape).save_pretrained(destination)


def main() -> None:
    """Generate the example configurations at the requested location."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dryrun/models"))
    args = parser.parse_args()
    prepare_example_configs(args.output_dir.resolve())


if __name__ == "__main__":
    main()
