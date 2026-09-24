# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Convert one completed HyperParallel DCP checkpoint to Hugging Face format."""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hyper_parallel.models._transformers import CheckpointManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed conversion arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert a completed multi-rank DCP checkpoint into a standard "
            "Transformers safetensors checkpoint in one CPU process."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Completed DCP step directory, for example global_step_100.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Original Hugging Face model directory containing config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty destination directory for the Transformers checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model-dir.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size accepted by save_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model and tokenizer code from the model directory.",
    )
    return parser.parse_args()


def _validate_paths(
    checkpoint_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Validate source and destination paths before allocating the model."""
    if not checkpoint_dir.is_dir():
        raise ValueError(f"DCP checkpoint directory does not exist: {checkpoint_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Hugging Face model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"config.json does not exist under model directory: {model_dir}")
    if checkpoint_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must differ from --checkpoint-dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")


def main() -> None:
    """Convert DCP model tensors and copy Transformers model assets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser()
    model_dir = args.model_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    tokenizer_dir = (args.tokenizer_dir or model_dir).expanduser()
    _validate_paths(checkpoint_dir, model_dir, output_dir)

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    CheckpointManager(model).save_pretrained(
        output_dir,
        dcp_checkpoint=checkpoint_dir,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)
    logger.info("Transformers checkpoint saved to %s", output_dir)


if __name__ == "__main__":
    main()
