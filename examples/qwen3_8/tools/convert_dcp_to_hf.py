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
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hyper_parallel.core.distributed_checkpoint import (
    FileSystemReader,
    TensorStorageMetadata,
    load as load_dcp_state,
)
from hyper_parallel.core.distributed_checkpoint.utils import str_to_dtype
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager

logger = logging.getLogger(__name__)


class DCPCheckpointManager(CheckpointManager):
    """Read a completed DCP snapshot without gathering the live model weights."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Bind an unreplaced HF model defining the expected export layout."""
        super().__init__(model)
        self._dcp_checkpoint: str | Path | None = None
        self._dcp_expert_layout: str | None = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        dcp_checkpoint: str | Path,
        dcp_expert_layout: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Export persisted model tensors through the shared HF writer.

        Args:
            save_directory: Destination HF checkpoint directory.
            dcp_checkpoint: Completed DCP directory to read on the writing rank.
            dcp_expert_layout: Optional auto, hf or grouped_gemm expert layout.
            **kwargs: Shared HF save options, including shard size and format.

        Returns:
            Whether this process wrote the checkpoint.
        """
        self._dcp_checkpoint = dcp_checkpoint
        self._dcp_expert_layout = dcp_expert_layout
        return super().save_pretrained(save_directory, **kwargs)

    def _gather_full_state_dict(self, *, keep_state_dict: bool) -> dict[str, Any]:
        """Replace live-weight gathering with offline DCP reconstruction."""
        if not keep_state_dict:
            return {}
        if self._dcp_checkpoint is None:
            raise ValueError("A completed DCP checkpoint is required for offline export")
        state_dict = self._load_model_state_dict_from_dcp(self._dcp_checkpoint)
        if self._dcp_expert_layout is not None:
            state_dict = self._normalize_dcp_export_layout(state_dict, self._dcp_expert_layout)
        return state_dict

    def _normalize_dcp_export_layout(
        self,
        state_dict: dict[str, torch.Tensor],
        layout: str,
    ) -> dict[str, torch.Tensor]:
        """Restore batched expert layout against an unreplaced HF model."""
        if layout not in ("auto", "hf", "grouped_gemm"):
            raise ValueError(f"Unsupported DCP expert layout: {layout}")
        if getattr(self.model, "_hp_used_replacement_weight_conversions", None):
            raise ValueError("DCP layout normalization requires an unreplaced Hugging Face model")
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state_dict))
        if missing:
            raise ValueError(f"DCP checkpoint is missing HF model parameters or buffers: {missing}")
        result = dict(state_dict)
        expert_groups: dict[str, list[str]] = defaultdict(list)
        for name in state_dict:
            if re.search(r"(?:^|\.)experts\.(?:gate_up_proj|up_proj|down_proj)$", name):
                expert_groups[name.rsplit(".", 1)[0]].append(name)
        for prefix, names in expert_groups.items():
            if any(name not in expected or state_dict[name].ndim != 3 for name in names):
                raise ValueError(f"Unsupported DCP expert parameters under {prefix}; expected batched HF weights")
            self._restore_expert_group(state_dict, expected, result, prefix, names, layout)
        for name, tensor in result.items():
            target = expected.get(name)
            if target is None:
                raise ValueError(f"DCP parameter {name!r} is absent from the HF model; check --model-dir")
            if tensor.shape != target.shape:
                raise ValueError(
                    f"DCP parameter {name!r} has shape {tuple(tensor.shape)}, "
                    f"but HF expects {tuple(target.shape)}; check model config and expert layout"
                )
        return result

    @staticmethod
    def _restore_expert_group(state_dict, expected, result, prefix, names, layout):
        """Restore one expert group while rejecting ambiguous or incompatible layouts."""
        hf_matches = all(state_dict[name].shape == expected[name].shape for name in names)
        transposed_matches = all(
            state_dict[name].transpose(-2, -1).shape == expected[name].shape for name in names
        )
        if layout == "auto" and hf_matches and transposed_matches:
            raise ValueError(
                f"Ambiguous expert layout under {prefix}; explicitly select hf or grouped_gemm"
            )
        transpose = layout == "grouped_gemm" or (layout == "auto" and not hf_matches)
        if transpose and not transposed_matches:
            raise ValueError(
                f"DCP expert shapes under {prefix} do not match the requested HF layout: "
                + ", ".join(
                    f"{name}: DCP {tuple(state_dict[name].shape)}, HF {tuple(expected[name].shape)}"
                    for name in names
                )
            )
        if transpose:
            for name in names:
                result[name] = state_dict[name].transpose(-2, -1).contiguous()
            logger.info("Restored HF expert layout for %s", prefix)

    @staticmethod
    def _model_key_from_dcp(
        fqn: str,
        planner_data: dict[str, tuple[Any, ...]] | None,
    ) -> str | None:
        """Return the model-local key for one flattened DCP entry."""
        path = planner_data.get(fqn) if planner_data is not None else None
        if path is not None:
            if len(path) < 2 or path[0] != "model":
                return None
            return ".".join(str(part) for part in path[1:])
        if fqn.startswith("model."):
            return fqn[len("model."):]
        return None

    def _load_model_state_dict_from_dcp(
        self,
        checkpoint_id: str | Path,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct only model tensors from a completed DCP on rank 0."""
        reader = FileSystemReader(checkpoint_id)
        metadata = reader.load_metadata()
        planner_data = metadata.planner_data
        if planner_data is not None and not isinstance(planner_data, dict):
            raise TypeError("DCP planner_data must be a mapping for Hugging Face export")

        dcp_state: dict[str, torch.Tensor] = {}
        model_keys: dict[str, str] = {}
        seen_model_keys: set[str] = set()
        for fqn, storage_metadata in metadata.state_dict_metadata.items():
            model_key = self._model_key_from_dcp(fqn, planner_data)
            if model_key is None:
                continue
            if not isinstance(storage_metadata, TensorStorageMetadata):
                raise TypeError(
                    f"DCP model entry {fqn!r} is not a tensor and cannot be exported"
                )
            if model_key in seen_model_keys:
                raise ValueError(f"Duplicate model key {model_key!r} in DCP checkpoint")
            dcp_state[fqn] = torch.empty(
                storage_metadata.size,
                dtype=str_to_dtype(storage_metadata.properties.dtype),
                device="cpu",
            )
            model_keys[fqn] = model_key
            seen_model_keys.add(model_key)

        if not dcp_state:
            raise ValueError(f"No model tensors found in DCP checkpoint {checkpoint_id}")

        load_dcp_state(
            dcp_state,
            checkpoint_id=checkpoint_id,
            no_dist=True,
            broadcast_replicated_tensors=False,
        )
        return {
            model_keys[fqn]: tensor.detach()
            for fqn, tensor in dcp_state.items()
        }


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
        "--expert-weight-layout",
        choices=("auto", "hf", "grouped_gemm"),
        default="auto",
        help=(
            "DCP batched expert layout. Auto restores transposed GroupedExperts weights "
            "by comparing with the HF model; use an explicit layout for ambiguous shapes."
        ),
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

    # A model built from config has no source conversion history. Keep fused
    # experts intact instead of letting HF's default reverse mapping split them.
    has_fused_moe = any(
        parameter.ndim == 3 and name.endswith((".experts.gate_up_proj", ".experts.down_proj"))
        for name, parameter in model.named_parameters()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    DCPCheckpointManager(model).save_pretrained(
        output_dir,
        dcp_checkpoint=checkpoint_dir,
        dcp_expert_layout=args.expert_weight_layout,
        save_original_format=False if has_fused_moe else None,
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
