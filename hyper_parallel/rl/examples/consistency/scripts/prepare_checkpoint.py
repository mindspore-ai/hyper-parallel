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
"""Copy the first decoder layers of an indexed Qwen3 checkpoint for acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

from rl.unified_execution.models import get_model_family


def _file_digest(path: Path) -> str:
    """Hash files without retaining checkpoint bytes in Python memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_checkpoint(source: Path, output: Path, layers: int) -> dict:
    """Retain original tensor values and experts, changing only decoder depth.

    Args:
        source: Original directory with an indexed safetensors checkpoint.
        output: New directory; existing directories are never overwritten.
        layers: Positive number of initial decoder layers to retain.

    Returns:
        Source identity and exported shard hashes for numerical acceptance.
    """
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    get_model_family(config["model_type"])
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= config["num_hidden_layers"]:
        raise ValueError("layers must retain between one and the original number of decoder layers")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    groups = {}
    for name, filename in index["weight_map"].items():
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match is None or int(match[1]) < layers:
            groups.setdefault(filename, []).append(name)
    output.mkdir(parents=True, exist_ok=False)
    config["num_hidden_layers"] = layers
    if "max_window_layers" in config:
        config["max_window_layers"] = min(config["max_window_layers"], layers)
    if isinstance(config.get("layer_types"), list):
        config["layer_types"] = config["layer_types"][:layers]
    if isinstance(config.get("mlp_only_layers"), list):
        config["mlp_only_layers"] = [index for index in config["mlp_only_layers"] if index < layers]
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for path in source.iterdir():
        if path.is_file() and path.name.startswith(("tokenizer", "vocab", "merges", "generation_config",
                                                   "special_tokens", "chat_template")):
            shutil.copy2(path, output / path.name)
    manifest = {"source": str(source.resolve()), "layers": layers,
                "source_config_sha256": _file_digest(source / "config.json"),
                "source_index_sha256": _file_digest(source / "model.safetensors.index.json"), "shards": []}
    exported = {"metadata": {"total_size": 0}, "weight_map": {}}
    for number, (filename, names) in enumerate(sorted(groups.items()), 1):
        with safe_open(source / filename, framework="pt", device="cpu") as checkpoint:
            weights = {name: checkpoint.get_tensor(name) for name in names}
        target = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        save_file(weights, output / target, metadata={"format": "pt"})
        exported["weight_map"].update({name: target for name in names})
        exported["metadata"]["total_size"] += sum(value.numel() * value.element_size() for value in weights.values())
        manifest["shards"].append({"source_file": filename, "output_file": target,
                                   "tensors": len(names), "sha256": _file_digest(output / target)})
        del weights
    (output / "model.safetensors.index.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    manifest["output_config_sha256"] = _file_digest(output / "config.json")
    manifest["output_index_sha256"] = _file_digest(output / "model.safetensors.index.json")
    (output / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_checkpoint(source: Path, output: Path, layers: int) -> None:
    """Verify a reusable crop against source metadata and every exported tensor shard."""
    manifest = json.loads((output / "crop_manifest.json").read_text(encoding="utf-8"))
    if manifest["layers"] != layers:
        raise ValueError("Existing checkpoint has a different decoder depth")
    paths = ((source / "config.json", "source_config_sha256"),
             (source / "model.safetensors.index.json", "source_index_sha256"),
             (output / "config.json", "output_config_sha256"),
             (output / "model.safetensors.index.json", "output_index_sha256"))
    for path, key in paths:
        if _file_digest(path) != manifest[key]:
            raise ValueError(f"Checkpoint metadata differs from crop manifest: {path}")
    for shard in manifest["shards"]:
        if _file_digest(output / shard["output_file"]) != shard["sha256"]:
            raise ValueError(f"Cropped checkpoint shard checksum failed: {shard['output_file']}")


def main() -> None:
    """Prepare a separate artifact without modifying the pretrained source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--reuse", action="store_true", help="Verify and reuse an existing crop")
    args = parser.parse_args()
    if args.reuse and args.output.exists():
        verify_checkpoint(args.source, args.output, args.layers)
    else:
        prepare_checkpoint(args.source, args.output, args.layers)


if __name__ == "__main__":
    main()
