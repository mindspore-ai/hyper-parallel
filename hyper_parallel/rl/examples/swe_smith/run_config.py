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
"""Build host-visible SWE-Smith run paths without importing the training runtime."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml


_SHA256 = re.compile(r"[0-9a-f]{64}")
LOCAL_DOCKER_HOST = "unix:///var/run/docker.sock"


def resolve_docker_host(endpoint: str | None) -> str:
    """Return the explicit local socket or validate a remote SSH endpoint.

    The same endpoint is used for candidates and graders. Ambient Docker
    contexts do not determine where a task runs.
    """
    if endpoint is None or endpoint == LOCAL_DOCKER_HOST:
        return LOCAL_DOCKER_HOST
    if not isinstance(endpoint, str) or any(character.isspace() for character in endpoint):
        raise ValueError("Remote Docker endpoint must be ssh:// with a nonempty host")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "ssh" or not parsed.hostname or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.password is not None):
        raise ValueError("Remote Docker endpoint must be ssh:// with a nonempty host")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Remote Docker endpoint has an invalid port") from error
    return endpoint


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _selected_identities(directory: Path) -> list[tuple[str, str]]:
    """Validate the selection manifest and preserve its ordered split identities."""
    selection = _read_json(directory / "selection.json")
    splits = selection.get("splits")
    if selection.get("schema_version") != 1 or not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("Prepared selection must declare train and validation splits with schema_version 1")
    expected = []
    for split, description in splits.items():
        identities = description.get("instance_ids") if isinstance(description, dict) else None
        if (not isinstance(identities, list) or not identities
                or any(not isinstance(identity, str) or not identity for identity in identities)
                or not _SHA256.fullmatch(str(description.get("sha256", "")))
                or not isinstance(description.get("source"), str) or not description["source"]):
            raise ValueError(f"Prepared {split} source and selected identities are incomplete")
        expected.extend((split, identity) for identity in identities)
    if len({identity for _, identity in expected}) != len(expected):
        raise ValueError("Prepared selection contains duplicate task identities")
    return expected


def _prepared_identity(directory: Path) -> tuple[str, set[str]]:
    """Cross-check prepared records and hash their public and private artifacts."""
    expected = _selected_identities(directory)
    files = ["selection.json", "instances.jsonl", "train.parquet", "validation.parquet"]
    actual, images = [], set()
    try:
        with (directory / "instances.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row["instance_id"]
                if not isinstance(identity, str) or not isinstance(row.get("image_name"), str):
                    raise ValueError("Prepared records require task and image identities")
                private = "instances/" + hashlib.sha256(identity.encode()).hexdigest() + ".json"
                public_metadata = dict(row)
                split = public_metadata.pop("split")
                if _read_json(directory / private) != public_metadata:
                    raise ValueError(f"Prepared private record differs for task {identity}")
                files.append(private)
                actual.append((split, identity))
                images.add(row["image_name"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Prepared instance records are incomplete or inconsistent") from error
    if actual != expected:
        raise ValueError("Prepared instance records differ from the selection manifest")
    digest = hashlib.sha256()
    for name in sorted(files):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Prepared data file is missing or empty: {path}")
        digest.update(name.encode() + b"\0" + _hash_file(path).encode() + b"\0")
    return digest.hexdigest(), images


def _image_identity(path: Path, required: set[str]) -> str:
    """Require pinned images and absolute Python paths before hashing the manifest."""
    manifest = _read_json(path)
    images = manifest.get("images")
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True or not isinstance(images, dict):
        raise ValueError("Image manifest must be complete with schema_version 1")
    for name in required:
        image = images.get(name)
        if (not isinstance(image, dict) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("image", "")))
                or not isinstance(image.get("python"), str) or not Path(image["python"]).is_absolute()):
            raise ValueError(f"Image manifest lacks a fixed image and absolute Python for {name}")
    return _hash_file(path)


def _resume_identity(resume: Path, config: dict, data_sha256: str, images_sha256: str) -> None:
    """Validate checkpoint progress and its saved prepared-data and image identities."""
    completion = _read_json(resume / "checkpoint_complete.json")
    step = completion.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or config["train"]["max_steps"] <= step:
        raise ValueError("Resume max_steps must be greater than the completed checkpoint step")
    try:
        old = yaml.safe_load((resume / "resolved_config.yaml").read_text(encoding="utf-8"))
        old_task = old["agentic"]["codex"]["task"]
        old_dir = Path(old_task["data_dir"]).resolve()
        old_data = old["data"]
        if any(Path(old_data[key]).resolve() != old_dir / name for key, name in
               (("train_path", "train.parquet"), ("test_path", "validation.parquet"))):
            raise ValueError("Resume configuration data source differs from the prepared task directory")
        if old_task["data_sha256"] != data_sha256 or old_task["images_sha256"] != images_sha256:
            raise ValueError("Resume data or image identity differs from the checkpoint configuration")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError("Resume requires a readable resolved_config.yaml with SWE-Smith identities") from error


def _set_training_steps(train: dict, steps: int | None) -> None:
    """Apply an optional step override and validate the resolved training budget."""
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        train["max_steps"] = steps
    if not isinstance(train["max_steps"], int) or isinstance(train["max_steps"], bool) or train["max_steps"] <= 0:
        raise ValueError("max_steps must be a positive integer")


def build_config(template: dict, *, data_dir: Path, images_path: Path, model_dir: Path, output_dir: Path,
                 checkpoint_dir: Path, docker_host: str | None, steps: int | None = None,
                 resume: Path | None = None) -> dict:
    """Resolve one SWE-Smith recipe for a local or SSH Docker sandbox.

    Prepared JSON metadata and private records are cross-checked; Parquet files
    are fingerprinted without importing a Parquet reader or the training runtime.
    The trainer retains ownership of checkpoint topology and tensor validation.

    Args:
        template: Existing SWE-Smith training or inference recipe.
        data_dir: Prepared data directory containing both public splits.
        images_path: Complete, prepared image manifest for the selected tasks.
        model_dir: Local model and tokenizer directory.
        output_dir: Run logs and Codex sessions directory.
        checkpoint_dir: Destination for newly saved checkpoints.
        docker_host: SSH Docker endpoint, local socket URI, or None for local execution.
        steps: Optional total training steps, including resumed steps.
        resume: Completed checkpoint directory to resume.

    Returns:
        Independent configuration with absolute host-visible paths.

    Raises:
        ValueError: Inputs are incomplete, identities differ, or outputs overlap inputs.
    """
    data_dir, images_path, model_dir, output_dir, checkpoint_dir = (
        Path(path).resolve() for path in (data_dir, images_path, model_dir, output_dir, checkpoint_dir))
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    endpoint = resolve_docker_host(docker_host)
    inputs = [data_dir, model_dir, images_path]
    if resume is not None:
        resume = Path(resume).resolve()
        inputs.append(resume)
    if any(destination == source or source in destination.parents or destination in source.parents
           for destination in (output_dir, checkpoint_dir) for source in inputs):
        raise ValueError("Run output directories must not overwrite, contain, or be inside input artifacts")
    data_sha256, required_images = _prepared_identity(data_dir)
    images_sha256 = _image_identity(images_path, required_images)
    config = deepcopy(template)
    codex = config["agentic"]["codex"]
    config.setdefault("data", {})
    train = config.setdefault("train", {})
    train.setdefault("max_steps", 1)
    train.setdefault("checkpoint", {})
    logging = config.setdefault("logging", {})
    logging.setdefault("wandb", {})
    previous_images = str(codex["task"].get("images_sha256", ""))
    if _SHA256.fullmatch(previous_images) and previous_images != images_sha256:
        raise ValueError("Prepared image manifest SHA256 differs from the pinned template")
    previous_data = codex["task"].get("data_sha256")
    if previous_data is not None and previous_data != data_sha256:
        raise ValueError("Prepared data SHA256 differs from the pinned template")
    _set_training_steps(train, steps)
    config["model"].update(weights_path=str(model_dir), tokenizer_path=str(model_dir))
    config["data"].update(train_path=str(data_dir / "train.parquet"), test_path=str(data_dir / "validation.parquet"))
    codex["task"].update(data_dir=str(data_dir), data_sha256=data_sha256,
                         images_path=str(images_path), images_sha256=images_sha256, docker_host=endpoint)
    codex["workspace"]["docker_host"] = endpoint
    codex["session_root"] = str(output_dir / "sessions")
    checkpoint = train["checkpoint"]
    checkpoint.update(output_dir=str(checkpoint_dir), load_path=None if resume is None else str(resume))
    logging["experiment_name"] = output_dir.name
    logging["wandb"]["directory"] = str(output_dir / "wandb")
    if resume is not None:
        _resume_identity(resume, config, data_sha256, images_sha256)
    return config
