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
"""Thin command-line interface for native-build codegen workflows."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterator, Optional

from hyper_parallel.codegen.build import generate_from_native_build
from hyper_parallel.codegen.check.drift import verify_drift_intact
from hyper_parallel.codegen.check.pin import verify_transformers_pin
from hyper_parallel.codegen.manager import preflight_integrity_check, resolve_layout_for_config
from hyper_parallel.codegen.meta import load_codegen_meta
from hyper_parallel.trainer.config import parser as config_parser

logger = logging.getLogger(__name__)

_SKIP_DIRS = {"generated", ".git", ".venv", "venv", "__pycache__", ".tmp"}


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CLI and return its process exit code."""
    parser = _build_parser()
    args, config_overrides = parser.parse_known_args(argv)
    setattr(args, "_config_overrides", config_overrides)
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        return 2
    _configure_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except Exception as exc:  # A CLI reports failures without a traceback by default.
        if getattr(args, "verbose", False):
            logger.exception("codegen: %s", exc)
        else:
            logger.error("codegen: %s", exc)
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Observe a meta model and generate or reuse its snapshot bundle."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "codegen generate is a single-process CPU/meta command; "
            "run it with python instead of torchrun"
        )
    config = _load_config(args.config, _cli_overrides(args))
    layout = generate_from_native_build(config, artifact_dir=args.output)
    meta = load_codegen_meta(layout.meta_path)
    signature = meta.signature if meta is not None else "?"
    logger.info("codegen: artifact %s (signature %s)", layout.artifact_dir, signature)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run full preflight for one configured artifact."""
    config = _load_config(args.config, _cli_overrides(args))
    preflight_integrity_check(config, args.config)
    logger.info("codegen: PASS %s", args.config)
    return 0


def cmd_check_all(args: argparse.Namespace) -> int:
    """Run preflight and drift checks for every discovered YAML."""
    if _cli_overrides(args):
        raise ValueError("check --all does not accept per-config CLI overrides")
    failed = 0
    for yaml_path in discover_yaml_files(args.root):
        try:
            config = _load_config(yaml_path)
            preflight_integrity_check(config, yaml_path)
            layout = resolve_layout_for_config(config, yaml_path)
            meta = load_codegen_meta(layout.meta_path)
            if meta is not None:
                drifts = verify_drift_intact(meta, yaml_path, artifact_dir=layout.artifact_dir)
                if drifts:
                    failed = 1
                    logger.error("codegen: drift in %s: %s", yaml_path, "; ".join(drifts))
                else:
                    logger.info("codegen: drift intact %s", yaml_path)
            logger.info("codegen: PASS %s", yaml_path)
        except Exception as exc:  # Aggregate failures while scanning all configs.
            failed = 1
            logger.error("codegen: FAIL %s: %s", yaml_path, exc)
    return failed


def cmd_check_pin(args: argparse.Namespace) -> int:
    """Validate a Transformers version constraint without generating artifacts."""
    status, installed, spec = verify_transformers_pin(args.transformers_pin)
    logger.info("codegen: transformers pin %s - installed %s (%s)", spec, installed, status)
    return 0


def discover_yaml_files(root: str) -> Iterator[str]:
    """Yield YAML files below root while skipping generated and environment directories."""
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith((".yaml", ".yml")):
                yield os.path.join(dirpath, name)


def _load_config(yaml_path: str, overrides: Optional[list[str]] = None):
    """Load a resolved TrainerConfig without building trainer runtime components."""
    return config_parser.load_training_config(yaml_path, overrides or [])


def _cli_overrides(args: argparse.Namespace) -> list[str]:
    """Return typed config overrides left after argparse processing."""
    return list(getattr(args, "_config_overrides", None) or [])


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyper_parallel.codegen",
        description="Generate and inspect snapshot-driven codegen artifacts.",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser(
        "generate",
        help="observe the model on meta and generate its bundle",
    )
    generate.add_argument("--config", required=True, help="path to the training YAML")
    generate.add_argument("--output", help="explicit artifact output directory")
    generate.set_defaults(func=cmd_generate)

    check = subparsers.add_parser("check", help="preflight integrity check")
    check.add_argument("--config", help="path to the training YAML")
    check.add_argument("--all", action="store_true", help="check every discovered YAML")
    check.add_argument("--root", default=".", help="root to search when --all")
    check.add_argument("--transformers-pin", help="Transformers version constraint")
    check.set_defaults(func=_cmd_check_dispatch)
    return parser


def _cmd_check_dispatch(args: argparse.Namespace) -> int:
    modes = [bool(args.all), bool(args.transformers_pin)]
    if sum(modes) > 1:
        raise ValueError("check accepts only one of --all / --transformers-pin")
    if args.all:
        return cmd_check_all(args)
    if args.transformers_pin:
        return cmd_check_pin(args)
    if not args.config:
        raise ValueError("check needs --config, --all, or --transformers-pin")
    return cmd_check(args)


__all__ = [
    "cmd_check",
    "cmd_check_all",
    "cmd_check_pin",
    "cmd_generate",
    "discover_yaml_files",
    "main",
]
