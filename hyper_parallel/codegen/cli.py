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
from typing import Optional

from hyper_parallel.codegen.build import generate_from_native_build
from hyper_parallel.trainer.config import parser as config_parser

logger = logging.getLogger(__name__)


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
    """Observe a meta model and generate a fresh bundle."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "codegen generate is a single-process CPU/meta command; "
            "run it with python instead of torchrun"
        )
    config = _load_config(args.config, _cli_overrides(args))
    layout = generate_from_native_build(config, artifact_dir=args.output)
    logger.info("codegen: artifact %s", layout.artifact_dir)
    return 0


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
    """Build the codegen command-line parser."""
    parser = argparse.ArgumentParser(
        prog="hyper_parallel.codegen",
        description="Generate a model bundle from a training YAML.",
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

    return parser


__all__ = ["cmd_generate", "main"]
