# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Codegen CLI: thin command-line wrapper over manager / check / artifact.

``python -m hyper_parallel.codegen`` dispatches to :func:`main`.  No business
logic — argument parsing, YAML discovery, and exit-code mapping only.

Exit codes are 0 for success or a skipped pin check, 1 for a check/clean
failure, and 2 for a usage error.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    """argparse entry; returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        return 2
    _configure_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - a CLI reports, does not traceback
        logger.error("codegen: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    """``generate --config <yaml>`` — build/reuse the bundle for one YAML."""
    config = _load_config(args.config)
    from hyper_parallel.codegen.manager import ensure_codegen_artifact

    if not getattr(config, "codegen", False):
        logger.info("codegen: generate skipped — codegen is false")
        return 0
    layout = ensure_codegen_artifact(config, args.config)
    if layout is None:
        return 0
    from hyper_parallel.codegen.meta import load_codegen_meta

    meta = load_codegen_meta(layout.meta_path)
    signature = meta.signature if meta is not None else "?"
    logger.info("codegen: artifact %s (signature %s)", layout.artifact_dir, signature)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """``check --config <yaml>`` — full preflight for one YAML."""
    config = _load_config(args.config)
    if not getattr(config, "codegen", False):
        logger.info("codegen: SKIP (codegen disabled)")
        return 0
    from hyper_parallel.codegen.manager import preflight_integrity_check

    preflight_integrity_check(config, args.config)
    logger.info("codegen: PASS %s", args.config)
    return 0


def cmd_check_all(args: argparse.Namespace) -> int:
    """Run preflight for every discovered YAML and check cached artifact drift."""
    failed = 0
    for yaml_path in discover_yaml_files(args.root):
        try:
            config = _load_config(yaml_path)
            if not getattr(config, "codegen", False):
                logger.info("codegen: SKIP (codegen disabled) %s", yaml_path)
                continue
            from hyper_parallel.codegen.manager import preflight_integrity_check

            preflight_integrity_check(config, yaml_path)
            logger.info("codegen: PASS %s", yaml_path)
            # Drift can only be checked when an artifact already exists.
            from hyper_parallel.codegen.check.drift import verify_drift_intact
            from hyper_parallel.codegen.meta import load_codegen_meta
            from hyper_parallel.codegen.artifact import resolve_artifact_layout

            layout = resolve_artifact_layout(yaml_path)
            meta = load_codegen_meta(layout.meta_path)
            if meta is not None:
                drifts = verify_drift_intact(meta, yaml_path)
                if drifts:
                    failed = 1
                    logger.error("codegen: drift in %s: %s", yaml_path, "; ".join(drifts))
                else:
                    logger.info("codegen: drift intact %s", yaml_path)
        except Exception as exc:  # noqa: BLE001 - aggregate, keep scanning
            failed = 1
            logger.error("codegen: FAIL %s: %s", yaml_path, exc)
    return failed


def cmd_check_fresh(args: argparse.Namespace) -> int:
    """``check --config <yaml> --require-fresh`` — stale artifact fails, no regeneration."""
    config = _load_config(args.config)
    if not getattr(config, "codegen", False):
        logger.info("codegen: SKIP (codegen disabled)")
        return 0
    from hyper_parallel.codegen.manager import _compute_signature
    from hyper_parallel.codegen.meta import load_codegen_meta
    from hyper_parallel.codegen.artifact import resolve_artifact_layout

    layout = resolve_artifact_layout(args.config, model_name=_model_name(config))
    meta = load_codegen_meta(layout.meta_path)
    if meta is None:
        logger.error("codegen: no artifact at %s — regenerate first", layout.artifact_dir)
        return 1
    current = _compute_signature(config, layout)
    if meta.signature != current:
        logger.error(
            "codegen: STALE %s (stored %s, current %s) — regenerate",
            args.config, meta.signature, current,
        )
        return 1
    logger.info("codegen: FRESH %s (signature %s)", args.config, meta.signature)
    return 0


def cmd_check_pin(args: argparse.Namespace) -> int:
    """Validate a Transformers version constraint without regenerating artifacts."""
    from hyper_parallel.codegen.check.pin import verify_transformers_pin

    status, installed, spec = verify_transformers_pin(args.transformers_pin)
    logger.info("codegen: transformers pin %s — installed %s (%s)", spec, installed, status)
    return 0  # OK and SKIP both exit 0


def cmd_clean(args: argparse.Namespace) -> int:
    """``clean --config <yaml>`` — delete the bundle dir + temp residue."""
    from hyper_parallel.codegen.artifact import clean_temp_artifacts, resolve_artifact_layout

    layout = resolve_artifact_layout(args.config, model_name=_model_name_for(args.config))
    if not os.path.isdir(layout.artifact_dir):
        logger.error("codegen: no artifact at %s", layout.artifact_dir)
        return 1
    clean_temp_artifacts(layout)
    import shutil

    shutil.rmtree(layout.artifact_dir)
    logger.info("codegen: removed %s", layout.artifact_dir)
    return 0


# ---------------------------------------------------------------------------
# discovery & helpers
# ---------------------------------------------------------------------------

_SKIP_DIRS = {"generated", ".git", ".venv", "venv", "__pycache__", ".tmp"}


def discover_yaml_files(root: str) -> Iterator[str]:
    """Walk ``root`` for ``*.yaml`` / ``*.yml``, skipping generated/venv/git dirs."""
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith((".yaml", ".yml")):
                yield os.path.join(dirpath, name)


def _load_config(yaml_path: str):
    """Load a ``TrainerConfig`` from a YAML path."""
    from hyper_parallel.trainer.config.manager import load_training_config

    return load_training_config(yaml_path)


def _model_name(config: object) -> Optional[str]:
    from hyper_parallel.codegen.manager import (  # pylint: disable=import-outside-toplevel
        _model_name,
        _resolve_hf_config_for_layout,
    )

    return _model_name(config, _resolve_hf_config_for_layout(config))


def _model_name_for(yaml_path: str) -> Optional[str]:
    """Model name from a YAML (used by ``clean`` before config load)."""
    try:
        return _model_name(_load_config(yaml_path))
    except Exception:  # noqa: BLE001 - clean falls back to default naming
        return None


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyper_parallel.codegen",
        description="Hyper codegen: generate/check/clean codegen artifacts.",
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")

    gen = sub.add_parser("generate", help="generate or reuse the bundle for one YAML")
    gen.add_argument("--config", required=True, help="path to the training YAML")
    gen.set_defaults(func=cmd_generate)

    check = sub.add_parser("check", help="preflight integrity check")
    check.add_argument("--config", help="path to the training YAML")
    check.add_argument("--all", action="store_true", help="check every discovered YAML")
    check.add_argument("--root", default=".", help="root to search when --all")
    check.add_argument("--require-fresh", action="store_true", help="fail on stale, never regenerate")
    check.add_argument("--transformers-pin", help="Transformers version constraint (e.g. ==5.15.0)")
    check.set_defaults(func=_cmd_check_dispatch)

    clean = sub.add_parser("clean", help="delete one YAML's artifact bundle")
    clean.add_argument("--config", required=True, help="path to the training YAML")
    clean.set_defaults(func=cmd_clean)

    return parser


def _cmd_check_dispatch(args: argparse.Namespace) -> int:
    """Route the ``check`` subcommand's mutually-exclusive modes."""
    modes = [bool(args.all), bool(args.require_fresh), bool(args.transformers_pin)]
    if sum(modes) > 1:
        raise ValueError("check accepts only one of --all / --require-fresh / --transformers-pin")
    if args.all:
        return cmd_check_all(args)
    if args.require_fresh:
        return cmd_check_fresh(args)
    if args.transformers_pin:
        return cmd_check_pin(args)
    if not args.config:
        raise ValueError("check needs --config, --all, --require-fresh, or --transformers-pin")
    return cmd_check(args)


__all__ = [
    "cmd_check",
    "cmd_check_all",
    "cmd_check_fresh",
    "cmd_check_pin",
    "cmd_clean",
    "cmd_generate",
    "discover_yaml_files",
    "main",
]
