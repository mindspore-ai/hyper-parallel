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
"""Tokenizer assets for the Qwen3-VL-MoE dual-arm codegen validation.

The probe checkpoint carries only ``config.json`` (no tokenizer files), but
``build_indexed_text_dataset`` is wired with ``tokenizer_type: pretokenized``
and ``vocab_size: 151936``, which matches the Qwen2.5 tokenizer shipped on the
hub. This module materializes that tokenizer into the validation workspace
(idempotently), either by copying an existing bundle (``--source``, for
offline hosts) or by downloading it from the hub once.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence

_TOKENIZER_HUB_ID = "Qwen/Qwen2.5-0.5B"
_TOKENIZER_PATTERNS = ["tokenizer*", "vocab*", "merges*", "special_tokens_map*"]
# Qwen2.5 tokenizer length (151643 base + 22 added tokens). The model-side
# vocab_size in the YAML stays 151936 (Qwen3-VL-MoE embedding size); the
# dataset is pretokenized, so the tokenizer length is only a bundle identity
# check here.
_EXPECTED_TOKENIZER_LEN = 151665
_TOKENIZER_MARKER = "tokenizer_config.json"


def _verify_vocab(output: Path) -> None:
    """Assert the tokenizer bundle in ``output`` has the expected vocab size."""
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    if len(tokenizer) != _EXPECTED_TOKENIZER_LEN:
        raise RuntimeError(
            f"tokenizer length mismatch: expected {_EXPECTED_TOKENIZER_LEN}, got {len(tokenizer)}"
        )


def prepare_tokenizer(output: Path, *, source: Path | None = None, endpoint: str | None = None) -> None:
    """Materialize the Qwen2.5 tokenizer bundle into ``output`` (idempotent).

    Args:
        output: Destination directory for the tokenizer bundle.
        source: Optional local directory already holding the tokenizer files;
            when given, files are copied instead of downloaded (offline hosts).
        endpoint: Optional Hugging Face endpoint override (e.g. a mirror);
            defaults to the ``HF_ENDPOINT`` environment variable when already
            set, otherwise the official hub.
    """
    output.mkdir(parents=True, exist_ok=True)
    if (output / _TOKENIZER_MARKER).is_file():
        try:
            _verify_vocab(output)
            return
        except Exception:  # noqa: BLE001 - partial bundle, refetch below
            pass

    if source is not None:
        for pattern in _TOKENIZER_PATTERNS:
            for src_file in source.glob(pattern):
                if src_file.is_file():
                    shutil.copy2(src_file, output / src_file.name)
        _verify_vocab(output)
        return

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

    snapshot_download(
        _TOKENIZER_HUB_ID,
        local_dir=str(output),
        allow_patterns=_TOKENIZER_PATTERNS,
    )
    _verify_vocab(output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse tokenizer preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare Qwen3-VL-MoE tokenizer assets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source",
        default=None,
        type=Path,
        help="Local dir already holding the tokenizer files (offline hosts)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="HF endpoint override (e.g. https://hf-mirror.com); "
        "falls back to HF_ENDPOINT when unset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare the tokenizer bundle for the validation workspace.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``); supports ``--output``,
            ``--source`` and ``--endpoint``.
    """
    args = _parse_args(argv)
    prepare_tokenizer(
        args.output.expanduser().resolve(),
        source=args.source.expanduser().resolve() if args.source else None,
        endpoint=args.endpoint,
    )


if __name__ == "__main__":
    main()
