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
"""Scan JSONL with a local tokenizer; render templates and exercise training masks.

Run from an editable HyperParallel installation, without torchrun or model weights:
    HYPER_PARALLEL_PLATFORM=torch python -m examples.qwen3_8.tools.validate_chat_templates \
        --data-path /path/to/data.jsonl --tokenizer-path /path/to/model \
        --tools-key tools --max-seq-len 4096 --output-dir outputs/template_check

Add --config examples/qwen3_6_moe/train_online.yaml to read only
dataset.model_assets.chat_template_kwargs (not dataset paths or sequence length).
With --config, --template-kwargs overrides individual YAML fields; {} keeps them.
Without --config, legacy defaults apply unless --template-kwargs is supplied;
an explicit {} then selects tokenizer defaults. YAML interpolation is not resolved.

Exit codes: 0 = all records passed; 1 = record failures or empty dataset;
2 = invalid configuration or setup failure. Output contains dataset text: keep it private.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import hyper_parallel
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.text.build_data_transform import (
    TextConversationTransform,
    _normalize_conversation,
    _normalize_tools,
)
from hyper_parallel.data.text.build_tokenizer import AutoTokenizer
from hyper_parallel.data.text.chat_template import TokenizerTemplate


def validate_record(line: str, line_number: int, transform: TextConversationTransform) -> dict[str, Any]:
    """Render and encode a record, retaining its stage and error instead of aborting the scan.

    Args:
        line: One non-empty JSONL line.
        line_number: One-based physical line number in the source file.
        transform: The actual conversation transform used by training.

    Returns:
        Per-record status, rendered text when available, shifted lengths, and any error.
    """
    result = {"line": line_number, "render_ok": False, "training_ok": False, "stage": "json"}
    try:
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Each JSONL record must be an object")
        result["stage"] = "normalize"
        # Reuse training normalization so tool JSON strings and ShareGPT roles behave identically.
        messages = _normalize_conversation(record[transform.text_keys])
        tools = None
        if transform.tools_key is not None:
            tools = _normalize_tools(record.get(transform.tools_key), transform.tools_key)
        result.update(message_count=len(messages), tool_count=len(tools or []))
        result["stage"] = "render"
        # This is the same native-template invocation as the training encoder, without truncation.
        result["template_text"] = transform.chat_template._apply_chat_template(
            messages, tokenize=False, return_dict=False, tools=tools,
        )
        result["render_ok"] = True
        result["stage"] = "training_encode"
        sample = transform(record)[0]
        input_ids, labels = sample["input_ids"], sample["labels"]
        if input_ids.shape != labels.shape or input_ids.numel() == 0:
            raise ValueError(f"Invalid shifted shapes: input_ids={input_ids.shape}, labels={labels.shape}")
        supervised = int((labels != IGNORE_INDEX).sum())
        result.update(training_ok=True, stage="complete", train_tokens=input_ids.numel(),
                      supervised_tokens=supervised, zero_supervision=supervised == 0)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # A broken sample must not hide later failures; the aggregate exit code still signals failure.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def scan_dataset(
        data_path: Path,
        output_dir: Path,
        transform: TextConversationTransform,
        metadata: dict[str, Any],
        progress_every: int = 100,
) -> dict[str, Any]:
    """Scan all non-empty lines and write templates.jsonl, errors.jsonl, and summary.json.

    Args:
        data_path: Input JSONL file, never modified.
        output_dir: New output directory; an existing directory is rejected.
        transform: Configured training transform.
        metadata: Tokenizer and configuration provenance for the summary.
        progress_every: Print progress every N records; zero disables progress.

    Returns:
        Summary with counts and an explicit all_passed flag.
    """
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"configuration": metadata, "checked": 0, "blank_lines": 0, "render_passed": 0,
               "training_passed": 0, "failed": 0, "zero_supervision": 0}
    with data_path.open(encoding="utf-8") as source, \
            (output_dir / "templates.jsonl").open("x", encoding="utf-8") as rendered, \
            (output_dir / "errors.jsonl").open("x", encoding="utf-8") as errors:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                summary["blank_lines"] += 1
                continue
            result = validate_record(line, line_number, transform)
            rendered.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary["checked"] += 1
            summary["render_passed"] += int(result["render_ok"])
            summary["training_passed"] += int(result["training_ok"])
            summary["zero_supervision"] += int(result.get("zero_supervision", False))
            if not result["training_ok"]:
                summary["failed"] += 1
                error = {key: value for key, value in result.items() if key != "template_text"}
                errors.write(json.dumps(error, ensure_ascii=False) + "\n")
                errors.flush()
                print(f"FAIL line={line_number} stage={result['stage']}: {result['error_type']}", flush=True)
            if progress_every and summary["checked"] % progress_every == 0:
                print(f"checked={summary['checked']} failed={summary['failed']}", flush=True)
    summary["all_passed"] = summary["checked"] > 0 and summary["failed"] == 0
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    return summary


def _resolve_template_kwargs(config_path: Path | None, overrides: str | None) -> dict[str, Any]:
    """Read template options without constructing the training dataset or model."""
    if config_path is None:
        # Preserve the old CLI contract, including explicit {} for native defaults.
        options = {"enable_thinking": True, "reasoning_effort": "medium"} if overrides is None else {}
    else:
        with config_path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")
        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("config.dataset must be a mapping")
        assets = dataset.get("model_assets")
        if not isinstance(assets, dict):
            raise ValueError("config.dataset.model_assets must be a mapping")
        options = assets.get("chat_template_kwargs")
        if options is None:
            options = {}
        if not isinstance(options, dict) or any(not isinstance(key, str) for key in options):
            raise ValueError("config.dataset.model_assets.chat_template_kwargs must be a mapping with string keys")
        options = dict(options)
    if overrides is not None:
        explicit_options = json.loads(overrides)
        if not isinstance(explicit_options, dict):
            raise ValueError("template-kwargs must be a JSON object")
        options.update(explicit_options)
    return options


def parse_args() -> argparse.Namespace:
    """Parse local-only tokenizer and dataset options, matching the Qwen training defaults."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True,
                        help="Local model/tokenizer directory; no weights loaded")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; contains private rendered data")
    parser.add_argument("--text-key", default="messages")
    parser.add_argument("--tools-key", default="tools", help="Top-level tools field; pass an empty string to disable")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Training truncation limit before label shift")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--config", type=Path,
                        help="Training YAML; read only dataset.model_assets.chat_template_kwargs")
    parser.add_argument("--template-kwargs",
                        help="JSON options: override YAML fields with --config; otherwise replace legacy defaults")
    args = parser.parse_args()
    if args.max_seq_len < 2 or args.progress_every < 0:
        parser.error("max-seq-len must be >= 2 and progress-every must be >= 0")
    if not args.data_path.is_file() or not args.tokenizer_path.is_dir():
        parser.error("data-path must be a file and tokenizer-path must be a local directory")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory to preserve previous results")
    try:
        args.template_kwargs = _resolve_template_kwargs(args.config, args.template_kwargs)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Invalid template configuration: {exc}")
    return args


def main() -> int:
    """Run the CPU-only full scan and return a nonzero exit status on failures."""
    args = parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True,
                                                 use_fast=True, trust_remote_code=False)
        template = TokenizerTemplate(tokenizer, chat_template_kwargs=args.template_kwargs)
        transform = TextConversationTransform(template, args.max_seq_len, args.text_key, args.tools_key or None)
        metadata = {"data_path": str(args.data_path.resolve()), "tokenizer_path": str(args.tokenizer_path.resolve()),
                    "hyper_parallel_path": hyper_parallel.__file__, "text_key": args.text_key,
                    "tools_key": args.tools_key or None, "max_seq_len": args.max_seq_len,
                    "template_kwargs": args.template_kwargs,
                    "config_path": str(args.config.resolve()) if args.config else None}
        summary = scan_dataset(args.data_path, args.output_dir, transform, metadata, args.progress_every)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Setup/scan failed: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
