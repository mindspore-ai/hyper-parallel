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
"""Analyze Decif JSONL lengths with the Qwen3.8 training chat template.

Example:
    python examples/qwen3_8/tools/analyze_decif_lengths.py \
        --data-path /home/chaoran/datasets/decif-30k.jsonl \
        --tokenizer-path /home/chaoran/models/Qwen3.8-27B \
        --max-seq-len 32768 \
        --output-dir ./outputs/decif_length_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.text.build_tokenizer import AutoTokenizer
from hyper_parallel.data.text.chat_template import build_chat_template


_FULL_SEQUENCE_LIMIT = 2**31 - 1
_CSV_FIELDS = (
    "source_index",
    "message_count",
    "assistant_message_count",
    "raw_character_count",
    "raw_reasoning_tokens",
    "raw_answer_tokens",
    "template_tokens_before_truncation",
    "train_tokens_after_truncation",
    "supervised_tokens_after_truncation",
    "masked_tokens_after_truncation",
    "supervision_ratio",
    "truncated_tokens",
    "was_truncated",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure Decif conversation lengths after the exact Qwen tokenizer chat template and loss mask."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Decif JSONL file containing messages.")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Local Qwen3.8 tokenizer/model directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/decif_length_stats"),
        help="Directory for the PNG, per-sample CSV, and summary JSON.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="Training max_seq_len used to calculate truncation and shifted labels (default: 32768).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default="medium",
        help="Qwen reasoning effort forwarded to the tokenizer chat template (default: medium).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render the chat template with enable_thinking=false instead of the Decif training default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N non-empty records for a quick check.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many records; use 0 to disable progress output.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow tokenizer code from the model directory.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate input paths and numeric options."""
    if not args.data_path.is_file():
        raise ValueError(f"Decif JSONL file does not exist: {args.data_path}")
    if not args.tokenizer_path.exists():
        raise ValueError(f"Tokenizer path does not exist: {args.tokenizer_path}")
    if args.max_seq_len <= 1:
        raise ValueError("--max-seq-len must be greater than one")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.progress_every < 0:
        raise ValueError("--progress-every must not be negative")


def _iter_jsonl(data_path: Path, limit: int | None) -> Iterable[tuple[int, Mapping[str, Any]]]:
    """Yield validated JSON objects from a Decif JSONL file.

    Args:
        data_path: Input JSONL path.
        limit: Optional maximum number of non-empty records.

    Yields:
        One-based source line number and decoded JSON object.

    Raises:
        ValueError: If a record is invalid JSON or is not an object.
    """
    emitted = 0
    with data_path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {data_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Expected a JSON object at {data_path}:{line_number}")
            yield line_number, record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _get_messages(record: Mapping[str, Any], line_number: int) -> list[dict[str, Any]]:
    """Return a validated OpenAI-style Decif conversation."""
    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(f"Record at line {line_number} must contain a messages list")

    normalized_messages = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"Message {message_index} at line {line_number} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message {message_index} at line {line_number} has an invalid role")
        if not isinstance(content, str):
            raise ValueError(f"Message {message_index} at line {line_number} must contain string content")
        normalized_messages.append(dict(message))
    if not normalized_messages:
        raise ValueError(f"Record at line {line_number} has no messages")
    return normalized_messages


def _count_field_tokens(tokenizer: Any, messages: Sequence[Mapping[str, Any]], field: str) -> int:
    """Count standalone tokens in one raw assistant text field."""
    token_count = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        value = message.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Assistant field {field!r} must be a string when present")
        token_count += len(tokenizer.encode(value, add_special_tokens=False))
    return token_count


def analyze_record(
        tokenizer: Any,
        chat_template: Any,
        record: Mapping[str, Any],
        line_number: int,
        max_seq_len: int,
) -> dict[str, Any]:
    """Measure one Decif record before and after the training truncation boundary.

    Args:
        tokenizer: Training tokenizer.
        chat_template: HyperParallel tokenizer chat-template wrapper.
        record: Decoded Decif JSON object.
        line_number: One-based source line number.
        max_seq_len: Training sequence-length limit before causal shifting.

    Returns:
        Per-record statistics suitable for CSV output.
    """
    messages = _get_messages(record, line_number)
    encoded = chat_template.encode_messages(messages, max_seq_len=_FULL_SEQUENCE_LIMIT)
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    if len(input_ids) != len(labels):
        raise ValueError(f"input_ids and labels differ in length at line {line_number}")

    template_tokens = len(input_ids)
    retained_labels = labels[-max_seq_len:]
    # Online conversation training removes the final input token and the first label
    # so each retained label is the next-token target of the aligned model input.
    shifted_labels = retained_labels[1:]
    train_tokens = len(shifted_labels)
    supervised_tokens = sum(label != IGNORE_INDEX for label in shifted_labels)
    masked_tokens = train_tokens - supervised_tokens
    supervision_ratio = supervised_tokens / train_tokens if train_tokens else 0.0

    raw_character_count = 0
    for message in messages:
        raw_character_count += len(message["content"])
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            if not isinstance(reasoning_content, str):
                raise ValueError("Assistant reasoning_content must be a string when present")
            raw_character_count += len(reasoning_content)

    return {
        "source_index": line_number,
        "message_count": len(messages),
        "assistant_message_count": sum(message["role"] == "assistant" for message in messages),
        "raw_character_count": raw_character_count,
        # These two fields exclude chat-template wrappers and are diagnostic, not additive parts of total length.
        "raw_reasoning_tokens": _count_field_tokens(tokenizer, messages, "reasoning_content"),
        "raw_answer_tokens": _count_field_tokens(tokenizer, messages, "content"),
        "template_tokens_before_truncation": template_tokens,
        "train_tokens_after_truncation": train_tokens,
        "supervised_tokens_after_truncation": supervised_tokens,
        "masked_tokens_after_truncation": masked_tokens,
        "supervision_ratio": supervision_ratio,
        "truncated_tokens": max(template_tokens - max_seq_len, 0),
        "was_truncated": template_tokens > max_seq_len,
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from no values")
    position = percentile / 100.0 * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction)


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize one non-empty numeric distribution."""
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "mean": float(fmean(sorted_values)),
        "p50": _percentile(sorted_values, 50),
        "p75": _percentile(sorted_values, 75),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
        "max": float(sorted_values[-1]),
    }


def build_summary(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Build the machine-readable aggregate report."""
    if not rows:
        raise ValueError("The input JSONL contained no records")
    template_lengths = [row["template_tokens_before_truncation"] for row in rows]
    train_lengths = [row["train_tokens_after_truncation"] for row in rows]
    supervised_lengths = [row["supervised_tokens_after_truncation"] for row in rows]
    reasoning_lengths = [row["raw_reasoning_tokens"] for row in rows]
    answer_lengths = [row["raw_answer_tokens"] for row in rows]
    truncated_count = sum(bool(row["was_truncated"]) for row in rows)
    zero_supervision_count = sum(row["supervised_tokens_after_truncation"] == 0 for row in rows)
    return {
        "data_path": str(args.data_path),
        "tokenizer_path": str(args.tokenizer_path),
        "max_seq_len": args.max_seq_len,
        "enable_thinking": not args.disable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "sample_count": len(rows),
        "truncated_sample_count": truncated_count,
        "truncated_sample_ratio": truncated_count / len(rows),
        "zero_supervision_sample_count": zero_supervision_count,
        "zero_supervision_sample_ratio": zero_supervision_count / len(rows),
        "template_tokens_before_truncation": _distribution_summary(template_lengths),
        "train_tokens_after_truncation": _distribution_summary(train_lengths),
        "supervised_tokens_after_truncation": _distribution_summary(supervised_lengths),
        "raw_reasoning_tokens": _distribution_summary(reasoning_lengths),
        "raw_answer_tokens": _distribution_summary(answer_lengths),
    }


def _bucket_counts(values: Sequence[int]) -> tuple[list[str], list[int]]:
    """Count fixed token-length buckets through 32K and overflow."""
    boundaries = (1024, 2048, 4096, 8192, 16384, 24576, 32768)
    labels = ("≤1K", "1–2K", "2–4K", "4–8K", "8–16K", "16–24K", "24–32K", ">32K")
    counts = [0] * len(labels)
    for value in values:
        bucket_index = 0
        while bucket_index < len(boundaries) and value > boundaries[bucket_index]:
            bucket_index += 1
        counts[bucket_index] += 1
    return list(labels), counts


def plot_report(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_path: Path) -> None:
    """Write the four-panel length report to a PNG file."""
    # Matplotlib is optional for core training and only required by this plotting script.
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # pylint: disable=C0415
        from matplotlib.figure import Figure  # pylint: disable=C0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Generating decif_length_distribution.png requires matplotlib; "
            "install it with 'python -m pip install matplotlib'."
        ) from exc

    template_lengths = [row["template_tokens_before_truncation"] for row in rows]
    train_lengths = [row["train_tokens_after_truncation"] for row in rows]
    supervised_lengths = [row["supervised_tokens_after_truncation"] for row in rows]
    max_seq_len = int(summary["max_seq_len"])

    figure = Figure(figsize=(15, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)

    histogram_bins = min(max(round(math.sqrt(len(rows))), 30), 120)
    axes[0, 0].hist(template_lengths, bins=histogram_bins, color="#4C78A8", alpha=0.85)
    axes[0, 0].axvline(max_seq_len, color="#E45756", linestyle="--", linewidth=2, label=f"max_seq_len={max_seq_len}")
    axes[0, 0].set_title("Template tokens before truncation")
    axes[0, 0].set_xlabel("Tokens")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].legend()

    axes[0, 1].hist(train_lengths, bins=histogram_bins, color="#72B7B2", alpha=0.7, label="Train tokens")
    axes[0, 1].hist(
        supervised_lengths,
        bins=histogram_bins,
        color="#F58518",
        alpha=0.65,
        label="Supervised tokens",
    )
    axes[0, 1].set_title("Actual train length and CE-supervised length")
    axes[0, 1].set_xlabel("Tokens")
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].legend()

    scatter_stride = max(len(rows) // 20000, 1)
    scatter_train = train_lengths[::scatter_stride]
    scatter_supervised = supervised_lengths[::scatter_stride]
    axes[1, 0].scatter(scatter_train, scatter_supervised, s=8, alpha=0.22, color="#54A24B", edgecolors="none")
    diagonal_max = max(train_lengths, default=1)
    axes[1, 0].plot((0, diagonal_max), (0, diagonal_max), color="#9D9D9D", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Supervised tokens per training sequence")
    axes[1, 0].set_xlabel("Train tokens after truncation")
    axes[1, 0].set_ylabel("Supervised tokens after truncation")

    bucket_labels, bucket_values = _bucket_counts(template_lengths)
    bars = axes[1, 1].bar(bucket_labels, bucket_values, color="#B279A2")
    axes[1, 1].set_title("Template-length buckets")
    axes[1, 1].set_xlabel("Tokens before truncation")
    axes[1, 1].set_ylabel("Samples")
    axes[1, 1].tick_params(axis="x", rotation=25)
    for bar, count in zip(bars, bucket_values):
        if count:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}\n({count / len(rows):.1%})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Decif length distribution | samples={len(rows):,} | truncated={summary['truncated_sample_ratio']:.2%}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)


def main() -> None:
    """Analyze Decif records and write the PNG, CSV, and JSON reports."""
    args = parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    args.tokenizer_path = args.tokenizer_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    _validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    chat_template = build_chat_template(
        "tokenizer",
        tokenizer,
        chat_template_kwargs={
            "enable_thinking": not args.disable_thinking,
            "reasoning_effort": args.reasoning_effort,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "decif_length_samples.csv"
    rows = []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record_count, (line_number, record) in enumerate(_iter_jsonl(args.data_path, args.limit), start=1):
            try:
                row = analyze_record(tokenizer, chat_template, record, line_number, args.max_seq_len)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Failed to analyze Decif record at line {line_number}: {exc}") from exc
            writer.writerow(row)
            rows.append(row)
            if args.progress_every and record_count % args.progress_every == 0:
                print(f"Analyzed {record_count:,} records")

    summary = build_summary(rows, args)
    summary_path = args.output_dir / "decif_length_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "decif_length_distribution.png"
    plot_report(rows, summary, figure_path)

    template_summary = summary["template_tokens_before_truncation"]
    supervised_summary = summary["supervised_tokens_after_truncation"]
    print(f"Wrote figure: {figure_path}")
    print(f"Wrote per-sample data: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Template tokens: "
        f"p50={template_summary['p50']:.0f}, p95={template_summary['p95']:.0f}, "
        f"p99={template_summary['p99']:.0f}, max={template_summary['max']:.0f}"
    )
    print(
        "Supervised tokens: "
        f"p50={supervised_summary['p50']:.0f}, p95={supervised_summary['p95']:.0f}, "
        f"p99={supervised_summary['p99']:.0f}, max={supervised_summary['max']:.0f}"
    )
    print(
        f"Truncated samples: {summary['truncated_sample_count']:,}/{summary['sample_count']:,} "
        f"({summary['truncated_sample_ratio']:.2%})"
    )


if __name__ == "__main__":
    main()
