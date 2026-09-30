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
"""Prepare a deterministic HotpotQA Distractor subset for the Search MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SOURCE = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet?download=true"
)


def _builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _builtin(value.tolist())
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("HotpotQA JSON must contain a top-level sample list")
    return pd.DataFrame(value)


def _download(url: str, destination: Path) -> None:
    """Download atomically so an interrupted transfer is never reused as data."""
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def _convert(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert HotpotQA rows to the stable search dataset schema."""
    required = {"question", "answer", "context", "supporting_facts"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"HotpotQA input is missing columns: {missing}")
    rows = []
    for index, row in frame.iterrows():
        rows.append({
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip(),
            "question_id": str(row.get("id", row.get("_id", index))),
            "question_type": str(row.get("type", "")),
            "question_level": str(row.get("level", "")),
            "context_json": json.dumps(_builtin(row["context"]), ensure_ascii=False),
            "supporting_facts_json": json.dumps(
                _builtin(row["supporting_facts"]), ensure_ascii=False
            ),
        })
    return pd.DataFrame(rows)


def split_samples(frame: pd.DataFrame, train_count: int, test_count: int, seed: int,
                  heldout: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve previous holdouts and reject ambiguous IDs before expanding training."""
    _validate_source(frame)
    if frame.question.duplicated().any():
        raise ValueError("Duplicate question IDs or question text in source")
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if heldout is None:
        heldout = frame.iloc[:0]
    if heldout.question_id.duplicated().any() or not set(heldout.question_id).issubset(set(frame.question_id)):
        raise ValueError("Previous holdout must have unique IDs present in source")
    if len(heldout) > test_count or train_count <= 0 or test_count <= 0:
        raise ValueError("Invalid split sizes")
    reserved = frame[frame.question_id.isin(heldout.question_id)]
    remaining = shuffled[~shuffled.question_id.isin(reserved.question_id)]
    extra = test_count - len(reserved)
    if len(remaining) < extra + train_count:
        raise ValueError("Not enough disjoint questions for requested split")
    test = pd.concat([reserved, remaining.iloc[:extra]], ignore_index=True)
    train = remaining.iloc[extra:extra + train_count].reset_index(drop=True)
    return train, test


def main() -> None:
    """Download or convert one source, then create disjoint train and test files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--additional-input", type=Path, action="append", default=[],
                        help="Additional shards of the same training source")
    parser.add_argument("--validation-input", type=Path,
                        help="Independent official dev source; never used for training")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--preserve-test", type=Path, help="Keep all previous holdout IDs out of training")
    args = parser.parse_args()
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train-samples and test-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("train.parquet", "test.parquet", "manifest.json")):
        raise FileExistsError("Use a new output directory; existing datasets are never overwritten")
    source = args.input
    if source is None:
        source = args.output_dir / "hotpotqa-distractor-validation-source.parquet"
        try:
            source_frame = _load(source) if source.is_file() else None
        except (OSError, ValueError):
            source.unlink()
            source_frame = None
        if source_frame is None:
            _download(args.source_url, source)
            source_frame = _load(source)
    else:
        source_frame = _load(source)
    frame = _convert(pd.concat([source_frame, *[_load(path) for path in args.additional_input]],
                               ignore_index=True))
    heldout = pd.read_parquet(args.preserve_test) if args.preserve_test else None
    if args.validation_input:
        validation = _convert(_load(args.validation_input))
        train, test = independent_splits(frame, validation, args.train_samples, args.test_samples,
                                         args.seed, heldout)
    else:
        train, test = split_samples(frame, args.train_samples, args.test_samples, args.seed, heldout)
    for name, split in (("train", train), ("test", test)):
        split.to_parquet(args.output_dir / f"{name}.parquet", index=False)
    manifest = {
        "split_protocol": "independent_sources" if args.validation_input else "single_source_holdout",
        "additional_sources": [{"path": str(path.resolve()),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                               for path in args.additional_input],
        "validation_source": ({"path": str(args.validation_input.resolve()),
                               "sha256": hashlib.sha256(args.validation_input.read_bytes()).hexdigest()}
                              if args.validation_input else None),
        "source": str(source.resolve()), "source_url": args.source_url if args.input is None else None,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_rows": len(frame),
        "seed": args.seed, "train_samples": len(train), "test_samples": len(test),
        "preserved_test": str(args.preserve_test) if args.preserve_test else None,
        "split_overlap": len(set(train.question_id) & set(test.question_id)),
        "sha256": {name: hashlib.sha256((args.output_dir / f"{name}.parquet").read_bytes()).hexdigest()
                   for name in ("train", "test")},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared train={args.train_samples} test={args.test_samples} at {args.output_dir}")


def _validate_source(frame: pd.DataFrame) -> None:
    """Reject ambiguous or empty records before selecting a split."""
    if frame.question_id.duplicated().any():
        raise ValueError("Duplicate question IDs in source")
    fields = frame[["question", "answer", "question_id"]]
    if fields.isna().any().any() or fields.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Empty question, answer or ID")


def independent_splits(train_source: pd.DataFrame, validation_source: pd.DataFrame,
                       train_count: int, test_count: int, seed: int,
                       heldout: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select fixed holdouts from dev and reject cross-source ID/question leakage."""
    for frame in (train_source, validation_source):
        _validate_source(frame)
    def normalize(values):
        """Normalize question text before checking split overlap."""
        return set(values.str.casefold().str.split().str.join(" "))
    if (set(train_source.question_id) & set(validation_source.question_id)
            or normalize(train_source.question) & normalize(validation_source.question)):
        raise ValueError("Training and validation sources overlap")
    if not 0 < train_count <= len(train_source) or not 0 < test_count <= len(validation_source):
        raise ValueError("Invalid independent split sizes")
    heldout = validation_source.iloc[:0] if heldout is None else heldout
    ids = set(heldout.question_id)
    if heldout.question_id.duplicated().any() or len(heldout) > test_count or not ids.issubset(
        set(validation_source.question_id)
    ):
        raise ValueError("Previous holdout must belong to validation source")
    reserved = validation_source.set_index("question_id", drop=False).loc[list(heldout.question_id)]
    remaining = validation_source[~validation_source.question_id.isin(ids)]
    test = pd.concat([reserved, remaining.sample(frac=1, random_state=seed).iloc[:test_count - len(reserved)]])
    train = train_source.sample(frac=1, random_state=seed).iloc[:train_count]
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    main()
