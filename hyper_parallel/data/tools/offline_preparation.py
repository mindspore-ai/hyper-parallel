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
# modified from https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py

"""Processing large data for pretraining."""

import argparse
import glob
import gzip
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars

    NLTK_AVAILABLE = True
except ImportError:
    PunktLanguageVars = object
    NLTK_AVAILABLE = False

# Store generated samples in the indexed ``.bin/.idx`` format.
from hyper_parallel.data.indexed import io as indexed_dataset


class CustomLanguageVars(PunktLanguageVars):
    """Preserve newline runs when Punkt detects sentence boundaries."""

    _period_context_fmt = r"""
        \S*
        %(SentEndChars)s
        \s*
        (?=(?P<after_tok>
            %(NonWord)s
            |
            (?P<next_tok>\S+)
        ))"""


def build_tokenizer(args: argparse.Namespace) -> Any:
    """Build the tokenizer configured for offline preprocessing."""
    warnings.warn(
        "Only Hugging Face tokenizers loaded through AutoTokenizer are currently supported; custom tokenizer "
        "classes are not supported.",
        UserWarning,
        stacklevel=2,
    )
    return build_huggingface_tokenizer(args)


def build_huggingface_tokenizer(args: argparse.Namespace) -> Any:
    """Build a Hugging Face tokenizer from command-line arguments."""
    tokenizer_path = args.tokenizer_name_or_path
    if not tokenizer_path:
        raise ValueError("tokenizer_name_or_path must be provided for HuggingFaceTokenizer")

    tokenizer_kwargs = {
        "use_fast": args.tokenizer_use_fast,
        "trust_remote_code": args.trust_remote_code,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        **tokenizer_kwargs,
    )

    if args.chat_template:
        tokenizer.chat_template = args.chat_template

    if args.add_special_tokens:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": args.add_special_tokens,
            }
        )

    return tokenizer


def append_eod(args: argparse.Namespace, tokenizer: Any) -> int | None:
    """Resolve the document-ending token requested for preprocessing."""
    if not args.append_eod:
        return None
    eod_id = getattr(tokenizer, "eos_token_id", None)
    if eod_id is None:
        # Prefer EOS as the document boundary and fall back to SEP for tokenizers
        # that do not define an EOS token.
        eod_id = getattr(tokenizer, "sep_token_id", None)
    if eod_id is None:
        raise ValueError("append_eod requires the tokenizer to define eos_token_id or sep_token_id")
    return int(eod_id)


class Encoder:
    """Encode JSON text fields into token ids and sentence lengths."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store the preprocessing arguments shared with worker processes.

        Args:
            args: Parsed offline preprocessing arguments.
        """
        self.args = args

    def initializer(self) -> None:
        """Build the tokenizer and sentence splitter in each worker process."""
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not NLTK_AVAILABLE:
                raise ImportError("NLTK is required when --split-sentences is enabled")
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(
                    os.environ["NLTK_DATA"],
                    "tokenizers",
                    "punkt",
                    f"{self.args.lang}.pickle",
                )
                splitter = nltk.load(f"file:{library}")
            else:
                splitter = nltk.load(f"tokenizers/punkt/{self.args.lang}.pickle")
            if self.args.keep_newlines:
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,  # pylint: disable=protected-access
                    lang_vars=CustomLanguageVars(),
                )
            else:
                Encoder.splitter = splitter

    def split(self, json_line: str) -> tuple[str, int]:
        """Split configured JSON text fields into sentence lists."""
        data = json.loads(json_line)
        output = {}
        max_chunk_length = 1_000_000
        for key in self.args.json_keys:
            text = data[key]
            sentences_by_chunk = [
                Encoder.splitter.tokenize(text[start : start + max_chunk_length])
                for start in range(0, len(text), max_chunk_length)
            ]
            output[key] = [sentence for chunk in sentences_by_chunk for sentence in chunk]
        return json.dumps(output), len(json_line)

    def encode(
        self,
        json_line: str,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]], int]:
        """Tokenize one JSON record into per-key token ids and sentence lengths.

        Args:
            json_line: One serialized JSON record.

        Returns:
            Per-key token ids, per-key sentence lengths, and the input byte length.
        """
        data = json.loads(json_line)
        ids = {}
        lens = {}
        keys = self.args.json_keys
        for key in keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.encode(sentence, add_special_tokens=False)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            eod = append_eod(self.args, Encoder.tokenizer)
            if len(doc_ids) > 0 and eod is not None:
                doc_ids.append(eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition:
    """Process one dataset partition with a pool of worker processes."""

    def __init__(self, args: argparse.Namespace, workers: int) -> None:
        """Store the partition configuration.

        Args:
            args: Parsed offline preprocessing arguments.
            workers: Number of worker processes used by this partition.
        """
        self.args = args
        self.workers = workers
        self.performance: list[float] = []

    def print_processing_stats(
        self,
        count: int,
        proc_start: float,
        total_bytes_processed: int,
    ) -> None:
        """Report encoding throughput at the configured log interval.

        Args:
            count: Number of documents processed so far.
            proc_start: Timestamp when processing started.
            total_bytes_processed: Total input bytes processed so far.
        """
        if count % self.args.log_interval != 0:
            return
        elapsed = time.time() - proc_start
        docs_per_second = count / elapsed
        megabytes_per_second = total_bytes_processed / elapsed / 1024 / 1024
        print(
            f"Processed {count} documents " f"({docs_per_second} docs/s, {megabytes_per_second} MB/s).",
            file=sys.stderr,
        )
        if self.args.find_optimal_num_workers:
            self.performance.append(docs_per_second)

    def split_sentences(self, file_name: tuple[str, str]) -> None:
        """Split every document in one JSONL partition into sentences."""
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        encoder = Encoder(self.args)
        with open(input_file_name, "r", encoding="utf-8") as input_file, open(
            output_file_name,
            "w",
            encoding="utf-8",
        ) as output_file:
            # multiprocessing.Pool must be shut down with close()+join() so
            # in-flight tasks finish; a 'with' block would terminate() them.
            pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)  # pylint: disable=R1732
            try:
                split_docs = pool.imap(encoder.split, input_file, 32)
                proc_start = time.time()
                total_bytes_processed = 0
                for count, (document, bytes_processed) in enumerate(split_docs, start=1):
                    total_bytes_processed += bytes_processed
                    output_file.write(document + "\n")
                    self.print_processing_stats(count, proc_start, total_bytes_processed)
            finally:
                pool.close()
                pool.join()

    def process_json_file(self, file_name: tuple[str, str]) -> list[float]:
        """Tokenize one JSONL partition into indexed dataset .bin/.idx files.

        Args:
            file_name: Input JSONL path and output prefix pair.

        Returns:
            Throughput measurements collected while benchmarking workers.
        """
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        # multiprocessing.Pool must be shut down with close()+join() so
        # in-flight tasks finish; a 'with' block would terminate() them.
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)  # pylint: disable=R1732

        level = "sentence" if self.args.split_sentences else "document"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        keys = self.args.json_keys
        for key in keys:
            output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(len(tokenizer)),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        pack_to_seq_len = getattr(self.args, "pack_to_seq_len", None)
        chunk_size = pack_to_seq_len + 1 if pack_to_seq_len is not None else None
        token_buffers = {key: [] for key in keys}
        with open(input_file_name, "r", encoding="utf-8") as fin:
            encoded_docs = pool.imap(encoder.encode, fin, 32)
            for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
                if self.args.find_optimal_num_workers and i > self.args.max_documents:
                    break
                total_bytes_processed += bytes_processed
                for key in keys:
                    if chunk_size is None:
                        builders[key].add_document(doc[key], sentence_lens[key])
                        continue
                    token_buffers[key].extend(doc[key])
                    complete_length = len(token_buffers[key]) // chunk_size * chunk_size
                    for offset in range(0, complete_length, chunk_size):
                        chunk = token_buffers[key][offset : offset + chunk_size]
                        builders[key].add_document(chunk, [chunk_size])
                    del token_buffers[key][:complete_length]
                self.print_processing_stats(i, proc_start, total_bytes_processed)

        keys = self.args.json_keys
        for key in keys:
            builders[key].finalize(output_idx_files[key])

        pool.close()
        pool.join()

        return self.performance


def _parse_bool(value: str) -> bool:
    """Parse an explicit command-line boolean value."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, but received {value!r}")


def _add_offline_source_arguments(group: Any) -> None:
    """Register source, tokenizer, and tokenization arguments."""
    arguments = (
        (("--dataset-name-or-path",), {"required": True, "help": "Input JSON/JSONL file, directory, or glob."}),
        (("--output-prefix",), {"required": True, "help": "Path prefix for generated .bin/.idx files."}),
        (("--json-keys",), {"nargs": "+", "default": ["text"], "help": "JSON fields to tokenize."}),
        (("--tokenizer-name-or-path",), {"required": True, "help": "Tokenizer name or local path."}),
        (("--chat-template",), {"default": None, "help": "Optional tokenizer chat template."}),
        (("--add-special-tokens",), {"nargs": "+", "default": None, "help": "Additional special tokens."}),
        (("--tokenizer-use-fast",), {"type": _parse_bool, "default": True, "help": "Use the fast tokenizer."}),
        (("--trust-remote-code",), {"action": "store_true", "help": "Allow tokenizer remote code."}),
        (("--split-sentences",), {"action": "store_true", "help": "Split text into sentences."}),
        (("--keep-newlines",), {"action": "store_true", "help": "Preserve newlines while splitting."}),
        (("--lang",), {"default": "english", "help": "Punkt language for sentence splitting."}),
        (("--append-eod",), {"type": _parse_bool, "default": True, "help": "Append an EOD token."}),
        (("--pack-to-seq-len",), {"type": int, "default": None, "help": "Fixed packed sequence length."}),
    )
    for flags, options in arguments:
        group.add_argument(*flags, **options)


def _add_offline_worker_arguments(group: Any) -> None:
    """Register partitioning and worker arguments."""
    arguments = (
        (("--keep-sequential-samples",), {"action": "store_true", "help": "Keep sequential samples."}),
        (("--keep-partition-files",), {"action": "store_true", "help": "Keep temporary partition files."}),
        (("--workers",), {"type": int, "default": 8, "help": "Number of worker processes."}),
        (("--partitions",), {"type": int, "default": 1, "help": "Number of data partitions."}),
        (("--find-optimal-num-workers",), {"action": "store_true", "help": "Benchmark worker counts."}),
        (
            ("--workers-to-check",),
            {"nargs": "+", "type": int, "default": [16, 32, 64], "help": "Candidate worker counts."},
        ),
        (("--max-documents",), {"type": int, "default": 100_000, "help": "Benchmark document limit."}),
        (("--log-interval",), {"type": int, "default": 1000, "help": "Progress-report interval."}),
    )
    for flags, options in arguments:
        group.add_argument(*flags, **options)


def get_args() -> argparse.Namespace:
    """Parse offline dataset preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare an offline dataset.")
    group = parser.add_argument_group(title="input data")
    _add_offline_source_arguments(group)
    _add_offline_worker_arguments(group)
    return parser.parse_args()


def get_file_name(args: argparse.Namespace, file_id: int) -> dict[str, str]:
    """Build input and output paths for one dataset partition.

    Args:
        args: Parsed preprocessing arguments.
        file_id: Zero-based partition identifier.

    Returns:
        Input, sentence-split, and output-prefix paths for the partition.
    """
    output_directory = os.path.dirname(args.output_prefix)
    output_name = os.path.basename(args.output_prefix)
    work_directory = os.path.join(output_directory, f".{output_name}_preprocess")
    input_directory = os.path.join(work_directory, f"partition{file_id}")
    partition_directory = os.path.join(output_directory, f"partition{file_id}")
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(partition_directory, exist_ok=True)
    input_file_name = os.path.join(input_directory, "input.jsonl")
    sentence_split_file = os.path.join(input_directory, "input_ss.jsonl")
    output_prefix = os.path.join(partition_directory, output_name)
    file_names = {"partition": input_file_name, "sentence_split": sentence_split_file, "output_prefix": output_prefix}
    return file_names


def check_files_exist(
    in_ss_out_names: list[dict[str, str]],
    key: str,
    num_partitions: int,
) -> bool:
    """Check whether every partition contains the requested file type."""
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def find_optimal_num_workers(
    performance: dict[int, list[float]],
    partitions: int,
) -> None:
    """Print candidate worker counts ordered by average throughput."""
    results = []
    for workers, measurements in performance.items():
        if measurements:
            results.append((workers, float(np.mean(measurements))))
    if not results:
        raise ValueError("No worker performance measurements were collected")
    results.sort(key=lambda item: item[1], reverse=True)
    print("\nWorker performance results:")
    for position, (workers, average_rate) in enumerate(results, start=1):
        print(f"{position}. {workers} workers: {average_rate:.4f} docs/s")
    best_workers, best_rate = results[0]
    print(
        f"Best configuration: {best_workers} total workers "
        f"({best_workers // partitions} per partition), {best_rate:.4f} docs/s."
    )


def _process_json_file_worker(
    args: argparse.Namespace,
    workers: int,
    name: dict[str, str],
    input_key: str,
    queue: Any,
) -> None:
    """Encode one partition and return performance or an error to the parent."""
    try:
        partition = Partition(args, workers)
        measurements = partition.process_json_file((name[input_key], name["output_prefix"]))
        queue.put((True, measurements))
    except Exception:  # pylint: disable=W0718
        queue.put((False, traceback.format_exc()))


def _split_sentences_worker(
    args: argparse.Namespace,
    workers: int,
    name: dict[str, str],
    queue: Any,
) -> None:
    """Split one partition and report completion or an error to the parent."""
    try:
        partition = Partition(args, workers)
        partition.split_sentences((name["partition"], name["sentence_split"]))
        queue.put((True, None))
    except Exception:  # pylint: disable=W0718
        queue.put((False, traceback.format_exc()))


def _wait_for_processes(processes: list[Any], queue: Any) -> list[Any]:
    """Collect child results, join processes, and surface child failures."""
    results = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    queue.close()
    queue.join_thread()
    errors = [payload for succeeded, payload in results if not succeeded]
    if errors:
        raise RuntimeError("Child preprocessing process failed:\n" + "\n".join(errors))
    return [payload for succeeded, payload in results if succeeded]


def _count_jsonl_lines(file_name: str) -> int:
    """Count records in a plain or gzip-compressed JSONL file."""
    open_file = gzip.open if file_name.endswith(".gz") else open
    open_kwargs = {"mode": "rt", "encoding": "utf-8"}
    with open_file(file_name, **open_kwargs) as input_file:
        return sum(1 for _ in input_file)


def _resolve_input_files(dataset_name_or_path: str) -> list[str]:
    """Resolve a JSON input file, directory, or glob into stable file order.

    Args:
        dataset_name_or_path: Input file, directory, or glob expression.

    Returns:
        Sorted input JSON/JSONL file paths.

    Raises:
        ValueError: If the input does not resolve to supported files.
    """
    input_path = Path(dataset_name_or_path).expanduser()
    supported_suffixes = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    if input_path.is_dir():
        input_files = [
            str(path.resolve())
            for path in input_path.iterdir()
            if path.is_file() and path.name.lower().endswith(supported_suffixes)
        ]
    elif input_path.is_file():
        input_files = [str(input_path.resolve())]
    else:
        input_files = [
            str(Path(file_name).resolve())
            for file_name in glob.glob(dataset_name_or_path)
            if Path(file_name).is_file() and file_name.lower().endswith(supported_suffixes)
        ]
    input_files.sort()
    if not input_files:
        raise ValueError(
            "dataset_name_or_path must be a JSON/JSONL file, a directory containing "
            f"JSON/JSONL files, or a matching glob; received {dataset_name_or_path!r}"
        )
    return input_files


def _merge_partition_outputs(
    args: argparse.Namespace,
    in_ss_out_names: list[dict[str, str]],
) -> None:
    """Merge all encoded partition datasets into final output files."""
    level = "sentence" if args.split_sentences else "document"
    tokenizer = build_tokenizer(args)
    for key in args.json_keys:
        output_bin_file = f"{args.output_prefix}_{key}_{level}.bin"
        output_idx_file = f"{args.output_prefix}_{key}_{level}.idx"
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin_file,
            dtype=indexed_dataset.DType.optimal_dtype(len(tokenizer)),
        )
        for name in in_ss_out_names:
            partition_prefix = f"{name['output_prefix']}_{key}_{level}"
            builder.add_index(partition_prefix)
        builder.finalize(output_idx_file)


def _cleanup_intermediate_files(
    args: argparse.Namespace,
    in_ss_out_names: list[dict[str, str]],
    input_files: list[str],
) -> None:
    """Remove generated JSON partition inputs after successful processing."""
    source_files = {str(Path(file_name).resolve()) for file_name in input_files}
    for name in in_ss_out_names:
        generated_files = [name["sentence_split"]]
        if str(Path(name["partition"]).resolve()) not in source_files:
            generated_files.append(name["partition"])
        for file_name in generated_files:
            if os.path.isfile(file_name):
                os.remove(file_name)

    output_path = Path(args.output_prefix).expanduser().resolve()
    work_directory = output_path.parent / f".{output_path.name}_preprocess"
    if work_directory.is_dir():
        for directory, _, _ in os.walk(work_directory, topdown=False):
            try:
                os.rmdir(directory)
            except OSError:
                pass


def _validate_preparation_args(args: argparse.Namespace) -> list[int]:
    """Validate preprocessing options and return worker candidates."""
    if args.partitions <= 0:
        raise ValueError("partitions must be greater than zero")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")
    if args.max_documents <= 0:
        raise ValueError("max_documents must be greater than zero")
    pack_to_seq_len = getattr(args, "pack_to_seq_len", None)
    if pack_to_seq_len is not None and pack_to_seq_len <= 0:
        raise ValueError("pack_to_seq_len must be greater than zero")
    if pack_to_seq_len is not None and not args.append_eod:
        warnings.warn(
            "pack_to_seq_len is enabled while append_eod is disabled; packed samples will not contain EOD tokens "
            "at original document boundaries.",
            UserWarning,
            stacklevel=2,
        )
    if not args.output_prefix:
        raise ValueError("output_prefix must be provided")
    if not args.tokenizer_name_or_path:
        raise ValueError("tokenizer_name_or_path must be provided")
    output_parent = Path(args.output_prefix).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    worker_candidates = list(args.workers_to_check) if args.find_optimal_num_workers else [args.workers]
    if not worker_candidates or any(workers <= 0 for workers in worker_candidates):
        raise ValueError("worker counts must be greater than zero")
    invalid_workers = [workers for workers in worker_candidates if workers % args.partitions != 0]
    if invalid_workers:
        raise ValueError(f"Worker counts {invalid_workers} must be divisible by partitions ({args.partitions})")

    if args.split_sentences and not NLTK_AVAILABLE:
        raise ImportError("NLTK is required when --split-sentences is enabled")
    return worker_candidates


def _partition_input_data(
    args: argparse.Namespace,
    input_files: list[str],
) -> list[dict[str, str]]:
    """Create or reuse input partitions and return their path descriptions."""
    if args.partitions == 1 and len(input_files) == 1:
        output_path = Path(args.output_prefix).expanduser().resolve()
        work_directory = output_path.parent / f".{output_path.name}_preprocess"
        work_directory.mkdir(parents=True, exist_ok=True)
        return [
            {
                "partition": input_files[0],
                "sentence_split": str(work_directory / "input_ss.jsonl"),
                "output_prefix": args.output_prefix,
            }
        ]

    partition_size = 1
    if args.keep_sequential_samples:
        total_sample_count = sum(_count_jsonl_lines(filename) for filename in input_files)
        if total_sample_count == 0:
            raise ValueError("Input JSON/JSONL files must contain at least one record")
        partition_size = math.ceil(total_sample_count / args.partitions)
    if args.partitions == 1:
        output_path = Path(args.output_prefix).expanduser().resolve()
        directory = output_path.parent / f".{output_path.name}_preprocess" / "partition0"
        directory.mkdir(parents=True, exist_ok=True)
        names = [
            {
                "partition": str(directory / "input.jsonl"),
                "sentence_split": str(directory / "input_ss.jsonl"),
                "output_prefix": args.output_prefix,
            }
        ]
    else:
        names = [get_file_name(args, file_id) for file_id in range(args.partitions)]
    keep_files = getattr(args, "keep_partition_files", False)
    partitions_present = keep_files and check_files_exist(names, "partition", args.partitions)
    split_files_present = keep_files and check_files_exist(names, "sentence_split", args.partitions)
    if not partitions_present and not split_files_present:
        _write_input_partitions(args, names, input_files, partition_size)
    return names


def _write_input_partitions(
    args: argparse.Namespace,
    names: list[dict[str, str]],
    input_files: list[str],
    partition_size: int | None,
) -> None:
    """Distribute input records across partition files."""
    outputs = [open(name["partition"], "w", encoding="utf-8") for name in names]  # pylint: disable=R1732
    try:
        partition_index = 0
        line_count = 0
        for input_file_name in input_files:
            open_file = gzip.open if input_file_name.endswith(".gz") else open
            with open_file(input_file_name, "rt", encoding="utf-8") as input_file:
                for line in input_file:
                    outputs[partition_index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0 and partition_index < args.partitions - 1:
                            partition_index += 1
                    else:
                        partition_index = (partition_index + 1) % args.partitions
    finally:
        for output in outputs:
            output.close()


def _split_partitions(args: argparse.Namespace, workers: int, names: list[dict[str, str]]) -> None:
    """Run sentence splitting for partitions that lack cached outputs."""
    split_files_present = getattr(args, "keep_partition_files", False) and check_files_exist(
        names, "sentence_split", args.partitions
    )
    if not args.split_sentences or split_files_present:
        return
    queue = multiprocessing.Queue()
    processes = []
    for name in names:
        process = multiprocessing.Process(target=_split_sentences_worker, args=(args, workers, name, queue))
        process.start()
        processes.append(process)
    _wait_for_processes(processes, queue)


def _encode_partitions(
    args: argparse.Namespace,
    workers: int,
    names: list[dict[str, str]],
) -> list[float]:
    """Encode every partition and flatten its performance measurements."""
    input_key = "sentence_split" if args.split_sentences else "partition"
    queue = multiprocessing.Queue()
    processes = []
    for name in names:
        process = multiprocessing.Process(
            target=_process_json_file_worker,
            args=(args, workers, name, input_key, queue),
        )
        process.start()
        processes.append(process)
    results = _wait_for_processes(processes, queue)
    return [measurement for measurements in results for measurement in (measurements or [])]


def prepare_offline_dataset(args: argparse.Namespace) -> None:
    """Prepare an indexed offline dataset from a parsed configuration.

    Args:
        args: Offline dataset preprocessing configuration.
    """
    worker_candidates = _validate_preparation_args(args)

    performance = {}
    input_files = _resolve_input_files(args.dataset_name_or_path)
    for workers in worker_candidates:
        print(f"Processing data with {workers} workers.")
        workers_per_partition = workers // args.partitions

        if args.split_sentences:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        in_ss_out_names = _partition_input_data(args, input_files)
        _split_partitions(args, workers_per_partition, in_ss_out_names)
        performance[workers] = _encode_partitions(args, workers_per_partition, in_ss_out_names)

        if args.partitions > 1:
            _merge_partition_outputs(args, in_ss_out_names)
        if not getattr(args, "keep_partition_files", False):
            _cleanup_intermediate_files(args, in_ss_out_names, input_files)

    if args.find_optimal_num_workers:
        find_optimal_num_workers(performance, args.partitions)


def main() -> None:
    """Parse command-line arguments and prepare the offline dataset."""
    prepare_offline_dataset(get_args())


if __name__ == "__main__":
    main()
