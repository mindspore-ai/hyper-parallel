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
"""Validate Qwen3.5 batch-one fresh-prefill alignment on Ascend NPU."""

# Select Torch and alignment before importing HyperParallel or vLLM.
# pylint: disable=wrong-import-position
import argparse
import gc
from functools import wraps
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["HYPER_VLLM_ALIGNMENT"] = "true"
os.environ["VLLM_BATCH_INVARIANT"] = "0"

import pandas as pd
import torch
from transformers import AutoTokenizer
import vllm as vllm_package
from vllm import LLM, SamplingParams
from vllm.forward_context import get_forward_context
from vllm.inputs import TokensPrompt
import vllm_ascend as vllm_ascend_package
from vllm_ascend.ops import gdn as vllm_ascend_gdn

from rl.roles.rollout.vllm_plugin import register_hyper_models

from hyper_parallel.models.qwen3_5 import Qwen3_5ForCausalLM, _build_config
from hyper_parallel.models.qwen3_5.state_dict import Qwen3_5StateDictAdapter


_HYPER_ARCHITECTURE = "HyperQwen3_5ForCausalLM"
_SUPPORTED_TRANSFORMERS_VERSION = "5.5.4"
_SUPPORTED_VLLM_VERSION = "0.22.1"
_SUPPORTED_VLLM_ASCEND_VERSION = "0.22.1rc1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gsm8k-path", type=Path, required=True)
    parser.add_argument("--gsm8k-index", type=int, default=170)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--capture-settle-seconds", type=float, default=10.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-mode",
        choices=("reference", "vllm"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--capture-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: Path) -> str:
    files = [model_path / "config.json"]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        files.append(index_path)
    weight_files = sorted(model_path.glob("*.safetensors"))
    files.extend(weight_files)
    if not weight_files or any(not path.is_file() for path in files):
        raise ValueError(f"Incomplete model fingerprint inputs under {model_path}")
    digest = sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _validate_dependency_versions() -> None:
    versions = {
        "transformers": package_version("transformers").split("+", maxsplit=1)[0],
        "vllm": package_version("vllm").split("+", maxsplit=1)[0],
        "vllm-ascend": package_version("vllm-ascend").split("+", maxsplit=1)[0],
    }
    expected = {
        "transformers": _SUPPORTED_TRANSFORMERS_VERSION,
        "vllm": _SUPPORTED_VLLM_VERSION,
        "vllm-ascend": _SUPPORTED_VLLM_ASCEND_VERSION,
    }
    if versions != expected:
        raise ValueError(f"Alignment gate requires dependency versions {expected}, got {versions}")


def _prompt_ids(args: argparse.Namespace) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    frame = pd.read_parquet(args.gsm8k_path)
    matches = [
        row
        for row in frame.itertuples(index=False)
        if int(row.extra_info["index"]) == args.gsm8k_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one GSM8K row for index {args.gsm8k_index}, got {len(matches)}"
        )
    row = matches[0]
    messages = [dict(message) for message in row.prompt.tolist()]
    prompt_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row.extra_info["answer"]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]
    if not prompt_ids:
        raise ValueError("Alignment validation prompt must contain at least one token")
    return list(prompt_ids)


def _model_config(model_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(weights_path=model_path, config_overrides={})
    )


def _token_hidden(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError(f"Alignment gate requires batch size 1, got {output.shape}")
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Expected token-hidden output, got shape {output.shape}")
    return output.detach()


def _capture_hook(captures: dict[str, torch.Tensor], name: str):
    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        """Capture one module output for exact comparison."""
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"{name} returned non-Tensor output {type(output)}")
        captures[name] = _token_hidden(output)

    return hook


def _install_layer_hooks(model: Any, captures: dict[str, torch.Tensor]) -> list[Any]:
    handles = [
        model.model.embed_tokens.register_forward_hook(
            _capture_hook(captures, "embed_tokens")
        ),
        model.model.norm.register_forward_hook(_capture_hook(captures, "norm")),
    ]
    handles.extend(
        layer.register_forward_hook(_capture_hook(captures, f"layer.{layer_index}"))
        for layer_index, layer in enumerate(model.model.layers)
    )
    return handles


def _capture_reference(
    model_path: str,
    prompt_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    config = _build_config(_model_config(model_path))
    model = Qwen3_5ForCausalLM(config)
    state_dict = Qwen3_5StateDictAdapter.load_hf_state_dict(
        model_path,
        config,
        dtype=torch.bfloat16,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    del state_dict

    captures: dict[str, torch.Tensor] = {}
    handles = _install_layer_hooks(model, captures)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_ids), dtype=torch.long, device=device)
    positions = positions.view(1, 1, -1).expand(3, 1, -1)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, position_ids=positions)["logits"][0]
    for handle in handles:
        handle.remove()
    payload = {
        "prompt_ids": prompt_ids,
        "captures": {name: tensor.cpu() for name, tensor in captures.items()},
        "logits": logits.detach().cpu(),
        "num_hidden_layers": len(model.model.layers),
    }
    del model, input_ids, positions, logits
    gc.collect()
    torch.npu.empty_cache()
    return payload


def _capture_vllm(
    args: argparse.Namespace,
    prompt_ids: list[int],
) -> dict[str, Any]:
    register_hyper_models()
    llm = LLM(
        model=args.model_path,
        hf_overrides={"architectures": [_HYPER_ARCHITECTURE]},
        dtype="bfloat16",
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        generation_config="vllm",
        disable_log_stats=True,
    )
    if llm.llm_engine.model_config.architecture != _HYPER_ARCHITECTURE:
        raise ValueError("Alignment gate did not load the Hyper Qwen3.5 adapter")

    def install_hooks(model: Any) -> None:
        """Install fresh-prefill evidence and tensor capture hooks."""
        state: dict[str, Any] = {
            "captures": {},
            "cache_evidence": {
                "conv_initial_state_modes": [],
                "gdn_initial_state_nonzero": [],
                "gdn_cu_seqlens": [],
                "forward_count": 0,
            },
            "runner_logits": [],
            "handles": [],
            "original_conv_args": vllm_ascend_gdn.get_non_spec_causal_conv1d_host_args,
            "original_recurrence": vllm_ascend_gdn.torch_gdn_recurrence,
        }
        model._hyper_alignment_capture_state = state  # pylint: disable=protected-access
        captures = state["captures"]
        cache_evidence = state["cache_evidence"]

        def metadata_hook(_module: Any, _inputs: Any) -> None:
            """Record attention metadata proving one complete fresh prefill."""
            cache_evidence["forward_count"] += 1
            if cache_evidence["forward_count"] != 1:
                raise ValueError("Alignment gate executed more than one model forward")
            grouped_metadata = get_forward_context().attn_metadata
            metadata_items = (
                grouped_metadata.items()
                if isinstance(grouped_metadata, dict)
                else (("default", grouped_metadata),)
            )
            records = []
            for name, metadata in metadata_items:
                required_fields = (
                    "num_prefills",
                    "num_decodes",
                    "num_decode_tokens",
                    "actual_seq_lengths_q",
                    "seq_lens_list",
                )
                if not all(hasattr(metadata, field) for field in required_fields):
                    continue
                records.append(
                    {
                        "name": str(name),
                        "num_prefills": int(metadata.num_prefills),
                        "num_decodes": int(metadata.num_decodes),
                        "num_decode_tokens": int(metadata.num_decode_tokens),
                        "actual_seq_lengths_q": [
                            int(value) for value in metadata.actual_seq_lengths_q
                        ],
                        "seq_lens_list": [int(value) for value in metadata.seq_lens_list],
                    }
                )
            cache_evidence["attention_metadata"] = records

        def position_hook(_module: Any, inputs: Any) -> None:
            """Capture full-attention positions and enforce cross-layer consistency."""
            position_ids = inputs[1].detach()
            if "positions" in captures and not torch.equal(captures["positions"], position_ids):
                raise ValueError("Qwen3.5 full-attention layers received different positions")
            captures["positions"] = position_ids

        state["handles"].append(
            model.model.layers[0].register_forward_pre_hook(metadata_hook)
        )
        state["handles"].append(
            model.model.rotary_emb.register_forward_pre_hook(position_hook)
        )
        state["handles"].extend(_install_layer_hooks(model, captures))
        state["handles"].append(
            model.lm_head.register_forward_hook(
                lambda _module, _inputs, output: state["runner_logits"].append(output.detach())
            )
        )

        @wraps(state["original_conv_args"])
        def capture_conv_args(metadata: Any) -> Any:
            """Record whether causal convolution reads historical state."""
            result = state["original_conv_args"](metadata)
            cache_evidence["conv_initial_state_modes"].append(
                [int(value) for value in result[2]]
            )
            return result

        @wraps(state["original_recurrence"])
        def capture_recurrence(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            cu_seqlens_host: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Record GDN state and request boundaries before recurrence."""
            cache_evidence["gdn_initial_state_nonzero"].append(
                int(torch.count_nonzero(initial_state))
            )
            cache_evidence["gdn_cu_seqlens"].append(
                [int(value) for value in cu_seqlens_host]
            )
            return state["original_recurrence"](
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                cu_seqlens_host,
            )

        vllm_ascend_gdn.get_non_spec_causal_conv1d_host_args = capture_conv_args
        vllm_ascend_gdn.torch_gdn_recurrence = capture_recurrence

    def cleanup_hooks(model: Any) -> None:
        """Restore patched functions and remove model hooks."""
        state = model._hyper_alignment_capture_state  # pylint: disable=protected-access
        vllm_ascend_gdn.get_non_spec_causal_conv1d_host_args = state["original_conv_args"]
        vllm_ascend_gdn.torch_gdn_recurrence = state["original_recurrence"]
        for handle in state["handles"]:
            handle.remove()

    def collect_capture(model: Any) -> dict[str, Any]:
        """Collect model tensors and evidence after the single forward."""
        state = model._hyper_alignment_capture_state  # pylint: disable=protected-access
        cleanup_hooks(model)
        captures = state["captures"]
        if len(state["runner_logits"]) != 1:
            raise ValueError(
                f"Expected one sampling logits tensor, got {len(state['runner_logits'])}"
            )
        with torch.inference_mode():
            full_logits = model.compute_logits(captures["norm"]).detach().cpu()
        payload = {
            "captures": {name: tensor.cpu() for name, tensor in captures.items()},
            "logits": full_logits,
            "runner_logits": state["runner_logits"][0].cpu(),
            "cache_evidence": state["cache_evidence"],
        }
        delattr(model, "_hyper_alignment_capture_state")
        return payload

    llm.apply_model(install_hooks)
    try:
        output = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
            use_tqdm=False,
        )[0]
    except Exception:
        llm.apply_model(cleanup_hooks)
        raise
    captured = llm.apply_model(collect_capture)[0]
    captures = captured["captures"]
    cache_evidence = captured["cache_evidence"]
    if list(output.prompt_token_ids) != prompt_ids:
        raise ValueError("vLLM executed different prompt token IDs")
    expected_positions = list(range(len(prompt_ids)))
    position_rows = captures["positions"].reshape(-1, len(prompt_ids)).tolist()
    if not position_rows or any(row != expected_positions for row in position_rows):
        raise ValueError("vLLM alignment gate read historical or chunked positions")
    if captures["norm"].shape[0] != len(prompt_ids):
        raise ValueError("vLLM did not execute the complete prompt in one fresh prefill")
    expected_boundary = [0, len(prompt_ids)]
    metadata_records = cache_evidence.get("attention_metadata", [])
    if cache_evidence.get("forward_count") != 1 or not metadata_records or any(
        record["num_prefills"] != 1
        or record["num_decodes"] != 0
        or record["num_decode_tokens"] != 0
        or record["actual_seq_lengths_q"] != [len(prompt_ids)]
        or record["seq_lens_list"] != [len(prompt_ids)]
        for record in metadata_records
    ):
        raise ValueError(f"vLLM did not execute one cache-free fresh prefill: {cache_evidence}")
    if not cache_evidence["conv_initial_state_modes"] or any(
        modes != [0] for modes in cache_evidence["conv_initial_state_modes"]
    ):
        raise ValueError(f"GDN convolution read historical state: {cache_evidence}")
    if not cache_evidence["gdn_initial_state_nonzero"] or any(
        cache_evidence["gdn_initial_state_nonzero"]
    ):
        raise ValueError(f"GDN recurrence received nonzero historical state: {cache_evidence}")
    if any(
        boundaries != expected_boundary for boundaries in cache_evidence["gdn_cu_seqlens"]
    ):
        raise ValueError(f"GDN recurrence used chunked request boundaries: {cache_evidence}")
    return {
        "prompt_ids": prompt_ids,
        "captures": captures,
        "logits": captured["logits"],
        "runner_logits": captured["runner_logits"],
        "generated_token": output.outputs[0].token_ids[0],
        "cache_evidence": cache_evidence,
    }


def _capture_to_file(args: argparse.Namespace) -> None:
    if args.capture_output is None:
        raise ValueError("--capture-output is required with --capture-mode")
    prompt_ids = _prompt_ids(args)
    if args.capture_mode == "reference":
        device = torch.device(args.device)
        torch.npu.set_device(device)
        payload = _capture_reference(args.model_path, prompt_ids, device)
    else:
        payload = _capture_vllm(args, prompt_ids)
    torch.save(payload, args.capture_output)


def _run_capture(args: argparse.Namespace, mode: str, output: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--model-path={args.model_path}",
        f"--gsm8k-path={args.gsm8k_path}",
        f"--gsm8k-index={args.gsm8k_index}",
        f"--device={args.device}",
        f"--max-model-len={args.max_model_len}",
        f"--capture-settle-seconds={args.capture_settle_seconds}",
        f"--gpu-memory-utilization={args.gpu_memory_utilization}",
        f"--kv-cache-memory-bytes={args.kv_cache_memory_bytes}",
        f"--output={args.output}",
        f"--capture-mode={mode}",
        f"--capture-output={output}",
    ]
    subprocess.run(command, check=True, shell=False)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "exact": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    difference = actual.float().sub(expected.float()).abs()
    return {
        "exact": torch.equal(actual, expected),
        "dtype": str(actual.dtype),
        "mismatch_count": int(torch.count_nonzero(actual != expected)),
        "max_abs_difference": float(difference.max()),
    }


def _compare(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="hyper-qwen3-5-alignment-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        vllm_path = temporary_path / "vllm.pt"
        reference_path = temporary_path / "reference.pt"
        _run_capture(args, "vllm", vllm_path)
        time.sleep(args.capture_settle_seconds)
        _run_capture(args, "reference", reference_path)
        vllm = torch.load(vllm_path, map_location="cpu", weights_only=True)
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)

    if vllm["prompt_ids"] != reference["prompt_ids"]:
        raise ValueError("Reference and vLLM prompt token IDs differ")
    num_hidden_layers = reference["num_hidden_layers"]
    capture_names = [
        "embed_tokens",
        *[f"layer.{layer_index}" for layer_index in range(num_hidden_layers)],
        "norm",
    ]
    capture_metrics = {
        name: _metrics(vllm["captures"][name], reference["captures"][name])
        for name in capture_names
    }
    first_divergence = next(
        (name for name in capture_names if not capture_metrics[name]["exact"]),
        None,
    )
    logits_metrics = _metrics(vllm["logits"], reference["logits"])
    vllm_log_probs = torch.log_softmax(vllm["logits"].float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference["logits"].float(), dim=-1)
    log_probs_metrics = _metrics(vllm_log_probs, reference_log_probs)
    runner_metrics = _metrics(vllm["runner_logits"][-1], vllm["logits"][-1])
    bf16_outputs = all(
        reference["captures"][name].dtype == torch.bfloat16
        and vllm["captures"][name].dtype == torch.bfloat16
        for name in capture_names
    ) and reference["logits"].dtype == vllm["logits"].dtype == torch.bfloat16
    fp32_log_probs = (
        reference_log_probs.dtype == vllm_log_probs.dtype == torch.float32
    )
    next_token = int(reference["logits"][-1].argmax())
    accepted = (
        first_divergence is None
        and logits_metrics["exact"]
        and log_probs_metrics["exact"]
        and runner_metrics["exact"]
        and bf16_outputs
        and fp32_log_probs
        and vllm["generated_token"] == next_token
    )
    prompt_ids = reference["prompt_ids"]
    report = {
        "accepted": accepted,
        "scope": "qwen3_5_tp1_bf16_batch1_fresh_prefill",
        "alignment_enabled": True,
        "batch_invariant_enabled": False,
        "prefix_caching_enabled": False,
        "chunked_prefill_enabled": False,
        "decode_forward_executed": False,
        "cache_evidence": vllm["cache_evidence"],
        "model_fingerprint": _model_fingerprint(Path(args.model_path)),
        "data_sha256": _file_sha256(args.gsm8k_path),
        "prompt_index": args.gsm8k_index,
        "prompt_length": len(prompt_ids),
        "prompt_ids_sha256": sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "torch_version": package_version("torch"),
        "torch_npu_version": package_version("torch-npu"),
        "transformers_version": package_version("transformers"),
        "vllm_version": package_version("vllm"),
        "vllm_ascend_version": package_version("vllm-ascend"),
        "vllm_module": str(Path(vllm_package.__file__).resolve()),
        "vllm_ascend_module": str(Path(vllm_ascend_package.__file__).resolve()),
        "bf16_outputs": bf16_outputs,
        "fp32_log_probs": fp32_log_probs,
        "first_hidden_divergence": first_divergence,
        "captures": capture_metrics,
        "full_logits": logits_metrics,
        "full_fp32_log_probs": log_probs_metrics,
        "runner_vs_reprojected_last_logits": runner_metrics,
        "reference_next_token": next_token,
        "vllm_generated_token": vllm["generated_token"],
    }
    _write_report(args.output, report)
    if not accepted:
        raise RuntimeError(f"Qwen3.5 fresh-prefill alignment is not bit-exact: {report}")


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output)


def main() -> None:
    """Run one capture subprocess or compare isolated reference captures."""
    args = _parse_args()
    if args.capture_mode is not None:
        _validate_dependency_versions()
        _capture_to_file(args)
        return
    if args.capture_output is not None:
        raise ValueError("--capture-output is valid only with --capture-mode")
    args.output.unlink(missing_ok=True)
    try:
        _validate_dependency_versions()
        _compare(args)
    except Exception as error:
        if not args.output.is_file():
            _write_report(
                args.output,
                {
                    "accepted": False,
                    "scope": "qwen3_5_tp1_bf16_batch1_fresh_prefill",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    main()
