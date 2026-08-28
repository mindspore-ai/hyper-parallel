# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Merge isolated per-SoC multicore vendors while retaining one host payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


_VENDOR_NAME = "hyper_parallel_multicore_nn"
_KERNEL_ROOT = Path("op_impl/ai_core/tbe/kernel")
_SOC_NAME_PATTERN = re.compile(r"^ascend[0-9a-z_]+$")
_HOST_ELF_PATHS = {
    "op_api/lib/libcust_opapi.so",
    "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
    "op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so",
}
_HOST_IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HOST_SOC_PRIORITY = {
    "ascend910_93": 0,
    "ascend910b": 1,
}


def _parse_args() -> argparse.Namespace:
    """Parse explicit per-SoC vendor inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="SOC=VENDOR_ROOT",
        help="Repeat once per isolated SoC build.",
    )
    parser.add_argument(
        "--host-input-identity",
        action="append",
        required=True,
        metavar="SOC=SHA256",
        help="Repeat once per input to prove an identical common host build input.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_soc_payload(relative_path: Path, soc: str) -> bool:
    """Return whether a file belongs to the selected compiled-kernel subtree."""
    try:
        kernel_relative = relative_path.relative_to(_KERNEL_ROOT)
    except ValueError:
        return False
    parts = kernel_relative.parts
    return bool(parts) and (
        parts[0] == soc
        or (len(parts) > 1 and parts[0] == "config" and parts[1] == soc)
    )


def _common_files(vendor_root: Path, soc: str) -> dict[str, str]:
    """Hash all host/source payload that must be byte-identical across SoC builds."""
    return {
        str(relative_path): _sha256(path)
        for path in sorted(vendor_root.rglob("*"))
        if path.is_file()
        for relative_path in (path.relative_to(vendor_root),)
        if not _is_soc_payload(relative_path, soc)
    }


def _readelf(path: Path, *arguments: str) -> str:
    """Return one readelf inspection, rejecting malformed host libraries."""
    result = subprocess.run(
        ["readelf", *arguments, str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Cannot inspect host ELF {path}: {result.stderr.strip()}")
    return result.stdout


def _host_elf_abi_fingerprint(path: Path) -> dict[str, list[str]]:
    """Describe the host ELF ABI without depending on link layout."""
    dynamic_entries = sorted(
        line.strip()
        for line in _readelf(path, "-dW").splitlines()
        if "(NEEDED)" in line or "(SONAME)" in line
    )
    dynamic_symbols: list[str] = []
    symbol_pattern = re.compile(
        r"^\s*\d+:\s+[0-9a-fA-F]+\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+(?:\s+(.*))?$"
    )
    for line in _readelf(path, "--dyn-syms", "-W").splitlines():
        match = symbol_pattern.match(line)
        if not match:
            continue
        size, symbol_type, bind, visibility, name = match.groups()
        dynamic_symbols.append(" ".join((size, symbol_type, bind, visibility, name or "")))
    if not dynamic_symbols:
        raise ValueError(f"Incomplete host ELF ABI fingerprint for {path}")
    return {
        "dynamic_entries": dynamic_entries,
        "dynamic_symbols": sorted(dynamic_symbols),
    }


def _host_elf_abi_mismatches(relative_path: str, base_file: Path, candidate_file: Path) -> list[str]:
    """Return mismatched ABI fields for one known, discarded host ELF variant."""
    if relative_path not in _HOST_ELF_PATHS:
        return ["unsupported_path"]
    try:
        base_fingerprint = _host_elf_abi_fingerprint(base_file)
        candidate_fingerprint = _host_elf_abi_fingerprint(candidate_file)
    except ValueError as error:
        return [f"inspection_error={error}"]
    return sorted(
        field
        for field in set(base_fingerprint) | set(candidate_fingerprint)
        if base_fingerprint.get(field) != candidate_fingerprint.get(field)
    )


def _require_soc_payload(vendor_root: Path, soc: str) -> tuple[Path, Path]:
    """Require both the kernel binaries and package config for one SoC."""
    kernel_path = vendor_root / _KERNEL_ROOT / soc
    config_path = vendor_root / _KERNEL_ROOT / "config" / soc
    for path in (kernel_path, config_path):
        if not path.is_dir() or not any(child.is_file() for child in path.rglob("*")):
            raise ValueError(f"Missing compiled {soc} vendor payload: {path}")
    return kernel_path, config_path


def _compare_host_payload(
    base_vendor: Path,
    base_common_files: dict[str, str],
    soc: str,
    vendor_root: Path,
) -> list[str]:
    """Validate one discarded host variant and return its ABI-compatible ELF paths."""
    common_files = _common_files(vendor_root, soc)
    if common_files == base_common_files:
        return []

    missing = sorted(set(base_common_files) - set(common_files))
    extra = sorted(set(common_files) - set(base_common_files))
    changed_candidates = sorted(
        path
        for path in set(base_common_files) & set(common_files)
        if base_common_files[path] != common_files[path]
    )
    discarded_host_variants: list[str] = []
    changed = []
    for relative_path in changed_candidates:
        mismatches = _host_elf_abi_mismatches(
            relative_path,
            base_vendor / relative_path,
            vendor_root / relative_path,
        )
        if not mismatches:
            discarded_host_variants.append(f"{soc}:{relative_path}")
        else:
            changed.append(f"{relative_path}[{','.join(mismatches)}]")
    if missing or extra or changed:
        raise ValueError(
            f"Per-SoC host vendor payload differs for {soc}: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return discarded_host_variants


def _normalize_inputs(inputs: list[tuple[str, Path]]) -> tuple[list[tuple[str, Path]], set[str]]:
    """Validate per-SoC inputs and return them in canonical host priority order."""
    if not inputs:
        raise ValueError("At least one per-SoC vendor input is required.")
    normalized_inputs: list[tuple[str, Path]] = []
    seen_socs: set[str] = set()
    for soc, vendor_root in inputs:
        if not _SOC_NAME_PATTERN.fullmatch(soc) or Path(soc).name != soc:
            raise ValueError(f"Invalid SoC name: {soc!r}")
        vendor_root = vendor_root.resolve()
        if soc in seen_socs:
            raise ValueError(f"Duplicate SoC vendor input: {soc}")
        if vendor_root.name != _VENDOR_NAME or not vendor_root.is_dir():
            raise ValueError(f"Expected input vendor root named {_VENDOR_NAME}: {vendor_root}")
        _require_soc_payload(vendor_root, soc)
        normalized_inputs.append((soc, vendor_root))
        seen_socs.add(soc)
    normalized_inputs.sort(
        key=lambda item: (_HOST_SOC_PRIORITY.get(item[0], len(_HOST_SOC_PRIORITY)), item[0])
    )
    return normalized_inputs, seen_socs


def _validate_host_input_identities(
    host_input_identities: dict[str, str],
    seen_socs: set[str],
) -> str:
    """Require one identical, well-formed common-host identity per SoC."""
    if set(host_input_identities) != seen_socs:
        raise ValueError(
            "Host input identities must match the SoC inputs exactly: "
            f"inputs={sorted(seen_socs)}, identities={sorted(host_input_identities)}"
        )
    invalid_identities = sorted(
        f"{soc}={identity}"
        for soc, identity in host_input_identities.items()
        if not _HOST_IDENTITY_PATTERN.fullmatch(identity)
    )
    if invalid_identities:
        raise ValueError(f"Invalid host input identities: {invalid_identities}")
    unique_host_input_identities = set(host_input_identities.values())
    if len(unique_host_input_identities) != 1:
        raise ValueError(f"Per-SoC common host input identities differ: {host_input_identities}")
    return next(iter(unique_host_input_identities))


def _validate_output(output: Path, normalized_inputs: list[tuple[str, Path]]) -> Path:
    """Resolve a narrow output root and reject overlap with any input."""
    output = output.resolve()
    if output.name != _VENDOR_NAME or output in {Path("/"), output.parent}:
        raise ValueError(f"Expected a narrow output vendor root named {_VENDOR_NAME}: {output}")
    for _, vendor_root in normalized_inputs:
        if (
            output == vendor_root
            or output.is_relative_to(vendor_root)
            or vendor_root.is_relative_to(output)
        ):
            raise ValueError(
                f"Output vendor must not overlap an input vendor: output={output}, input={vendor_root}"
            )
    return output


def merge_vendors(
    inputs: list[tuple[str, Path]],
    output: Path,
    host_input_identities: dict[str, str],
) -> dict[str, Any]:
    """Retain one provenanced host vendor and merge verified per-SoC kernels."""
    normalized_inputs, seen_socs = _normalize_inputs(inputs)
    host_input_identity = _validate_host_input_identities(host_input_identities, seen_socs)
    output = _validate_output(output, normalized_inputs)
    base_soc, base_vendor = normalized_inputs[0]
    base_common_files = _common_files(base_vendor, base_soc)
    base_host_libraries = sorted(set(base_common_files) & _HOST_ELF_PATHS)
    if not base_host_libraries:
        raise ValueError(f"Base vendor contains no recognized host ELF: {base_vendor}")
    for relative_path in base_host_libraries:
        _host_elf_abi_fingerprint(base_vendor / relative_path)

    discarded_host_variants: list[str] = []
    for soc, vendor_root in normalized_inputs[1:]:
        discarded_host_variants.extend(
            _compare_host_payload(base_vendor, base_common_files, soc, vendor_root)
        )

    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(base_vendor, output)
    for soc, vendor_root in normalized_inputs[1:]:
        kernel_path, config_path = _require_soc_payload(vendor_root, soc)
        shutil.copytree(kernel_path, output / _KERNEL_ROOT / soc)
        shutil.copytree(config_path, output / _KERNEL_ROOT / "config" / soc)

    libraries = sorted(output.rglob("libcust_opapi.so"))
    if len(libraries) != 1:
        raise ValueError(f"Merged vendor must contain one libcust_opapi.so, found {libraries}")
    return {
        "schema_version": 1,
        "status": "PASSED",
        "vendor_root": str(output),
        "supported_socs": [soc for soc, _ in normalized_inputs],
        "host_vendor_soc": base_soc,
        "host_payload_files": len(base_common_files),
        "host_input_identity": host_input_identity,
        "discarded_host_variants": discarded_host_variants,
        "libcust_opapi_sha256": _sha256(libraries[0]),
    }


def main() -> int:
    """Merge command-line inputs and optionally write a report."""
    args = _parse_args()
    inputs: list[tuple[str, Path]] = []
    for value in args.input:
        soc, separator, path = value.partition("=")
        if not separator or not soc or not path:
            raise ValueError(f"Invalid --input {value!r}; expected SOC=VENDOR_ROOT.")
        inputs.append((soc, Path(path)))
    host_input_identities: dict[str, str] = {}
    for value in args.host_input_identity:
        soc, separator, identity = value.partition("=")
        if not separator or not soc or not identity:
            raise ValueError(f"Invalid --host-input-identity {value!r}; expected SOC=SHA256.")
        if soc in host_input_identities:
            raise ValueError(f"Duplicate host input identity: {soc}")
        host_input_identities[soc] = identity
    report = merge_vendors(inputs, Path(args.output), host_input_identities)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(report_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
