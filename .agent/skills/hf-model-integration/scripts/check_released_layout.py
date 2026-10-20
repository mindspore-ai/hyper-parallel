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
"""Pin any released HF checkpoint layout against a meta-device build.

Model-agnostic version of the released-layout contract test: given a local
model directory, build the architecture on the ``meta`` device from its
``config.json`` and compare the parameter catalog against the released
weight files — every released entry must exist with the released shape, and
the build must not invent entries the release lacks. No weight tensor is
loaded and no accelerator is touched.

Shape sources, best first:

* ``*.safetensors`` shard headers (a length-prefixed JSON block; read
  without loading tensor data) when the shards are present;
* otherwise ``model.safetensors.index.json`` provides the entry names and
  the comparison degrades to the name sets.

Tied weights are handled: build-side alias names that the release omits
(for example ``lm_head.weight`` under ``tie_word_embeddings``) are reported
and excluded from the mismatch set instead of flagged as invented entries.

Fused MoE experts are handled: released per-expert 2D weights are reconciled
against the stacked/concatenated 3D expert tensors that transformers 5.x
builds and converts to at load time (``experts.<i>.gate/up/down_proj.weight``
vs ``experts.gate_up_proj``/``experts.down_proj``), with expert counts and
shapes checked, instead of reported as missing/invented entries.

Exit codes: 0 all checks pass; 1 usage/input problems; 2 layout mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"


def read_shard_header(path: Path) -> dict:
    """Return the safetensors header of one shard without loading tensors."""
    with path.open("rb") as shard:
        (header_len,) = struct.unpack("<Q", shard.read(8))
        return json.loads(shard.read(header_len).decode("utf-8"))


def released_entries(model_dir: Path) -> dict[str, tuple[int, ...] | None]:
    """Map every released weight entry to its shape when discoverable."""
    index_path = model_dir / INDEX_NAME
    single = model_dir / "model.safetensors"
    entries: dict[str, tuple[int, ...] | None] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        entries = dict.fromkeys(weight_map)
        for shard_name in sorted(set(weight_map.values())):
            shard = model_dir / shard_name
            if not shard.is_file():
                continue
            for name, meta in read_shard_header(shard).items():
                if name != "__metadata__":
                    entries[name] = tuple(meta["shape"])
    elif single.is_file():
        for name, meta in read_shard_header(single).items():
            if name != "__metadata__":
                entries[name] = tuple(meta["shape"])
    return entries


def built_entries_and_aliases(
        model_dir: Path) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    """Meta-build the architecture and list its parameters plus tied aliases.

    Tied weights (for example ``lm_head.weight`` under ``tie_word_embeddings``)
    appear in ``state_dict`` under every alias, while a release legitimately
    stores the tensor once. The alias set is derived from parameter identity
    (``named_parameters`` with and without duplicate removal), so any tying
    pattern is covered without model-family knowledge.
    """
    # Heavy imports stay inside so --help works anywhere.
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoConfig,
        AutoModelForCausalLM,
    )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True,
                                        trust_remote_code=False)
    config.use_cache = False
    model = _meta_build(config, torch, AutoModelForCausalLM)
    entries = {name: tuple(tensor.shape)
               for name, tensor in model.state_dict().items()}
    deduped = {name for name, _ in model.named_parameters()}
    aliases = {name for name, _
               in model.named_parameters(remove_duplicate=False)} - deduped
    ignored = tuple(getattr(type(model), "_keys_to_ignore_on_load_unexpected",
                            None) or ())
    return entries, aliases, ignored


def built_entries(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Meta-build the architecture from config.json and list its parameters."""
    return built_entries_and_aliases(model_dir)[0]


def split_declared_ignored(released: dict, built: dict, patterns) -> tuple[dict, dict]:
    """Separate released entries the architecture declares it will not load.

    A release may legitimately carry a subtree the modelling code does not
    build: an extra prediction head, a tower a task class leaves out. The
    architecture states this itself, in ``_keys_to_ignore_on_load_unexpected``,
    and ``from_pretrained`` drops those tensors without complaint — so a pin
    that calls them "absent from the build" reports upstream's own contract as
    a defect, and the operator is left deciding whether a real difference is
    hiding behind a known one.

    Only *released-only* entries are reclassified: an entry the build has, with
    a shape that drifted, stays a failure however the patterns read. The split
    is reported, never silent — the declaration is a reason to label the
    difference, not a reason to stop looking at it.
    """
    if not patterns:
        return released, {}
    compiled = [re.compile(pattern) for pattern in patterns]
    only = set(released) - set(built)
    declared = {name: shape for name, shape in released.items()
                if name in only and any(rx.search(name) for rx in compiled)}
    kept = {name: shape for name, shape in released.items()
            if name not in declared}
    return kept, declared


_EXPERT = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<leaf>[A-Za-z0-9_]+)\.weight$")

# Task auto-classes tried in order: a causal LM first (the common case), then
# the multimodal heads, then the bare encoder. The released catalog belongs to
# whichever class the family registers, and a bring-up should not have to name
# it by hand.
AUTO_CLASS_NAMES = ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                    "AutoModelForVision2Seq", "AutoModel")


def _meta_build(config, torch_module, first_class=None):
    """Meta-build the architecture the checkpoint declares, else by auto-class.

    ``config.architectures`` names the class the released checkpoint belongs
    to, so it is tried first: an auto-class may happily build a *different*
    member of the same family (the text tower of a composite model, say) and
    produce a catalog that is a strict subset of the release. The auto-class
    chain remains the fallback for configurations that name no architecture
    or name one this transformers version does not export.
    """
    import transformers  # pylint: disable=import-outside-toplevel

    candidates = []
    for architecture in getattr(config, "architectures", None) or []:
        declared = getattr(transformers, architecture, None)
        if declared is not None:
            candidates.append(declared)
    if first_class is not None and first_class not in candidates:
        candidates.append(first_class)
    for name in AUTO_CLASS_NAMES:
        auto_class = getattr(transformers, name, None)
        if auto_class is not None and auto_class not in candidates:
            candidates.append(auto_class)

    failures = []
    for candidate in candidates:
        builder = (getattr(candidate, "from_config", None)
                   or getattr(candidate, "_from_config", None))
        if builder is None:
            continue
        try:
            with torch_module.device("meta"):
                return builder(config)
        except (ValueError, TypeError, AttributeError) as error:
            head = str(error).splitlines()[0]
            failures.append(f"{getattr(candidate, '__name__', candidate)}: {head}")
    raise ValueError("no model class could build this configuration:\n  "
                     + "\n  ".join(failures))


def reconcile_fused_experts(released: dict, built: dict):
    """Pair per-expert released weights with fused 3D built expert tensors.

    Transformers 5.x builds MoE experts as stacked 3D parameters
    (``experts.down_proj`` of shape ``(E, out, in)``) and, for some families,
    concatenates two released projections into one (``experts.gate_up_proj``
    of shape ``(E, 2*out, in)``), converting released per-expert 2D weights
    (``experts.<i>.<leaf>.weight``) at load time. This reconciles the two
    views generically: a fused leaf consumes either the same-named released
    leaf (stack) or a pair of released leaves whose output dims concatenate
    to it. Returns ``(released_rest, built_rest, notes, problems)`` with
    explained entries removed from both sides.
    """
    rel_groups: dict = {}
    rel_rest = dict(released)
    for name, shape in released.items():
        match = _EXPERT.match(name)
        if match:
            rel_groups.setdefault(match["prefix"], {}).setdefault(
                match["leaf"], {})[int(match["idx"])] = shape
    built_rest = dict(built)
    notes, problems = [], []
    matched_groups = 0
    for prefix, leaves in sorted(rel_groups.items()):
        fused = {name[len(prefix) + 1:]: shape for name, shape in built.items()
                 if name.startswith(prefix + ".") and
                 not _EXPERT.match(name) and len(shape) == 3}
        if not fused:
            continue  # built side is per-expert too; strict compare applies
        consumed_leaves = set()
        names_only = False
        for fused_leaf, fshape in sorted(fused.items()):
            n_experts = fshape[0]
            NOSHAPE = object()  # index names present, shard headers absent

            def uniform(leaf):
                entries = leaves.get(leaf, {})
                if set(entries) != set(range(n_experts)):
                    return None
                shapes = set(entries.values())
                if shapes == {None}:
                    return NOSHAPE
                if len(shapes) != 1 or None in shapes:
                    return None
                return shapes.pop()

            if fused_leaf in leaves:  # stacked: (E, out, in) from E x (out, in)
                shape = uniform(fused_leaf)
                if shape is NOSHAPE:
                    names_only = True
                elif shape is None or (n_experts,) + shape != fshape:
                    problems.append(f"{prefix}.{fused_leaf}: released experts do "
                                    f"not stack to {fshape}")
                    continue
                consumed_leaves.add(fused_leaf)
            else:  # concatenated pair: (E, a+b, in) from E x (a, in) + E x (b, in)
                pair = [leaf for leaf in sorted(leaves)
                        if leaf not in consumed_leaves and leaf not in fused
                        and uniform(leaf) is not None]
                shaped_pair = [leaf for leaf in pair if uniform(leaf) is not NOSHAPE]
                if shaped_pair:
                    take = [leaf for leaf in shaped_pair
                            if uniform(leaf)[1] == fshape[2]][:2]
                    ok = (len(take) == 2 and
                          uniform(take[0])[0] + uniform(take[1])[0] == fshape[1])
                elif len(pair) == 2:  # names-only mode: counts checked, shapes not
                    take, ok, names_only = pair, True, True
                else:
                    take, ok = [], False
                if ok:
                    consumed_leaves.update(take)
                else:
                    problems.append(f"{prefix}.{fused_leaf}: no released leaf "
                                    f"pair concatenates to {fshape}")
                    continue
            built_rest.pop(f"{prefix}.{fused_leaf}", None)
        if names_only and consumed_leaves:
            notes.append(f"{prefix}: expert counts matched by name only "
                         f"(no shard headers, shapes unverified)")
        for leaf in consumed_leaves:
            for idx in leaves[leaf]:
                rel_rest.pop(f"{prefix}.{idx}.{leaf}.weight", None)
        if consumed_leaves:
            matched_groups += 1
    if matched_groups:
        notes.append(f"fused-expert reconciliation: {matched_groups} expert "
                     f"groups matched against stacked/concatenated build tensors")
    return rel_rest, built_rest, notes, problems


def reconcile_renamed_prefixes(released: dict, built: dict):
    """Pair entries a wholesale rename separated, by suffix and shape.

    Some families keep the released parameter tree but move it under a new
    prefix in the build (``model.layers.0...`` versus
    ``model.language_model.layers.0...`` once a text tower becomes a subtree
    of a multimodal model), so a strict name comparison reports every entry
    as both missing and invented. This pairs the two sides on the longest
    name suffix that is unique with the same shape on both sides, which is a
    rename-agnostic identity: no family knowledge, and an entry that is
    genuinely absent or reshaped still fails to pair. Returns
    ``(released_rest, built_rest, notes, problems)`` with paired entries
    removed from both sides.
    """
    rel_index: dict = {}
    built_index: dict = {}
    rel_by_suffix: dict = {}
    built_by_suffix: dict = {}
    for source, table, by_suffix in ((released, rel_index, rel_by_suffix),
                                     (built, built_index, built_by_suffix)):
        for name, shape in source.items():
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                suffix = ".".join(parts[-depth:])
                table.setdefault((suffix, shape), []).append(name)
                by_suffix.setdefault(suffix, []).append(name)

    pairs: dict = {}
    taken_built = set()
    shapeless = 0
    for name in sorted(released):
        parts = name.split(".")
        # Without shard headers the released side has no shapes, so pairing
        # falls back to suffix uniqueness alone - weaker, and reported as such.
        by_name_only = released[name] is None
        for depth in range(len(parts), 0, -1):
            suffix = ".".join(parts[-depth:])
            if by_name_only:
                candidates_rel = rel_by_suffix.get(suffix, [])
                candidates_built = built_by_suffix.get(suffix, [])
            else:
                key = (suffix, released[name])
                candidates_rel = rel_index.get(key, [])
                candidates_built = built_index.get(key, [])
            if len(candidates_rel) == 1 and len(candidates_built) == 1:
                candidate = candidates_built[0]
                if candidate not in taken_built:
                    pairs[name] = candidate
                    taken_built.add(candidate)
                    shapeless += 1 if by_name_only else 0
                break

    rel_rest = {k: v for k, v in released.items() if k not in pairs}
    built_rest = {k: v for k, v in built.items() if k not in taken_built}
    notes, problems = [], []
    if pairs:
        prefix_counts: dict = {}
        for source_name, build_name in pairs.items():
            source_parts, build_parts = source_name.split("."), build_name.split(".")
            common = 0
            while (common < min(len(source_parts), len(build_parts))
                   and source_parts[-1 - common] == build_parts[-1 - common]):
                common += 1
            mapping = (".".join(source_parts[:len(source_parts) - common]),
                       ".".join(build_parts[:len(build_parts) - common]))
            prefix_counts[mapping] = prefix_counts.get(mapping, 0) + 1
        renames = {m: n for m, n in prefix_counts.items() if m[0] != m[1]}
        if renames:
            summary = ", ".join(
                f"{source or '<root>'} -> {build or '<root>'} ({count})"
                for (source, build), count in
                sorted(renames.items(), key=lambda item: -item[1])[:3])
            basis = ("suffix only (no shard headers, shapes unverified)"
                     if shapeless else "suffix and shape")
            notes.append(f"rename reconciliation: {len(pairs)} entries paired "
                         f"by {basis}; prefix map {summary}")
    return rel_rest, built_rest, notes, problems


def compare(released: dict, built: dict) -> list[str]:
    """Return one message per layout violation; empty means the pin holds."""
    problems = []
    missing = sorted(set(released) - set(built))
    unexpected = sorted(set(built) - set(released))
    if missing:
        problems.append(f"released entries absent from the build ({len(missing)}): "
                        + ", ".join(missing[:5]))
    if unexpected:
        problems.append(f"build has entries the release lacks ({len(unexpected)}): "
                        + ", ".join(unexpected[:5]))
    mismatched = {name: (shape, built[name])
                  for name, shape in released.items()
                  if shape is not None and name in built and tuple(shape) != built[name]}
    if mismatched:
        sample = next(iter(mismatched.items()))
        problems.append(f"shape drift on {len(mismatched)} entries, e.g. "
                        f"{sample[0]}: released {sample[1][0]} vs built {sample[1][1]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/hf-model-integration/scripts/check_released_layout.py",
        description="Pin a released HF checkpoint layout against a meta build.")
    parser.add_argument("--model-dir", required=True,
                        help="local directory with config.json and the released "
                             "index/safetensors files")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        print(f"[layout] no config.json under {model_dir}", flush=True)
        return 1
    released = released_entries(model_dir)
    if not released:
        print(f"[layout] no {INDEX_NAME} or model.safetensors under {model_dir}",
              flush=True)
        return 1
    built, aliases, ignore_patterns = built_entries_and_aliases(model_dir)
    tied_skipped = sorted(name for name in aliases
                          if name in built and name not in released)
    for name in tied_skipped:
        built.pop(name)
    if tied_skipped:
        print(f"[layout] tied aliases excluded (the release stores the tensor "
              f"once): {', '.join(tied_skipped[:5])}", flush=True)

    shaped = sum(1 for shape in released.values() if shape is not None)
    released_total = len(released)
    released, declared = split_declared_ignored(released, built, ignore_patterns)
    if declared:
        prefixes = sorted({name.split(".")[0] for name in declared})
        print(f"[layout] declared-ignored subtree: {len(declared)} released "
              f"entries under {prefixes} match the architecture's own "
              f"_keys_to_ignore_on_load_unexpected {list(ignore_patterns)} — "
              f"upstream does not build these; reported, not failed",
              flush=True)
    released_rest, built_rest, notes, problems = reconcile_fused_experts(
        released, built)
    strict = compare(released_rest, built_rest)
    if strict:
        # A wholesale prefix rename reports every entry as both missing and
        # invented; retry on the rename-agnostic pairing before failing.
        renamed_rel, renamed_built, rename_notes, rename_problems = (
            reconcile_renamed_prefixes(released_rest, built_rest))
        retried = rename_problems + compare(renamed_rel, renamed_built)
        if not retried:
            notes += rename_notes
            strict = []
    problems += strict
    for note in notes:
        print(f"[layout] {note}", flush=True)
    print(f"[layout] released entries={released_total} (with shapes: {shaped}), "
          f"built={len(built)}, compared={len(released)}", flush=True)
    for problem in problems:
        print(f"[layout] FAIL {problem}", flush=True)
    if not problems:
        print("[layout] PASS: the meta build matches the released catalog", flush=True)
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
