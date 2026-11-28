---
name: ascend-profiling-timeline
description: >-
  Generate single-file Ascend NPU profiling HTML reports combining an interactive
  zoomable Canvas timeline (merged MAC%/Vec% step-lines, zoom +/-, wheel zoom, pan,
  optional --window-ms lite slicing, sortable op table) and publication-grade
  single-layer 3x zoomed SVG reports (Free-breakdown card, multi-stream Gantt,
  green overlap strip, 90-degree rotated op labels). Supports kernel_details.csv,
  msprof op_summary*.csv, and CANN SQLite msprof_*.db directly.
---

# Ascend Profiling Timeline Visualization Skill

Use this skill to turn Ascend NPU (`torch_npu.profiler` / `msprof`) profiling exports into self-contained, single-file HTML visual reports. Supports:
- `ASCEND_PROFILER_OUTPUT` (`kernel_details.csv` + `step_trace_time.csv`)
- CANN `mindstudio_profiler_output` (`op_summary*.csv`)
- CANN SQLite database (`msprof_*.db`) directly

## Unified Basic Color Palette (Consistent Across All Charts)
- **Blue (`#2563eb`)**: Main Compute / Cube (`AI_CORE` / `MIX_AIC`)
- **Yellow (`#d97706`)**: Vector / Auxiliary Compute (`AI_VECTOR_CORE`)
- **Red (`#dc2626`)**: HCCL Communication (`COMMUNICATION` / `hcom_*`)
- **Purple (`#7c3aed`)**: `AI_CPU` helper tasks
- **Green (`#16a34a`)**: Compute∥Comm Overlap & `MAC %` step-line
- **Brown (`#92400e`)**: `Vec %` step-line
- **Gray (`#94a3b8`)**: Free / Other

## Recommended One-Command Unified Report (`scripts/render_combined_report.py`)

Generates **one self-contained HTML** containing:
1. Full-step Free breakdown & bottleneck summary card
2. Interactive zoomable Canvas timeline (`+`/`-`, mouse wheel zoom, drag zoom, `Shift+drag` pan, merged `0–100%` `MAC %` green + `Vec %` brown step-lines)
3. Typical Forward & Backward single-layer 3× zoomed static SVG reports (for PPT/PDF screenshots) with one-click jump buttons back to the interactive Canvas
4. Sortable full-step operator utilization & duration ranking table

```bash
# Full-step unified report
python3 .agent/skills/ascend-profiling-timeline/scripts/render_combined_report.py \
  <PROF_DIR_OR_MSPROF_DB> \
  <OUT_HTML> \
  --title "Model · CP=8 · Step 2 Profiling" \
  --scope "Rank 0 · Step 2"

# Fast lightweight Lite report (slices interactive Canvas to 2500-4500 ms while keeping full-step stats)
python3 .agent/skills/ascend-profiling-timeline/scripts/render_combined_report.py \
  <PROF_DIR_OR_MSPROF_DB> \
  <OUT_LITE_HTML> \
  --title "Model · CP=8 · Step 2 (Lite)" \
  --scope "Rank 0 · Step 2 · 2500-4500 ms slice" \
  --window-ms 2500:4500
```

## Individual Scripts
- `scripts/render_interactive_timeline.py`: Standalone interactive zoomable Canvas report (`--window-ms`, `--min-duration-us` supported).
- `scripts/render_layer_svg.py`: Standalone single-layer 3× zoomed SVG report.


## Scope and interpretation

This skill visualizes existing evidence. Use [profiling](../profiling/SKILL.md)
for capture and cross-file consistency checks; charts do not replace numerical
or performance acceptance. The renderers require Python 3.9+ and only the
standard library. The interactive report needs a browser supporting
`DecompressionStream` (gzip).

- Select one rank/device and state the captured steps in `--scope`. Direct CSV
  file paths are accepted. If a directory contains multiple databases, select
  a specific `.db` rather than combining independent device clocks.
- SQLite input supports the CANN `STRING_IDS`, `COMPUTE_TASK_INFO`, and `TASK`
  schema, with optional `TASK_PMU_INFO` and `COMMUNICATION_OP` tables. It is read
  only. Other profiler schema versions require an explicit adapter.
- `--cube-cores N` supplies the verified Cube core count for database export.
  Without it, Cube utilization remains unavailable; no device count is assumed.
- MAC/Vector PMU activity ratios are not model MFU. Missing measurements are
  not evidence of zero utilization. The interactive table marks missing values;
  unavailable SVG averages are marked `N/A`.
- Free is the complement of classified compute and communication coverage. It
  may contain AI_CPU or unclassified activity, so it does not prove device idle.
  Full-input statistics and optional step CSV totals may cover different ranges;
  check their scope before comparison. Lite slicing only filters the Canvas.
- Automatic layer windows use anchor clustering (40 ms minimum spacing) and a
  median-duration heuristic, not module IDs. Verify the anchors against model
  markers; with insufficient anchors, the SVG covers the full input window.
- SVG drawings omit tasks at most 10 microseconds long and show up to seven
  dominant lanes. Statistics retain these tasks. PMU lines are visual samples,
  not a reconstructed device-wide utilization signal.

Imported from `weich97/hyper-parallel-dev` PR #3 at
`e6ba6f042c84cc1ece924f737e817b519fd18a26`, including its template and interactive
renderer dependencies, with integration compatibility fixes.


## Authorship and integration

The enhancements in [source PR #3](https://github.com/weich97/hyper-parallel-dev/pull/3)
were authored by [hxsong-GitHub](https://github.com/hxsong-GitHub) (original Git
signature: `huangxs`): the combined report, direct SQLite input, explanatory
captions, and clustered/median-duration layer selection. Original commits:
`cd70abfa`, `dfa80b9f`, and `e6ba6f04`.

This upstream integration was submitted by
[Goethedinger](https://github.com/Goethedinger), with compatibility fixes and
validation, on top of the workflow implementation contributed by
[weich97](https://github.com/weich97). Integration authorship does not replace
credit for the original implementation.
