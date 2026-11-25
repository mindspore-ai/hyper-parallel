#!/usr/bin/env python3
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
"""Generate publication-grade single-layer 3x zoomed SVG profiling report from Ascend CSV exports.

Supports both kernel_details.csv and msprof op_summary*.csv automatically.
"""
import argparse
from collections import defaultdict
import csv
from html import escape
import math
from pathlib import Path
from typing import Optional

def locate_csv(input_dir: Path) -> tuple[Path, bool]:
    """Locate a kernel CSV and identify its column schema."""
    if input_dir.is_file() and input_dir.suffix == ".csv":
        return input_dir, input_dir.name.startswith("op_summary")
    kd = input_dir / "kernel_details.csv"
    if kd.is_file():
        return kd, False
    for cand in sorted(input_dir.glob("op_summary*.csv")):
        if cand.is_file():
            return cand, True
    raise FileNotFoundError(f"Neither kernel_details.csv nor op_summary*.csv found in {input_dir}")


def load_ops(path: Path, is_op_summary: bool) -> list[dict]:
    """Parse and sort valid timed tasks, preserving missing PMU values."""
    name_col = "Op Name" if is_op_summary else "Name"
    type_col = "OP Type" if is_op_summary else "Type"
    core_col = "Task Type" if is_op_summary else "Accelerator Core"
    start_col = "Task Start Time(us)" if is_op_summary else "Start Time(us)"
    dur_col = "Task Duration(us)" if is_op_summary else "Duration(us)"

    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                st = float(row[start_col])
                du = float(row[dur_col])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(st) and math.isfinite(du)) or st < 0 or du <= 0:
                continue
            tp = (row.get(type_col) or "Unknown").strip()
            nm = (row.get(name_col) or tp).strip()
            core = (row.get(core_col) or "").strip()
            sid = (row.get("Stream ID") or "N/A").strip()
            tid = (row.get("Task ID") or "N/A").strip()
            shapes = (row.get("Input Shapes") or "").strip().strip('"')

            def flt(k: str, scale: float = 1.0) -> Optional[float]:
                """Read a finite nonnegative metric from the current CSV row."""
                try:
                    v = float(row[k])
                    return v * scale if math.isfinite(v) and v >= 0 else None
                except (KeyError, TypeError, ValueError):
                    return None

            mac = flt("aic_mac_ratio", 100.0)
            vec = flt("aiv_vec_ratio", 100.0)
            aic = flt("aicore_time(us)", 1.0) or 0.0
            aiv = flt("aiv_time(us)", 1.0) or 0.0

            norm_tp = tp.lower().replace("_", "")
            if core == "COMMUNICATION" or norm_tp.startswith(("hcom", "hccl")):
                cat, lane = "comm", "HCCL 通信"
            elif core == "AI_CPU":
                cat, lane = "aicpu", f"流s{sid} AI_CPU"
            elif core in ("AI_CORE", "MIX_AIC", "AI_VECTOR_CORE", "MIX_AIV"):
                cat, lane = "compute", f"流s{sid} {core}"
            else:
                cat, lane = "other", f"流s{sid} 其他"

            rows.append(
                {
                    "start": st,
                    "dur": du,
                    "end": st + du,
                    "type": tp,
                    "name": nm,
                    "core": core,
                    "stream": sid,
                    "task_id": tid,
                    "shapes": shapes,
                    "mac": mac,
                    "vec": vec,
                    "aic": aic,
                    "aiv": aiv,
                    "cat": cat,
                    "lane": lane,
                }
            )
    if not rows:
        raise ValueError(f"No timed tasks in {path}")
    rows.sort(key=lambda r: r["start"])
    return rows


def merge_intervals(iv: list) -> list:
    """Merge overlapping intervals into disjoint time coverage."""
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def interval_len(iv: list) -> float:
    """Return the total duration of disjoint intervals."""
    return sum(e - s for s, e in iv)


def intersect_intervals(a: list, b: list) -> list:
    """Intersect two disjoint sorted interval lists."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def format_metric(value: Optional[float]) -> str:
    """Format an observed percentage, leaving missing readings explicit."""
    return "N/A" if value is None else f"{value:.1f}%"


def window_breakdown(ops: list[dict], t0: Optional[float] = None, t1: Optional[float] = None) -> dict:
    """Compute interval coverage and available task-weighted PMU metrics."""
    if not ops:
        return {}
    if t0 is None:
        t0 = ops[0]["start"]
    if t1 is None:
        t1 = max(op["end"] for op in ops)
    span = max(t1 - t0, 1.0)
    comp_iv = merge_intervals(
        [
            (max(t0, op["start"]), min(t1, op["end"]))
            for op in ops
            if op["cat"] == "compute" and op["end"] > t0 and op["start"] < t1
        ]
    )
    comm_iv = merge_intervals(
        [
            (max(t0, op["start"]), min(t1, op["end"]))
            for op in ops
            if op["cat"] == "comm" and op["end"] > t0 and op["start"] < t1
        ]
    )
    ov_iv = intersect_intervals(comp_iv, comm_iv)

    comp_t, comm_t, ov_t = interval_len(comp_iv), interval_len(comm_iv), interval_len(ov_iv)
    pure_comp, pure_comm = max(0.0, comp_t - ov_t), max(0.0, comm_t - ov_t)
    free_t = max(0.0, span - pure_comp - pure_comm - ov_t)

    mac_w = mac_sum = vec_w = vec_sum = 0.0
    for op in ops:
        if op["aic"] > 0 and op["mac"] is not None:
            mac_sum += op["mac"] * op["aic"]
            mac_w += op["aic"]
        if op["aiv"] > 0 and op["vec"] is not None:
            vec_sum += op["vec"] * op["aiv"]
            vec_w += op["aiv"]

    return {
        "span_us": span,
        "count": len(ops),
        "comp_count": sum(1 for op in ops if op["cat"] == "compute"),
        "comm_count": sum(1 for op in ops if op["cat"] == "comm"),
        "aicpu_count": sum(1 for op in ops if op["cat"] == "aicpu"),
        "pure_comp_pct": 100.0 * pure_comp / span,
        "pure_comm_pct": 100.0 * pure_comm / span,
        "overlap_pct": 100.0 * ov_t / span,
        "free_pct": 100.0 * free_t / span,
        "comm_masked_pct": (100.0 * ov_t / comm_t) if comm_t > 0 else 0.0,
        "weighted_mac_pct": (mac_sum / mac_w) if mac_w > 0 else None,
        "weighted_vec_pct": (vec_sum / vec_w) if vec_w > 0 else None,
        "ov_iv": ov_iv,
        "ov_us": ov_t,
    }


def pick_layer_window(
    ops: list[dict], anchor_type: str, is_backward: bool, target_idx: int, min_layer_gap_us: float = 40000.0
) -> tuple[float, float, list[dict]]:
    """Select a median-duration anchor window; fall back to the full input."""
    raw_anchors = [i for i, op in enumerate(ops) if op["type"] == anchor_type]
    if not raw_anchors:
        return ops[0]["start"], max(op["end"] for op in ops), ops
    # Cluster multiple occurrences of the same anchor inside one layer (e.g. 2x GroupedMatmul or 6x SoftmaxGrad)
    anchors = [raw_anchors[0]]
    for idx in raw_anchors[1:]:
        if ops[idx]["start"] - ops[anchors[-1]]["start"] >= min_layer_gap_us:
            anchors.append(idx)
    if len(anchors) < 2:
        return ops[0]["start"], max(op["end"] for op in ops), ops

    # Pick a typical (median-duration) layer among interior candidates, avoiding PP bubble outliers
    cands = []
    lo = min(2, len(anchors) - 2)
    hi = max(lo + 1, len(anchors) - 2)
    for k in range(lo, hi):
        dt = ops[anchors[k + 1]]["start"] - ops[anchors[k]]["start"]
        if dt >= min_layer_gap_us:
            cands.append((dt, k))
    if cands:
        cands.sort(key=lambda x: x[0])
        med_dt = cands[len(cands) // 2][0]
        # Among layers close to median duration, pick one near target_idx
        near_med = [k for dt, k in cands if abs(dt - med_dt) <= max(med_dt * 0.15, 5000.0)]
        idx = min(near_med, key=lambda k: abs(k - min(target_idx, len(anchors) - 2)))
    else:
        idx = min(target_idx, len(anchors) - 2)

    a0 = anchors[idx]
    a1 = anchors[idx + 1] if idx + 1 < len(anchors) else len(ops) - 1
    if not is_backward:
        s_idx = a0
        for j in range(a0 - 1, max(0, a0 - 40), -1):
            if "Mhc" in ops[j]["type"] or "HcPre" in ops[j]["type"]:
                s_idx = j
                break
        e_idx = a1
        for j in range(a1 - 1, max(s_idx + 1, a1 - 40), -1):
            if "Mhc" in ops[j]["type"] or "HcPre" in ops[j]["type"]:
                e_idx = j
                break
    else:
        s_idx = max(0, a0 - 2)
        e_idx = a1
        for j in range(a1 - 1, max(s_idx + 1, a1 - 25), -1):
            if "MhcGrad" in ops[j]["type"]:
                e_idx = j + 1
                break
    t0, t1 = ops[s_idx]["start"], ops[e_idx]["start"]
    return t0, t1, [op for op in ops if op["end"] > t0 and op["start"] < t1]


def short_op_label(tp: str, nm: str) -> str:
    """Shorten operator names for dense labels while retaining full tooltips."""
    for prefix, short in (
        ("hcom_alltoallv_", "hcom_alltoallv"),
        ("hcom_allGather_", "hcom_allGather"),
        ("hcom_allReduce_", "hcom_allReduce"),
        ("hcom_reduceScatter_", "hcom_reduceSc…"),
    ):
        if tp == prefix:
            return short
    if "partial_rope" in nm:
        return "PartialRope"
    return (tp[:14] + "…") if len(tp) > 15 else tp


def make_tip(idx: int, op: dict, t0: float, tot_op_us: float) -> str:
    """Build escaped task details for an SVG tooltip."""
    mac_s = f"{op['mac']:.2f}%" if op["mac"] is not None else "n/a"
    vec_s = f"{op['vec']:.2f}%" if op["vec"] is not None else "n/a"
    share = 100.0 * op["dur"] / tot_op_us if tot_op_us > 0 else 0.0
    return escape(
        f"#{idx+1}  {op['type']}\n"
        f"类别 {op['cat']} · 流 s{op['stream']} · 类型 {op['core']}\n"
        f"时间 {(op['start']-t0):.1f} → {(op['end']-t0):.1f} µs（相对层窗）\n"
        f"时长 {op['dur']:.1f} µs（占算子总耗时 {share:.2f}%）\n"
        f"op_name {op['name']}\n"
        f"Task ID {op['task_id']} · 输入 \"{op['shapes'] or 'N/A'}\"\n"
        f"aic_mac_ratio×100：{mac_s}\n"
        f"aiv_vec_ratio×100：{vec_s}"
    )


def semantic_lane_color(lane: str, cat: str) -> str:
    """Basic unmistakable colors for verbal presentation: 蓝/黄/红/紫/绿/棕/灰."""
    if cat == "comm":
        return "#dc2626"  # 红
    if cat == "aicpu":
        return "#7c3aed"  # 紫
    if cat == "other":
        return "#94a3b8"
    return "#d97706" if "AI_VECTOR_CORE" in lane or "MIX_AIV" in lane else "#2563eb"


def build_step_polyline(
    segments: list[tuple[float, float, float]], y_zero: float, color: str, stroke_w: float = 1.5
) -> str:
    """Build a crisp step-function SVG polyline connecting (xs, xe, y) plateaus with vertical risers."""
    if not segments:
        return ""
    segs = sorted(segments, key=lambda s: s[0])
    pts = [f"{segs[0][0]:.2f},{y_zero:.2f}"]
    prev_x = segs[0][0]
    for xs, xe, yy in segs:
        if xs > prev_x + 1.2:
            pts.append(f"{prev_x:.2f},{y_zero:.2f}")
            pts.append(f"{xs:.2f},{y_zero:.2f}")
        elif xs < prev_x:
            xs = prev_x
        xe = max(xe, xs)
        pts.append(f"{xs:.2f},{yy:.2f}")
        pts.append(f"{xe:.2f},{yy:.2f}")
        prev_x = xe
    pts.append(f"{prev_x:.2f},{y_zero:.2f}")
    return (
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{stroke_w}"'
        ' stroke-linejoin="miter"/>'
    )


def render_svg(title_txt: str, ops: list[dict], t0: float, t1: float, stats: dict) -> str:
    """Render task lanes, sampled PMU lines and three enlarged subwindows."""
    W, x0, x1 = 1280, 106.0, 1104.0
    pw, span = x1 - x0, max(t1 - t0, 1.0)
    tot_op_us = sum(op["dur"] for op in ops)
    small_ops = [op for op in ops if op["dur"] <= 10.0]
    small_us = sum(op["dur"] for op in small_ops)

    lane_dur, lane_cat = defaultdict(float), {}
    for op in ops:
        lane_dur[op["lane"]] += op["dur"]
        lane_cat[op["lane"]] = op["cat"]
    cat_order = {"compute": 0, "comm": 1, "aicpu": 2}
    lanes = sorted(lane_dur.keys(), key=lambda ln: (cat_order.get(lane_cat[ln], 3), -lane_dur[ln]))[:7]
    lane_color = {}
    for ln in lanes:
        c = lane_cat[ln]
        lane_color[ln] = semantic_lane_color(ln, c)

    def x_of(t: float, a: float, b: float) -> float:
        """Map a timestamp to the clipped plotting range."""
        return x0 + max(0.0, min(1.0, (t - a) / max(b - a, 1.0))) * pw

    util_top, util_bot = 114.0, 316.0
    lane_top, lane_h, lane_step = 370.0, 21.0, 28.0
    ov_title_y = lane_top + len(lanes) * lane_step + 16.0
    ov_y = ov_title_y + 8.0
    axis_y = ov_y + 24.0
    zoom_title_y = axis_y + 26.0
    seg_block_h = 262.0
    legend_y = zoom_title_y + 28.0 + 3 * seg_block_h + 18.0
    H = int(legend_y + 64)

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"'
            " font-family=\"'Noto Sans CJK SC','Source Han Sans SC','DejaVu Sans',sans-serif\">"
        ),
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    max_op = max(ops, key=lambda op: op["dur"])
    out.append(
        f'<text x="{x0}" y="36" font-size="18.5" fill="#1b1b1b" font-weight="600">{escape(title_txt)}（本层总耗时'
        f" {span/1000:.2f} ms）</text>"
    )
    out.append(
        f'<text x="{x0}" y="58" font-size="11" fill="#444">本层共执行 {stats["count"]} 次算子（计算'
        f' {stats["comp_count"]} 次 / HCCL 通信 {stats["comm_count"]} 次 / AI_CPU {stats["aicpu_count"]} 次） ·'
        f' 单次最慢算子：{escape(max_op["type"])}（{max_op["dur"]/1000:.2f} ms）</text>'
    )
    out.append(
        f'<text x="{x0}" y="76" font-size="10.8" fill="#555">多条流并行累加耗时 {tot_op_us/1000:.2f} ms（因多流同时跑，'
        f"累计耗时为本层总耗时的 {100.0*tot_op_us/span:.1f}%） ·"
        f' 算力加权平均：MAC（矩阵）{format_metric(stats["weighted_mac_pct"])} ·'
        f' Vec（向量）{format_metric(stats["weighted_vec_pct"])}</text>'
    )
    out.append(
        f'<text x="{x0}" y="102" font-size="10.5" fill="#1b1b1b" font-weight="600">① 算力忙不忙（全层总览，'
        "0–100%）：绿线 = 矩阵单元 MAC 利用率（越高越好） · 棕线 = 向量单元 Vec 利用率</text>"
    )

    for pct in (0, 25, 50, 75, 100):
        yy = util_bot - (pct / 100.0) * (util_bot - util_top)
        out.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        out.append(f'<text x="{x0-8}" y="{yy+4:.2f}" font-size="9" fill="#1b1b1b" text-anchor="end">{pct}%</text>')

    # Free breakdown card (unified basic colors: 蓝/红/绿/灰)
    out.append(
        '<rect x="1124" y="114" width="146" height="142" fill="#fafafa" stroke="#d9d9d9" stroke-width="1" rx="3"/>'
    )
    out.append('<text x="1132" y="132" font-size="9.5" fill="#1b1b1b" font-weight="600">本层时间去向（共100%）</text>')
    cy = 152
    for lbl, val, col in (
        ("仅计算在跑 (蓝)", f'{stats["pure_comp_pct"]:.1f}%', "#2563eb"),
        ("仅通信在等 (红)", f'{stats["pure_comm_pct"]:.1f}%', "#dc2626"),
        ("计算∥通信 (绿)", f'{stats["overlap_pct"]:.1f}%', "#16a34a"),
        ("其余时间 (灰)", f'{stats["free_pct"]:.1f}%', "#64748b"),
    ):
        out.append(f'<text x="1132" y="{cy}" font-size="9" fill="#333">{lbl}</text>')
        out.append(
            f'<text x="1262" y="{cy}" font-size="9" fill="{col}" font-weight="600" text-anchor="end">{val}</text>'
        )
        cy += 16
    out.append('<line x1="1132" y1="214" x2="1262" y2="214" stroke="#e2e2e2" stroke-width="0.8"/>')
    out.append('<text x="1132" y="229" font-size="9" fill="#333">通信被计算盖住</text>')
    out.append(
        '<text x="1262" y="229" font-size="9" fill="#16a34a" font-weight="600"'
        f' text-anchor="end">{stats["comm_masked_pct"]:.1f}%</text>'
    )
    out.append('<text x="1132" y="246" font-size="8.3" fill="#777">数值越高说明通信掩盖越好</text>')

    tips = [make_tip(i, op, t0, tot_op_us) for i, op in enumerate(ops)]
    mac_segs_top, vec_segs_top = [], []
    for i, op in enumerate(ops):
        if op["dur"] <= 10.0:
            continue
        xs, xe = x_of(max(t0, op["start"]), t0, t1), x_of(min(t1, op["end"]), t0, t1)
        bw = max(0.8, xe - xs)
        if op["mac"] is not None and op["mac"] > 0:
            ym = util_bot - (min(100.0, op["mac"]) / 100.0) * (util_bot - util_top)
            mac_segs_top.append((xs, xs + bw, ym))
        if op["vec"] is not None and op["vec"] > 0:
            yv = util_bot - (min(100.0, op["vec"]) / 100.0) * (util_bot - util_top)
            vec_segs_top.append((xs, xs + bw, yv))
    if mac_segs_top:
        out.append(build_step_polyline(mac_segs_top, util_bot, "#16a34a", 1.6))
    if vec_segs_top:
        out.append(build_step_polyline(vec_segs_top, util_bot, "#92400e", 1.4))

    out.append(
        f'<text x="{x0}" y="358" font-size="10.5" fill="#1b1b1b" font-weight="600">②'
        " 各通道执行时间线（每行一条流/通道，横条越宽=耗时越长，上下对齐=同一时刻并行；悬停色块看详情）</text>"
    )
    lane_y = {ln: lane_top + idx * lane_step for idx, ln in enumerate(lanes)}
    for ln, yy in lane_y.items():
        out.append(
            f'<rect x="{x0}" y="{yy:.1f}" width="{pw:.1f}" height="{lane_h}" fill="#f7f7f7" stroke="#e6e6e6"'
            ' stroke-width="0.6"/>'
        )
        out.append(
            f'<text x="{x0-8}" y="{yy+14:.1f}" font-size="9.5" fill="#1b1b1b" text-anchor="end">{escape(ln)}</text>'
        )

    for i, op in enumerate(ops):
        if op["dur"] <= 10.0 or op["lane"] not in lane_y:
            continue
        xs, xe = x_of(max(t0, op["start"]), t0, t1), x_of(min(t1, op["end"]), t0, t1)
        bw, yy = max(0.8, xe - xs), lane_y[op["lane"]]
        out.append(
            f'<rect x="{xs:.2f}" y="{yy:.1f}" width="{bw:.2f}" height="{lane_h}" fill="{lane_color[op["lane"]]}"'
            f' rx="1"><title>{tips[i]}</title></rect>'
        )
        if bw >= 52:
            out.append(
                f'<text x="{xs+bw/2:.1f}" y="{yy+14:.1f}" font-size="9" fill="#ffffff" text-anchor="middle"'
                f' font-weight="600">{escape(short_op_label(op["type"], op["name"]))}</text>'
            )

    # Small ops summary card
    out.append(
        f'<rect x="1124" y="{lane_top:.1f}" width="146" height="90" fill="#fafafa" stroke="#d9d9d9" stroke-width="1"'
        ' rx="3"/>'
    )
    out.append(
        f'<text x="1132" y="{lane_top+17:.1f}" font-size="9.5" fill="#1b1b1b" font-weight="600">未画出的极短算子</text>'
    )
    out.append(f'<text x="1132" y="{lane_top+33:.1f}" font-size="8.8" fill="#555">≤10 µs 共 {len(small_ops)} 次</text>')
    out.append(f'<text x="1132" y="{lane_top+49:.1f}" font-size="8.8" fill="#555">合计 {small_us/1000:.2f} ms</text>')
    out.append(
        f'<text x="1132" y="{lane_top+65:.1f}" font-size="8.8" fill="#555">占多流累计'
        f" {100.0*small_us/max(tot_op_us,1):.2f}%</text>"
    )
    out.append(
        f'<text x="1132" y="{lane_top+81:.1f}" font-size="8.8" fill="#555">占本层跨度 {100.0*small_us/span:.2f}%</text>'
    )

    # Overlap strip: title ABOVE the bar (consistent with all other sections)
    out.append(
        f'<text x="{x0}" y="{ov_title_y:.1f}" font-size="10.2" fill="#1b1b1b" font-weight="600">③'
        " 计算与通信同时跑的区间（绿条 = 计算流与 HCCL 通信重叠时刻，越满越好）：共"
        f' {stats["ov_us"]/1000:.2f} ms（占本层 {stats["overlap_pct"]:.1f}% · 盖住 {stats["comm_masked_pct"]:.1f}%'
        " 通信）</text>"
    )
    out.append(
        f'<rect x="{x0}" y="{ov_y:.1f}" width="{pw:.1f}" height="10" fill="#f7f7f7" stroke="#e6e6e6"'
        ' stroke-width="0.6"/>'
    )
    out.append(
        f'<text x="{x0-8}" y="{ov_y+8.5:.1f}" font-size="9" fill="#1b1b1b" text-anchor="end">并行重叠(绿)</text>'
    )
    for s, e in stats["ov_iv"]:
        xs, xe = x_of(s, t0, t1), x_of(e, t0, t1)
        out.append(f'<rect x="{xs:.2f}" y="{ov_y+1:.1f}" width="{max(1.0, xe-xs):.2f}" height="8" fill="#16a34a"/>')

    out.append(f'<line x1="{x0}" y1="{axis_y-4:.1f}" x2="{x1}" y2="{axis_y-4:.1f}" stroke="#bbb" stroke-width="1"/>')
    for k in range(5):
        tx = x0 + (k / 4.0) * pw
        lbl = "0 ms" if k == 0 else f"{(k / 4.0) * (span / 1000.0):.1f} ms"
        out.append(
            f'<line x1="{tx:.1f}" y1="{axis_y-7:.1f}" x2="{tx:.1f}" y2="{axis_y-1:.1f}" stroke="#999"'
            ' stroke-width="1"/>'
        )
        out.append(f'<text x="{tx:.1f}" y="{axis_y+9:.1f}" font-size="9" fill="#444" text-anchor="middle">{lbl}</text>')
    out.append(f'<text x="{x1+14}" y="{axis_y+2:.1f}" font-size="9" fill="#666">本层相对时间</text>')

    # 3x zoomed 3-segment strips
    out.append(
        f'<text x="{x0}" y="{zoom_title_y:.1f}" font-size="10.5" fill="#1b1b1b" font-weight="600">④ 全层切 3 段各放大 3'
        " 倍（把上方总览均分为前/中/后 3 段；每段内「折线 + 泳道细条 + 竖排算子名」上下对齐，悬停看全名）</text>"
    )
    seg_names = ("前段", "中段", "后段")
    for seg in range(3):
        s0, s1 = t0 + (seg / 3.0) * span, t0 + ((seg + 1) / 3.0) * span
        by = zoom_title_y + 24.0 + seg * seg_block_h
        seg_ops = [(i, op) for i, op in enumerate(ops) if op["end"] > s0 and op["start"] < s1 and op["dur"] > 10.0]
        out.append(
            f'<text x="{x0}" y="{by:.1f}" font-size="10" fill="#1b1b1b" font-weight="600">第 {seg+1}/3'
            f" 段（{seg_names[seg]}）：{(s0-t0)/1000:.2f}–{(s1-t0)/1000:.2f} ms（横向拉宽 3 倍，含"
            f" {len(seg_ops)} 次算子）</text>"
        )
        tick_y = by + 16.0
        for k in range(5):
            tx = x0 + (k / 4.0) * pw
            tv = ((s0 - t0) + (k / 4.0) * (s1 - s0)) / 1000.0
            out.append(
                f'<text x="{tx:.1f}" y="{tick_y:.1f}" font-size="8.5" fill="#555" text-anchor="middle">{tv:.2f}</text>'
            )
        out.append(f'<text x="{x1+8}" y="{tick_y:.1f}" font-size="8.5" fill="#555">ms</text>')

        zu_top, zu_bot = by + 24.0, by + 128.0
        for pct in (0, 25, 50, 75, 100):
            yy = zu_bot - (pct / 100.0) * (zu_bot - zu_top)
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#efefef" stroke-width="0.9"/>')
            out.append(f'<text x="{x0-8}" y="{yy+3:.1f}" font-size="8.5" fill="#555" text-anchor="end">{pct}%</text>')

        zb_top, zb_lane_h, zb_step = zu_bot + 6.0, 3.6, 4.6
        zb_mid = zb_top + (len(lanes) * zb_step) / 2.0 + 2.0
        out.append(f'<text x="{x0-8}" y="{zb_mid:.1f}" font-size="8.2" fill="#555" text-anchor="end">各流条形</text>')
        lbl_y = zb_top + len(lanes) * zb_step + 11.0
        out.append(f'<text x="{x0-8}" y="{lbl_y+14:.1f}" font-size="8.2" fill="#555" text-anchor="end">算子短名</text>')
        cands = []
        mac_segs, vec_segs = [], []
        for i, op in seg_ops:
            xs, xe = x_of(max(s0, op["start"]), s0, s1), x_of(min(s1, op["end"]), s0, s1)
            bw = max(0.8, xe - xs)
            if op["mac"] is not None and op["mac"] > 0:
                ym = zu_bot - (min(100.0, op["mac"]) / 100.0) * (zu_bot - zu_top)
                mac_segs.append((xs, xs + bw, ym))
            if op["vec"] is not None and op["vec"] > 0:
                yv = zu_bot - (min(100.0, op["vec"]) / 100.0) * (zu_bot - zu_top)
                vec_segs.append((xs, xs + bw, yv))
            if op["lane"] in lane_y:
                yy = zb_top + lanes.index(op["lane"]) * zb_step
                out.append(
                    f'<rect x="{xs:.2f}" y="{yy:.2f}" width="{bw:.2f}" height="{zb_lane_h}"'
                    f' fill="{lane_color[op["lane"]]}"><title>{tips[i]}</title></rect>'
                )
                vis_dur = min(s1, op["end"]) - max(s0, op["start"])
                if (
                    vis_dur >= (s1 - s0) * 0.002
                    or "Mhc" in op["type"]
                    or "SparseFlashMla" in op["type"]
                    or op["cat"] == "comm"
                ):
                    cands.append((xs + bw * 0.5, vis_dur, short_op_label(op["type"], op["name"])))

        if mac_segs:
            out.append(build_step_polyline(mac_segs, zu_bot, "#16a34a", 1.5))
        if vec_segs:
            out.append(build_step_polyline(vec_segs, zu_bot, "#92400e", 1.3))

        cands.sort(key=lambda c: -c[1])
        placed, chosen = [], []
        for cx, _vd, lbl in cands:
            if all(abs(cx - px) >= 10.5 for px in placed):
                placed.append(cx)
                chosen.append((cx, lbl))
        for cx, lbl in sorted(chosen, key=lambda c: c[0]):
            out.append(
                f'<text x="{cx:.2f}" y="{lbl_y:.2f}" font-size="8" fill="#1b1b1b" transform="rotate(90'
                f' {cx:.2f} {lbl_y:.2f})">{escape(lbl)}</text>'
            )

    lx = x0
    for ln in lanes:
        out.append(f'<rect x="{lx:.1f}" y="{legend_y-7:.1f}" width="12" height="8" fill="{lane_color[ln]}"/>')
        out.append(f'<text x="{lx+16:.1f}" y="{legend_y:.1f}" font-size="9" fill="#1b1b1b">{escape(ln)}</text>')
        lx += 126
    ly2 = legend_y + 20
    out.append(
        f'<rect x="{x0}" y="{ly2-3}" width="12" height="2.2" fill="#16a34a"/><text x="{x0+16}" y="{ly2}" font-size="9"'
        ' fill="#1b1b1b">MAC 矩阵利用率 % (绿线)</text>'
    )
    out.append(
        f'<rect x="{x0+185}" y="{ly2-3}" width="12" height="2.2" fill="#92400e"/><text x="{x0+201}" y="{ly2}"'
        ' font-size="9" fill="#1b1b1b">Vec 向量利用率 % (棕线)</text>'
    )
    out.append(
        f'<rect x="{x0+370}" y="{ly2-4}" width="12" height="4" fill="#16a34a"/><text x="{x0+386}" y="{ly2}"'
        ' font-size="9" fill="#1b1b1b">计算与通信并行重叠 (绿条)</text>'
    )
    out.append(
        f'<text x="{x0}" y="{ly2+20}" font-size="9" fill="#6b6b6b">颜色说明（全报告统一）：蓝=Cube 计算、黄=Vector'
        " 计算、红=HCCL 通信、紫=AI_CPU、绿=MAC 折线与并行重叠、棕=Vec 折线、灰=其余时间 Free。</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    """Parse command-line inputs and write the profiling report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Profiling directory")
    ap.add_argument("--output", type=Path, required=True, help="Output HTML path")
    ap.add_argument("--title", default="Ascend NPU 单层逐算子时间线报告")
    ap.add_argument("--scope", default="Rank 0 · Free 口径占比")
    ap.add_argument("--anchor-fwd", default="SparseFlashMla", help="Forward layer anchor OP Type")
    ap.add_argument("--anchor-bwd", default="SparseFlashMlaGrad", help="Backward layer anchor OP Type")
    ap.add_argument("--layer-idx", type=int, default=20, help="Target anchor index for forward layer")
    args = ap.parse_args()

    csv_path, is_op_sum = locate_csv(args.input)
    ops = load_ops(csv_path, is_op_sum)
    full = window_breakdown(ops)

    tf0, tf1, wf = pick_layer_window(ops, args.anchor_fwd, False, args.layer_idx)
    tb0, tb1, wb = pick_layer_window(ops, args.anchor_bwd, True, max(1, args.layer_idx // 2))
    sf = window_breakdown(wf, tf0, tf1)
    sb = window_breakdown(wb, tb0, tb1)

    svg_f = render_svg(f"{args.title} · 典型前向层", wf, tf0, tf1, sf)
    svg_b = render_svg(f"{args.title} · 典型反向层", wb, tb0, tb1, sb)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(args.title)}</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#1b1b1b}}
 header{{background:#1b1b1b;color:#fff;padding:22px 32px}} header h1{{margin:0 0 6px;font-size:22px}} header p{{margin:0;color:#cfcfcf;font-size:13px}}
 .sum{{max-width:1200px;margin:20px auto 0;padding:0 16px}}
 table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 th,td{{padding:8px 10px;border-bottom:1px solid #e6e6e6;text-align:center}} th{{background:#2b2b2b;color:#fff}} td:first-child,th:first-child{{text-align:left}}
 a{{color:#0072B2;text-decoration:none}} a:hover{{text-decoration:underline}}
 .fig{{max-width:1280px;margin:26px auto;padding:14px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);overflow-x:auto}}
 .fig h2{{font-size:17px;margin:0 0 4px}} .sub{{color:#6b6b6b;font-size:12.5px;margin:0 0 10px}} .wrap svg{{width:100%;height:auto;display:block}}
 nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 16px;z-index:9;font-size:13px}} nav a{{margin-right:14px}}
</style></head><body>
<header><h1>{escape(args.title)}</h1><p>{escape(args.scope)}</p></header>
<nav><a href="#summary">汇总</a><a href="#fwd">1. 典型前向层</a><a href="#bwd">2. 典型反向层</a></nav>
<div class="sum" id="summary">
  <h2 style="font-size:16px;margin:10px 0">占比汇总（Free 口径）</h2>
  <table><thead><tr><th>窗口</th><th>算子数</th><th>纯计算</th><th>纯通信</th><th>重叠</th><th>Free</th><th>通信被掩盖</th><th>加权 MAC</th><th></th></tr></thead><tbody>
    <tr><td><b>全步采样窗</b>（{full["span_us"]/1e6:.2f}s）</td><td>{full["count"]}</td><td>{full["pure_comp_pct"]:.1f}%</td><td>{full["pure_comm_pct"]:.1f}%</td><td>{full["overlap_pct"]:.1f}%</td><td>{full["free_pct"]:.1f}%</td><td>{full["comm_masked_pct"]:.1f}%</td><td>{format_metric(full["weighted_mac_pct"])}</td><td>—</td></tr>
    <tr><td>典型前向层（{sf["span_us"]/1000:.2f} ms）</td><td>{sf["count"]}</td><td>{sf["pure_comp_pct"]:.1f}%</td><td>{sf["pure_comm_pct"]:.1f}%</td><td>{sf["overlap_pct"]:.1f}%</td><td>{sf["free_pct"]:.1f}%</td><td>{sf["comm_masked_pct"]:.1f}%</td><td>{format_metric(sf["weighted_mac_pct"])}</td><td><a href="#fwd">跳转</a></td></tr>
    <tr><td>典型反向层（{sb["span_us"]/1000:.2f} ms）</td><td>{sb["count"]}</td><td>{sb["pure_comp_pct"]:.1f}%</td><td>{sb["pure_comm_pct"]:.1f}%</td><td>{sb["overlap_pct"]:.1f}%</td><td>{sb["free_pct"]:.1f}%</td><td>{sb["comm_masked_pct"]:.1f}%</td><td>{format_metric(sb["weighted_mac_pct"])}</td><td><a href="#bwd">跳转</a></td></tr>
  </tbody></table>
</div>
<section id="fwd" class="fig"><h2>1. 典型前向层</h2><p class="sub">层窗 {sf["span_us"]:.0f} µs · 纯计算 {sf["pure_comp_pct"]:.1f}% / 纯通信 {sf["pure_comm_pct"]:.1f}% / 重叠 {sf["overlap_pct"]:.1f}% / Free {sf["free_pct"]:.1f}%</p><div class="wrap">{svg_f}</div></section>
<section id="bwd" class="fig"><h2>2. 典型反向层</h2><p class="sub">层窗 {sb["span_us"]:.0f} µs · 纯计算 {sb["pure_comp_pct"]:.1f}% / 纯通信 {sb["pure_comm_pct"]:.1f}% / 重叠 {sb["overlap_pct"]:.1f}% / Free {sb["free_pct"]:.1f}%</p><div class="wrap">{svg_b}</div></section>
</body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{args.output} · full {full['count']} ops · fwd {sf['count']} ops · bwd {sb['count']} ops")


if __name__ == "__main__":
    main()
