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
"""Generate a single-file combined Ascend NPU profiling report:
  1. Full-step Free breakdown & bottleneck summary card
  2. Interactive zoomable Canvas timeline (merged MAC%/Vec% step-lines, optional --window-ms lite slice)
  3. Typical Forward & Backward single-layer 3x zoomed SVG reports (with one-click zoom jump buttons)
  4. Sortable operator utilization & duration ranking table

Supports kernel_details.csv, op_summary*.csv, or direct CANN SQLite msprof_*.db input.
"""
import argparse
from collections import Counter
from contextlib import closing
import csv
from html import escape
import json
from pathlib import Path
import sqlite3
import tempfile

import render_interactive_timeline as rit
import render_layer_svg as rls


def export_sqlite_db(db_path: Path, out_dir: Path, cube_cores: int = 0) -> Path:
    """Export one read-only CANN database to the common CSV schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as con:
        strings = dict(con.execute("SELECT id,value FROM STRING_IDS"))
        cols = [r[1] for r in con.execute("PRAGMA table_info(COMPUTE_TASK_INFO)")]
        has_shapes = "inputShapes" in cols
        sel = (
            "SELECT globalTaskId,opType,name,taskType,blockNum"
            + (",inputShapes" if has_shapes else "")
            + " FROM COMPUTE_TASK_INFO"
        )
        names = {}
        for row in con.execute(sel):
            task, kind, name, core, blocks = row[:5]
            sh = strings.get(row[5], "") if has_shapes else ""
            names[task] = ((strings.get(kind, ""), strings.get(name, ""), strings.get(core, "")), blocks, sh)
        measure = ("aic_total_time", "aic_mac_ratio", "aiv_total_time", "aiv_vec_ratio", "aic_mte2_ratio")
        metric_ids = [key for key, value in strings.items() if value in measure]
        pmu = {}
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if metric_ids and "TASK_PMU_INFO" in tables:
            qmarks = ",".join("?" for _ in metric_ids)
            for task, name, value in con.execute(
                f"SELECT globalTaskId,name,value FROM TASK_PMU_INFO WHERE name IN ({qmarks})",
                metric_ids,
            ):
                pmu.setdefault(task, {})[strings[name]] = value
        columns = (
            "Start Time(us)",
            "Duration(us)",
            "Type",
            "Name",
            "Accelerator Core",
            "Stream ID",
            "Task ID",
            "Input Shapes",
            "aicore_time(us)",
            "aic_mac_ratio",
            "aiv_time(us)",
            "aiv_vec_ratio",
            "aic_mte2_ratio",
            "cube_utilization(%)",
        )
        with (out_dir / "kernel_details.csv").open("w", newline="", encoding="utf-8") as dest:
            w = csv.writer(dest)
            w.writerow(columns)
            for start, end, task, stream in con.execute(
                "SELECT startNs,endNs,globalTaskId,streamId FROM TASK WHERE endNs>startNs"
            ):
                item = names.get(task)
                if not item or item[0][2] == "COMMUNICATION":
                    continue
                identity, blocks, sh = item
                vals = pmu.get(task, {})
                aic, mac, aiv, vector, mte2 = [vals.get(k) for k in measure]
                cube = (
                    100 * aic / (end - start) * blocks / cube_cores
                    if aic and cube_cores > 0 and blocks and 0 < blocks <= cube_cores
                    else None
                )
                w.writerow(
                    (
                        start / 1000,
                        (end - start) / 1000,
                        *identity,
                        stream,
                        task,
                        sh,
                        None if aic is None else aic / 1000,
                        mac,
                        None if aiv is None else aiv / 1000,
                        vector,
                        mte2,
                        cube,
                    )
                )
            comm_rows = (
                con.execute("SELECT opType,opName,startNs,endNs FROM COMMUNICATION_OP WHERE endNs>startNs")
                if "COMMUNICATION_OP" in tables
                else []
            )
            for kind, name, start, end in comm_rows:
                w.writerow(
                    (
                        start / 1000,
                        (end - start) / 1000,
                        strings.get(kind, "HCCL"),
                        strings.get(name, "HCCL"),
                        "COMMUNICATION",
                        "",
                        "N/A",
                        "",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                )
    return out_dir


def resolve_input_dir(inp: Path, tmp_dir: Path, cube_cores: int = 0) -> Path:
    """Resolve one CSV source or export one explicitly selected database."""
    if inp.is_file() and inp.suffix == ".csv":
        return inp
    if inp.is_file() and inp.suffix == ".db":
        return export_sqlite_db(inp, tmp_dir, cube_cores)
    if inp.is_dir():
        if (inp / "kernel_details.csv").is_file() or list(inp.glob("op_summary*.csv")):
            return inp
        dbs = sorted(inp.glob("**/msprof_*.db"))
        if len(dbs) > 1:
            raise ValueError("Multiple databases found; select one rank/device .db explicitly")
        if dbs:
            return export_sqlite_db(dbs[0], tmp_dir, cube_cores)
    raise FileNotFoundError(f"No kernel_details.csv, op_summary*.csv, or msprof_*.db found in {inp}")


def build_bottleneck_alert(ops: list[dict], full: dict) -> str:
    """Summarize observed operator durations and communication overlap."""
    comm_dur = Counter()
    comp_dur = Counter()
    comp_cnt = Counter()
    for op in ops:
        if op["cat"] == "comm":
            comm_dur[op["type"]] += op["dur"]
        elif op["cat"] == "compute":
            comp_dur[op["type"]] += op["dur"]
            comp_cnt[op["type"]] += 1
    top_comm = "、".join(f"<code>{escape(k)}</code> {v/1e6:.2f}s" for k, v in comm_dur.most_common(4))
    top_comp = "、".join(
        f"<code>{escape(k)}</code> {v/1e6:.2f}s ({comp_cnt[k]} 次)" for k, v in comp_dur.most_common(3)
    )
    return (
        '<div class="alert-box"><b>核心瓶颈诊断：</b>① <b>通信占比</b>：未掩盖纯通信'
        f' <b>{full["pure_comm_pct"]:.1f}%</b>（通信掩盖率 <b>{full["comm_masked_pct"]:.1f}%</b>），'
        f'主要通信：{top_comm or "无"}；② <b>计算 Top 耗时</b>：{top_comp or "无"}（加权 MAC'
        f' <b>{rls.format_metric(full["weighted_mac_pct"])}</b> / Vec'
        f' <b>{rls.format_metric(full["weighted_vec_pct"])}</b>）。</div>'
    )


def main() -> None:
    """Parse command-line inputs and write the profiling report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Profiling dir (CSV/DB) or msprof_*.db file")
    ap.add_argument("output", type=Path, help="Output single-file HTML report")
    ap.add_argument("--title", default="Ascend NPU 训练逐算子时间线报告", help="Report title")
    ap.add_argument("--scope", required=True, help="Run / hardware / step scope subtitle")
    ap.add_argument(
        "--window-ms", default="", help="Optional interactive Canvas slice START:END in ms (for fast Lite HTML)"
    )
    ap.add_argument(
        "--min-duration-us", type=float, default=0.0, help="Optional minimum duration (us) for interactive Canvas"
    )
    ap.add_argument(
        "--anchor-fwd", default="", help="Forward layer anchor OP Type (auto-detects SparseFlashMla or GroupedMatmul)"
    )
    ap.add_argument(
        "--anchor-bwd",
        default="",
        help="Backward layer anchor OP Type (auto-detects SparseFlashMlaGrad or GroupedMatmul)",
    )
    ap.add_argument(
        "--cube-cores",
        type=int,
        default=0,
        help="Verified Cube core count per profiled device; 0 leaves DB Cube utilization unavailable",
    )
    args = ap.parse_args()
    if args.cube_cores < 0:
        raise ValueError("cube-cores must be nonnegative")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        prof_dir = resolve_input_dir(args.input, td_path / "exported", args.cube_cores)
        inter_tmp = td_path / "inter.html"

        timeline_args = [str(prof_dir), str(inter_tmp), "--sequence", args.title, "--scope", args.scope]
        if args.window_ms:
            timeline_args += ["--window-ms", args.window_ms]
        timeline_args += ["--min-duration-us", str(args.min_duration_us)]
        rit.main(timeline_args)

        # 2) Build forward & backward single-layer 3x zoomed SVGs
        csv_path, is_op_sum = rls.locate_csv(prof_dir)
        ops = rls.load_ops(csv_path, is_op_sum)
        full = rls.window_breakdown(ops)
        origin = ops[0]["start"]

        op_types = {op["type"] for op in ops}
        fwd_anchor = args.anchor_fwd or next(
            (
                c
                for c in ("SparseFlashMla", "FlashAttentionScore", "GroupedMatmul", "RotaryPositionEmbedding")
                if c in op_types
            ),
            ops[0]["type"],
        )
        bwd_anchor = args.anchor_bwd or next(
            (c for c in ("SparseFlashMlaGrad", "FlashAttentionScoreGrad", "SoftmaxGrad") if c in op_types),
            fwd_anchor,
        )

        tf0, tf1, wf = rls.pick_layer_window(ops, fwd_anchor, False, 10)
        tb0, tb1, wb = rls.pick_layer_window(ops, bwd_anchor, True, 10 if bwd_anchor != fwd_anchor else 40)

        sf = rls.window_breakdown(wf, tf0, tf1)
        sb = rls.window_breakdown(wb, tb0, tb1)
        svg_f = rls.render_svg(f"{args.title} · 典型前向层逐算子时间线", wf, tf0, tf1, sf)
        svg_b = rls.render_svg(f"{args.title} · 典型反向层逐算子时间线", wb, tb0, tb1, sb)

        anchor_json = json.dumps(fwd_anchor).replace("<", "\\u003c")

        # 3) Combine into one self-contained HTML
        html = inter_tmp.read_text(encoding="utf-8")
        extra_css = (
            ".wrap{overflow-x:auto}.wrap svg{width:100%;height:auto;display:block}"
            ".alert-box{padding:10px 14px;background:#fff5f5;border-left:3px solid"
            " #dc2626;color:#7f1d1d;border-radius:0 6px 6px 0;margin:12px 0;font-size:13.5px}"
            ".guide-box{padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid"
            " #2563eb;border-radius:0 6px 6px 0;margin:8px 0 12px;font-size:13px;line-height:1.65;color:#334155}"
            ".guide-box b{color:#0f172a}"
            ".sec-head{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}"
        )
        html = html.replace("</style>", extra_css + "\n</style>", 1)
        html = html.replace(
            '<a href="#ranking">算子利用率</a>',
            '<a href="#fwd">典型前向层 SVG</a>　<a href="#bwd">典型反向层 SVG</a>　<a href="#ranking">算子利用率</a>',
            1,
        )
        html = html.replace(
            '</section>\n<section id="timeline">',
            build_bottleneck_alert(ops, full) + '</section>\n<section id="timeline">',
            1,
        )

        def layer_guide(st: dict, phase_desc: str) -> str:
            """Explain the coverage categories and plot layout."""
            return (
                '<div class="guide-box">'
                f'<b>本层一眼结论（{phase_desc}，总耗时 {st["span_us"]/1000:.2f} ms）：</b>'
                f'只有计算在跑 <b>{st["pure_comp_pct"]:.1f}%</b>（蓝） · '
                f'只有通信在等 <b>{st["pure_comm_pct"]:.1f}%</b>（红） · '
                f'计算和通信同时跑 <b>{st["overlap_pct"]:.1f}%</b>（绿） · '
                f'其余时间 <b>{st["free_pct"]:.1f}%</b>（灰）；'
                f'总通信时间里有 <b>{st["comm_masked_pct"]:.1f}%</b> 被计算成功盖住。<br>'
                "<b>读图顺序（自上而下 4 块）：</b>"
                "<b>① 算力忙不忙</b>（绿线=矩阵 MAC 利用率、棕线=向量 Vec 利用率，越高越忙）→ "
                "<b>② 各通道时间线</b>（每行一条执行流/通信通道，条越宽耗时越长，上下对齐=同一时刻并行）→ "
                "<b>③ 并行重叠绿条</b>（计算流与通信流同时在跑的区间，绿条越满越好）→ "
                "<b>④ 切 3 段各放大 3 倍</b>（把上面全层均分为前/中/后 3 段展开，每段内折线、细彩条、"
                "竖排算子名上下对齐；鼠标悬停任意色块看完整算子名和耗时）。"
                "</div>"
            )

        sec_f = (
            f'<section id="fwd"><div class="sec-head"><h2>1. 典型前向层（总耗时 {sf["span_us"]/1000:.2f} ms · 静态 3×'
            " 放大拆解图）</h2>"
            '<button id="zoom-this-fwd">🔍 在上方交互图中自由缩放查看此层</button></div>'
            f'{layer_guide(sf, "单层前向计算")}'
            f'<div class="wrap">{svg_f}</div></section>'
        )
        sec_b = (
            f'<section id="bwd"><div class="sec-head"><h2>2. 典型反向层（总耗时 {sb["span_us"]/1000:.2f} ms · 静态 3×'
            " 放大拆解图）</h2>"
            '<button id="zoom-this-bwd">🔍 在上方交互图中自由缩放查看此层</button></div>'
            f'{layer_guide(sb, "单层反向求梯度与通信")}'
            f'<div class="wrap">{svg_b}</div></section>'
        )
        html = html.replace('<section id="ranking">', sec_f + "\n" + sec_b + '\n<section id="ranking">', 1)

        if not args.window_ms:
            rel_f0, rel_f1 = tf0 - origin, tf1 - origin
            rel_b0, rel_b1 = tb0 - origin, tb1 - origin
            html = html.replace(
                'q("jump-fwd").onclick=()=>jumpLayer("SparseFlashMla");\n '
                ' q("jump-bwd").onclick=()=>jumpLayer("SparseFlashMlaGrad");',
                f'q("jump-fwd").onclick=()=>{{limit({rel_f0:.1f},{rel_f1-rel_f0:.1f});}};\n'
                f'  q("jump-bwd").onclick=()=>{{limit({rel_b0:.1f},{rel_b1-rel_b0:.1f});}};\n'
                "  const"
                f' zf=q("zoom-this-fwd");if(zf)zf.onclick=()=>{{limit({rel_f0:.1f},{rel_f1-rel_f0:.1f});'
                'q("timeline").scrollIntoView({behavior:"smooth"});};\n'
                "  const"
                f' zb=q("zoom-this-bwd");if(zb)zb.onclick=()=>{{limit({rel_b0:.1f},{rel_b1-rel_b0:.1f});'
                'q("timeline").scrollIntoView({behavior:"smooth"});};',
            )
        else:
            html = html.replace(
                'q("jump-fwd").onclick=()=>jumpLayer("SparseFlashMla");\n '
                ' q("jump-bwd").onclick=()=>jumpLayer("SparseFlashMlaGrad");',
                f'q("jump-fwd").onclick=()=>jumpLayer({anchor_json});\n  q("jump-bwd").style.display="none";\n  const'
                f' zf=q("zoom-this-fwd");if(zf)zf.onclick=()=>{{jumpLayer({anchor_json});'
                'q("timeline").scrollIntoView({behavior:"smooth"});};\n'
                '  const zb=q("zoom-this-bwd");if(zb)zb.style.display="none";',
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
        print(f"COMBINED_HTML={args.output} SIZE={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
