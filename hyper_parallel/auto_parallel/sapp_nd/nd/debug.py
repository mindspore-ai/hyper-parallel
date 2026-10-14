# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Debugging utilities"""

import os
import colorsys
import csv
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from math import isnan, sqrt
from typing import Any, Iterator, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import yaml

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.recompute_dimension import read_recompute_modes

# The column a configuration's activation checkpoint mode goes in, written by a
# search with a recompute dimension and read from a measured CSV that states it.
RECOMPUTE_COLUMN = "recompute"

# The results table's recompute rows: the activation checkpoint mode a
# configuration runs, and the plan per layer that names the exceptions to it.
MODE_ROW = "RMOD"
LAYER_ROW = "RLAYER"
# Rows whose cells hold text, which carries no shading and no largest value.
_TEXT_ROWS = (MODE_ROW, LAYER_ROW)


def _recompute_of(row: dict) -> Optional[str]:
    """Take a measured row's activation checkpoint mode out of it, None where the CSV states none."""
    stated = (row.pop(RECOMPUTE_COLUMN, None) or "").strip()
    if not stated:
        return None
    modes = read_recompute_modes(stated, f"the {RECOMPUTE_COLUMN} column")
    if len(modes) != 1:
        raise ValueError(f"the {RECOMPUTE_COLUMN} column: a measured run ran one mode, not {stated!r}")
    return modes[0]


def _has_recompute(entries: list) -> bool:
    """Whether any entry's configuration states an activation checkpoint mode."""
    return any(getattr(entry[0], "recompute", None) for entry in entries)


# Where the CSVs and the plots go. Defaults to a directory beside this file,
# which is unwritable in an installed package and mixes consecutive runs
# together, so ``run_nd -o`` overrides it.
_OUTPUT_DIR = None
# The CSVs Debug.write appends a row a candidate to.
DEBUG_CSVS = ("debug.csv", "debug_mem.csv")


def set_output_dir(path: Optional[str], fresh: bool = False) -> None:
    """Send the debug artifacts to *path* instead of the package directory.

    Args:
        path: the run's output folder, or None for the package's own.
        fresh: remove the debug CSVs an earlier run left there. ``Debug.write`` appends and writes the column
            titles only into a new file, so a run into a folder another run wrote would add its rows under that
            run's. The process that starts a run passes True; the pool's workers, which append to the files
            the run started, do not.
    """
    global _OUTPUT_DIR  # pylint: disable=global-statement
    _OUTPUT_DIR = str(path) if path else None
    if _OUTPUT_DIR:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
    if fresh:
        for name in DEBUG_CSVS:
            stale = os.path.join(output_dir(), name)
            if os.path.isfile(stale):
                os.remove(stale)


def output_dir() -> str:
    """Return the directory the debug artifacts are written to."""
    if _OUTPUT_DIR:
        return _OUTPUT_DIR
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class PerfParts(Enum):
    """decomposition of performance"""

    FW_COMPUTE = auto()
    BW_COMPUTE = auto()
    RECOMPUTE = auto()
    # FSDP's gathers and reduce-scatters, and a run's other DP traffic.
    DP_COMM = auto()
    # The all-reduce among the copies of an FSDP shard: HSDP's once a step,
    # every gradient's where FSDP shards nothing.
    DP_REDUCE = auto()
    MP_COMM = auto()
    EP_COMM = auto()
    CP_COMM = auto()
    PP_COMM = auto()
    BUBBLE = auto()
    TOTAL = auto()
    MEMORY = auto()

    def __str__(self):
        return self.name

    def short_name(self):
        """Returns short component name"""
        name = "Perf"
        if self == self.FW_COMPUTE:
            name = "FW"
        elif self == self.BW_COMPUTE:
            name = "BW"
        elif self == self.RECOMPUTE:
            name = "Rec"
        elif self == self.DP_COMM:
            name = "DP"
        elif self == self.DP_REDUCE:
            name = "AR"
        elif self == self.MP_COMM:
            name = "MP"
        elif self == self.EP_COMM:
            name = "EP"
        elif self == self.CP_COMM:
            name = "CP"
        elif self == self.PP_COMM:
            name = "P2P"
        elif self == self.BUBBLE:
            name = "BBL"
        elif self == self.MEMORY:
            name = "MEM"
        return name


class RealParts(Enum):
    """decomposition of performance"""

    COMP = auto()
    DP_WAIT = auto()
    MP_WAIT = auto()
    EP_WAIT = auto()
    CP_WAIT = auto()
    PP_WAIT = auto()
    IDLE = auto()
    TOTAL = auto()

    def __str__(self):
        return self.name.lower()


# An optional column of a classified CSV, and the key it is read into: the
# trainer's own step time in milliseconds, measured on steps the profiler left
# alone. It is not a RealParts member, so every consumer that walks that enum
# ignores it. ``examples/training_demo/sweep_qwen3_5_moe.py`` writes the column
# and spells the name again rather than importing it, because that launcher
# runs on the cluster's control node with nothing but the standard library.
TRAINER_STEP = "step_trainer"


class MemParts(Enum):
    """decomposition of memory"""

    TOTAL = auto()

    def __str__(self):
        return self.name


class Debug:
    """Debugging tools"""

    def __init__(
        self,
        parallel_dimensions,
        info_type,
        enable=True,
        output_file="debug.csv",
    ):
        self.enable = enable
        if self.enable:
            self.parallel_dimensions = parallel_dimensions
            self.info = {p: 0 for p in info_type}
            self.output_file = os.path.join(output_dir(), output_file)

    def is_enabled(self):
        """Check whether debugging is enabled"""
        return self.enable

    def column_titles(self):
        """Parameters to debug"""
        titles = self.parallel_dimensions.keys() + list(self.info.keys())
        titles = [str(t) for t in titles]
        return ",".join(titles) + "\n"

    def values(self):
        """values debugged"""
        str_dims = [str(v) for v in self.parallel_dimensions.values()]
        str_score = [str(int(v)) for v in self.info.values()]
        return ",".join(str_dims + str_score) + "\n"

    def write(self):
        """Append this row to the CSV, under the column titles where the file is new.

        Inside :func:`holding_open` the file stays open for the next row.
        """
        if self.enable:
            logger.info("debug written")
            held = _HELD.files.get(self.output_file)
            if held is not None:
                held.write(self.values())
                return
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            is_new = not os.path.exists(self.output_file)
            outfile = open(self.output_file, "a", encoding="utf-8")  # pylint: disable=consider-using-with
            if _HELD.depth:
                _HELD.files[self.output_file] = outfile
            try:
                if is_new:
                    outfile.write(self.column_titles())
                outfile.write(self.values())
            finally:
                if not _HELD.depth:
                    outfile.close()


class _HeldFiles:
    """The debug CSVs a search holds open while it writes them, by path."""

    def __init__(self) -> None:
        """No block open and no file held."""
        self.depth = 0
        self.files = {}


_HELD = _HeldFiles()


@contextmanager
def holding_open() -> Iterator[None]:
    """Keep each debug CSV written inside the block open until the block ends.

    A search writes a row a candidate, and opening and closing the file for
    each row cost 3.3 s of the Qwen3.5 example's 14507 rows in a temporary
    folder and 6.5 s inside OneDrive.  The files are closed, so whole on
    disk, when the outermost block ends; a row written outside any block
    still opens and closes its file.
    """
    _HELD.depth += 1
    try:
        yield
    finally:
        _HELD.depth -= 1
        if not _HELD.depth:
            while _HELD.files:
                _HELD.files.popitem()[1].close()


def pastel(color, l_delta=0.0, lbl=None, sat=None):
    "Pastel (lighter) color of the input"
    if color == "white":
        return (1.0, 1.0, 1.0)
    if color == "black":
        return (0.5, 0.5, 0.5)
    try:
        color = mc.cnames[color]
    except KeyError:
        pass
    color_hls = colorsys.rgb_to_hls(*mc.to_rgb(color))
    lgt = 0.7
    if lbl is not None:
        lgt = lbl
    lgt = lgt + l_delta

    if sat is None:
        sat = 0.6
    return colorsys.hls_to_rgb(color_hls[0], lgt, sat)


def near_white(color, ratio):
    "Very light color of the input for background"
    rgb = mc.to_rgb(color)
    if rgb is None:
        return "white"
    (red, green, blue) = rgb
    red += (1 - red) * ratio
    green += (1 - green) * ratio
    blue += (1 - blue) * ratio
    return (red, green, blue)


def dim_color(dim, default="black"):
    """Color of parallel dimensions for plot"""
    color = {
        Dim.DP: "orange",
        Dim.OP: "orange",
        Dim.TP: "red",
        Dim.EP: "blue",
        Dim.CP: "teal",
        Dim.PP: "green",
        Dim.VPP: "green",
        Dim.MBN: "green",
    }
    try:
        dim = Dim.get_dim(dim)
        if dim in color:
            return color[dim]
        return default
    except ValueError:
        return default


def gen_colors(categories):
    """Color of each time component"""
    compute_color = "purple"
    idle_color = "grey"
    col_d = {
        str(PerfParts.FW_COMPUTE): pastel(compute_color, -0.2),
        str(PerfParts.BW_COMPUTE): pastel(compute_color, -0.1),
        str(PerfParts.RECOMPUTE): pastel(compute_color),
        str(PerfParts.DP_COMM): pastel(dim_color(Dim.DP)),
        str(PerfParts.DP_REDUCE): pastel(dim_color(Dim.DP), -0.15),
        str(PerfParts.MP_COMM): pastel(dim_color(Dim.TP), -0.1),
        str(PerfParts.EP_COMM): pastel(dim_color(Dim.EP)),
        str(PerfParts.CP_COMM): pastel(dim_color(Dim.CP)),
        str(PerfParts.PP_COMM): pastel(dim_color(Dim.PP)),
        str(PerfParts.BUBBLE): pastel(dim_color(Dim.PP), -0.15),
        "IDLE": idle_color,
        "COMPUTATION": pastel(compute_color, -0.2),
    }
    return [col_d.get(cat) for cat in categories]


def set_twin_handles(ax1, data_frame, dbg_cols):
    """Set legend for estimation and real"""
    handle1, label1 = ax1.get_legend_handles_labels()
    ax2 = plt.twinx()
    data_frame[dbg_cols].plot.bar(
        stacked=True,
        sharex=True,
        ax=ax2,
        position=0,
        color=gen_colors(dbg_cols),
        width=0.4,
        rot=0,
    )

    handle2, label2 = ax2.get_legend_handles_labels()  # type: ignore
    # Patches compare by identity, so testing the handle admitted every one of
    # them while the labels deduplicated by name: the legend then drew more
    # swatches than it had names and mislabelled every entry past the first
    # shared one. Deduplicate on the label and keep its handle in step.
    for handle, lbl in zip(handle2, label2):
        if lbl not in label1:
            handle1.append(handle)
            label1.append(lbl)
    handles = handle1
    labels = label1
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
    leg = ax2.get_legend()
    pp_color = gen_colors(["PP_COMM"])[0]
    leg.legend_handles[-1].set_facecolor(pp_color)  # type: ignore


# The measured parts of the comparison plot, named by the ND part each is set
# against: FSDP waits count as DP and sequence-parallel waits as MP, as they
# do in the correlations.
MEASURED_BARS = ("COMPUTATION", "DP_COMM", "MP_COMM", "EP_COMM", "CP_COMM", "BUBBLE")


def measured_bars(waits: dict, plot_idle: bool = False) -> list:
    """Return one configuration's measured parts in ``MEASURED_BARS`` order.

    Every wait column of the classified CSV lands in one bar, ``op_wait`` in
    DP's and ``sp_wait`` in MP's as ``real_in_parts`` counts them, so with idle
    the stack adds up to the measured step. A part the CSV lacks is zero.

    Args:
        waits: A configuration's measured parts, as ``get_comm_classified_data`` reads them.
        plot_idle: Whether to append the idle remainder.
    """
    def part(name: str) -> float:
        """The measured part *name*, zero when the CSV does not have it."""
        return waits.get(name) or 0.0

    bars = [
        part("comp"),
        part("dp_wait") + part("op_wait"),
        part("mp_wait") + part("sp_wait"),
        part("ep_wait"),
        part("cp_wait"),
        part("BUBBLE"),
    ]
    if plot_idle:
        bars.append(part("IDLE"))
    return bars


def _cell_number(text):
    """Read one cell of the degree table as a number.

    A boolean dimension such as SP prints as True or False, which float()
    refuses, so every search that varied SP failed as its plot was drawn.
    """
    if text in ("True", "False"):
        return float(text == "True")
    return float(text)


class Plot:
    """plot ND top configs"""

    title: str
    col_title: list[str]
    row_title: list[str]
    cell_text: list[list[str]]
    data: list[tuple]
    dbg_cols: list[str]
    top: int

    def __init__(
        self,
        title: Optional[str],
        rows: list,
        debug_parts: list,
        top: Optional[int] = None,
        show_top: bool = False,
    ) -> None:
        """Initialize the plot table, optionally displaying each configuration's rank."""
        self.title = title
        self.top = top if top is not None else 20
        self.row_title = rows + ["MEM"]
        if show_top:
            self.row_title.insert(0, "TOP")
        self.dbg_cols = list(map(str, debug_parts))
        self.col_title = []
        self.cell_text = []
        self.data = []

    def make_table(self):
        """Make the table below the plot with each parallelism degree and the recompute rows.

        A recompute row holds text rather than a degree, so it keeps one
        shade and its cells grow with the lines its plan takes.
        """
        self.cell_text = list(map(list, zip(*self.cell_text)))  # transpose
        max_rows = [
            None if title in _TEXT_ROWS else max(map(_cell_number, cells))
            for title, cells in zip(self.row_title, self.cell_text)
        ]
        the_table = plt.table(
            cellText=self.cell_text,
            rowLabels=self.row_title,
            colLabels=self.col_title,
            cellLoc="center",
            loc="bottom",
        )
        row_colors = list(map(dim_color, self.row_title))
        for row in range(len(self.row_title)):
            lines = max(str(text).count("\n") + 1 for text in self.cell_text[row])
            cell = the_table[row + 1, -1]
            cell.set_edgecolor("none")
            cell.get_text().set_color(row_colors[row])
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))
            cell.set_height(cell.get_height() * lines)
            for col in range(len(self.cell_text[0])):
                self._shade_cell(the_table[row + 1, col], row, max_rows[row], lines, row_colors[row])

        for col in range(len(self.cell_text[0])):
            the_table[0, col].set_edgecolor("none")

        the_table.scale(xscale=1, yscale=1.2)  # +len(rows)/5)

    def _shade_cell(self, cell: Any, row: int, largest: Optional[float], lines: int, color: Any) -> None:
        """Shade one cell of the table by its value against the row's largest.

        Args:
            cell: The cell to shade.
            row: Its row, which names the degree or the recompute setting.
            largest: The largest value the row holds; ``None`` for a row of
                text, which keeps the row's lightest shade.
            lines: The lines the row's tallest cell takes.
            color: The row's colour.
        """
        cell.set_height(cell.get_height() * lines)
        if largest is None:
            ratio = 1
        else:
            try:
                ratio = 1 - (_cell_number(str(cell.get_text().get_text())) / largest)
            except ZeroDivisionError:
                ratio = 0
        if self.row_title[row] == LAYER_ROW:
            cell.set_text_props(ha="center", fontfamily="monospace")
        logger.debug(
            "tmax = %s, ratio = %f, col=%s, newcolor=%s",
            str(largest),
            ratio,
            str(mc.to_rgb(color)),
            str(near_white(color, ratio)),
        )
        cell.set_facecolor(near_white(pastel(color), ratio))
        cell.set_edgecolor("none")

    def close(self, output_path, filename):
        """Plot closing statements"""
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.xlim([-0.5, len(self.data) - 0.5])
        if self.title is not None:
            plt.title(self.title)
        plt.subplots_adjust(left=0.1, bottom=0.047 * (2 + len(self.row_title)))
        plotfile = os.path.join(output_path, filename + ".pdf")
        plt.savefig(plotfile, bbox_inches="tight")
        plt.clf()

    def parse_data(
        self,
        configs_estimated,
        **kwargs,
    ):
        """Parse test data for plot"""
        real_data = kwargs.get("real_data", None)
        plot_idle = kwargs.get("plot_idle", False)
        include_all = kwargs.get("include_all", False)
        recompute_plans = kwargs.get("recompute_plans")
        min_e = configs_estimated[0][2]
        i = 0
        for index, cfg_e in enumerate(configs_estimated):
            cells = cfg_e[0].values() + self._recompute_cells(cfg_e[0], recompute_plans) + [cfg_e[1]]
            if self.row_title[0] == "TOP":
                cells.insert(0, cfg_e[0].rank or index + 1)
            self.cell_text.append(cells)
            self.col_title.append("")
            try:
                self.data.append(
                    tuple([cfg_e[0], cfg_e[2], cfg_e[3]] + cfg_e[4])
                )
                if real_data is not None:
                    logger.info(cfg_e[5])
                    real_data.append(tuple(measured_bars(cfg_e[5], plot_idle)))
            except IndexError:
                score = cfg_e[2]
                if not include_all and (
                    i >= self.top or (min_e is not None and score > min_e * 20)
                ):
                    self.cell_text.pop()
                    break
                self.data.append(tuple([cfg_e[0], score] + cfg_e[3]))
                i += 1

    def _recompute_cells(self, config: Any, recompute_plans: Optional[dict]) -> list:
        """The recompute cells of one configuration's column, empty where the table has no such row.

        Args:
            config: The configuration the column states.
            recompute_plans: ``{configuration: (mode, layers)}`` from a
                choice among named modes per layer; ``None`` where the search
                made none.

        Returns:
            The mode alone, the mode and its plan per layer, or nothing.
        """
        if MODE_ROW not in self.row_title:
            return []
        mode, layers = (recompute_plans or {}).get(config, (getattr(config, "recompute", None), None))
        cells = [mode.upper() if mode else "-"]
        if LAYER_ROW in self.row_title:
            cells.append(
                yaml.safe_dump(layers, default_flow_style=False, sort_keys=False).strip()
                if layers is not None else "-"
            )
        return cells


def top_plot_configs(configs_estimated: list, max_num: Optional[int] = None) -> list:
    """Return the configurations the standard ND plot would include."""
    if not configs_estimated:
        return []
    top = max_num if max_num is not None else 20
    if top <= 0:
        return []
    best_score = configs_estimated[0][2]
    selected = []
    for config in configs_estimated:
        if len(selected) >= top or config[2] > best_score * 20:
            break
        selected.append(config)
    return selected


def plot_nd(
    configs_estimated: list,
    output_path: str,
    debug_parts: list,
    title: Optional[str] = None,
    max_num: Optional[int] = None,
    include_all: bool = False,
    top_result_count: Optional[int] = None,
    recompute_plans: Optional[dict] = None,
) -> None:
    """Plot estimation with optional recompute rows and a divider before additions.

    Args:
        configs_estimated: Ranked configurations with memory, score and its parts.
            TOP uses the rank stored on each configuration, falling back to its
            position in the supplied list when it has not been ranked.
        output_path: Directory in which to save results.pdf.
        debug_parts: Performance components to plot.
        title: Optional plot title.
        max_num: Maximum number of configurations in the normal plot.
        include_all: Include every supplied configuration without normal plot limits.
        top_result_count: Number of normal results preceding the additions.
        recompute_plans: ``{configuration: (mode, layers)}``, the activation
            checkpoint mode and the plan per layer a choice among named modes
            made. ``None`` omits the plan row; a configuration that states a
            mode of its own still shows it.
    """
    plot = Plot(
        title, configs_estimated[0][0].keys(), debug_parts,
        top=max_num, show_top=True,
    )
    if _has_recompute(configs_estimated) or recompute_plans is not None:
        plot.row_title.insert(-1, MODE_ROW)
    if recompute_plans is not None:
        plot.row_title.insert(-1, LAYER_ROW)
    plot.parse_data(configs_estimated, include_all=include_all, recompute_plans=recompute_plans)

    data_frame = pd.DataFrame(
        plot.data, columns=(["config", "estim"] + plot.dbg_cols)
    )
    axis = data_frame[plot.dbg_cols].plot.bar(
        stacked=True, color=gen_colors(plot.dbg_cols), width=0.4, rot=0
    )
    axis.set_ylim(ymin=1)
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if top_result_count is not None and 0 < top_result_count < len(plot.data):
        axis.axvline(top_result_count - 0.5, color="black", linewidth=1.5)

    plot.make_table()
    plot.close(output_path, "results")


def plot_recompute(
    groups: list,
    output_path: str,
    debug_parts: list,
    filename: str,
    title: Optional[str] = None,
    columns_per_page: int = 12,
) -> None:
    """Write a summary of every selected recompute plan, then a detail page per strategy.

    Each bar is the cost model's estimate of the step under that plan, split
    into its parts, so the step between two bars is what the cost model
    expects the extra recomputed layer to cost.

    Args:
        groups: The ranked parallel configurations and the records of their
            selected plans, as ``recompute_plot_data`` prepares them.
        output_path: Directory in which to save the PDF.
        debug_parts: The performance components in each record's parts.
        filename: The PDF's basename, without its extension.
        title: Optional description of the model and the selection.
        columns_per_page: The most plans on one detail page. The summary
            holds every selected plan, with a divider between strategies.

    Raises:
        ValueError: The page size is not positive, or a record's score parts
            do not cover every component.
    """
    if columns_per_page <= 0:
        raise ValueError("columns_per_page must be positive")
    if not groups:
        return
    os.makedirs(output_path, exist_ok=True)
    heading = title or "ND recompute"
    with PdfPages(os.path.join(output_path, filename + ".pdf")) as document:
        _save_page(document, _recompute_summary(groups, debug_parts, heading))
        for config, records in groups:
            _save_detail_pages(document, config, records, debug_parts, heading, columns_per_page)


def _save_page(document: Any, figure: Any) -> None:
    """Add one figure to the PDF and release it."""
    document.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _recompute_summary(groups: list, debug_parts: list, heading: str) -> Any:
    """The page that holds every selected plan, the strategies it covers named where it has none."""
    plans = [(config, record) for config, records in groups for record in records]
    labels = [f"{config.rank}.{index + 1}" for config, records in groups for index in range(len(records))]
    page_title = f"{heading}\nSummary | {len(plans)} plans across {len(groups)} configurations"
    if not plans:
        return _empty_recompute_page([config for config, _ in groups], page_title)
    figure = _recompute_page(plans, debug_parts, page_title, labels)
    empty = [str(config.rank) for config, records in groups if not records]
    if empty:
        figure.text(0.5, 0.01, "No qualifying plans for TOP " + ", ".join(empty), ha="center", fontsize=8)
    return figure


def _save_detail_pages(
    document: Any, config: Any, records: list, debug_parts: list, heading: str, columns_per_page: int
) -> None:
    """Add one strategy's plans to the PDF, paginated, or a page saying it has none."""
    if not records:
        _save_page(document, _empty_recompute_page([config], f"{heading}\nTOP {config.rank}"))
        return
    for start in range(0, len(records), columns_per_page):
        page = records[start:start + columns_per_page]
        page_title = (
            f"{heading}\n"
            f"TOP {config.rank} | plans {start + 1}-{start + len(page)} of {len(records)}"
        )
        labels = [str(start + index + 1) for index in range(len(page))]
        _save_page(document, _recompute_page([(config, record) for record in page], debug_parts, page_title, labels))


def _empty_recompute_page(configs: list, title: str) -> Any:
    """Say that a selection is empty, and name the strategies it covers."""
    figure = plt.figure(figsize=(12, max(5, 3 + 0.3 * len(configs))))
    figure.text(0.5, 0.8, title, ha="center", fontsize=12)
    figure.text(0.5, 0.6, "No recompute plan satisfies the selection for these configurations.", ha="center")
    descriptions = [
        f"TOP {config.rank}: " + "  ".join(f"{dim}={value}" for dim, value in config.dims_val.items())
        for config in configs
    ]
    figure.text(0.5, 0.45, "\n".join(descriptions), ha="center", va="top")
    return figure


def _recompute_page(plans: list, debug_parts: list, title: str, labels: list) -> Any:
    """Draw one page of plans, with a layer table under the chart and a divider between strategies."""
    if any(len(record["parts"]) != len(debug_parts) for _, record in plans):
        raise ValueError("recompute plot records must include every performance component")
    rows = ["TOP"] + plans[0][0].keys() + ["RFULL", MODE_ROW, LAYER_ROW, "MEM", "SCORE"]
    cells = list(map(list, zip(*[_recompute_column(config, record) for config, record in plans])))
    lines = [max(str(text).count("\n") + 1 for text in row) for row in cells]
    heights = [0.20] + [0.18 * count + 0.06 for count in lines]
    table_height = sum(heights)
    figure = plt.figure(figsize=(max(12, 1.3 * len(plans) + 3), 4.5 + table_height))
    grid = figure.add_gridspec(2, 1, height_ratios=[3.5, table_height], hspace=0.10)
    axis = _recompute_chart(figure.add_subplot(grid[0]), plans, debug_parts, title, labels)
    table_axis = figure.add_subplot(grid[1])
    _recompute_table(table_axis, rows, cells, labels, heights)
    for index in range(1, len(plans)):
        if plans[index][0] != plans[index - 1][0]:
            axis.axvline(index - 0.5, color="black", linewidth=1.5)
            table_axis.axvline(index / len(plans), color="black", linewidth=1.5, zorder=10)
    figure.subplots_adjust(left=0.09, right=0.84, top=0.88, bottom=0.035)
    return figure


def _recompute_column(config: Any, record: dict) -> list:
    """One plan's column: its strategy, how many layers it fully recomputes, its plan, memory and estimate."""
    layers = record["layers"]
    return [str(config.rank)] + config.values() + [
        str(record["full_layers"]) if record["full_layers"] is not None else "-",
        record["mode"].upper() if record["mode"] else "-",
        yaml.safe_dump(layers, default_flow_style=False, sort_keys=False).strip() if layers is not None else "-",
        f"{record['memory']:.0f} MB",
        f"{record['score']:.6g}",
    ]


def _recompute_chart(axis: Any, plans: list, debug_parts: list, title: str, labels: list) -> Any:
    """Stack each plan's estimated parts on one bar, and return the axis."""
    parts = list(map(str, debug_parts))
    frame = pd.DataFrame([record["parts"] for _, record in plans], columns=parts, index=labels)
    frame.plot.bar(ax=axis, stacked=True, color=gen_colors(parts), width=0.65, rot=0)
    axis.set_xlim(-0.5, len(plans) - 0.5)
    axis.set_ylim(
        min(0, float(frame.clip(upper=0).sum(axis=1).min()) * 1.05),
        max(1, float(frame.clip(lower=0).sum(axis=1).max()) * 1.05),
    )
    axis.set_ylabel("Estimated step (lower is better)")
    axis.set_xlabel("")
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    axis.set_title(title, fontsize=11)
    return axis


def _recompute_table(table_axis: Any, rows: list, cells: list, labels: list, heights: list) -> None:
    """Draw the table of plans under the chart, each row shaded like its dimension."""
    table_height = sum(heights)
    table_axis.axis("off")
    table = table_axis.table(cellText=cells, rowLabels=list(map(str, rows)), colLabels=labels,
                             cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(heights[row] / table_height)
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            continue
        row_title = rows[row - 1]
        if column == -1:
            cell.get_text().set_color(dim_color(row_title))
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(near_white(pastel(dim_color(row_title)), 0.9))
            if row_title == LAYER_ROW:
                cell.get_text().set_fontfamily("monospace")
                cell.get_text().set_ha("center" if cells[row - 1][column] == "{}" else "left")


def _score_parts() -> list:
    """The parts a score splits into, in the order the debugger fills them."""
    return [part for part in PerfParts if part not in {PerfParts.TOTAL, PerfParts.MEMORY}]


def _make_parent(path: str) -> None:
    """Create the directory *path* is to be written in, when it is missing."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_ranking_csv(scored_space: list, path: str) -> None:
    """Write a search's configurations in ND's order, best first.

    One row per configuration that fits memory: its rank, its degrees, the
    activation checkpoint mode where the search had a recompute dimension,
    the peak memory in MB, the score and the parts the score splits into,
    which are blank when the search ran without debug output. Scores keep
    full precision so that a reader can tell a tie from a near miss.

    Args:
        scored_space: ``(config, memory, score, parts)`` entries, as
            ``ParallelizeLayer.order_search_space`` sorts them. Stored ranks
            are preserved; unranked configurations use their input positions.
        path: CSV file to write; its directory is created when missing.
    """
    parts = _score_parts()
    _make_parent(path)
    dims = [str(dim) for dim in scored_space[0][0].keys()] if scored_space else []
    moded = _has_recompute(scored_space)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank"] + dims + ([RECOMPUTE_COLUMN] if moded else []) + ["memory_mb", "score"]
                        + [str(part) for part in parts])
        for rank, (config, memory, score, values) in enumerate(scored_space, start=1):
            split = [repr(float(value)) for value in values] if values else [""] * len(parts)
            writer.writerow([config.rank or rank] + config.values() + ([config.recompute] if moded else [])
                            + [memory, repr(float(score))] + split)


def write_estimates_csv(configs_estimated: list, path: str) -> None:
    """Write ND's estimate of every configuration of a classified comparison.

    One row per measured configuration, in the comparison's order: its
    degrees, the activation checkpoint mode where the measured CSV states
    one, the measured step, ND's peak memory in MB, its score and the parts
    of the score. The plots show these only as bars; a sweep needs ND's
    memory as a number, to set it beside the peak the trainer logged.

    Args:
        configs_estimated: ``(config, peak_mem, real_time, score, parts,
            real_parts)`` entries, as ``ParallelizeLayer.compare_with_csv``
            returns them.
        path: CSV file to write; its directory is created when missing.
    """
    parts = _score_parts()
    _make_parent(path)
    dims = [str(dim) for dim in configs_estimated[0][0].keys()] if configs_estimated else []
    moded = _has_recompute(configs_estimated)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dims + ([RECOMPUTE_COLUMN] if moded else []) + ["time", "memory_mb", "score"]
                        + [str(part) for part in parts])
        for config, memory, step, score, values, _ in configs_estimated:
            split = [repr(float(value)) for value in values[:len(parts)]]
            writer.writerow(config.values() + ([config.recompute] if moded else [])
                            + [step, memory, repr(float(score))] + split)


def busy_time(entry: tuple) -> float:
    """The measured step of a comparison entry less its idle remainder."""
    return entry[2] - (entry[5].get("IDLE") or 0.0)


def plot_vs_real(
    configs_estimated, csv_f, output_path, debug_parts, title=None
):
    """Plot estimation vs real global time"""
    plot = Plot(title, configs_estimated[0][0].keys(), debug_parts)
    plot.parse_data(configs_estimated)

    data_frame = pd.DataFrame(
        plot.data, columns=(["config", "Real", "estim"] + plot.dbg_cols)
    )
    ax1 = data_frame["Real"].plot.bar(
        position=1.1, width=0.4, secondary_y="real", color="grey", rot=0
    )

    set_twin_handles(ax1, data_frame, plot.dbg_cols)
    plot.make_table()
    plot.close(output_path, Path(os.path.basename(csv_f)).stem)


def plot_vs_real_comm_classified(
    configs_estimated,
    csv_f,
    output_path,
    debug_parts,
    **kwargs,
):
    """Plot estimation vs real detailed time.

    Written to ``<csv stem><suffix>.pdf`` in *output_path*: ``plot_idle`` adds
    the measured idle remainder to the measured bars, and ``suffix`` (empty
    by default) tells apart several plots of one CSV.
    """
    plot_idle = kwargs.get("plot_idle", False)
    title = kwargs.get("title", None)
    suffix = kwargs.get("suffix", "")
    real_data = []

    plot = Plot(title, configs_estimated[0][0].keys(), debug_parts)
    plot.parse_data(
        configs_estimated,
        real_data=real_data,
        plot_idle=plot_idle,
    )

    data_frame = pd.DataFrame(
        plot.data, columns=(["config", "real", "estim"] + plot.dbg_cols)
    )
    real_cols = list(MEASURED_BARS)
    if plot_idle:
        real_cols.append("IDLE")
    real_df = pd.DataFrame(real_data, columns=real_cols)

    ax1 = real_df[real_cols].plot.bar(
        stacked=True,
        sharex=True,
        position=1,
        secondary_y="real",
        color=gen_colors(real_cols),
        width=0.4,
        rot=0,
        legend=False,
    )

    set_twin_handles(ax1, data_frame, plot.dbg_cols)
    plot.make_table()
    plot.close(
        output_path,
        Path(os.path.basename(csv_f)).stem + suffix,
    )


def correlation_topk(configs_estimated, csv_f):
    """Computes correlation & top-k between real & estimation"""
    times = []
    estims = []
    for _, _, time, score, _ in configs_estimated:
        times.append(time)
        estims.append(score)
    correl = pearsonr(times, estims).statistic  # type: ignore
    if isnan(correl):
        logger.critical(
            "An input array is constant: %s or %s", str(times), str(estims)
        )
    topk = 0
    for i, score in enumerate(estims):
        if not score == min(estims[i:]):
            break
        topk += 1
    if topk == 0:
        for i, score in enumerate(estims):
            if score == min(estims[i:]):
                break
            topk -= 1

    logger.info("Correlation for file %s is: %.3f", csv_f, correl * 100)
    return correl, topk


def get_real_data(csv_f):
    """Read execution time of different configurations on a given csv file"""
    configs = []
    row_num = 0
    with open(csv_f, newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            row_num += 1
            logger.info(row)
            real_time = float(row.pop("time"))
            recompute = _recompute_of(row)
            config = []
            for dim_str, value in row.items():
                try:
                    dim = Dim.get_dim(dim_str)
                    logger.debug("%s : %s", str(dim), str(dim.from_str(value)))
                    config.append((dim, dim.from_str(value)))
                except ValueError:
                    pass
            configs.append((Dim.Dimensions(config, recompute=recompute), real_time))
    return configs, row_num


def get_diff_dims(csv_f):
    """Read execution time of different configurations on a given csv file"""
    dims = []
    data_frame = pd.read_csv(csv_f)
    for dim_str, degrees in data_frame.items():
        try:
            dim = Dim.get_dim(dim_str)
            diff_values = len(set(degrees))
            if diff_values > 1:
                dims.append(dim)
        except ValueError:
            pass
    return dims


def get_comm_classified_data(csv_f, plot_idle=False):
    """Read time components of different configurations on a given csv file.

    ``time`` is the mean of the PROFILED steps, which is what the instrument
    read and not what the model costs: on Ascend the profiler adds 0.33 to
    0.84 s a step at EP 1 and 0.01 to 0.10 at EP 8, which is enough to reverse
    a ranking. A CSV may therefore carry ``step_trainer`` as well, the
    trainer's own step time over steps it did not profile, and this reads it
    into the parts under the same name for whoever ranks the round.

    It is deliberately NOT substituted for ``time`` here. The parts carry the
    profiler's cost too, so the step less its parts goes negative against the
    honest total on 4 of the 15 points of the 2 October round, to -499 ms at
    EP 1, where against the profiled step the same residual is +96 to +454.
    Idle stays a profiled quantity until the parts have a correction of their
    own; only the total is honest enough to rank on.
    """
    configs = []
    with open(csv_f, newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            logger.info(row)
            time = float(row.pop("time"))
            trainer_step = (row.pop(TRAINER_STEP, "") or "").strip()
            recompute = _recompute_of(row)
            config = []
            comm_wait_time_classified = {}
            total_wait = 0

            for component, value_str in row.items():
                if "wait" in component:
                    value_float = float(value_str)
                    logger.info(
                        "Comm_wait = %s, v = %f", component, value_float
                    )
                    comm_wait_time_classified[component] = value_float
                    total_wait += value_float
                elif "comp" in component:
                    value_float = float(value_str)
                    logger.info("Computation = %f", value_float)
                    comm_wait_time_classified["comp"] = value_float
                    total_wait += value_float
                else:
                    logger.info("d = %s, v = %s", component, value_str)
                    dim = Dim.get_dim(component)
                    config.append((dim, dim.from_str(value_str)))
            comm_wait_time_classified["BUBBLE"] = comm_wait_time_classified.get(str(RealParts.PP_WAIT))
            if trainer_step:
                comm_wait_time_classified[TRAINER_STEP] = float(trainer_step)
            if plot_idle:
                comm_wait_time_classified["IDLE"] = time - total_wait
                logger.info(
                    "idle = total time - total waits = %.3f - %.3f",
                    time,
                    total_wait,
                )
            configs.append(
                (Dim.Dimensions(config, recompute=recompute), time, comm_wait_time_classified)
            )
    return configs


def estimation_in_real_parts(
    estimations_in_real_components, estimations, score
):
    """Transform the estimation components into the RealParts components for comparison with real time"""
    estimations_in_real_components[RealParts.TOTAL].append(score)
    estimations_in_real_components[RealParts.COMP].append(
        estimations[PerfParts.FW_COMPUTE.value - 1]
        + estimations[PerfParts.BW_COMPUTE.value - 1]
        + estimations[PerfParts.RECOMPUTE.value - 1]
    )
    estimations_in_real_components[RealParts.DP_WAIT].append(
        estimations[PerfParts.DP_COMM.value - 1]
        + estimations[PerfParts.DP_REDUCE.value - 1]
    )
    estimations_in_real_components[RealParts.MP_WAIT].append(
        estimations[PerfParts.MP_COMM.value - 1]
    )
    estimations_in_real_components[RealParts.CP_WAIT].append(
        estimations[PerfParts.CP_COMM.value - 1]
    )
    estimations_in_real_components[RealParts.EP_WAIT].append(
        estimations[PerfParts.EP_COMM.value - 1]
    )
    estimations_in_real_components[RealParts.PP_WAIT].append(
        estimations[PerfParts.BUBBLE.value - 1]
        + estimations[PerfParts.PP_COMM.value - 1]
    )
    return estimations_in_real_components


def real_in_parts(parts, real, time):
    """Transform the real time components into the RealParts components for comparison with estimation"""
    parts[RealParts.TOTAL].append(time)
    for part in RealParts:
        if part not in {RealParts.TOTAL, RealParts.IDLE}:
            if str(part) in real.keys():
                parts[part].append(real[str(part)])
            else:
                logger.warning(
                    "part = %s not in real keys = %s", part, real.keys()
                )
                parts[part].append(0)

    op = "op_wait"
    if op in real.keys():
        parts[RealParts.DP_WAIT][-1] += real["op_wait"]

    sp = "sp_wait"
    if sp in real.keys():
        parts[RealParts.MP_WAIT][-1] += real["sp_wait"]

    return parts


def correlation_with_classified_comms(configs_estimated: list) -> tuple:
    """Computes correlation and distance between components time & estimation."""
    score_classified = {}
    time_classified = {}
    distances = {}

    for wait in RealParts:
        if wait not in {RealParts.IDLE}:
            score_classified[wait] = []
            time_classified[wait] = []
            distances[wait] = []

    topk = 0
    still_top_k = True

    for i, (_, _, time, score, values, real_values) in enumerate(
        configs_estimated
    ):
        if (
            still_top_k
            and score == (min(configs_estimated[i:], key=lambda t: t[3]))[3]
        ):
            topk += 1
        else:
            still_top_k = False

        score_classified = estimation_in_real_parts(
            score_classified, values, score
        )

        time_classified = real_in_parts(time_classified, real_values, time)

        square_distances_sum = 0
        for wait in RealParts:
            if wait not in {RealParts.TOTAL, RealParts.IDLE}:
                distance = (
                    time_classified[wait][-1]
                    / time_classified[RealParts.TOTAL][-1]
                    - score_classified[wait][-1]
                    / score_classified[RealParts.TOTAL][-1]
                )
                square_distances_sum += distance * distance
                distances[wait].append(abs(distance))
        distances[RealParts.TOTAL].append(sqrt(square_distances_sum))

    correls = {}
    for wait in RealParts:
        pearson_wait(correls, time_classified, score_classified, wait)
    return correls, distances, topk, len(configs_estimated)


def _share(value, total):
    """Fraction of a total, 0 when the total is 0"""
    return value / total if total else 0.0


def format_classified_comparison(configs_estimated: list) -> str:
    """Side-by-side measured and estimated parts of every configuration.

    Shares are over each side's own total. ND has no idle term, so the measured
    idle share is the part of the step the estimate does not account for.

    Args:
        configs_estimated: Entries of ``ParallelizeLayer.order_space_test_comm_classified``.

    Returns:
        One table per configuration: measured value and share, ND share, and their difference.
    """
    parts = [part for part in RealParts if part not in {RealParts.IDLE, RealParts.TOTAL}]
    lines = []
    for config, _, time, score, values, real_values in configs_estimated:
        estim = estimation_in_real_parts({part: [] for part in RealParts}, values, score)
        real = real_in_parts({part: [] for part in RealParts}, real_values, time)
        lines.append(f"{config}: measured {time:.6g}, ND score {score:.6g}")
        lines.append(f"  {'part':8s} {'measured':>12s} {'share':>8s} {'ND share':>9s} {'diff':>8s}")
        for part in parts:
            real_share = _share(real[part][-1], time)
            estim_share = _share(estim[part][-1], score)
            lines.append(
                f"  {str(part):8s} {real[part][-1]:12.6g} {real_share:8.1%} {estim_share:9.1%} "
                f"{real_share - estim_share:+8.1%}"
            )
        idle = time - sum(real[part][-1] for part in parts)
        lines.append(f"  {str(RealParts.IDLE):8s} {idle:12.6g} {_share(idle, time):8.1%} {'-':>9s}")
    return "\n".join(lines)


def color_diff(diff):
    """Color difference"""
    if diff > 0:
        return f"\033[92m improved by {diff:.3f}%\033[00m"
    return f"\033[91m worsened by {-diff:.3f}%\033[00m"


def color_correl(correlation):
    """Color correlation"""
    res = f"{correlation*100:.3f}%"
    if correlation > 0.9:
        res = f" \033[92m{res}\033[00m "
    elif correlation < 0:
        res = f"\033[91m{res}\033[00m "
    elif correlation < 0.5:
        res = f" \033[91m{res}\033[00m "
    else:
        res = f" \033[00m{res}\033[00m "
    return res


def print_diff(case, prev, new, **kwargs):
    """Print difference of correlation"""
    topk = kwargs.get("topk", None)
    total = kwargs.get("total", None)
    tabsize = kwargs.get("tabsize", 40)
    diff = (new - prev) * 100
    msg = ""
    if -0.1 < diff < 0.1:
        msg = f"{case} \tcorrelation :{color_correl(new)}  \033[00m\033[00m"
    else:
        msg = (
            f"{case} \tcorrelation ({color_correl(new)}) is{color_diff(diff)}"
        )
    if topk is not None and total is not None:
        msg += f"   topk = {topk}/{total}"
    logger.output(msg.expandtabs(tabsize))


def get_distance_i(part: RealParts, data_i: tuple) -> float:
    """get the average distance of a given part"""
    _, distance, _, _ = data_i
    return sum(distance[part]) / len(distance[part])


def get_correl_i(part, data_i):
    """get the correlation of a given part"""
    f_correl, _, _, _ = data_i
    return f_correl[part]


def print_part_x_file(data, fun):
    """Prints a metric computed by fun for each couple (part, file)"""
    msg = ""
    for part in RealParts:
        if part is not RealParts.IDLE:
            msg += "\n" + str(part) + "\t"
            col_sum = 0
            col_num = 0
            for data_i in data:
                try:
                    info = fun(part, data_i)
                    msg += f"\t{(info*100):.1f}%"
                    col_sum += info
                    col_num += 1
                except KeyError:
                    msg += "\t  -"
            if col_num > 0:
                msg += f"\t\t{(col_sum/col_num)*100:.1f}%"
    return msg


def print_correlations_classified(data):
    """Printer for estimation vs detailed profiling"""
    msg = "\n\t"
    for i, _ in enumerate(data):
        msg += "\tFile " + str(i + 1)
    msg += "\t\tavg"

    msg += "\nCorrelation (higher is better)"
    msg += print_part_x_file(data, get_correl_i)

    msg += "\ntop_k\t"
    for _, _, top_k, total in data:
        msg += "\t" + str(top_k) + "/" + str(total)

    msg += "\n\nEuclidean Distance (lower is better)"
    msg += print_part_x_file(data, get_distance_i)

    logger.output(msg)


def is_constant(array):
    """Whether the given array only has the same elements"""
    if len(array) == 0:
        return True
    value = array[0]
    return all(vi == value for vi in array)


def pearson_wait(correls, real, estim, wait):
    """Compute Pearson correlation if inputs are not empty"""
    if wait not in {RealParts.IDLE}:
        logger.debug(
            "correlation for %s between real = %s && estim = %s",
            str(wait),
            str(real[wait]),
            str(estim[wait]),
        )
        if not is_constant(real[wait]) and not is_constant(estim[wait]):
            pearson = pearsonr(
                real[wait], estim[wait]
            ).statistic  # type: ignore
            logger.info(
                "correlation[%s] of real %s vs estim %s = %f",
                wait,
                real[wait],
                estim[wait],
                pearson,
            )
            correls[wait] = pearson
        else:
            logger.warning(
                "either estim[%s] = %s is constant", wait, str(estim[wait])
            )
            logger.warning(
                "or      real[%s] = %s is constant", wait, str(real[wait])
            )
