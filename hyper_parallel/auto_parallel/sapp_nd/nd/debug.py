# Copyright 2025 Huawei Technologies Co., Ltd
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
from enum import Enum, auto
from pathlib import Path
from functools import partial
from math import isnan, sqrt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim


class PerfParts(Enum):
    """decomposition of performance"""

    FW_COMPUTE = auto()
    BW_COMPUTE = auto()
    RECOMPUTE = auto()
    DP_COMM = auto()
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
            self.output_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "output",
                output_file,
            )
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
        """Parameters to debug"""
        if self.enable:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            is_new = not os.path.exists(self.output_file)
            logger.info("debug written")
            with open(self.output_file, "a", encoding="utf-8") as outfile:
                if is_new:
                    outfile.write(self.column_titles())
                outfile.write(self.values())


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
    for handle in handle2:
        if handle not in handle1:
            handle1.append(handle)
    for lbl in label2:
        if lbl not in label1:
            label1.append(lbl)
    handles = handle1
    labels = label1
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
    leg = ax2.get_legend()
    pp_color = gen_colors(["PP_COMM"])[0]
    leg.legend_handles[-1].set_facecolor(pp_color)  # type: ignore


class Plot:
    """plot ND top configs"""

    title: str
    col_title: list[str]
    row_title: list[str]
    cell_text: list[list[str]]
    data: list[tuple]
    dbg_cols: list[str]
    top: int

    def __init__(self, title, rows, debug_parts, top=None):
        self.title = title
        self.top = top if top is not None else 20
        self.row_title = rows + ["MEM"]
        self.dbg_cols = list(map(str, debug_parts))
        self.col_title = []
        self.cell_text = []
        self.data = []

    def make_table(self):
        """Make table below plot with each parallelism degree"""
        self.cell_text = list(map(list, zip(*self.cell_text)))  # transpose
        max_rows = list(map(max, map(partial(map, float), self.cell_text)))
        the_table = plt.table(
            cellText=self.cell_text,
            rowLabels=self.row_title,
            colLabels=self.col_title,
            cellLoc="center",
            loc="bottom",
        )
        row_colors = list(map(dim_color, self.row_title))
        for row in range(len(self.row_title)):
            cell = the_table[row + 1, -1]
            cell.set_edgecolor("none")
            cell.get_text().set_color(row_colors[row])
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))
            for col in range(len(self.cell_text[0])):
                cell = the_table[row + 1, col]
                value = float(str(cell.get_text().get_text()))
                try:
                    ratio = 1 - (value / max_rows[row])
                except ZeroDivisionError:
                    ratio = 0
                logger.debug(
                    "tmax = %s, ratio = %f, col=%s, newcolor=%s",
                    str(max_rows[row]),
                    ratio,
                    str(mc.to_rgb(row_colors[row])),
                    str(near_white(row_colors[row], ratio)),
                )
                cell.set_facecolor(near_white(pastel(row_colors[row]), ratio))
                cell.set_edgecolor("none")

        for col in range(len(self.cell_text[0])):
            the_table[0, col].set_edgecolor("none")

        the_table.scale(xscale=1, yscale=1.2)  # +len(rows)/5)

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
        min_e = configs_estimated[0][2]
        i = 0
        for cfg_e in configs_estimated:
            self.cell_text.append(cfg_e[0].values() + [cfg_e[1]])
            self.col_title.append("")
            try:
                self.data.append(
                    tuple([cfg_e[0], cfg_e[2], cfg_e[3]] + cfg_e[4])
                )
                if real_data is not None:
                    waits = cfg_e[5]
                    logger.info(waits)
                    wait_list = [
                        waits["comp"],
                        waits["dp_wait"],
                        waits["mp_wait"],
                        waits["ep_wait"],
                        waits["BUBBLE"],
                    ]
                    if plot_idle:
                        wait_list.append(waits["IDLE"])
                    real_data.append(tuple(wait_list))
            except IndexError:
                score = cfg_e[2]
                if i >= self.top or (min_e is not None and score > min_e * 20):
                    self.cell_text.pop()
                    break
                self.data.append(tuple([cfg_e[0], score] + cfg_e[3]))
                i += 1


def plot_nd(
    configs_estimated, output_path, debug_parts, title=None, max_num=None
):
    """Plot estimation"""
    plot = Plot(
        title, configs_estimated[0][0].keys(), debug_parts, top=max_num
    )
    plot.parse_data(configs_estimated)

    data_frame = pd.DataFrame(
        plot.data, columns=(["config", "estim"] + plot.dbg_cols)
    )
    axis = data_frame[plot.dbg_cols].plot.bar(
        stacked=True, color=gen_colors(plot.dbg_cols), width=0.4, rot=0
    )
    axis.set_ylim(ymin=1)
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plot.make_table()
    plot.close(output_path, "results")


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
    """Plot estimation vs real detailed time"""
    plot_idle = kwargs.get("plot_idle", False)
    title = kwargs.get("title", None)
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
    real_cols = [
        "COMPUTATION",
        "DP_COMM",
        "MP_COMM",
        "EP_COMM",
        "BUBBLE",
    ]
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
        Path(os.path.basename(csv_f)).stem,
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
            config = []
            for dim_str, value in row.items():
                try:
                    dim = Dim.get_dim(dim_str)
                    logger.debug("%s : %s", str(dim), str(dim.from_str(value)))
                    config.append((dim, dim.from_str(value)))
                except ValueError:
                    pass
            configs.append((Dim.Dimensions(config), real_time))
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
    """Read time components of different configurations on a given csv file"""
    configs = []
    with open(csv_f, newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            logger.info(row)
            time = float(row.pop("time"))
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
            if plot_idle:
                comm_wait_time_classified["IDLE"] = time - total_wait
                logger.info(
                    "idle = total time - total waits = %.3f - %.3f",
                    time,
                    total_wait,
                )
            configs.append(
                (Dim.Dimensions(config), time, comm_wait_time_classified)
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


def correlation_with_classified_comms(configs_estimated):
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
        distances[RealParts.TOTAL] = sqrt(square_distances_sum)

    correls = {}
    for wait in RealParts:
        pearson_wait(correls, time_classified, score_classified, wait)
    return correls, distances, topk, len(configs_estimated)


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


def get_distance_i(part, data_i):
    """get the average distance of a given part"""
    _, distance, _, _ = data_i
    if part is RealParts.TOTAL:
        return distance[part]
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
