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
"""run parallelization"""

import argparse
import json
import os
import sys

import yaml

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import set_strict
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger, set_verbose_level
import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd import _pool
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.ratios as Ratios
from hyper_parallel.auto_parallel.sapp_nd.nd.recompute_dimension import read_recompute_modes
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import trainer_runs_layer_plans
from hyper_parallel.auto_parallel.sapp_nd.nd.verify import (
    census_spec_yaml,
    report,
    report_spec,
    traffic_report,
    verify_activations,
    verify_estimate,
    verify_flops,
    verify_parameters,
    verify_spec,
    verify_traffic,
)


def _link_ratios(cli_parser, cache_file):
    """The ratios of a ratios file that price the copies: FORWARD, else COMPUTE, for a copy; COMPUTE for its cost.

    FORWARD, where a round measured the forward on its own, is the
    milliseconds one unit of the forward the estimate prices stands for: the
    window the copies to the host hide in. COMPUTE stands for the forward and
    the backward together, which the estimate may split otherwise than the
    device runs them, and for the score's units, which the copies' cost to
    the step is charged in.

    Args:
        cli_parser: The parser, to report a file that states neither.
        cache_file: The ratios file of -c, or ``None``.

    Returns:
        ``(copy, cost)``, each ``None`` without a file, the second also
        where the file states no COMPUTE ratio.
    """
    if cache_file is None:
        return None, None
    with open(cache_file, encoding="utf-8") as handle:
        ratios = json.load(handle)
    name = "FORWARD" if "FORWARD" in ratios else "COMPUTE"
    ratio = ratios.get(name)
    if not isinstance(ratio, (int, float)) or ratio <= 0:
        cli_parser.error(f"{cache_file} states no positive {name} ratio to calibrate the host link with")
    compute = ratios.get("COMPUTE")
    return float(ratio), float(compute) if isinstance(compute, (int, float)) and compute > 0 else None


def _link_figures(cli_parser, cli_args):
    """The host link figures the CLI states for -ao, ``None`` for each it leaves to the device.

    With a ratios file (-c), its FORWARD or COMPUTE ratio turns a copy's
    seconds into the estimate's units, in place of the device's sustained
    throughput, and its COMPUTE ratio the copies' cost (:func:`_link_ratios`).

    Args:
        cli_parser: The parser, to report a ratios file with neither ratio.
        cli_args: The parsed CLI namespace.

    Returns:
        ``gib_per_s``, ``sustained_tflops``, ``overlap``, ``ms_per_unit``,
        ``score_ms_per_unit`` and ``copy_cost_ms_per_gib``.
    """
    copy_ratio, cost_ratio = _link_ratios(cli_parser, getattr(cli_args, "cache_file", None))
    return {
        "gib_per_s": cli_args.host_link_gibps,
        "sustained_tflops": cli_args.sustained_tflops,
        "overlap": getattr(cli_args, "host_link_overlap", None),
        "ms_per_unit": copy_ratio,
        "score_ms_per_unit": cost_ratio,
        "copy_cost_ms_per_gib": getattr(cli_args, "host_link_cost", None),
    }


def _host_link(cli_parser, cli_args, device):
    """The host link -ao offloads over: the device's, with what the CLI states replacing its figures.

    Args:
        cli_parser: The parser, to report a device with no link to offload over.
        cli_args: The parsed CLI namespace.
        device: The device type the search prices.

    Returns:
        The link, or ``None`` without -ao.
    """
    if not cli_args.auto_offload:
        return None
    try:
        return Hard.HostLink.of(device.host_link, _link_figures(cli_parser, cli_args))
    except ValueError as error:
        cli_parser.error(f"device {device}: {error}")
        return None


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _apply_cli_overrides(search_cfg, cli_args):
    """Override the search config's batch, memory budget and devices from the CLI.

    Args:
        search_cfg: The search config read from ``-s/--search-config``.
        cli_args: The parsed CLI namespace.

    Raises:
        ValueError: When ``-d`` is not a whole number of nodes.
    """
    if cli_args.device_type is not None:
        # -A sets the device the search prices, as it does on the CLI path;
        # only the search config's device_type reached the search before.
        search_cfg.cluster_spec["device_type"] = cli_args.device_type
    if cli_args.global_batch_size is not None:
        search_cfg.constraint["global_batch_size"] = cli_args.global_batch_size
    if getattr(cli_args, "auto_recompute", False):
        search_cfg.estimator["recompute_strategy"] = "auto"
    if cli_args.max_mem is not None:
        # -M sets the device budget the search checks against, exactly as it
        # does on the CLI path, instead of being silently ignored here.
        search_cfg.cluster_spec["device_memory_gb"] = (
            Memory.from_string(cli_args.max_mem.strip()).to_gb().size
        )
    if cli_args.devices is not None:
        cards_per_node = search_cfg.cluster_spec.get("cards_per_node")
        if not cards_per_node:
            # The device type knows its node size (A3: 16); defaulting to 8
            # silently halves an A3 node and invalidates every candidate.
            device = Hard.device_map.get(cli_args.device_type)
            cards_per_node = device.intra_node_num() if device else 8
            search_cfg.cluster_spec["cards_per_node"] = cards_per_node
            logger.info(
                "cluster.cards_per_node not set, using %d from device type %s",
                cards_per_node, cli_args.device_type,
            )
        cards_per_node = max(1, cards_per_node)
        nodes = max(1, cli_args.devices // cards_per_node)
        if cli_args.devices % cards_per_node:
            # A search config counts whole nodes, so a part of one was rounded
            # and another cluster than the one asked for priced.
            raise ValueError(
                f"-d {cli_args.devices} is not a whole number of nodes of {cards_per_node} cards: a search "
                f"config counts whole nodes and would price {nodes * cards_per_node} devices. Give -d a "
                f"multiple of {cards_per_node}, or price exactly {cli_args.devices} with -y and no -s"
            )
        search_cfg.cluster_spec["num_nodes"] = nodes
    if getattr(cli_args, "recompute", None) is not None:
        # --recompute states the search config's parallelism.recompute, as the
        # other flags state their settings: the modes the search chooses
        # among for each strategy, or for each layer where the search config's
        # recompute says per_layer, and auto every mode it can price.
        modes = read_recompute_modes(cli_args.recompute, "--recompute")
        if search_cfg.estimator.get("recompute_strategy") != "per_layer":
            search_cfg.estimator["recompute_strategy"] = "auto"
        if [str(value).strip().lower() for value in cli_args.recompute] == ["auto"]:
            search_cfg.estimator.pop("recompute_modes", None)
        else:
            search_cfg.estimator["recompute_modes"] = modes
    if getattr(cli_args, "cache_file", None) is not None:
        # -c names the ratios the search scores with, as on the CLI path;
        # the search config path dropped it and ranked on the raw score.
        search_cfg.estimator["ratios_file"] = cli_args.cache_file


def _recompute_modes(cli_args):
    """The recompute dimension of a search: --recompute, else a hyper_v2 yaml's context.recompute.

    A search config's ``parallelism.recompute`` is not a dimension: there the
    search chooses a mode for each strategy, and --recompute states it
    (:func:`_apply_cli_overrides`).

    Args:
        cli_args: The parsed CLI namespace.

    Returns:
        The modes, or None where nothing states the dimension.

    Raises:
        ValueError: A value that is neither a mode nor auto.
    """
    if cli_args.recompute is not None:
        return read_recompute_modes(cli_args.recompute, "--recompute")
    if cli_args.framework != "hyper_v2" or cli_args.search_config or not cli_args.yaml_config:
        return None
    if not os.path.isfile(cli_args.yaml_config):
        return None
    with open(cli_args.yaml_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    context = raw.get("context") if isinstance(raw, dict) else None
    stated = context.get("recompute") if isinstance(context, dict) else None
    return read_recompute_modes(stated, f"{cli_args.yaml_config}: context.recompute")


def _recompute_kwargs(auto_recompute, modes):
    """The search's recompute arguments for the modes --recompute or a yaml's context states.

    Without -ar the modes are a dimension, every candidate ranked under each
    of them. With it they are the modes each layer of a candidate chooses
    among, the choice the trainer runs as activation_checkpoint.layers, as a
    search config's ``recompute: per_layer`` makes it.

    Args:
        auto_recompute: Whether -ar chooses each layer's recompute.
        modes: The modes stated, or None.

    Returns:
        The keyword arguments for Parallelize.
    """
    if modes is None:
        return {}
    if auto_recompute:
        return {"recompute_modes": modes, "recompute_mode_per_layer": True}
    return {"recompute_dimension": modes}


def _check_recompute_potentials(cli_parser, cli_args, modes) -> None:
    """Refuse -rp/--recompute_potentials where there is no plan per layer to extend, or nowhere to write it.

    Args:
        cli_parser: The argument parser, which reports the refusal.
        cli_args: The parsed CLI namespace.
        modes: The recompute modes stated, or None.
    """
    if not cli_args.auto_recompute or modes is None:
        cli_parser.error("-rp/--recompute_potentials extends a recompute plan per layer: it needs "
                         "-ar/--auto_recompute with the modes each layer chooses among, from --recompute or a "
                         "hyper_v2 yaml's context.recompute")
    if cli_args.real_csv is not None or cli_args.search_config or cli_args.verify:
        cli_parser.error("-rp/--recompute_potentials applies to the standard ND search, not --real_csv, "
                         "--search-config, or --verify")
    if cli_args.verbosity < 2:
        cli_parser.error("-rp/--recompute_potentials writes a CSV and two plots, which a verbosity of at least 2 "
                         "turns on")


def _priced_train_yaml(search_config: str) -> str:
    """The train.yaml a search config names, which its search prices.

    Args:
        search_config: The search config's path.

    Returns:
        Its ``train_yaml``, or an empty string where it names none, as a
        standalone search config does.
    """
    with open(search_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    named = raw.get("train_yaml") if isinstance(raw, dict) else None
    return named if isinstance(named, str) else ""


def _check_train_yaml(cli_parser: argparse.ArgumentParser, cli_args: argparse.Namespace) -> None:
    """Refuse a -y that is not the train.yaml the search config prices.

    The search prices the train.yaml its config names, and -y is the one
    the resolved strategy is written over: two files would put one run's
    strategy over another's run.

    Args:
        cli_parser: The parser, whose ``error()`` exits.
        cli_args: The parsed CLI namespace, both files found.
    """
    priced = _priced_train_yaml(cli_args.search_config)
    if priced and os.path.isfile(priced) and not os.path.samefile(priced, cli_args.yaml_config):
        cli_parser.error(
            f"-y names {cli_args.yaml_config}, but the search config prices its train_yaml, {priced}: "
            "pass the same file"
        )


def _compare_with_real_csv(runner, cli_args):
    """Print ND's estimate next to the configurations measured in a classified CSV.

    Args:
        runner: The ND runner built from the CLI arguments.
        cli_args: The parsed CLI namespace. Requires ``real_csv``; ND's
            real-versus-estimate plot goes to ``output_dir`` when it is set.
    """
    if cli_args.output_dir is not None:
        os.makedirs(cli_args.output_dir, exist_ok=True)
    # Keep debug.csv beside the plot rather than inside the installed package.
    Debug.set_output_dir(cli_args.output_dir)
    configs_estimated, metrics = runner.compare_with_csv(
        cli_args.real_csv, output_path=cli_args.output_dir, plot_idle=True
    )
    logger.output("%s", Debug.format_classified_comparison(configs_estimated))
    Debug.print_correlations_classified([metrics])
    if cli_args.write_ratios is not None:
        ratios = Ratios.fit_ratios(configs_estimated)
        Ratios.write_ratios(cli_args.write_ratios, ratios, configs_estimated, cli_args.real_csv)
        for text in Ratios.report(configs_estimated, ratios):
            logger.output("%s", text)
        logger.output("Ratios written to %s; run_nd -c reads them", cli_args.write_ratios)


def _run_hyper_v2_search(cli_parser, cli_args):
    """Run the HyperParallel V2 strategy search via ``config_adapter``.

    This branch is activated when ``-f hyper_v2`` is combined with
    ``-s/--search-config``.  It reads the Search Config YAML, validates
    it, runs the ND search engine through :func:`search_strategies`,
    and writes the resolved strategy back into a copy of the original
    ``train.yaml``.

    Args:
        cli_parser: The :class:`argparse.ArgumentParser` (used for ``error()``).
        cli_args: The parsed CLI namespace.  Requires ``yaml_config``,
            ``search_config``, and optionally ``output_dir``.

    Raises:
        SystemExit: If validation fails (via ``parser.error``).
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.config_adapter import (
        read_search_config,
        validate,
        search_strategies,
        write_resolved_yaml,
    )

    if not os.path.isfile(cli_args.search_config):
        cli_parser.error(f"search-config not found: {cli_args.search_config}")
    if not os.path.isfile(cli_args.yaml_config):
        cli_parser.error(f"yaml-config not found: {cli_args.yaml_config}")
    _check_train_yaml(cli_parser, cli_args)

    set_verbose_level(cli_args.verbosity)
    Debug.set_output_dir(cli_args.output_dir, fresh=True)

    search_cfg = read_search_config(cli_args.search_config)
    try:
        _apply_cli_overrides(search_cfg, cli_args)
    except ValueError as exc:
        cli_parser.error(str(exc))
    offload = None
    if getattr(cli_args, "auto_offload", False):
        if search_cfg.estimator.get("recompute_strategy") != "per_layer":
            cli_parser.error("-ao offloads in a choice of a recompute mode per layer: give the search config "
                             "recompute: per_layer, and leave out -ar, which chooses one mode for every layer")
        offload = _link_figures(cli_parser, cli_args)

    if getattr(search_cfg, "parallelism_summary", ""):
        logger.output("Parallelism: %s", search_cfg.parallelism_summary)

    errors = validate(search_cfg)
    hard_errors = [e for e in errors if e.severity == "error"]
    warnings = [e for e in errors if e.severity == "warning"]
    for w in warnings:
        logger.warning("%s: %s", w.field_path, w.message)
    if hard_errors:
        for e in hard_errors:
            logger.error("%s: %s", e.field_path, e.message)
        cli_parser.error(
            f"Search config validation failed with {len(hard_errors)} error(s)."
        )

    result = search_strategies(search_cfg) if offload is None else search_strategies(search_cfg, offload=offload)
    search_cfg.resolved_strategy = result

    output_dir = cli_args.output_dir or "."
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    resolve_path = os.path.join(output_dir, "resolved.yaml")
    write_resolved_yaml(search_cfg, cli_args.yaml_config, resolve_path)
    logger.output("Resolved strategy written to %s", resolve_path)
    if "activation_checkpoint" in result:
        _log_activation_checkpoint(result)
    logger.output(
        "Optimal strategy: dp=%(dp)s tp=%(tp)s pp=%(pp)s "
        "cp=%(cp)s ep=%(ep)s mb_num=%(micro_batch_num)s "
        "recompute=%(activation_checkpoint)s "
        "mem=%(memory_estimate_mb).0f MB score=%(score).2e",
        result,
    )


def _log_activation_checkpoint(result):
    """Log the activation checkpointing a search chose: its mode, the layers that run another, and each layer's own.

    A plan per layer the installed trainer cannot run, as it declares no
    ``activation_checkpoint.layers``, is logged as advice: the resolved yaml
    runs every layer the mode.

    Args:
        result: The search's result, which states ``activation_checkpoint``.
    """
    plan = result.get("activation_checkpoint_layers")
    if plan and trainer_runs_layer_plans():
        logger.output("Activation checkpoint mode %s, and per layer: %s", result["activation_checkpoint"], plan)
    elif plan:
        logger.output("Activation checkpoint mode for every layer: %s, as the resolved yaml states it. The plan per "
                      "layer %s is advice: the installed trainer's activation_checkpoint has no layers field",
                      result["activation_checkpoint"], plan)
    else:
        logger.output("Activation checkpoint mode for every layer: %s", result["activation_checkpoint"])
    if result.get("offloaded_layers"):
        moved = result.get("offloaded_gib", {})
        shares = result.get("offloaded_share", {})
        layers = ", ".join(
            f"{name} ({moved[name]:.2f} GiB a micro-batch, {shares[name]:.0%} of what it keeps)"
            if name in moved and name in shares else name
            for name in result["offloaded_layers"]
        )
        logger.output("Offloaded to the host, priced as offload should run: layers %s. The trainer runs no offload "
                      "yet, so the plan keeps them off", layers)
    per_layer = result.get("recompute_per_layer")
    if per_layer:
        logger.output(
            "With each layer run its own way, the score would be %.2e at %.0f MB: %s",
            per_layer["score"], per_layer["memory_estimate_mb"], per_layer["ranges"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python run_nd.py",
        description=("Provides a degree to *N* parallelism dimensions"),
        epilog="",
    )

    parser.add_argument(
        "-y",
        "--yaml_config",
        type=str,
        required=True,
        help="Path to yaml configuration file",
    )
    parser.add_argument(
        "-f",
        "--framework",
        default="mindformers",
        type=str,
        required=False,
        help="Framework to evaluate in "
        "[mindformers, mindspeed, hyperparallel, hyper_v2, torchtitan]",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        default=None,
        help="Number of devices. Takes yaml value if unspecified",
    )
    parser.add_argument(
        "-b",
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size. Takes yaml value if unspecified",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model Name to use. Takes yaml value if unspecified",
    )
    # parser.add_argument(
    #     "-g",
    #     "--generate_yaml_in",
    #     type=str,
    #     default=None,
    #     help="Generate all fitting yaml configurations in the given folder",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--csv",
    #     type=str,
    #     default=None,
    #     help="Computes correlation coefficient from csv results file",
    # )
    parser.add_argument(
        "-l",
        "--dimensions",
        nargs="*",
        type=str,
        default=None,
        help="list of varying (output) dimensions",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Worker processes that price the search's candidates; 1 prices them in this one. "
        "Default: ND_JOBS, else half the logical cores and at most 4, and 1 above -v 2, whose "
        "line per estimate only this process prints. A search with -ar, --recompute, a recompute "
        "dimension in its yaml or a multimodal model, every mode but the plain search, and a "
        "search of fewer than 2000 candidates price them here",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Level of verbosity in range [0,6], "
        "0 being no output and 6 being debug level output. "
        "Plot and debug csv are generated from 2",
    )
    parser.add_argument(
        "-k",
        "--ppb_k",
        type=int,
        default=None,
        help="Write the pipeline balancer's layer description, with per-layer "
        "times for every recompute option, of the k-th ranked configuration "
        "(0 is the best) to the output directory.",
    )
    parser.add_argument(
        "-A",
        "--device_type",
        default=None,
        help="choose device type between A2 or A3: A2 unless given, or a search config states one",
    )
    parser.add_argument(
        "-swap_os",
        "--swap_opt_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate swap optimiezr state",
    )
    # parser.add_argument(
    #     "-lm",
    #     "--less_memory",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Activate less memory schedule",
    # )
    parser.add_argument(
        "-mppb",
        "-–manual_pipeline_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Takes offset and recompute from yaml",
    )
    parser.add_argument(
        "-ar",
        "--auto_recompute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Give every layer of each configuration the fastest recompute "
        "option that fits, instead of scoring it fully recomputed. With "
        "--recompute, or a hyper_v2 yaml's context.recompute, each layer "
        "chooses among those modes, a plan the trainer runs as "
        "activation_checkpoint.layers; without, among sets of recomputed ops "
        "the trainer does not run",
    )
    parser.add_argument(
        "-ao",
        "--auto_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With -ar, let each pipeline stage's first layers offload their "
        "activations to the host instead of recomputing them, over the "
        "device's host link",
    )
    parser.add_argument(
        "--host_link_gibps",
        type=float,
        default=None,
        help="The host link's sustained copy bandwidth in GiB/s, for -ao, as "
        "hyper_offload's profile_transfer_bandwidth measures it; the device's "
        "placeholder when omitted",
    )
    parser.add_argument(
        "--sustained_tflops",
        type=float,
        default=None,
        help="The device's sustained TFLOP/s at the training precision, for "
        "-ao; the device's placeholder when omitted, and unused with -c, whose "
        "FORWARD or COMPUTE ratio converts a copy's seconds instead",
    )
    parser.add_argument(
        "--host_link_overlap",
        type=float,
        default=None,
        help="The share of the forward the copies may take, for -ao: 1 where "
        "they meet nothing, less where the step's collectives share what they "
        "need; the device's 0.8 when omitted",
    )
    parser.add_argument(
        "--host_link_cost",
        type=float,
        default=None,
        help="What the copies cost the step, in ms a GiB moved either way, for "
        "-ao; the device's when omitted (A3: 2.8, measured inside the Demo 2 "
        "step), 0 to price offload as free",
    )
    parser.add_argument(
        "-t",
        "--top_config_number",
        type=int,
        default=None,
        help="Number of top configs to print & plot",
    )
    parser.add_argument(
        "-e",
        "--exhaustive",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Ensure each requested dimension is > 1 in at least N plotted "
            "results; default: 0"
        ),
    )
    parser.add_argument(
        "-ee",
        "--force_exhaustive",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "When N >= --exhaustive, append N further results per requested "
            "dimension with degree > 1; otherwise append at least N and top up "
            "to --exhaustive total; default: 0"
        ),
    )
    parser.add_argument(
        "-eee",
        "--fforce-exhaustive",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Append N distinct results per requested dimension with degree > 1; "
            "each added result is reserved for one dimension and this option "
            "takes precedence over -e/-ee; default: 0"
        ),
    )
    parser.add_argument(
        "-mem",
        "--mem_for_ppb",
        type=str,
        default="0GB",
        help="Memory to reserve for pipeline balancing, taken out of the "
        "memory budget ND allows (default 0GB).",
    )
    parser.add_argument(
        "-c",
        "--cache_file",
        type=str,
        default=None,
        help="Cache file with ratios to recalibrate ND scores. With -ao its "
        "FORWARD ratio, where a round measured the forward on its own, else "
        "its COMPUTE ratio, also turns a copy's seconds into the estimate's "
        "units, and its COMPUTE ratio the copies' cost; with a search config "
        "(-s) that is all it does. "
        "Will be defaulted to 'None'.",
    )

    parser.add_argument(
        "-M",
        "--max_mem",
        type=str,
        default=None,
        help="Device memory budget the search must fit in, e.g. '58GB'. "
        "Overrides the yaml capacity and cluster.device_memory_gb. "
        "To reserve memory instead of capping it, use -mem/--mem_for_ppb.",
    )
    parser.add_argument(
        "--train-yaml",
        type=str,
        default=None,
        help="Path to training configuration yaml file (for hyperparallel2)",
    )
    parser.add_argument(
        "--accelerate-yaml",
        type=str,
        default=None,
        help="Path to accelerate configuration yaml file (for hyperparallel2)",
    )
    parser.add_argument(
        "-s",
        "--search-config",
        type=str,
        default=None,
        help="Path to Search Config YAML for fine-grained search-space control "
        "(hyper_v2 only). Scalar=fixed, list=candidates, 'auto'=ND decides.",
    )
    parser.add_argument(
        "-V",
        "--verify",
        action="store_true",
        help="Verify mode (hyper_v2 only): set the parameters and FLOPs ND prices "
        "of each part of the model -y trains beside those of the Transformers "
        "layers its checkpoint builds, and the model spec the resolver reads "
        "beside the one the census measures, with the run priced with each; "
        "with -o, write the census's spec there; and exit.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files when using --search-config "
        "(default: current directory), and for the real-versus-estimate "
        "plot of --real_csv (no plot when omitted).",
    )
    parser.add_argument(
        "--real_csv",
        type=str,
        default=None,
        help="Instead of searching, compare ND's estimate with the configurations "
        "measured in a classified profiling CSV (see nd.trace_classify).",
    )
    parser.add_argument(
        "--ranking_csv",
        type=str,
        default=None,
        help="Also write every configuration the search keeps, in ND's order, "
        "to this CSV: rank, degrees, memory in MB, score and its parts.",
    )
    parser.add_argument(
        "--write_ratios",
        type=str,
        default=None,
        help="With --real_csv, fit a ratio per part, measured over ND's estimate, "
        "on the configurations the CSV measured, write them to this JSON file "
        "(the file -c reads) and print how well they predict each configuration "
        "when fitted on the others.",
    )
    parser.add_argument(
        "--recompute",
        nargs="+",
        default=None,
        metavar="MODE",
        help="Search recompute as a dimension: the activation checkpoint modes "
        "(off, selective, full) a candidate may run, or auto for all three. "
        "Every candidate is priced under each and the pairs are ranked together, "
        "so a mode left out is never proposed. Overrides context.recompute of a "
        "hyper_v2 yaml and the search config's parallelism.recompute. Without "
        "either, every candidate keeps the recompute it derives, full unless "
        "-mppb. With -ar, the modes each layer chooses among instead. With "
        "--real_csv, the mode of rows that state none.",
    )
    parser.add_argument(
        "-rp",
        "--recompute_potentials",
        action="store_true",
        help="With -ar and a mode chosen per layer, also report what the cost "
        "model estimates each further recomputed layer would cost: for every "
        "configuration that fits, the sequence of plans that fully recompute one "
        "more layer at the same degrees and memory budget, written to "
        "recompute_potentials.csv and to two comparison PDFs beside the results "
        "plot. Off by default, and the search does none of that work without it: "
        "the report prices far more plans than the search itself and takes "
        "accordingly longer. Every figure is the cost model's estimate of the "
        "step, which a measured run can disagree with.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Refuse a config field no parser set where the estimate reads it, "
        "instead of pricing it as 0. Without it, a search names the fields it "
        "read that way under its ranking.",
    )

    args = parser.parse_args()
    set_strict(args.strict)
    if args.jobs is not None and args.jobs < 1:
        parser.error(f"-j/--jobs counts processes, at least 1, got {args.jobs}")
    if args.real_csv is not None and not os.path.isfile(args.real_csv):
        parser.error(f"real_csv not found: {args.real_csv}")
    if args.write_ratios is not None and args.real_csv is None:
        parser.error("--write_ratios fits the ratios on a comparison: it needs --real_csv")
    try:
        recompute_modes = _recompute_modes(args)
    except ValueError as exc:
        parser.error(str(exc))

    exhaustive_count = args.exhaustive
    force_exhaustive = args.force_exhaustive
    fforce_exhaustive = args.fforce_exhaustive
    if (exhaustive_count or force_exhaustive or fforce_exhaustive) and (
        args.real_csv is not None or args.search_config or args.verify
    ):
        parser.error(
            "--exhaustive, --force_exhaustive, and --fforce-exhaustive apply to "
            "the standard ND search, not --real_csv, --search-config, or --verify"
        )

    max_mem = (
        Memory.from_string(args.max_mem.strip())
        if args.max_mem is not None
        else None
    )

    if args.cache_file is not None:
        if not os.path.exists(args.cache_file):
            logger.error(
                f"cache file not found:"
                f" {args.cache_file}"
                "\nProceeding without cache file..."
            )
            args.cache_file = None

    if args.auto_recompute and args.mppb:
        parser.error("-ar/--auto_recompute chooses the recompute, so it cannot take it from the yaml (-mppb)")
    if args.auto_offload and args.search_config and args.framework != "hyper_v2":
        parser.error("-ao/--auto_offload with a search config (-s) needs -f hyper_v2, whose search config chooses "
                     "a recompute mode per layer")
    if args.auto_offload and not args.auto_recompute and not args.search_config:
        parser.error("-ao/--auto_offload offloads in the choice per layer -ar/--auto_recompute makes")
    if not args.auto_offload and (args.host_link_gibps is not None or args.sustained_tflops is not None
                                  or args.host_link_overlap is not None or args.host_link_cost is not None):
        parser.error("--host_link_gibps, --sustained_tflops, --host_link_overlap and --host_link_cost price "
                     "offload, which needs -ao/--auto_offload")
    if args.recompute_potentials:
        _check_recompute_potentials(parser, args, recompute_modes)
    if args.verify:
        if args.framework != "hyper_v2":
            parser.error("-V/--verify needs -f hyper_v2: it builds the Transformers checkpoint -y names")
        set_verbose_level(args.verbosity)
        logger.output("Parameters")
        for line in report(verify_parameters(args.yaml_config)):
            logger.output(line)
        logger.output("Forward FLOPs of one sequence; the time model prices the backward at twice them")
        for line in report(verify_flops(args.yaml_config)):
            logger.output(line)
        logger.output("Activations a layer keeps for its backward, bytes a token by op: the records' and the census's")
        for line in report(verify_activations(args.yaml_config)):
            logger.output(line)
        logger.output("Model spec as ND prices it: read from the checkpoint's config, and measured by the census")
        for line in report_spec(verify_spec(args.yaml_config)):
            logger.output(line)
        logger.output("The run priced with each spec: the resolver's in the ND column, the census's beside it")
        for line in report(verify_estimate(args.yaml_config, args.device_type or "A2")):
            logger.output(line)
        if args.output_dir:
            logger.output(f"census spec written to {census_spec_yaml(args.yaml_config, args.output_dir)}")
        logger.output("Bytes a forward of one sequence moves, the whole layer, beside the FLOPs the time model "
                      "prices of it")
        for line in traffic_report(verify_traffic(args.yaml_config)):
            logger.output(line)
        sys.exit(0)

    if args.framework == "hyper_v2" and args.search_config:
        _run_hyper_v2_search(parser, args)
        sys.exit(0)

    if args.framework == "hyper_v2" and args.devices is None:
        # An AutoModels train.yaml carries no world size: the runtime derives
        # the data-parallel replicate degree from it at launch. Without -d the
        # cluster would be inferred as d*t*cp*p, which understates HSDP runs
        # and silently invalidates every candidate in the search.
        parser.error(
            "-d/--devices is required for hyper_v2: the device count is not "
            "expressible in an AutoModels train.yaml. Alternatively set "
            "context.device_num in the config."
        )

    set_verbose_level(args.verbosity)
    Debug.set_output_dir(args.output_dir, fresh=True)
    dims = Dim.get_dims(args.dimensions)
    YAML_FOLDER = None  # args.generate_yaml_in
    machine = Hard.Machine(args.devices, args.device_type or "A2")
    host_link = _host_link(parser, args, machine.device)

    if args.framework == "hyperparallel2":
        if args.yaml_config is None or args.train_yaml is None or args.accelerate_yaml is None:
            parser.error("-y (model yaml), --train-yaml, and --accelerate-yaml are required for hyperparallel2")
        input_config = {
            "model": args.yaml_config,
            "train": args.train_yaml,
            "accelerate": args.accelerate_yaml,
            "machine": args.devices
        }
    elif args.framework == "torchtitan":
        module, config = args.yaml_config.split(":")
        input_config = {
            "module": module,
            "config": config,
            "machine": machine,
        }
    else:
        input_config = args.yaml_config

    # The workers start now and import while this process parses the run; each
    # rebuilds its runner from these values, so only the plain search takes them.
    workers = args.jobs if args.jobs is not None else _pool.default_workers(args.verbosity)
    pool = None
    if workers > 1 and isinstance(input_config, str) and args.real_csv is None and not args.auto_recompute \
            and recompute_modes is None:
        pool = _pool.Workers(workers, {
            "framework": args.framework, "yaml_config": input_config, "devices": args.devices,
            "device_type": args.device_type or "A2", "global_batch_size": args.global_batch_size,
            "dimensions": args.dimensions, "swap_os": args.swap_opt_state, "mppb": args.mppb,
            "model": args.model, "max_mem": args.max_mem.strip() if args.max_mem is not None else None,
            "mem_for_ppb": args.mem_for_ppb.strip(), "verbosity": args.verbosity,
            "output_dir": args.output_dir, "strict": args.strict,
        })

    nd_runner = Par.Parallelize(
        args.framework,
        input_config,
        machine,
        global_batch_size=args.global_batch_size,
        dimensions=dims,
        swap_os=args.swap_opt_state,
        mppb=args.mppb,
        auto_recompute=args.auto_recompute,
        auto_offload=args.auto_offload,
        host_link=host_link,
        model=args.model,
        # model="Telecom",  # args.model ====ONLY FOR XINYU BRANCH====
        max_mem=max_mem,
        mem_for_ppb=Memory.from_string(args.mem_for_ppb.strip()),
        # vpp_less_mem=args.less_memory,
        workers=pool,
        **_recompute_kwargs(args.auto_recompute, recompute_modes),
    )

    if args.real_csv is not None:
        _compare_with_real_csv(nd_runner, args)
        sys.exit(0)

    if YAML_FOLDER and not os.path.exists(YAML_FOLDER):
        os.makedirs(YAML_FOLDER)

    space = nd_runner.run_generation_to_ordering(
        YAML_FOLDER,
        threads_num=workers if pool is not None else None,
        top_num=args.top_config_number,
        cache_file=args.cache_file,
        ranking_csv=args.ranking_csv,
        exhaustive=exhaustive_count,
        force_exhaustive=force_exhaustive,
        fforce_exhaustive=fforce_exhaustive,
        dimensions=dims,
        recompute_potentials=args.recompute_potentials,
    )

    if args.ppb_k is not None:
        if not 0 <= args.ppb_k < len(space):
            parser.error(f"-k/--ppb_k: {len(space)} configurations were ranked, got {args.ppb_k}")
        yaml_name = os.path.splitext(os.path.basename(str(args.yaml_config)))[0]
        nd_runner.to_ppb(space, args.ppb_k, yaml_name)
