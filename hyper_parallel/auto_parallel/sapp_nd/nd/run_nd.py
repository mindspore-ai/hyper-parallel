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
import os
import sys

import yaml

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import set_strict
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger, set_verbose_level
import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard


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
    if getattr(cli_args, "cache_file", None) is not None:
        # -c names the ratios the search scores with, as on the CLI path;
        # the search config path dropped it and ranked on the raw score.
        search_cfg.estimator["ratios_file"] = cli_args.cache_file


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
    Debug.set_output_dir(cli_args.output_dir)

    search_cfg = read_search_config(cli_args.search_config)
    try:
        _apply_cli_overrides(search_cfg, cli_args)
    except ValueError as exc:
        cli_parser.error(str(exc))
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

    result = search_strategies(search_cfg)
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
    """Log the activation checkpoint mode a search chose for every layer.

    Args:
        result: The search's result, which states ``activation_checkpoint``.
    """
    logger.output("Activation checkpoint mode for every layer: %s", result["activation_checkpoint"])


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
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Level of verbosity in range [0,6], "
        "0 being no output and 6 being debug level output. "
        "Plot and debug csv are generated from 2",
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
        "-t",
        "--top_config_number",
        type=int,
        default=None,
        help="Number of top configs to print & plot",
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
        help="Cache file with ratios to recalibrate ND scores. "
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
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files when using --search-config "
        "(default: current directory), and for the search's plot and debug CSVs.",
    )
    parser.add_argument(
        "--ranking_csv",
        type=str,
        default=None,
        help="Also write every configuration the search keeps, in ND's order, "
        "to this CSV: rank, degrees, memory in MB, score and its parts.",
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
    Debug.set_output_dir(args.output_dir)
    dims = Dim.get_dims(args.dimensions)
    YAML_FOLDER = None  # args.generate_yaml_in
    machine = Hard.Machine(args.devices, args.device_type or "A2")

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

    nd_runner = Par.Parallelize(
        args.framework,
        input_config,
        machine,
        global_batch_size=args.global_batch_size,
        dimensions=dims,
        swap_os=args.swap_opt_state,
        mppb=args.mppb,
        model=args.model,
        # model="Telecom",  # args.model ====ONLY FOR XINYU BRANCH====
        max_mem=max_mem,
        mem_for_ppb=Memory.from_string(args.mem_for_ppb.strip()),
        # vpp_less_mem=args.less_memory,
    )

    if YAML_FOLDER and not os.path.exists(YAML_FOLDER):
        os.makedirs(YAML_FOLDER)

    space = nd_runner.run_generation_to_ordering(
        YAML_FOLDER,
        threads_num=None,
        top_num=args.top_config_number,
        cache_file=args.cache_file,
        ranking_csv=args.ranking_csv,
    )
