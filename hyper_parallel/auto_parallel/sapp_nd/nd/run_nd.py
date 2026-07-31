# Copyright 2024-2026 Huawei Technologies Co., Ltd
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

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger, set_verbose_level
import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard


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

    set_verbose_level(cli_args.verbosity)

    search_cfg = read_search_config(cli_args.search_config)

    if cli_args.global_batch_size is not None:
        search_cfg.constraint["global_batch_size"] = cli_args.global_batch_size
    if cli_args.devices is not None:
        cards_per_node = max(1, search_cfg.cluster_spec.get("cards_per_node", 8))
        search_cfg.cluster_spec["num_nodes"] \
            = max(1, cli_args.devices // cards_per_node)

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
    logger.output(
        "Optimal strategy: dp=%(dp)s tp=%(tp)s pp=%(pp)s "
        "cp=%(cp)s ep=%(ep)s mb_num=%(micro_batch_num)s "
        "mem=%(memory_estimate_mb).0f MB score=%(score).2e",
        result,
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
    # parser.add_argument(
    #     "-j",
    #     "--threads_num",
    #     type=int,
    #     default=None,
    #     help="Number of threads for the space generation",
    # )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Level of verbosity in range [0,6], "
        "0 being no output and 6 being debug level output. "
        "Plot and debug csv are generated from 2",
    )
    # parser.add_argument(
    #     "-k",
    #     "--ppb_k",
    #     type=int,
    #     default=None,
    #     help="choose configuration number k for ppb",
    # )
    parser.add_argument(
        "-A",
        "--device_type",
        default="A2",
        help="choose device type between A2 or A3",
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
        help="Memory to reserve for pipeline balancing. "
        "Will be decreased from the memory budget allowed by ND (default 0GB)",
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
        help="Memory to reserve for pipeline balancing. "
        "Will be decreased from the memory budget allowed by ND (default 0GB)",
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
        "(default: current directory).",
    )

    args = parser.parse_args()

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

    set_verbose_level(args.verbosity)
    dims = Dim.get_dims(args.dimensions)
    YAML_FOLDER = None  # args.generate_yaml_in
    machine = Hard.Machine(args.devices, args.device_type)

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
        threads_num=None,  # args.threads_num
        top_num=args.top_config_number,
        cache_file=args.cache_file,
    )
