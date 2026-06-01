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
"""CLI entrypoint for SAPP-PPB pipeline balancing."""
import argparse
import os
import sys

from sapp_ppb.sapp.sapp_pipeline import SappPipeline
from sapp_ppb.utils import interactive
from sapp_ppb.utils.compute_memory import compute_memories
from sapp_ppb.utils.config import initialize_layer_json
from sapp_ppb.utils.layer import generate_layers_list
from sapp_ppb.utils.logger import logger


def _str2bool(value: str) -> bool:
    """Parse a truthy string value coming from ``argparse``."""
    return str(value).lower() in ('true', '1', 'yes')


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline-balance CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog='SAPP AutoBalancing',
        description='Balance layers onto pipeline stages, considering recomputation and interleaving',
        epilog='')

    # Pipeline info
    parser.add_argument('-s', '--stage', type=int, default=4, help="Number of stages")
    parser.add_argument('-mb', '--micro_batch', type=int, default=4, help="Number of micro batch")
    parser.add_argument('-i', '--interleave_degree', type=int, default=1, help="Interleave level")

    # Memory size
    parser.add_argument('-mem', '--max_memory', type=int, default=56000,
                        help="Maximum memory available (MB)")
    parser.add_argument('-lm', '--less_memory', type=_str2bool, default=False,
                        help="Compute Memory with 'Less Memory interleave' option")
    parser.add_argument('-dual', '--dualpipe_v', type=_str2bool, default=False,
                        help="Compute Memory with 'DualpipeV' option")
    parser.add_argument('-mc', '--constant_memory', type=int, default=0,
                        help="Constant memory per stages")

    parser.add_argument('-o', '--output_folder', type=str, default="./output",
                        help="output files location")

    # Model info
    parser.add_argument('-m', '--model_name', type=str, default="model_name", help="")

    # Search time
    parser.add_argument('-t', '--time_limit', type=int, default=90,
                        help="Limitation on searching time")

    # Optimization level
    parser.add_argument('-O', '--optimization_level', type=int, default=1,
                        help="Defines optimization level when Stage (S) = Micro Batch number (M). "
                             "0 for same approach as M > S. "
                             "1 (default) generally better. "
                             "2 better for memory constrained cases.")

    # Simulate naive or manual config
    parser.add_argument('-naive', '--simulate_naive', type=_str2bool, default=False,
                        help="Simulate naive configs")
    parser.add_argument('-manual', '--manual_config', type=str, default=None,
                        help="Path of manual config")

    # Layer info
    parser.add_argument('-lf', '--layer_folder', type=str, default="./layers/",
                        help="Path to the layer folder")
    parser.add_argument('-dump', '--dump_layer', type=_str2bool, default=False,
                        help="Dump the layers")

    # For Computation of memory
    parser.add_argument('-mf', '--memory_folder', type=str, default="./memory/",
                        help="Path to the profiler memory folder")

    # For Initialization
    parser.add_argument('-init', '--init', type=str, default=None,
                        help="Path to the init file")

    # Computation argument
    parser.add_argument('-cm', '--compute_memory', type=_str2bool, default=False,
                        help="Parse Mindspore log to generate MEMORY of the layer (unavailable)")
    parser.add_argument('-exec', '--exec', type=_str2bool, default=True,
                        help="Compute solver")
    return parser


def _resolve_path(base_dir: str, path: str) -> str:
    """Return ``path`` resolved relative to ``base_dir`` unless it is already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def run(args: argparse.Namespace, base_dir: str) -> None:
    """Execute the pipeline balancing workflow for the given arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        base_dir (str): Directory used to resolve relative input / output paths.
    """
    if args.init:
        init_file = _resolve_path(base_dir, args.init)
        initialize_layer_json(args.model_name, init_file)

    output_folder = _resolve_path(base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    manual_config = None
    if args.manual_config:
        candidate = _resolve_path(base_dir, args.manual_config)
        if candidate.endswith(('yaml', 'yml')):
            manual_config = candidate

    layers = generate_layers_list(args.layer_folder, args.model_name)
    if args.compute_memory:
        layers = compute_memories(layers=layers, memory_folder=args.memory_folder,
                                  number_of_stage=args.stage)
    for layer in layers:
        logger.output("%s", layer)

    if args.dump_layer:
        for layer in layers:
            layer.dump()

    pipe = SappPipeline(model_name=args.model_name, num_of_stage=args.stage,
                        num_of_micro_batch=args.micro_batch, max_memory=args.max_memory,
                        layers=layers, num_of_interleave=args.interleave_degree,
                        vpp_less_memory=args.less_memory, dual=args.dualpipe_v,
                        constant_memory=args.constant_memory,
                        optimization_level=args.optimization_level)

    pipe.construct_problem(solver="pulp")

    if args.exec:
        pipe.solve_problem(time_limit=args.time_limit, dump_folder=output_folder)
        pipe.print_yaml_results()
        total_time = pipe.simulate(show=True, file_name=os.path.join(output_folder, "result.svg"))

        logger.output("total_time: %d", total_time)
        logger.output("time: %s", pipe.get_time())
        logger.output("mem_par: %s", pipe.get_memory_parameter())
        logger.output("mem_act: %s", pipe.get_memory_activation())

        if manual_config:
            logger.output("Simulating manual configs")
            pipe.simulate_comparison(manual_config, output_folder)
        if args.simulate_naive:
            logger.output("Simulating naive configs")
            pipe.simulate_naive(layers, output_folder)
    elif manual_config:
        logger.output("Simulating manual configs")
        pipe.simulate_only_manual(manual_config, output_folder)


def main() -> None:
    """Entry point invoked when the module is run as a script."""
    if len(sys.argv) == 1:
        interactive.main()
        return
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args, base_dir=os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
