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
"""Parallel-model/MindIO acceptance worker and isolated management harness.

The default harness replaces TaskD/ClusterD. Optional TaskD mode exercises
real Controller adapters and Agent/Torch restart callbacks. Upstream decisions
and Manager-to-Agent queue delivery remain test-driven, without Kubernetes.
"""

import argparse
import faulthandler
import importlib
import json
import logging
import os
import queue
import random
import secrets
import subprocess
import sys
import time
import threading
from datetime import timedelta
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import (
    ColwiseParallel, DTensor, Replicate, RowwiseParallel, Shard, SkipDTensorDispatch, distribute_tensor,
    fully_shard, init_device_mesh, parallelize_module,
)
from hyper_parallel.core.fault_tolerance import RecoveryConfig, RecoverySession, TrainState
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.core.utils.communication import differentiable_all_to_all_single
from hyper_parallel.data.indexed.indexed_data_config import resolve_data_paths
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext, build_dataset_for_dataloader
from tests.common.port_utils import allocate_port


class Cursor:
    """Consumed-batch position independent of an iterator instance."""

    def __init__(self) -> None:
        """Start at the first sample."""
        self.position = 0

    def state_dict(self) -> dict[str, int]:
        """Capture consumed samples."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore consumed samples."""
        self.position = state["position"]

    def __iter__(self) -> Iterator[int]:
        """Continue data consumption from the restored committed cursor."""
        while True:
            self.position += 1
            yield self.position


class _ExpertModel(torch.nn.Module):
    """Unique local expert weights with real differentiable EP token dispatch."""

    def __init__(self, group: Any, experts: int) -> None:
        """Bind the expert group without performing a collective during construction."""
        super().__init__()
        bias = os.getenv("HP_RECOVERY_TEST_OPTIMIZER", "adamw") != "muon"
        self.layer = torch.nn.Linear(8, 8, bias=bias)
        self.dropout = torch.nn.Dropout(0.2)
        self.output = torch.nn.Linear(8, 8, bias=bias)
        self.group = group
        self.experts = experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Dispatch two tokens per expert before local expert computation."""
        dispatched = differentiable_all_to_all_single(inputs, [2] * self.experts, [2] * self.experts, self.group)
        return self.output(self.dropout(self.layer(dispatched)))


def _fault_ranks(world: int) -> tuple[int, ...]:
    ranks = tuple(int(rank) for rank in os.getenv("HP_RECOVERY_TEST_FAULT_RANKS", "1").split(","))
    if not ranks or len(set(ranks)) != len(ranks) or any(rank < 0 or rank >= world for rank in ranks):
        raise ValueError("fault ranks must be distinct members of the training world")
    if len(ranks) == world or (0 in ranks and os.getenv("HP_RECOVERY_TEST_CONTROLLER") != "taskd"):
        raise ValueError("fault injection requires surviving replicas and a surviving Controller")
    return ranks


def worker(options: argparse.Namespace) -> None:
    """Run the selected model or complete Trainer acceptance worker.

    Args:
        options: Worker rank, run directory, ports and fault-injection mode.
    """
    if os.getenv("HP_RECOVERY_TEST_TEXT_TRAINER"):
        importlib.import_module("tests.torch.fault_tolerance.text_trainer_recovery").worker(options)
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    rank, world = options.rank, options.world
    fault_ranks = _fault_ranks(world)
    run_dir = Path(options.directory)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    taskd_hosted = os.getenv("HP_RECOVERY_TEST_CONTROLLER") == "taskd"
    config = RecoveryConfig(True) if taskd_hosted else RecoveryConfig(True, "127.0.0.1", options.controller_port)
    session = RecoverySession(config)
    logging.basicConfig(level=logging.INFO)
    if rank == 0 and not taskd_hosted:
        _start_control_bridge(run_dir)
    session.initialize()
    store = dist.TCPStore("127.0.0.1", options.store_port, None, False, timedelta(seconds=120))
    dist.init_process_group("hccl", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    mode = os.getenv("HP_RECOVERY_TEST_LAYOUT", "hsdp")
    if mode not in ("hsdp", "tp", "ep", "tp_ep"):
        raise ValueError(f"unknown recovery acceptance layout {mode}")
    ep_size = world if mode == "ep" else (2 if mode == "tp_ep" else 1)
    tp_size = world // ep_size if mode in ("tp", "tp_ep") else 1
    torch.manual_seed(42 + (rank // tp_size if ep_size > 1 else 0))
    dtype = torch.bfloat16 if os.getenv("HP_RECOVERY_TEST_DTYPE") == "bf16" else torch.float32
    with session.groups:
        _bootstrap_dataset(run_dir)
        if ep_size > 1:
            mesh = init_device_mesh("npu", (ep_size, tp_size), mesh_dim_names=("ep", "tp"))
            model = _ExpertModel(mesh["ep"].get_group(), ep_size).to(device="npu", dtype=dtype)
            if tp_size > 1:
                parallelize_module(model, mesh["tp"], {"layer": ColwiseParallel(use_local_output=True),
                                                       "output": RowwiseParallel(use_local_output=True)})
        else:
            bias = os.getenv("HP_RECOVERY_TEST_OPTIMIZER", "adamw") != "muon"
            layers = [torch.nn.Linear(8, 8, bias=bias), torch.nn.Dropout(0.2)]
            if mode == "tp":
                layers.append(torch.nn.Linear(8, 8, bias=bias))
            model = torch.nn.Sequential(*layers).to(device="npu", dtype=dtype)
            if mode == "hsdp":
                replicas = int(os.getenv("HP_RECOVERY_TEST_HSDP_REPLICAS", "2"))
                mesh = init_device_mesh("npu", (replicas, world // replicas), mesh_dim_names=("replicate", "shard"))
                policy = MixedPrecisionPolicy(reduce_dtype=torch.float32,
                                              apply_grad_on_fp32_main_grad=dtype == torch.bfloat16)
                fully_shard(model, mesh=mesh, mp_policy=policy)
            elif mode == "tp":
                mesh = init_device_mesh("npu", (world,), mesh_dim_names=("tp",))
                parallelize_module(model, mesh, {"0": ColwiseParallel(use_local_output=True),
                                                 "2": RowwiseParallel(use_local_output=True)})
    with session.groups:
        optimizer, leaves = _build_optimizer(model, dtype)
    schedulers = {name: torch.optim.lr_scheduler.StepLR(leaf, step_size=3, gamma=0.8)
                  for name, leaf in leaves.items()}
    loader = Cursor()
    random.seed(100 + rank)
    np.random.seed(100 + rank)
    torch.manual_seed(100 + rank)
    torch.npu.manual_seed(100 + rank)
    session.attach(model, optimizer, TrainState(loader, replica_providers=schedulers, device=torch.npu))
    verified_generation = 0

    def _body(active: RecoverySession) -> int:
        nonlocal verified_generation
        active.publish_initial_state(active.step)
        if mode == "hsdp" and active.groups.generation > verified_generation:
            source = torch.full((8, 4), float(rank), device="npu")
            distributed = distribute_tensor(source, mesh, [Replicate(), Shard(0)], src_data_rank=0)
            local = distributed.to_local()
            torch.testing.assert_close(local, torch.zeros_like(local), rtol=0, atol=0)
            verified_generation = active.groups.generation
            logging.getLogger(__name__).info("Recovery training communication resumed: rank=%s generation=%s",
                                             rank, verified_generation)
        iterator = active.make_iterator()
        if active.train_state.rng_restore_pending:
            active.finish_restore()
        while active.step < 12:
            step = active.step + 1
            # On the first execution only, let the real Controller detect
            # SIGKILL before the management harness requests stop_train.
            if options.fault and step == 9 and active.groups.generation == 0:
                forward_fault = os.getenv("HP_RECOVERY_FAULT_PHASE") == "forward"
                while not forward_fault or rank in fault_ranks or not (run_dir / "forward_go").exists():
                    active.pause()
                    time.sleep(0.02)
                (run_dir / f"forward_started{rank}").touch()
            next(iterator)
            inputs = torch.arange(16 * ep_size, dtype=dtype, device="npu").reshape(2 * ep_size, 8) / 16
            inputs = inputs + loader.position * 0.01 + (rank // tp_size) * 0.001
            inputs.requires_grad_(True)
            loss = model(inputs).square().mean()
            loss.backward()
            if dtype == torch.bfloat16:
                _prepare_main_gradients(model)
            clip_grad_norm_(model, 1.0)
            with active.update(step):
                with SkipDTensorDispatch(no_skip={torch.zeros_like}):
                    optimizer.step()
                optimizer.zero_grad()
                for scheduler in schedulers.values():
                    scheduler.step()
            record = {"rank": rank, "pid": os.getpid(), "step": active.step, "cursor": loader.position}
            temporary = run_dir / f"rank{rank}.json.tmp"
            temporary.write_text(json.dumps(record))
            temporary.replace(run_dir / f"rank{rank}.json")
        # CPU copies here are acceptance artifacts after training has finished;
        # neither recovery nor normal-step protection reads these files.
        torch.save({"step": active.step, "model": {name: tensor.detach().cpu().clone()
                                                  for name, tensor in active.model_state.tensor_items()},
                    "optimizer_metadata": active.model_state.metadata(), "training": active.train_state.capture()},
                   run_dir / f"final{rank}.pt")
        for name, leaf in leaves.items():
            key = "momentum_buffer" if name == "muon" else "exp_avg"
            nonzero = sum(torch.count_nonzero(getattr(state[key], "_local_tensor", state[key]).cpu()).item()
                          for state in leaf.state.values() if key in state)
            total = torch.tensor(nonzero, dtype=torch.int64, device="npu")
            dist.all_reduce(total)
            assert total.item() > 0, f"acceptance must exercise actual {name} updates"
        # Controller reports success only after observing a post-repair
        # heartbeat. Tiny models can otherwise finish before that heartbeat.
        if options.fault:
            time.sleep(5)
        return active.step

    session.run(_body)
    dist.destroy_process_group()


def _bootstrap_dataset(directory: Path) -> None:
    """Exercise Trainer data discovery and cache synchronization during worker bootstrap."""
    paths = resolve_data_paths(str(directory / "bootstrap_data"), distributed_walk=True)
    context = DataLoaderParallelContext(distributed_enabled=True, build_on_rank=lambda: True,
                                        build_cache_on_rank=lambda: dist.get_rank() == 0, barrier=dist.barrier)
    dataset = build_dataset_for_dataloader(lambda: tuple(paths), context, barrier_needed=True)
    if dataset is None or len(dataset) != 4:
        raise RuntimeError("bootstrap acceptance must retain both weighted indexed source paths")


def _prepare_main_gradients(model: torch.nn.Module) -> None:
    """Supply the mixed-precision contract when the TP demo returns local gradients."""
    for parameter in model.parameters():
        if (isinstance(parameter, DTensor) and getattr(parameter, "main_grad", None) is None
                and parameter.grad is not None):
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                parameter.main_grad = gradient.float()
            else:
                parameter.main_grad = DTensor.from_local(
                    gradient.float(), parameter.device_mesh, parameter.placements, shape=tuple(parameter.shape))
            parameter.grad = None


def _build_optimizer(model: torch.nn.Module, dtype: torch.dtype) -> tuple[Any, dict[str, Any]]:
    """Exercise pure Muon and mixed Muon/AdamW through the same recovery path."""
    kind = os.getenv("HP_RECOVERY_TEST_OPTIMIZER", "adamw")
    if kind == "adamw":
        leaves = {"adamw": AdamW(model.parameters(), lr=0.01)}
    elif kind in ("muon", "muon_adamw"):
        matrices = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
        vectors = [parameter for parameter in model.parameters() if parameter.ndim < 2]
        replicas = int(os.getenv("HP_RECOVERY_TEST_MUON_REPLICA_COUNT", "1"))
        leaves = {"muon": Muon(matrices, lr=0.01, ns_steps=2, hsdp_replica_count=replicas)}
        if vectors:
            leaves["adamw"] = AdamW(vectors, lr=0.001)
    else:
        raise ValueError(f"unknown recovery acceptance optimizer {kind}")
    chain = ChainedOptimizer(model, leaves)
    optimizer = Float16OptimizerWithFloat16Params(chain, model) if dtype == torch.bfloat16 else chain
    return optimizer, leaves


def manage_phase(directory: Path, world: int, fault: bool) -> None:
    """Give every child and log a lifetime bounded by this management process."""
    with ExitStack() as resources:
        _manage_phase(directory, world, fault, resources)


@contextmanager
def _spawn(command: list[str], log: TextIO) -> Iterator[subprocess.Popen]:
    """Keep Popen's context open until the harness has terminated every child."""
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8") as child:
        yield child


def _start_control_bridge(directory: Path) -> None:
    """Expose test management commands while the real Controller lives in worker zero."""
    controller = importlib.import_module("mindio_ttp.controller_ttp")
    lock = threading.Lock()
    for action in ("report_fault_ranks", "report_stop_complete", "report_strategies", "report_result"):
        def _callback(*values: Any, action: str = action) -> None:
            with lock, (directory / "control_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"action": action, "values": values}) + "\n")
        controller.tft_register_mindx_callback(action, _callback)

    def _receive_commands() -> None:
        for line in sys.stdin:
            command = json.loads(line)
            values = command["values"]
            action = command["action"]
            if action == "stop":
                ranks = {int(key): value for key, value in values[0].items()}
                controller.tft_notify_controller_stop_train(ranks, "stop")
            elif action == "strategies":
                ranks = {int(key): value for key, value in values[0].items()}
                controller.tft_notify_controller_on_global_rank(ranks, 1)
            elif action == "recover":
                controller.tft_notify_controller_change_strategy("recover", "")

    threading.Thread(target=_receive_commands, name="test-management-bridge", daemon=True).start()


def _manage_phase(directory: Path, world: int, fault: bool, resources: ExitStack) -> None:
    """Drive rank zero's Controller, keeping only the rendezvous store independent."""
    npu_port = 30000 + secrets.randbelow(20000)
    fault_ranks = _fault_ranks(world)
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{npu_port}-{npu_port + 127}"

    directory.mkdir()
    for index in range(2):
        shard = directory / "bootstrap_data" / str(index)
        shard.mkdir(parents=True)
        (shard / f"corpus_source_{index}_0.bin").touch()
    controller_port, store_port = allocate_port(), allocate_port()
    store = dist.TCPStore("127.0.0.1", store_port, None, True,
                          timedelta(seconds=120), wait_for_workers=False)
    events = queue.Queue()
    event_offset = 0
    children = {}
    taskd = None
    supervisor = None

    def _launch(rank: int, generation: int) -> None:
        if supervisor is not None:
            children[rank] = supervisor.launch(rank, generation)
            return
        log = resources.enter_context((directory / f"worker{rank}.{generation}.log").open("w", encoding="utf-8"))
        command = [sys.executable, "-u", __file__, "--worker", "--rank", str(rank), "--world", str(world),
                   "--directory", str(directory), "--controller-port", str(controller_port),
                   "--store-port", str(store_port)]
        if fault:
            command.append("--fault")
        children[rank] = resources.enter_context(_spawn(command, log))

    injected = recovered = stopped = strategy = False
    evidence = []

    def _command(action: str, *values: Any) -> None:
        control = taskd if taskd is not None else children[0]
        control.stdin.write(json.dumps({"action": action, "values": values}) + "\n")
        control.stdin.flush()

    try:
        if os.getenv("HP_RECOVERY_TEST_CONTROLLER") == "taskd":
            os.environ.update(MINDX_TASK_ID="hp-taskd-validation", PROCESS_RECOVER="on", POD_IP="127.0.0.1",
                              TTP_PORT=str(controller_port), MASTER_ADDR="127.0.0.1", WORLD_SIZE=str(world),
                              MINDX_SERVER_IP="127.0.0.1")
            log = resources.enter_context((directory / "taskd_manager.log").open("w", encoding="utf-8"))
            taskd = resources.enter_context(_spawn(
                [sys.executable, "-u", __file__, "--taskd-manager", "--world", str(world),
                 "--directory", str(directory)], log))
        if os.getenv("HP_RECOVERY_TEST_AGENT") == "taskd":
            supervisor = importlib.import_module("tests.torch.fault_tolerance.taskd_agent").TaskdProcessSupervisor(
                directory, world, store, __file__, controller_port, store_port, fault)
        for rank in range(world):
            _launch(rank, 0)
        original_pids = {rank: child.pid for rank, child in children.items()}
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            event_path = directory / "control_events.jsonl"
            if event_path.exists():
                with event_path.open(encoding="utf-8") as stream:
                    stream.seek(event_offset)
                    for line in stream:
                        event = json.loads(line)
                        events.put((event["action"], event["values"]))
                    event_offset = stream.tell()
            records = [directory / f"rank{rank}.json" for rank in range(world)]
            if fault and not injected and all(path.exists() for path in records):
                progress = [json.loads(path.read_text()) for path in records]
                if min(record["step"] for record in progress) >= 8:
                    if os.getenv("HP_RECOVERY_FAULT_PHASE") == "forward":
                        (directory / "forward_go").touch()
                        forward_rank = next(rank for rank in range(world) if rank not in fault_ranks)
                        if not (directory / f"forward_started{forward_rank}").exists():
                            time.sleep(0.01)
                            continue
                        time.sleep(0.2)
                    injection_start = time.monotonic()
                    for rank in fault_ranks:
                        children[rank].kill()
                    injected = True
                    evidence.append({"event": "fault", "fault_ranks": fault_ranks, "progress": progress,
                                     "signal_span_seconds": time.monotonic() - injection_start})
            if all(child.poll() == 0 for child in children.values()):
                if fault and not recovered:
                    raise AssertionError("workers exited without successful Controller recovery result")
                final_pids = {rank: child.pid for rank, child in children.items()}
                if fault:
                    assert all(final_pids[rank] != original_pids[rank] for rank in fault_ranks), (
                        f"fault_ranks={fault_ranks}, original_pids={original_pids}, final_pids={final_pids}")
                    assert all(final_pids[rank] == original_pids[rank]
                               for rank in range(world) if rank not in fault_ranks), (
                                   f"fault_ranks={fault_ranks}, original_pids={original_pids}, final_pids={final_pids}")
                evidence.append({"event": "complete", "original_pids": original_pids, "final_pids": final_pids})
                (directory / "evidence.json").write_text(json.dumps(evidence, indent=2))
                return
            failures = {rank: child.returncode for rank, child in children.items()
                        if child.poll() is not None and child.returncode != 0
                        and not (rank in fault_ranks and injected and not strategy)}
            if failures:
                raise AssertionError(f"worker failures {failures}; logs: {directory}")
            try:
                action, values = events.get(timeout=0.05)
            except queue.Empty:
                continue
            evidence.append({"event": action, "values": values})
            if action == "report_fault_ranks" and not stopped:
                stopped = True
                _command("stop", values[0])
            elif action == "report_stop_complete":
                assert values[0] == 0, values
                _command("strategies", values[2])
            elif action == "report_strategies" and not strategy:
                assert "recover" in values[1], values
                assert set(map(int, values[0])) == set(fault_ranks), (
                    f"reported_fault_ranks={values[0]}, injected_fault_ranks={fault_ranks}")
                strategy = True
                _command("recover")
                for rank in fault_ranks:
                    children[rank].wait(timeout=10)
                if supervisor is not None:
                    children.update(supervisor.launch_workers(fault_ranks, 1))
                else:
                    for rank in fault_ranks:
                        _launch(rank, 1)
            elif action == "report_result":
                assert values[0] == 0, values
                recovered = True
        raise TimeoutError(f"recovery acceptance deadline exceeded; logs: {directory}")
    finally:
        for child in children.values():
            if child.poll() is None:
                child.kill()
        for child in children.values():
            child.wait(timeout=15)
        if supervisor is not None:
            supervisor.close()
        if taskd is not None:
            try:
                if taskd.poll() is None:
                    _command("shutdown")
                    taskd.wait(timeout=15)
            finally:
                if taskd.poll() is None:
                    taskd.kill()
                    taskd.wait(timeout=15)
            if taskd.returncode != 0:
                raise AssertionError(f"TaskD Manager exited with {taskd.returncode}; logs: {directory}")
        del store


def run_phase(directory: Path, world: int, fault: bool) -> None:
    """Give each Controller a separate SDK lifetime and native log file."""
    command = [sys.executable, "-u", __file__, "--manager", "--world", str(world), "--directory", str(directory)]
    if fault:
        command.append("--fault")
    with (directory.parent / f"{directory.name}.manager.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=330, check=False)
    assert result.returncode == 0, f"acceptance manager failed: {directory.parent / (directory.name + '.manager.log')}"


def assert_same_tree(actual: Any, expected: Any) -> None:
    """Compare exact final state including moments, scheduler, RNG and cursor."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_same_tree(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            assert_same_tree(item, reference)
    else:
        assert actual == expected


def test_worker_replacement(tmp_path):
    """Compare a killed/replaced worker run with uninterrupted execution."""
    os.environ.pop("MINDX_TASK_ID", None)
    os.environ["RESTART_FAULT_PROCESS_TYPE"] = "worker"
    os.environ["TORCH_DIST_INIT_BARRIER"] = "0"
    os.environ["HCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["HCCL_EXEC_TIMEOUT"] = "60"
    os.environ["HCCL_DETERMINISTIC"] = "true"
    world = int(os.getenv("HP_RECOVERY_TEST_WORLD_SIZE", "8"))
    assert world in (4, 8)
    assert torch_npu.npu.device_count() >= world
    run_phase(tmp_path / "baseline", world, False)
    run_phase(tmp_path / "recovery", world, True)
    for rank in range(world):
        expected = torch.load(tmp_path / "baseline" / f"final{rank}.pt", weights_only=False)
        actual = torch.load(tmp_path / "recovery" / f"final{rank}.pt", weights_only=False)
        assert_same_tree(actual, expected)


def test_text_trainer_replacement(tmp_path, monkeypatch):
    """Exercise the real demo main, TextTrainer lifecycle and indexed loader.

    Args:
        tmp_path: Isolated pytest directory for assets and training artifacts.
        monkeypatch: Fixture restoring the TaskD test environment on exit.
    """
    helper = importlib.import_module("tests.torch.fault_tolerance.text_trainer_recovery")
    config = helper.prepare_assets(tmp_path / "assets")
    monkeypatch.setenv("HP_RECOVERY_TEST_TEXT_TRAINER", str(config))
    monkeypatch.setenv("HP_RECOVERY_TEST_CONTROLLER", "taskd")
    monkeypatch.setenv("HP_RECOVERY_TEST_AGENT", "taskd")
    test_worker_replacement(tmp_path)
    for phase in ("baseline", "recovery"):
        for rank in range(8):
            records = [json.loads(line) for line in (tmp_path / phase / f"begin{rank}.jsonl").read_text().splitlines()]
            actual = [(record["step"], record["generation"]) for record in records]
            expected = [(0, 0), (8, 1)] if phase == "recovery" else [(0, 0)]
            assert actual == expected, f"{phase} rank={rank} begin callbacks: expected={expected}, actual={actual}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manager", action="store_true")
    parser.add_argument("--taskd-manager", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--controller-port", type=int)
    parser.add_argument("--store-port", type=int)
    parser.add_argument("--directory")
    parser.add_argument("--fault", action="store_true")
    parsed_options = parser.parse_args()
    if parsed_options.taskd_manager:
        importlib.import_module("tests.torch.fault_tolerance.taskd_control").run(
            Path(parsed_options.directory), parsed_options.world)
    elif parsed_options.manager:
        manage_phase(Path(parsed_options.directory), parsed_options.world, parsed_options.fault)
    else:
        worker(parsed_options)
