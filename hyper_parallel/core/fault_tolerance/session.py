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
"""MindIO-coordinated process recovery independent of parameter partitioning."""

import importlib
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist

from hyper_parallel.core.fault_tolerance.config import RecoveryConfig
from hyper_parallel.core.fault_tolerance.adapters import ModelRecoveryAdapter
from hyper_parallel.core.fault_tolerance.groups import GroupRegistry
from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState, TrainState, local_tensor
from hyper_parallel.core.fault_tolerance.topology import ReplicaTopology
from hyper_parallel.core.fault_tolerance.transport import DeviceMetadataBank, ReplicaTransport

logger = logging.getLogger(__name__)


class RecoverySession:
    """Protect an explicit training body with MindIO Processor callbacks.

    Construct before the model, use ``groups`` around deterministic local
    mesh/model construction, then ``attach`` the persistent local state.
    Call ``run(body)`` once: the SDK reenters only this body after rollback.
    The body must recreate its iterator from the restored loader and use
    ``update`` around optimizer, main-param copyback and scheduler updates.
    Setup, checkpoint loads and teardown belong outside the reentrant body.

    Rank zero hosts the Controller, as in MindSpeed's ordinary launch path.
    Worker restart remains owned by the existing launcher/TaskD Agent.
    """

    def __init__(self, config: RecoveryConfig) -> None:
        """Validate the Ascend runtime and allocate process-local recovery bookkeeping."""
        self.config = config
        if not config.enable_worker_reboot:
            raise ValueError("enable_worker_reboot must be true to construct a RecoverySession")
        self.rank = dist.get_rank() if dist.is_initialized() else int(os.environ["RANK"])
        self.world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.sdk = importlib.import_module("mindio_ttp.framework_ttp")
        self.npu = importlib.import_module("torch_npu").npu
        required = ("stop_device", "restart_device", "set_device", "synchronize")
        if any(not callable(getattr(self.npu, name, None)) for name in required):
            raise RuntimeError("torch-npu stop/restart/synchronize APIs are required")
        if not callable(getattr(dist, "reinit_process_group", None)):
            raise RuntimeError("torch-npu reinit_process_group is required")
        if os.getenv("TORCH_DIST_INIT_BARRIER", "0") != "0":
            raise ValueError("TORCH_DIST_INIT_BARRIER must be 0 for replacement worker construction")
        self.npu.set_device(self.local_rank)
        self.groups = GroupRegistry()
        self.bank = DeviceMetadataBank(self.rank, torch.device("npu", self.local_rank))
        self.channel = None
        self.topology = None
        self.adapter = None
        self.step = 0
        self._live_step = None
        self.reboot = False
        self.model_state = None
        self.train_state = None
        self._model = None
        self._stopped = False
        self._in_update = False
        self._initialized = False
        self._running = False
        self._owns_controller = False
        self._resume_iterator = False

    def initialize(self) -> None:
        """Start an explicitly configured Controller or connect to TaskD's Controller."""
        if self._initialized:
            raise RuntimeError("Processor is already initialized")
        controller_ip, controller_port = self._controller_endpoint()
        try:
            if self.rank == 0 and self.config.controller_ip:
                # MindIO builds with separate Controller/Processor native
                # loggers need distinct paths when both live in rank zero.
                log_path = os.environ.get("TTP_LOG_PATH")
                os.environ["TTP_LOG_PATH"] = os.path.join(log_path or "logs", "controller")
                try:
                    self.sdk.tft_init_controller(0, self.world_size, False, True, False)
                    self._owns_controller = True
                    self.sdk.tft_start_controller(controller_ip, controller_port, False, "")
                finally:
                    if log_path is None:
                        os.environ.pop("TTP_LOG_PATH", None)
                    else:
                        os.environ["TTP_LOG_PATH"] = log_path
            self.sdk.tft_init_processor(self.rank, self.world_size, False, False, "", False, True, False)
            self._initialized = True
            self.sdk.tft_start_processor(controller_ip, controller_port)
            self.reboot = self.sdk.tft_is_reboot_node()
            self.groups.reboot = self.reboot
            self._initialized = True
        except BaseException:
            self.close()
            raise

    def _controller_endpoint(self) -> tuple[str, int]:
        """Resolve TaskD deployment metadata without using a worker's local Pod IP."""
        if self.config.controller_ip:
            return self.config.controller_ip, self.config.controller_port
        names = ("PROCESSOR_ADDR", "CONTROLLER_ADDR", "TTP_ADDR", "MASTER_ADDR")
        address = next((os.environ[name] for name in names
                        if os.environ.get(name)), "")
        port = os.getenv("TTP_PORT", "")
        if not address or not port:
            raise ValueError("TaskD Controller requires PROCESSOR_ADDR/CONTROLLER_ADDR/TTP_ADDR/MASTER_ADDR "
                             "and TTP_PORT pointing to its reachable deployment endpoint")
        try:
            port = int(port)
        except ValueError as error:
            raise ValueError("TaskD TTP_PORT must be an integer between 1 and 65535") from error
        if not 0 < port < 65536:
            raise ValueError("TaskD TTP_PORT must be between 1 and 65535")
        return address, port

    def attach(self, model: torch.nn.Module, optimizer: Any, train_state: TrainState, step: int = 0) -> None:
        """Bind every protected object and register the generic Torch callbacks.

        The caller must have built an eager model without
        activation/optimizer offload or external mutable state. Initial
        checkpoint loading is permitted only on original workers.
        """
        if not self._initialized or self.model_state is not None:
            raise RuntimeError("initialize once before attaching training state")
        if not dist.is_initialized() or dist.get_rank() != self.rank or str(dist.get_backend()) != "hccl":
            raise ValueError("attach requires the matching initialized HCCL world")
        if dist.get_world_size() != self.world_size or step < 0:
            raise ValueError("world size or checkpoint step disagrees with recovery configuration")
        self.adapter = ModelRecoveryAdapter(model, reboot=self.reboot)
        self.model_state = ModelOptimizerState(model, optimizer)
        self.topology = ReplicaTopology.discover(self.model_state, self.rank, self.world_size, self.reboot)
        self.train_state = train_state
        self.bank.replica_providers = train_state.replica_providers
        self._model = model
        self.step = step
        self._live_step = None if self.reboot else step
        ranks = list(self.topology.members(self.rank))
        replicas = len(ranks)
        replica_info = [{"rank_list": ranks, "replica_cnt": replicas, "replica_shift": 0}]
        self.sdk.tft_set_optimizer_replica(self.rank, replica_info)
        handlers = {
            "stop": self._stop,
            "clean": self._clean,
            "repair": self._repair,
            "rollback": self._rollback,
            "rebuild_group": self._rebuild,
            "stream_sync": self._sync,
            "save_ckpt": self._unsupported_dump,
            "rename": self._unsupported_dump
        }
        for name, handler in handlers.items():
            getattr(self.sdk, f"tft_register_{name}_handler")(handler)
        with self.groups:
            group = self.topology.create_group()
        self.channel = ReplicaTransport(group, self.bank, self.model_state, torch.device("npu", self.local_rank))
        self.groups.sealed = True

    def _capture_metadata(self, step: int) -> None:
        """Keep rank-specific metadata and mutable buffers; never snapshot weights or moments."""
        state = self.model_state.rank_state()
        self.bank.put(step, {"state": self.train_state.capture(), "extra": state["extra"]}, state["buffers"])

    @contextmanager
    def update(self, step: int) -> Iterator[None]:
        """Guard one positive, consecutive optimizer/scheduler update transaction.

        Data consumption and forward/backward precede this context. All
        writes to optimizer-owned values, main-to-model copyback, scheduler
        and committed training counters must complete inside the context.
        No ``end_updating_os`` is reported after a partial or failed update.
        """
        if not self._running or self._in_update or step != self.step + 1:
            raise RuntimeError("updates must be consecutive transactions inside session.run")
        # Keep blocked forward/backward/gradient communication outside the
        # SDK update window so PRELOCK can fence the last completed step.
        self._sync()
        if dist.is_initialized():
            dist.barrier()
        self.sdk.tft_start_updating_os()
        self._in_update = True
        self._live_step = None
        try:
            yield
            self._sync()
            self._live_step = step
            self._capture_metadata(step)
            self.sdk.tft_end_updating_os(step)
            self.step = step
            self.channel.mirror_metadata(step)
            self.pause()
        finally:
            self._in_update = False

    def pause(self) -> None:
        """Cooperate at a Python step boundary, including CPU-only waiting loops.

        ``tft_pause_train`` implements SDK pause commands; PRELOCK itself
        fences optimizer entry rather than raising from arbitrary Python
        code. After device stop, a boundary must also unwind into the SDK
        exception loop so its stop callback can finish the main-thread ACK.
        """
        if self._stopped:
            raise RuntimeError("FORCE STOP")
        self.sdk.tft_pause_train(self.step)

    def run(self, body: Callable[["RecoverySession"], Any]) -> Any:
        """Enter MindIO's exception loop and release the embedded Controller on termination."""
        if self.model_state is None or self._running:
            raise RuntimeError("attach once before running a recovery body")
        self._running = True
        entered = False

        @self.sdk.tft_exception_handler
        def _train(session):
            nonlocal entered
            entered = True
            if session.reboot:
                session.reboot = False
                raise RuntimeError("ARF FINISH")
            result = body(session)
            # Rank zero's SDK wrapper destroys the embedded Controller on
            # return. Disconnect Processors first so Controller shutdown
            # cannot race their live native polling callbacks.
            dist.barrier()
            # Keep the Controller alive while exit handshakes are processed;
            # rank zero disconnects last, after the remote Processors finish.
            for rank in reversed(range(self.world_size)):
                if self.rank == rank:
                    self.sdk.tft_destroy_processor()
                    self._initialized = False
                dist.barrier()
            return result

        try:
            if not self.reboot:
                # Original workers rendezvous once before the first mirror.
                # Replacement workers must enter the SDK repair wait instead.
                dist.barrier()
                self._capture_metadata(self.step)
                self.channel.mirror_metadata(self.step)
                if self.step:
                    self.sdk.tft_report_load_ckpt_step(self.step)
            with self.groups:
                result = _train(self)
            if not os.getenv("MINDX_TASK_ID"):
                self._owns_controller = False
            return result
        finally:
            self._running = False
            # The SDK exception wrapper owns Processor destruction after
            # entering its loop. close() handles pre-loop setup failures.
            if entered:
                self._initialized = False
            self.close()

    def close(self) -> None:
        """Release setup resources when construction or attachment fails."""
        if self._initialized and not self._running:
            self.sdk.tft_destroy_processor()
            self._initialized = False
        if self._owns_controller and not self._running:
            self.sdk.tft_destroy_controller()
            self._owns_controller = False

    def _sync(self) -> None:
        self.npu.set_device(self.local_rank)
        self.npu.synchronize()

    def _stop(self, train_args, ctx) -> None:
        del train_args, ctx
        self._stopped = True
        result = self.npu.stop_device(self.local_rank)
        if result not in (None, 0):
            raise RuntimeError(f"NPU stop_device failed: {result}")
        logger.info("Recovery device stopped: rank=%s step=%s", self.rank, self.step)

    def _clean(self, is_uce_error, train_args, ctx) -> int:
        del train_args, ctx
        if is_uce_error or not self._stopped:
            raise RuntimeError("clean requires a completed device stop; UCE recovery is unsupported")
        self.npu.set_device(self.local_rank)
        self.adapter.abort_iteration()
        self.npu.restart_device(self.local_rank)
        recovery = importlib.import_module("torch_npu.npu._recovery")
        tensors = [tensor for _, tensor in self.model_state.tensor_items()]
        tensors.extend(self.bank.tensors())
        tensors.extend(self.model_state.buffers.values())
        for tensor in tensors:
            recovery.update_npu_tensor_to_safe(local_tensor(tensor))
        self._stopped = False
        logger.info("Recovery device cleaned: rank=%s", self.rank)
        return 0

    def _rebuild(self, fault_ranks, train_args, ctx) -> None:
        del train_args, ctx
        if 0 in fault_ranks and self.config.controller_ip:
            raise RuntimeError("rank-zero Controller loss requires restarting the job in this hosting mode")
        self.groups.rebuild()
        self.model_state.rebuild_runtime_state()
        logger.info("Recovery groups rebuilt: rank=%s generation=%s", self.rank, self.groups.generation)

    def _repair(self, step, need_rebuild, error_ranks, repair_info, train_args, ctx) -> None:
        del need_rebuild, error_ranks, train_args, ctx
        repair_type = repair_info["repair_type"]
        if repair_type == self.sdk.RepairType.RT_SEND.value:
            if self._live_step != step:
                raise RuntimeError("donor does not hold valid device state at the requested recovery step")
            for destination in repair_info["dst"]:
                self.channel.send(destination, step)
            return
        if repair_type != self.sdk.RepairType.RT_RECV_REPAIR.value:
            raise RuntimeError(f"replica-only recovery does not support repair strategy {repair_type}")
        pairs = list(zip(repair_info["src"], repair_info["dst"]))
        sources = [source for source, destination in pairs if destination == self.rank]
        if len(sources) != 1:
            raise RuntimeError("dense recovery requires exactly one source per destination")
        self.channel.receive(sources[0], step)
        self._live_step = step
        logger.info("Recovery state received: rank=%s source=%s step=%s", self.rank, sources[0], step)

    def _rollback(self, step, train_args, ctx) -> None:
        del train_args, ctx
        if self._live_step != step:
            raise RuntimeError("device state requires replica repair before rollback; no CPU snapshot fallback exists")
        snapshot = self.bank.get(self.rank, step)
        self.npu.set_device(self.local_rank)
        metadata = self.bank.metadata(self.rank, step)
        self.model_state.restore_rank_state(snapshot["buffers"], metadata["extra"])
        self.train_state.restore(metadata["state"])
        self._sync()
        self.step = step
        self._resume_iterator = True
        logger.info("Recovery rollback completed: rank=%s step=%s", self.rank, step)

    def make_iterator(self) -> Iterator[Any]:
        """Recreate the consumed cursor without advancing committed RNG on reentry."""
        iterator = self.train_state.make_iterator(preserve_rng=self._resume_iterator)
        self._resume_iterator = False
        return iterator

    @staticmethod
    def _unsupported_dump(*args: Any) -> None:
        """Reject SDK dump strategies rather than acknowledge nonexistent files."""
        del args
        raise RuntimeError("replica-only recovery does not support checkpoint dumping")
