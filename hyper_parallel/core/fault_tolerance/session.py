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


class MetadataMirrorError(Exception):
    """Terminate training when required recovery metadata cannot be mirrored."""


class RecoverySession:
    """Protect an explicit training body with MindIO Processor callbacks.

    Construct before the model, use ``groups`` around deterministic local
    mesh/model construction, then ``attach`` the persistent local state.
    Call ``run(body)`` once: the SDK reenters only this body after rollback.
    The body must recreate its iterator from the restored loader and use
    ``update`` around optimizer, main-param copyback and scheduler updates.
    Setup, checkpoint loads and teardown belong outside the reentrant body.

    Rank zero hosts the explicitly configured Controller; otherwise TaskD does.
    Worker restart remains owned by the existing launcher/TaskD Agent.
    """

    def __init__(self, config: RecoveryConfig) -> None:
        """Validate the Ascend runtime and allocate process-local recovery bookkeeping."""
        self.config = config
        if not config.enable_worker_reboot:
            raise ValueError("enable_worker_reboot must be true to construct a RecoverySession")
        self.rank = dist.get_rank() if dist.is_initialized() else int(os.environ["RANK"])
        self.world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ["WORLD_SIZE"])
        # Reads LOCAL_RANK from the environment without requiring an initialized default process group.
        self.local_rank = dist.get_node_local_rank()
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
        self._stopped = False
        self._in_update = False
        self._initial_state_pending = False
        self._processor_initialized = False
        self._running = False
        self._controller_initialized = False

    def initialize(self) -> None:
        """Start an explicitly configured Controller or connect to TaskD's Controller."""
        if self._processor_initialized:
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
                    self._controller_initialized = True
                    self.sdk.tft_start_controller(controller_ip, controller_port, False, "")
                finally:
                    if log_path is None:
                        os.environ.pop("TTP_LOG_PATH", None)
                    else:
                        os.environ["TTP_LOG_PATH"] = log_path
            self.sdk.tft_init_processor(self.rank, self.world_size, False, False, "", False, True, False)
            # Allow close() to release the Processor if startup or reboot detection fails.
            self._processor_initialized = True
            self.sdk.tft_start_processor(controller_ip, controller_port)
            self.reboot = self.sdk.tft_is_reboot_node()
            self.groups.reboot = self.reboot
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
        """Bind protected objects and register recovery callbacks.

        The caller must have built an eager model without
        activation/optimizer offload or external mutable state. Initial
        checkpoint loading is permitted only on original workers.

        Args:
            model: Materialized model whose persistent local state is protected.
            optimizer: Optimizer or collection of supported optimizer leaves.
            train_state: Rank-owned and replica-equivalent training state adapter.
            step: Number of completed optimizer updates before training starts.
        """
        if not self._processor_initialized or self.model_state is not None:
            raise RuntimeError("initialize once before attaching training state")
        if not dist.is_initialized() or dist.get_rank() != self.rank or str(dist.get_backend()) != "hccl":
            raise ValueError("attach requires the matching initialized HCCL world")
        if dist.get_world_size() != self.world_size or step < 0:
            raise ValueError("world size or checkpoint step disagrees with recovery configuration")
        self.adapter = ModelRecoveryAdapter(model, reboot=self.reboot)
        self.model_state = ModelOptimizerState(model, optimizer)
        self.topology = ReplicaTopology.discover(self.model_state, self.rank, self.world_size, self.reboot)
        self.train_state = train_state
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
        group = self.topology.get_metadata_group(self.rank, self.groups)
        self.channel = ReplicaTransport(group, self.bank, self.model_state,
                                        torch.device("npu", self.local_rank))
        self.groups.sealed = True

    def _capture_metadata(self, step: int) -> None:
        """Keep rank-specific metadata and mutable buffers; never snapshot weights or moments."""
        state = self.model_state.rank_state()
        self.bank.put(step, {"state": self.train_state.capture(), "extra": state["extra"]}, state["buffers"])

    def publish_initial_state(self, step: int) -> None:
        """Publish the initial committed state after native start callbacks finish.

        Args:
            step: Completed updates after any normal initial checkpoint load.
        """
        if not self._initial_state_pending:
            return
        state = ModelOptimizerState(self.model_state.model, self.model_state.leaves)
        if (state.schema != self.model_state.schema
                or state.optimizer_ownership != self.model_state.optimizer_ownership):
            raise RuntimeError("training initialization changed the validated recovery layout")
        self.model_state = state
        self.channel.state = state
        self.step = step
        self._live_step = step
        self._capture_metadata(step)
        self._mirror_metadata(step)
        if step:
            self.sdk.tft_report_load_ckpt_step(step)
        self._initial_state_pending = False

    @contextmanager
    def update(self, step: int) -> Iterator[None]:
        """Guard one positive, consecutive optimizer/scheduler update transaction.

        Data consumption and forward/backward precede this context. All
        writes to optimizer-owned values, main-to-model copyback and scheduler
        must complete inside the context. Progress providers must capture the
        committed step even if the native Trainer advances its counter later.
        No ``end_updating_os`` is reported after a partial update or failed metadata mirror.

        Args:
            step: Next positive, consecutive committed optimizer step.

        Yields:
            Control to the native optimizer and scheduler update.
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
            self._mirror_metadata(step)
            self.sdk.tft_end_updating_os(step)
            self.step = step
            self.pause()
        finally:
            self._in_update = False

    def _mirror_metadata(self, step: int) -> None:
        """Require a complete metadata mirror before publishing a recoverable step."""
        try:
            self.channel.mirror_metadata(step)
        except Exception as error:
            # MindIO retries classified RuntimeErrors; incomplete mirrors must
            # use its non-retry exception path even if retry is enabled later.
            raise MetadataMirrorError(f"metadata mirror failed at step {step}; recovery is unavailable") from error

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

    def finish_restore(self) -> None:
        """Apply pending RNG once after native epoch initialization completes."""
        self.train_state.finish_restore()

    def run(self, body: Callable[["RecoverySession"], Any]) -> Any:
        """Enter the exception loop and release owned resources on termination.

        The body must publish initial state after its startup callbacks finish
        and before consuming training data. Publication is disabled on replacement
        workers and becomes a no-op after the original startup publication.

        Args:
            body: Reentrant training function receiving this attached session.

        Returns:
            The result returned by the completed training function.

        Raises:
            RuntimeError: The session is uninitialized, unattached, or already running.
        """
        if not self._processor_initialized or self.model_state is None or self._running:
            raise RuntimeError("initialize and attach once before running a recovery body")
        # The SDK retries training and teardown in one loop. A teardown error
        # followed by RETRY may reenter an already completed training body;
        # handling that exit-stage retry is outside the current recovery scope.
        @self.sdk.tft_exception_handler
        def _train(session):
            if session.reboot:
                session.reboot = False
                raise RuntimeError("ARF FINISH")
            return body(session)

        self._running = True
        self._initial_state_pending = not self.reboot
        try:
            with self.groups:
                result = _train(self)
            self._processor_initialized = False
            self._controller_initialized = False
        finally:
            self._running = False
            self.close()
        return result

    def close(self) -> None:
        """Release initialized resources outside the active SDK training loop."""
        if self._running:
            return
        if self._processor_initialized:
            self.sdk.tft_destroy_processor()
            self._processor_initialized = False
        if self._controller_initialized:
            self.sdk.tft_destroy_controller()
            self._controller_initialized = False

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
        logger.info("Recovery groups rebuilt: rank=%s generation=%s", self.rank, self.groups.generation)

    def _repair(self, step, need_rebuild, error_ranks, repair_info, train_args, ctx) -> None:
        del need_rebuild, error_ranks, train_args, ctx
        repair_type = repair_info["repair_type"]
        sending = repair_type == self.sdk.RepairType.RT_SEND.value
        if not sending and repair_type != self.sdk.RepairType.RT_RECV_REPAIR.value:
            raise RuntimeError(f"replica-only recovery does not support repair strategy {repair_type}")
        if sending and self._live_step != step:
            raise RuntimeError("donor does not hold valid device state at the requested recovery step")
        pairs = list(zip(repair_info["src"], repair_info["dst"]))
        sources = [source for source, destination in pairs if destination == self.rank]
        if not sending and len(sources) != 1:
            raise RuntimeError("dense recovery requires exactly one source per destination")
        # Reinit clears old links; the default world may still be lazy here.
        # Only the peers named by MindIO participate in this temporary group.
        group = dist.new_group(ranks=repair_info["rank_list"], backend="hccl", use_local_synchronization=True)
        try:
            if sending:
                for destination in repair_info["dst"]:
                    self.channel.send(destination, step, group)
            else:
                self.channel.receive(sources[0], step, group)
                self._live_step = step
                logger.info("Recovery state received: rank=%s source=%s step=%s", self.rank, sources[0], step)
        finally:
            dist.destroy_process_group(group)

    def _rollback(self, step, train_args, ctx) -> None:
        del train_args, ctx
        if self._live_step != step:
            raise RuntimeError("device state requires replica repair before rollback; no CPU snapshot fallback exists")
        # MindIO waits for all rebuild and repair replies before global rollback.
        # Muon's shape collectives must not run while a peer still clears HCCL keys.
        self.model_state.rebuild_runtime_state()
        snapshot = self.bank.get(self.rank, step)
        self.npu.set_device(self.local_rank)
        metadata = self.bank.metadata(self.rank, step)
        self.model_state.restore_rank_state(snapshot["buffers"], metadata["extra"])
        self.train_state.restore(metadata["state"])
        self._sync()
        self.step = step
        logger.info("Recovery rollback completed: rank=%s step=%s", self.rank, step)

    def make_iterator(self, factory: Callable[[], Iterator[Any]] = None) -> Iterator[Any]:
        """Recreate the consumed cursor before epoch initialization finalizes RNG.

        Args:
            factory: Native iterator constructor; defaults to iterating the protected loader.

        Returns:
            The iterator returned by the original constructor.
        """
        return self.train_state.make_iterator(factory=factory)

    @staticmethod
    def _unsupported_dump(*args: Any) -> None:
        """Reject SDK dump strategies rather than acknowledge nonexistent files."""
        del args
        raise RuntimeError("replica-only recovery does not support checkpoint dumping")
