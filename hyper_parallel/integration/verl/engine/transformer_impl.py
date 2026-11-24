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
"""VeRL policy execution over native HyperParallel model and loss interfaces."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import logprobs_from_logits as local_logprobs
from verl.workers.engine.base import BaseEngine, BaseEngineCtx
from verl.workers.engine.utils import detach_tree, postprocess_batch_func, prepare_micro_batches

from hyper_parallel import DTensor, HSDPModule, Shard
from hyper_parallel.integration.verl.engine.checkpoint import VeRLCheckpointAdapter
from hyper_parallel.integration.verl.engine.trainer import VerlTrainerAdapter
from hyper_parallel.components.losses._vocab_parallel_cross_entropy import (
    entropy_from_logits,
    vocab_parallel_cross_entropy_local,
)
from hyper_parallel.components.losses.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.trainer.config import normalize_distributed_setup_overrides
from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config
from hyper_parallel.integration.verl.engine.weight_export import get_per_tensor_param


if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)


def logprobs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, *, vocab_size: int, mesh: DeviceMesh,
) -> torch.Tensor:
    """Return log probabilities; TP1 uses VeRL's fused CE to bound vocabulary temporaries."""
    if mesh.size() == 1 and logits.device.type == "npu":
        # Preserve the original FP32 loss arithmetic without materializing
        # full-vocabulary exp/log-softmax buffers in Python.
        return local_logprobs(logits.float(), labels, inplace_backward=False)
    loss = vocab_parallel_cross_entropy_local(
        logits.float(),
        labels,
        vocab_size=vocab_size,
        mesh=mesh,
        reduction="none",
    ).reshape(labels.shape)
    # Native unreduced CE is vocabulary-local. Sum values without adding a
    # second reduction to its already-correct local backward derivatives.
    replicated = loss.detach().clone()
    torch.distributed.all_reduce(replicated, group=mesh.get_group())
    return -(loss - loss.detach() + replicated)


class HyperParallelEngineCtx(BaseEngineCtx):
    """Add HyperParallel mode and cleanup semantics to VeRL's load/offload context."""

    def __enter__(self) -> "HyperParallelEngineCtx":
        """Load the requested state and select the model's execution mode."""
        super().__enter__()
        try:
            self.engine.module.train(self.mode == "train")
        except BaseException as exc:
            super().__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up gradients or gathered weights before VeRL offloads state."""
        try:
            if self.mode == "train":
                if exc_type is not None or self.zero_grad_on_exit:  # pylint: disable=no-member
                    self.engine.optimizer_zero_grad()
            else:
                for module in self.engine.module.modules():
                    if isinstance(module, HSDPModule):
                        module.reshard()
        finally:
            super().__exit__(exc_type, exc_val, exc_tb)


class HyperParallelEngineWithLMHead(BaseEngine):
    """HP owns model parallelism; VeRL supplies policy losses and batch results."""

    def __init__(self, model_config: Any, engine_config: Any, optimizer_config: Any, checkpoint_config: Any) -> None:
        """Bind VeRL configuration to the native trainer topology."""
        # VeRL can register this engine while its configuration module is still importing VeRL.
        from hyper_parallel.integration.verl.config import (  # pylint: disable=import-outside-toplevel
            validate_and_resolve_hyper_parallel_config,
        )

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.hyper_parallel_config = validate_and_resolve_hyper_parallel_config(
            engine_config,
            model_config,
            optimizer_config=optimizer_config,
        )
        if self.hyper_parallel_config.fsdp_config.enable_offload:
            engine_config.param_offload = False
            engine_config.optimizer_offload = False
            logger.info(
                "Native FSDP CPU offload is enabled. VeRL param_offload and optimizer_offload are disabled."
            )

        # VeRL queries topology before initialize() allocates the training model.
        self.distributed_setup = create_distributed_setup_from_config(self.hyper_parallel_config)
        normalize_distributed_setup_overrides(self.distributed_setup, self.hyper_parallel_config)
        self.mesh_context = self.distributed_setup.mesh_context
        self.trainer = None

    @property
    def is_param_offload_enabled(self) -> bool:
        """Return whether VeRL manages parameter offload."""
        return self.engine_config.param_offload

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        """Return whether VeRL manages optimizer offload."""
        return self.engine_config.optimizer_offload

    def initialize(self) -> None:
        """Materialize native training state after worker device assignment."""
        self.trainer = VerlTrainerAdapter(
            self.hyper_parallel_config,
            distributed_setup=self.distributed_setup,
            tokenizer=self.model_config.tokenizer,
            forward_only=self.engine_config.forward_only,
        )
        self.module = self.trainer.model
        self.mesh_context = self.trainer.mesh_context
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self._model_offloaded = False
        self._optimizer_offloaded = False
        self.to(
            "cpu",
            model=self.is_param_offload_enabled,
            optimizer=self.is_optimizer_offload_enabled,
            grad=self.is_param_offload_enabled,
        )
        self.checkpoint_manager = VeRLCheckpointAdapter(
            self.trainer,
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
        )

    def get_data_parallel_rank(self) -> int:
        """Return the native data parallel rank."""
        return self.mesh_context.dp_rank

    def get_data_parallel_size(self) -> int:
        """Return the native data parallel size."""
        return self.mesh_context.dp_size

    def get_data_parallel_group(self) -> dist.ProcessGroup:
        """Return the native data parallel process group."""
        return self.mesh_context.device_mesh["dp"].get_group()

    def is_mp_src_rank_with_outputs(self) -> bool:
        """Identify the model parallel rank owning policy outputs."""
        return self.mesh_context.tp_rank == 0 and self.mesh_context.cp_rank == 0

    def forward_backward_batch(
        self, data: Any, loss_function: Callable | None = None, forward_only: bool = False,
    ) -> dict[str, Any]:
        """Execute VeRL micro-batches using HyperParallel's model and gradient contexts."""
        if not forward_only:
            if self.engine_config.forward_only:
                raise ValueError("A reference engine cannot train")
            if loss_function is None:
                raise ValueError("Training requires the VeRL loss function")
        batches, indices = self._process_inputs(data)
        return_model_output = tu.get_non_tensor_data(data, key="return_model_output", default=forward_only)
        results = []
        for micro_step, micro_batch in enumerate(batches):
            if not forward_only:
                self.trainer.model_reshard(micro_step, len(batches))
            grad_context = torch.no_grad() if forward_only else nullcontext()
            with grad_context, aux_loss_scale_context():
                native_input = micro_batch.to(self.trainer.device)
                outputs, labels = self.trainer.forward_step(
                    native_input["input_ids"],
                    micro_step=micro_step,
                    num_micro_steps=len(batches),
                    forward_only=forward_only,
                )
                model_outputs = self._process_outputs(
                    outputs,
                    labels,
                    native_input,
                    native_input["input_ids"].offsets()[-1].item(),
                )
                if loss_function is None:
                    loss = None
                    metrics = {}
                else:
                    loss, metrics = loss_function(
                        model_output=model_outputs, data=native_input, dp_group=self.get_data_parallel_group(),
                    )
                result = {"metrics": metrics}
                if loss is not None:
                    result["loss"] = loss.detach().item()
                if return_model_output:
                    result["model_output"] = detach_tree(model_outputs)
                del outputs, model_outputs
                if loss is not None:
                    bind_aux_loss_scale(loss)
                if not forward_only:
                    if self.hyper_parallel_config.training.empty_cache_before_backward:
                        get_torch_device().empty_cache()
                    with self.trainer.model_bwd_context:
                        loss.backward()
            results.append(result)
        outputs = postprocess_batch_func(results, indices, data)
        return outputs

    @staticmethod
    def _validate_inputs(data: Any) -> None:
        """Reject unsupported options and input layouts before batch preparation."""
        unsupported = ("calculate_sum_pi_squared", "distillation_use_topk", "distillation_only")
        enabled = [name for name in unsupported if tu.get_non_tensor_data(data, key=name, default=False)]
        if enabled:
            raise NotImplementedError(f"Unsupported VeRL policy-output options: {', '.join(enabled)}")
        if tu.get_non_tensor_data(data, key="pad_mode", default=DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING:
            raise NotImplementedError("The HyperParallel policy adapter requires NO_PADDING batches")
        if data["position_ids"].dim() != 2 or any(
            tu.unwrap_non_tensor_data(item) for item in data.get("multi_modal_inputs", [])
        ):
            raise NotImplementedError("The HyperParallel policy adapter supports text batches only")

    def _process_inputs(self, data: Any) -> tuple[list, Any]:
        """Prepare validated inputs using VeRL's native micro-batch planner."""
        self._validate_inputs(data)
        dp_group = self.get_data_parallel_group()
        tu.assign_non_tensor(
            data,
            sp_size=self.trainer.mesh_context.cp_size,
            dp_size=self.trainer.mesh_context.dp_size,
        )
        batch_tokens = data["loss_mask"].sum().to(self.trainer.device)
        dist.all_reduce(batch_tokens, group=dp_group)
        tu.assign_non_tensor(data, batch_num_tokens=batch_tokens.item())
        batches, indices = prepare_micro_batches(
            data,
            dp_group=dp_group,
            same_micro_num_in_dp=True,
        )
        return batches, indices

    def _process_outputs(
        self, output: Any, labels: torch.Tensor, data: Any, length: int,
    ) -> dict[str, torch.Tensor]:
        """Convert vocabulary/context-parallel logits to VeRL policy outputs."""
        logits = output.logits.to_local() if isinstance(output.logits, DTensor) else output.logits
        temperature = data["temperature"]
        is_unit_temperature = not isinstance(temperature, torch.Tensor) and float(temperature) == 1.0
        if not is_unit_temperature:
            temperature = torch.as_tensor(temperature, device=logits.device, dtype=torch.float32).reshape(-1)
            if temperature.numel() > 1:
                temperature = temperature.repeat_interleave(data["input_ids"].offsets().diff())
                local_length = logits.shape[-2]
                temperature = torch.nn.functional.pad(
                    temperature, (0, local_length * self.mesh_context.cp_size - length), value=1,
                )
                start = self.mesh_context.cp_rank * local_length
                temperature = temperature[start : start + local_length]
            logits = logits / temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype)
        tp_mesh = self.mesh_context.device_mesh["tp"]
        outputs = {"log_probs": logprobs_from_logits(
            logits, labels, vocab_size=self.module.config.vocab_size, mesh=tp_mesh,
        )}
        if tu.get_non_tensor_data(data, key="calculate_entropy", default=False):
            outputs["entropy"] = entropy_from_logits(logits, mesh=tp_mesh)
        for name, values in outputs.items():
            if self.mesh_context.cp_size > 1:
                values = DTensor.from_local(values, self.mesh_context.cp_mesh, (Shard(-1),)).full_tensor()
            outputs[name] = torch.nested.nested_tensor_from_jagged(values[0, :length], data["input_ids"].offsets())
        return outputs

    def optimizer_zero_grad(self) -> None:
        """Delegate gradient cleanup to the trainer."""
        self.trainer.optimizer_zero_grad()

    def optimizer_step(self) -> float:
        """Delegate gradient clipping and updates at VeRL's native optimizer boundary."""
        return self.trainer.optimizer_step()

    def lr_scheduler_step(self) -> float:
        """Advance HyperParallel schedulers at VeRL's scheduler boundary."""
        return self.trainer.lr_scheduler_step()

    def train_mode(self, **kwargs: Any) -> HyperParallelEngineCtx:
        """Create the HyperParallel training context over VeRL residency transitions."""
        return HyperParallelEngineCtx(
            self,
            mode="train",
            **kwargs,
        )

    def eval_mode(self, **kwargs: Any) -> HyperParallelEngineCtx:
        """Create the HyperParallel evaluation context with exit-time resharding."""
        return HyperParallelEngineCtx(
            self,
            mode="eval",
            **kwargs,
        )

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True) -> None:
        """Apply VeRL residency transitions to native live objects."""
        super().to(
            device=device,
            model=model,
            optimizer=optimizer,
            grad=grad,
        )
        if device not in ("cpu", get_device_name()):
            raise ValueError(f"Unsupported target device: {device}")
        offloaded = device == "cpu"
        if model and self._model_offloaded != offloaded:
            with torch.no_grad():
                if offloaded:
                    # FSDP repadding reads CPU storage inside Module.to(); waiting afterward is too late.
                    # Match the blocking D2H fix in https://github.com/verl-project/verl/pull/8187.
                    self.module.to("cpu", non_blocking=False)
                    get_torch_device().empty_cache()
                else:
                    self.module.to(get_device_id(), non_blocking=True)
            self._model_offloaded = offloaded
        if optimizer and self.optimizer is not None and self._optimizer_offloaded != offloaded:
            for optimizer_part in self.optimizer.chained_optimizers:
                if offloaded:
                    offload_fsdp_optimizer(optimizer_part)
                else:
                    load_fsdp_optimizer(optimizer_part, device)
            self._optimizer_offloaded = offloaded

    def save_checkpoint(
        self, local_path: str | None, hdfs_path: str | None = None, global_step: int = 0,
        max_ckpt_to_keep: int | None = None, **kwargs: Any,
    ) -> None:
        """Delegate framework checkpoint saves to the policy adapter."""
        if kwargs:
            raise TypeError(f"Unsupported checkpoint save arguments: {sorted(kwargs)}")
        return self.checkpoint_manager.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    def load_checkpoint(
        self, local_path: str | None, hdfs_path: str | None = None,
        del_local_after_load: bool = False, **kwargs: Any,
    ) -> None:
        """Delegate framework checkpoint restores to the policy adapter."""
        if kwargs:
            raise TypeError(f"Unsupported checkpoint load arguments: {sorted(kwargs)}")
        return self.checkpoint_manager.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    def get_per_tensor_param(self, **kwargs: Any) -> tuple[Iterator[tuple[str, torch.Tensor]], None]:
        """Export native weights one tensor at a time in rollout format."""
        del kwargs

        return get_per_tensor_param(self.module), None
