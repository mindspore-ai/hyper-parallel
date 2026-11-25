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
"""Native HyperParallel execution components used by the VeRL engine."""

import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.fully_shard.api import HSDPModule

from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer as CHAINED_OPTIMIZER
from hyper_parallel.trainer.runtime.model_integration import (
    DisabledModelIntegrationSession as DISABLED_MODEL_INTEGRATION_SESSION,
)
from hyper_parallel.core.optimizer.lr_scheduler import (
    LRSchedulersContainer as LRSCHEDULERS_CONTAINER,
)
from hyper_parallel.trainer.config.parser import load_training_config
from hyper_parallel.distributed.mesh import MeshContext as MESH_CONTEXT
from hyper_parallel.data.parallel import create_dataloader_parallel_context
from tests.ut.integration.verl._contracts import ADAPTERS, MODULES, list_of_dict_to_tensordict

transformer_impl = ADAPTERS.engine
HYPER_PARALLEL_ENGINE_WITH_LMHEAD = ADAPTERS.engine.HyperParallelEngineWithLMHead
HYPER_PARALLEL_ENGINE_CONFIG = ADAPTERS.config.HyperParallelEngineConfig
VERL_TRAINER_ADAPTER = ADAPTERS.trainer.VerlTrainerAdapter
tu = MODULES["verl.utils.tensordict_utils"]


class WrappedModule(HSDPModule, torch.nn.Module):
    """Small HSDP mixin instance without distributed initialization."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        HSDPModule.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(4))


class TestParameterConversion(unittest.TestCase):
    """Keep communication storage consistent across engine parameter conversions."""

    def test_dtype_conversion_refreshes_communication_storage(self) -> None:
        """The alias follows the converted parameter instead of keeping old storage."""
        model = WrappedModule()
        parameter = SimpleNamespace(
            sharded_param=model.weight,
            sharded_size=(4,),
            padded_sharded_param_size=(4,),
            alias=model.weight.data,
        )

        def reset():
            parameter.alias = model.weight.data

        parameter.reset_sharded_param = reset
        model.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(shard=lambda: None, hsdp_params=[parameter], param_group=None),
        )
        result = model.to(dtype=torch.float64)
        self.assertIs(result, model, f"result={result}, expected={model}")
        self.assertEqual(parameter.alias.dtype, torch.float64, f"dtype={parameter.alias.dtype}, expected=torch.float64")
        self.assertEqual(
            parameter.alias.data_ptr(), model.weight.data_ptr(),
            f"alias pointer={parameter.alias.data_ptr()}, expected={model.weight.data_ptr()}",
        )


class TestNativeBatchPreparation(unittest.TestCase):
    """Check the native input contract used by the adapter."""

    def test_native_text_batch_constructor_and_sequence_boundaries(self) -> None:
        """Build real CPU text batches without a pipeline or distributed group."""
        runtime = VERL_TRAINER_ADAPTER.__new__(VERL_TRAINER_ADAPTER)
        runtime.mesh_context = MESH_CONTEXT()
        runtime.device = torch.device("cpu")
        recipe = Path(__file__).with_name("native_cpu.yaml")
        runtime.config = load_training_config(recipe)
        runtime.model_integration = Mock()
        runtime.model_assets = []
        runtime.forward_only = True
        runtime.tokenizer = None
        runtime.dataloader_context = create_dataloader_parallel_context(runtime.mesh, tp_source_rank=None)
        runtime._build_collate_fn()
        runtime._build_get_batch()
        batch = runtime.collate_fn([
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([5, 6, -100])},
            {"input_ids": torch.tensor([7, 8]), "labels": torch.tensor([8, -100])},
        ])
        model, loss = runtime.get_batch(iter([batch]))
        length = int(model["input_ids"].numel())
        self.assertFalse(runtime.get_batch.pp_shared_data,
                         f"pipeline shared data={runtime.get_batch.pp_shared_data}, expected=False")
        self.assertEqual(length, 5, f"packed length={length}, expected=5")
        self.assertEqual(model["input_ids"].tolist(), [[4, 5, 6, 7, 8]],
                         f"packed inputs={model['input_ids'].tolist()}, expected=[[4, 5, 6, 7, 8]]")
        self.assertEqual(loss["labels"].tolist(), [[5, 6, -100, 8, -100]],
                         f"shifted labels={loss['labels'].tolist()}, expected=[[5, 6, -100, 8, -100]]")


class TinyLanguageModel(torch.nn.Module):
    """CPU model exercising the adapter's real forward and backward interfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        """Accept native model inputs without owning the external loss."""
        if "labels" in kwargs or "shift_labels" in kwargs:
            raise ValueError("Loss labels reached the model forward")
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class TestEngineNativeExecution(unittest.TestCase):
    """Exercise real HyperParallel input, gradient and optimizer helpers through the Engine."""

    @staticmethod
    def _engine():
        """Build a lightweight engine backed by native trainer components."""
        trainer = VERL_TRAINER_ADAPTER.__new__(VERL_TRAINER_ADAPTER)
        trainer.config = HYPER_PARALLEL_ENGINE_CONFIG(recipe_path=str(
            Path(__file__).with_name("native_cpu.yaml"),
        )).trainer_config
        trainer.config.fsdp_config.dp_shard_size = 2
        trainer.config.fsdp_config.requires_grad_sync = False
        trainer.device = torch.device("cpu")
        trainer.mesh_context = MESH_CONTEXT()
        trainer.tokenizer = None
        trainer.dataloader_context = create_dataloader_parallel_context(trainer.mesh_context, tp_source_rank=None)
        trainer._build_collate_fn()
        trainer._build_get_batch()
        trainer.model = TinyLanguageModel()
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.model_integration = DISABLED_MODEL_INTEGRATION_SESSION()
        trainer.hsdp_model_parts = [Mock(spec=HSDPModule)]
        trainer.fsdp_runtime_diagnostics = None
        trainer.optimizer = CHAINED_OPTIMIZER(
            trainer.model, {"sgd": torch.optim.SGD(trainer.model.parameters(), lr=0.1, foreach=False)},
        )
        trainer.lr_scheduler = LRSCHEDULERS_CONTAINER(
            trainer.optimizer, partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
        )
        engine = object.__new__(HYPER_PARALLEL_ENGINE_WITH_LMHEAD)
        engine.trainer = trainer
        engine.module = trainer.model
        engine.hyper_parallel_config = trainer.config
        engine.mesh_context = trainer.mesh_context
        engine.engine_config = SimpleNamespace(forward_only=False, param_offload=False, optimizer_offload=False)
        engine._model_offloaded = False
        engine._optimizer_offloaded = False
        engine.mode = None
        engine.get_data_parallel_group = Mock(return_value=None)
        # Policy-output arithmetic has separate numerical tests; keep the real HyperParallel execution path here.
        engine._process_outputs = lambda output, *_: {"scores": output.logits}
        return engine

    def test_infer_then_update_uses_native_hyper_execution(self) -> None:
        """Inference is read-only; two native micro-batches produce one clipped update."""
        engine = self._engine()
        trainer = engine.trainer
        rows = [
            {"input_ids": torch.tensor([1, 2, 3]), "position_ids": torch.arange(3),
             "loss_mask": torch.ones(3), "temperature": torch.tensor(1.0)},
            {"input_ids": torch.tensor([4, 5]), "position_ids": torch.arange(2),
             "loss_mask": torch.ones(2), "temperature": torch.tensor(1.0)},
        ]
        data = list_of_dict_to_tensordict(rows)
        tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1, return_model_output=True)
        trainer.config.training.empty_cache_before_backward = True
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        with (
            patch.object(transformer_impl.dist, "all_reduce"),
            patch.object(transformer_impl, "get_torch_device") as torch_device,
            patch.object(transformer_impl, "prepare_micro_batches", return_value=(
                [list_of_dict_to_tensordict([row]) for row in rows], None,
            )) as plan,
            patch.object(transformer_impl, "postprocess_batch_func", side_effect=lambda results, *_: results),
            patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as update,
        ):
            with engine.eval_mode():
                engine.forward_backward_batch(data, forward_only=True)
            update.assert_not_called()
            self.assertTrue(
                all(parameter.grad is None for parameter in trainer.model.parameters()),
                f"inference gradients={[parameter.grad for parameter in trainer.model.parameters()]}, expected=None",
            )
            for actual, expected in zip(trainer.model.parameters(), before):
                torch.testing.assert_close(actual, expected)
            with engine.train_mode(zero_grad_on_exit=False):
                engine.optimizer_zero_grad()
                results = engine.forward_backward_batch(
                    data, lambda model_output, **_: (model_output["scores"].square().mean() / 2, {}),
                )
                grad_norm = engine.optimizer_step()
            update.assert_called_once_with()
            self.assertEqual(plan.call_count, 2)
            self.assertEqual(torch_device.return_value.empty_cache.call_count, 2)
            self.assertFalse(results[0]["model_output"]["scores"].requires_grad)
            old_epoch = trainer.lr_scheduler.schedulers[0].last_epoch
            self.assertEqual(
                trainer.optimizer.param_groups[0]["lr"], 0.1,
                f"lr={trainer.optimizer.param_groups[0]['lr']}, expected=0.1 before scheduler",
            )
            actual_lr = engine.lr_scheduler_step()
            self.assertEqual(actual_lr, 0.05, f"lr={actual_lr}, expected=.05")
            self.assertEqual(
                trainer.lr_scheduler.schedulers[0].last_epoch, old_epoch + 1,
                f"scheduler epoch={trainer.lr_scheduler.schedulers[0].last_epoch}, expected={old_epoch + 1}",
            )
        self.assertGreater(
            grad_norm, 0, f"gradient norm={grad_norm}, expected>0",
        )
        self.assertTrue(
            any(not torch.equal(actual, expected) for actual, expected in zip(trainer.model.parameters(), before)),
            f"parameters unchanged after native optimizer update, expected a nonzero update: {results}",
        )
        sync_calls = trainer.hsdp_model_parts[0].set_requires_gradient_sync.call_args_list
        self.assertEqual(
            [call.args[0] for call in sync_calls], [False, True], f"sync={sync_calls}, expected=False/True",
        )

    def test_offload_and_reload_model_and_optimizer(self) -> None:
        """Move live state once per transition, with blocking device-to-host transfer."""
        engine = self._engine()
        engine.module = Mock()
        optimizer_parts = [Mock()]
        engine.optimizer = SimpleNamespace(chained_optimizers=optimizer_parts)
        with (
            patch.object(transformer_impl, "get_device_name", return_value="npu"),
            patch.object(transformer_impl, "get_device_id", return_value=0),
            patch.object(transformer_impl, "get_torch_device") as device,
            patch.object(transformer_impl, "offload_fsdp_optimizer") as offload,
            patch.object(transformer_impl, "load_fsdp_optimizer") as reload,
        ):
            engine.to("cpu")
            engine.to("cpu")
            engine.module.to.assert_called_once_with("cpu", non_blocking=False)
            device.return_value.empty_cache.assert_called_once()
            offload.assert_called_once_with(optimizer_parts[0])
            engine.module.to.reset_mock()

            engine.to("npu")
            engine.module.to.assert_called_once_with(0, non_blocking=True)
            reload.assert_called_once_with(optimizer_parts[0], "npu")

            with self.assertRaisesRegex(ValueError, "Unsupported target device"):
                engine.to("unavailable")

    def test_checkpoint_methods_forward_supported_arguments(self) -> None:
        """Forward the VeRL checkpoint contract and reject extra unsupported options."""
        engine = self._engine()
        engine.checkpoint_manager = Mock()
        engine.save_checkpoint("local", global_step=3)
        engine.load_checkpoint("local")
        engine.checkpoint_manager.save_checkpoint.assert_called_once_with("local", None, 3, None)
        engine.checkpoint_manager.load_checkpoint.assert_called_once_with("local", None, False)
        for method in (engine.save_checkpoint, engine.load_checkpoint):
            with self.subTest(method=method.__name__), self.assertRaisesRegex(TypeError, "unsupported_option"):
                method("local", unsupported_option=True)

    def test_rejects_invalid_training_entrypoints(self) -> None:
        """Reject a missing loss or an update against a forward-only engine."""
        engine = self._engine()
        with self.assertRaisesRegex(ValueError, "requires the VeRL loss"):
            engine.forward_backward_batch(None)
        engine.engine_config.forward_only = True
        with self.assertRaisesRegex(ValueError, "reference engine cannot train"):
            engine.forward_backward_batch(None)

    def test_initialize_uses_assigned_topology_and_bound_configuration(self) -> None:
        """Wire the trainer and checkpoint manager after the worker selects a device."""
        engine_config = HYPER_PARALLEL_ENGINE_CONFIG(
            recipe_path=str(Path(__file__).with_name("native_cpu.yaml")),
            trainer_config={"fsdp_config": {"enable_offload": True}},
            param_offload=True, optimizer_offload=True,
        )
        model_config = Mock(path="model", tokenizer=None, override_config={})
        dp_mesh = Mock()
        mesh = Mock(dp_rank=1, dp_size=4, device_mesh={"dp": dp_mesh})
        setup = Mock(mesh_context=mesh)
        with (
            patch.object(transformer_impl, "create_distributed_setup_from_config", return_value=setup),
            patch.object(transformer_impl, "normalize_distributed_setup_overrides") as normalize,
            patch.object(transformer_impl, "VerlTrainerAdapter") as build_trainer,
            patch.object(transformer_impl, "VeRLCheckpointAdapter") as build_checkpoint,
        ):
            with self.assertLogs(transformer_impl.__name__, level="INFO") as captured:
                engine = HYPER_PARALLEL_ENGINE_WITH_LMHEAD(model_config, engine_config, None, None)
            self.assertIn("VeRL param_offload and optimizer_offload are disabled", captured.output[0])
            normalize.assert_called_once_with(setup, engine.hyper_parallel_config)
            self.assertFalse(engine.is_param_offload_enabled)
            self.assertFalse(engine.is_optimizer_offload_enabled)
            self.assertEqual((engine.get_data_parallel_rank(), engine.get_data_parallel_size()), (1, 4))
            self.assertIs(engine.get_data_parallel_group(), dp_mesh.get_group.return_value)
            for tp_rank, cp_rank, owns_output in ((0, 0, True), (1, 0, False), (0, 1, False), (1, 1, False)):
                with self.subTest(tp_rank=tp_rank, cp_rank=cp_rank):
                    mesh.tp_rank, mesh.cp_rank = tp_rank, cp_rank
                    self.assertEqual(engine.is_mp_src_rank_with_outputs(), owns_output)
            engine.initialize()
            build_trainer.assert_called_once_with(
                engine.hyper_parallel_config, distributed_setup=setup, tokenizer=None, forward_only=False,
            )
            self.assertIs(engine.module, build_trainer.return_value.model)
            self.assertIs(engine.checkpoint_manager, build_checkpoint.return_value)
            with patch.object(transformer_impl, "get_per_tensor_param", return_value=iter(())) as export:
                tensors, adapter = engine.get_per_tensor_param()
                self.assertEqual(list(tensors), [])
                self.assertIsNone(adapter)
                export.assert_called_once_with(engine.module)

    def test_rejects_unsupported_input_metadata_and_layout(self) -> None:
        """Reject features that cannot be evaluated with the packed text path."""
        engine = self._engine()
        for key, value in (("distillation_only", True), ("pad_mode", "right_padding")):
            data = list_of_dict_to_tensordict([{"position_ids": torch.arange(2)}])
            tu.assign_non_tensor(data, **{key: value})
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                engine._process_inputs(data)
        data = list_of_dict_to_tensordict([{"position_ids": torch.zeros((2, 2), dtype=torch.long)}])
        with self.assertRaisesRegex(NotImplementedError, "text batches only"):
            engine._process_inputs(data)

    def test_process_outputs_scales_per_sample_temperature_and_calculates_entropy(self) -> None:
        """Compare packed per-sample log-probability and entropy to native PyTorch."""
        engine = self._engine()
        del engine._process_outputs
        engine.module.config = SimpleNamespace(vocab_size=3)
        tp_mesh = Mock(ndim=1)
        tp_mesh.size.return_value = 1
        tp_mesh.get_local_rank.return_value = 0
        engine.mesh_context = SimpleNamespace(device_mesh={"tp": tp_mesh}, cp_size=1, cp_rank=0)
        data = list_of_dict_to_tensordict([
            {"input_ids": torch.tensor([0, 1]), "temperature": torch.tensor(1.0)},
            {"input_ids": torch.tensor([2]), "temperature": torch.tensor(2.0)},
        ])
        tu.assign_non_tensor(data, calculate_entropy=True)
        logits = torch.tensor([[[1., 2., 4.], [3., 1., 2.], [2., 4., 1.]]])
        labels = torch.tensor([[0, 1, 2]])
        with (
            patch.object(transformer_impl.dist, "all_reduce"),
            patch(
                "hyper_parallel.components.losses._vocab_parallel_cross_entropy._differentiable_all_reduce",
                side_effect=lambda values, **_: values,
            ),
        ):
            outputs = engine._process_outputs(SimpleNamespace(logits=logits), labels, data, 3)
        expected = (logits / torch.tensor([1., 1., 2.]).reshape(1, -1, 1)).log_softmax(dim=-1)
        torch.testing.assert_close(outputs["log_probs"].values(), expected.gather(-1, labels.unsqueeze(-1)).view(-1))
        torch.testing.assert_close(outputs["entropy"].values(), -(expected.exp() * expected).sum(-1).view(-1))

    def test_single_npu_shard_uses_fused_logprobs(self) -> None:
        """Select the fused path on NPU and retain the FP32 input arithmetic."""
        values = torch.tensor([[1, 2]], dtype=torch.bfloat16)
        logits = Mock(device=SimpleNamespace(type="npu"), float=Mock(return_value=values.float()))
        labels = torch.tensor([1])
        mesh = Mock(size=Mock(return_value=1))
        with patch.object(transformer_impl, "local_logprobs") as kernel:
            actual = transformer_impl.logprobs_from_logits(logits, labels, vocab_size=2, mesh=mesh)
        self.assertIs(actual, kernel.return_value)
        kernel.assert_called_once_with(logits.float.return_value, labels, inplace_backward=False)

    def test_model_context_cleans_up_on_error_and_eval_exit(self) -> None:
        """Clear mode on entry errors and reshard gathered weights after evaluation."""
        engine = self._engine()
        engine.module = WrappedModule()
        with patch.object(engine.module, "train", side_effect=RuntimeError("cannot set mode")):
            with self.assertRaisesRegex(RuntimeError, "cannot set mode"), engine.train_mode(disable_auto_offload=True):
                self.fail("mode entry should have failed")
        self.assertIsNone(engine.mode)
        with patch.object(engine.module, "reshard") as reshard:
            with engine.eval_mode(disable_auto_offload=True):
                self.assertFalse(engine.module.training)
            reshard.assert_called_once_with()
        with patch.object(engine, "optimizer_zero_grad") as clear:
            with engine.train_mode(disable_auto_offload=True):
                self.assertTrue(engine.module.training)
            clear.assert_called_once_with()

    def test_context_parallel_outputs_reassemble_local_shards(self) -> None:
        """Gather CP log probabilities before restoring each sample's token boundaries."""
        engine = self._engine()
        del engine._process_outputs
        engine.module.config = SimpleNamespace(vocab_size=3)
        tp_mesh = Mock()
        cp_mesh = Mock()
        engine.mesh_context = SimpleNamespace(device_mesh={"tp": tp_mesh}, cp_size=2, cp_rank=1, cp_mesh=cp_mesh)
        data = list_of_dict_to_tensordict([
            {"input_ids": torch.tensor([0, 1]), "temperature": torch.tensor(1.0)},
            {"input_ids": torch.tensor([2]), "temperature": torch.tensor(2.0)},
        ])
        local_logits = torch.tensor([[[2., 4., 6.], [1., 2., 3.]]])
        labels = torch.tensor([[2, 0]])
        local = torch.tensor([[-0.3, 0.]])
        complete = torch.tensor([[-0.1, -0.2, -0.3, 0.]])

        class ShardedLogits:
            """Represent a local shard without a process group."""

            from_local = Mock()

            @staticmethod
            def to_local() -> torch.Tensor:
                """Expose the CP rank's logits."""
                return local_logits

        ShardedLogits.from_local.return_value.full_tensor.return_value = complete
        with (
            patch.object(transformer_impl, "DTensor", ShardedLogits),
            patch.object(transformer_impl, "logprobs_from_logits", return_value=local) as logprobs,
        ):
            output = engine._process_outputs(SimpleNamespace(logits=ShardedLogits()), labels, data, 3)
        expected_logits = local_logits / torch.tensor([2., 1.]).reshape(1, -1, 1)
        torch.testing.assert_close(logprobs.call_args.args[0], expected_logits)
        self.assertIs(ShardedLogits.from_local.call_args.args[1], cp_mesh)
        torch.testing.assert_close(output["log_probs"].values(), complete[0, :3])
