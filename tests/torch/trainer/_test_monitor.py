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
"""Real NPU Muon and checkpoint monitor acceptance; run under torchrun."""
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- register the NPU device backend
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.components.monitor import statistics_scope
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.config.monitor import MonitorConfig, SamplingConfig
from hyper_parallel.trainer.runtime.monitor.session import MonitorSession
from hyper_parallel.trainer.state import TrainerState


def _run_case(root: Path, asynchronous: bool) -> None:
    """Verify real NPU summaries, checkpoint timing, export, and exact restoration."""
    rank = dist.get_rank()
    torch.manual_seed(123)
    case = root / ('async' if asynchronous else 'sync')
    config = MonitorConfig(sinks=['jsonl'], output_dir=str(case / 'metrics'))
    config.local_loss = SamplingConfig(True, 1)
    config.local_grad_norm = replace(config.local_grad_norm, enabled=True, sampler_steps=1)
    config.optimizer.ns_norm = SamplingConfig(True, 1)
    model_config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = Qwen3ForCausalLM(model_config).npu()
    optimizer = Muon([p for p in model.parameters() if p.ndim == 2], ns_steps=5, lr=0.001)
    checkpoint = CheckpointingConfig(checkpoint_dir=str(case / 'checkpoint'), save_steps=1,
                                    save_epochs=0, is_async=asynchronous, save_optimizer=False,
                                    save_hf_weights=True, restore_optimizer=False)
    trainer = SimpleNamespace(
        config=SimpleNamespace(monitor=config, training=SimpleNamespace(log_tflops=False),
                               checkpoint=checkpoint, model_init_dtype='float32'),
        model=model, model_config=model_config, optimizer=optimizer, global_rank=rank,
        world_size=dist.get_world_size(), mesh=None, state=TrainerState(),
        lr_scheduler=None, train_dataloader=None, train_steps=3,
    )
    monitor = MonitorSession(trainer)
    trainer.monitor = monitor
    monitor.initialize()
    monitor.bind_model()
    callback = CheckpointerCallback(trainer)
    callback.on_train_begin(trainer.state)
    monitor.start_output()
    for step in (1, 2):
        monitor.begin_step(step)
        optimizer.zero_grad()
        ids = torch.tensor([[1, 2, 3, 4]], device='npu')
        loss = model(input_ids=ids, labels=ids).loss
        monitor.record_loss(loss, {'foundation_tokens': torch.tensor(3, device='npu')})
        loss.backward()
        monitor.record_gradients()
        with statistics_scope(monitor):
            optimizer.step()
        metrics = monitor.end_step()
        trainer.state.global_step = step
        monitor.log_step(trainer.state, metrics)
        callback.on_step_end(trainer.state)
    callback.on_train_end(trainer.state)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    run_id = monitor.run_id
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    trainer.config.checkpoint = replace(checkpoint, save_ckpt=False, save_hf_weights=False,
                                        restore_from=str(case / 'checkpoint/global_step_2'))
    trainer.state.global_step = 0
    restored = CheckpointerCallback(trainer)
    restored.on_train_begin(trainer.state)
    assert trainer.state.global_step == 2
    assert monitor.run_id == run_id
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    monitor.close()
    records = [json.loads(line) for line in (monitor.writer.directory / f'rank_{rank}.jsonl').read_text().splitlines()]
    ns = [r for r in records if r['event'] == 'ns_norm']
    assert ns and {r['step'] for r in ns} == {1, 2}
    assert all(owner['parameter'] != '<unmapped>' for r in ns for owner in r['matrix_owners'])
    assert all(math.isfinite(value) for record in ns for value in record['values'])
    assert {record['step'] for record in records if record['event'] == 'local_grad_norm'} == {1, 2}
    if asynchronous:
        assert {r['step'] for r in records if r.get('phase') == 'checkpoint_persist'} == {1, 2}
    phases = {r['phase'] for r in records if r['event'] == 'timing'}
    required = {'weights_save', 'checkpoint_restore', 'checkpoint_wait'}
    required |= {'checkpoint_submit', 'checkpoint_persist'} if asynchronous else {'checkpoint_save'}
    assert required <= phases, (required, phases)
    assert all(r['outcome'] == 'ok' for r in records if r['event'] == 'timing')
    assert monitor.transfer.dropped == 0
    if rank == 0:
        assert list((case / 'checkpoint/global_step_2').rglob('*.safetensors'))
        print(json.dumps({'case': case.name, 'rank': rank, 'ns_records': len(ns),
                          'phases': sorted(phases), 'restored_step': trainer.state.global_step}), flush=True)
    dist.barrier()


def test_monitor_npu_components() -> None:
    """Exercise NPU-only recorder and persistence boundaries on eight ranks.

    Feature: Trainer monitor device integration.
    Description: Execute real Muon updates and sync/async DCP plus HF export.
    Expectation: Complete per-rank summaries and timing, no lost device copies,
        and bitwise identical weights after checkpoint restoration.
    """
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    root = Path(os.environ['MONITOR_ACCEPTANCE_DIR'])
    try:
        _run_case(root, False)
        _run_case(root, True)
    finally:
        dist.destroy_process_group()
