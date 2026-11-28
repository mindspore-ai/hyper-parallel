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
"""Two-step actual sampling and publication with explicit nonzero fixture rewards."""

import argparse
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from rl.config.loader import load_config
from rl.trainer.build import build_trainer
from examples.consistency.scripts import run_consistency


def _samples(model):
    result = {}
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
        values = local.detach().reshape(-1)
        if values.numel():
            stride = max(1, values.numel() // 1024)
            result[name] = values[::stride][:1024].float().cpu().clone()
    return result


def main() -> None:
    """Observe real optimizer changes; probabilities and weight transport are unchanged."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, [])
    config['agentic'].update(module_path=str(Path(__file__).with_name('_context_parallel_reward.py')),
                            environment='cp_update_fixture')
    config['train']['optimizer']['lr'] = .001
    config['train']['learning_gate'].update(enabled=True, min_gradient_norm=1e-8, require_mixed_rewards=True)
    observations = []

    def observed_build(values):
        trainer = build_trainer(values)
        update = trainer.actor.update

        def checked_update(experience):
            before = _samples(trainer.actor.actor_model)
            metrics = update(experience)
            after = _samples(trainer.actor.actor_model)
            changed = any(not torch.equal(value, after[name]) for name, value in before.items())
            valid = torch.tensor(int(changed and metrics.gradient_norm > 0), device=trainer.device)
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            if not bool(valid):
                raise AssertionError('Every rank must observe nonzero gradients and changed local weights')
            observations.append({'step': trainer.state.global_step + 1,
                                 'gradient_norm': metrics.gradient_norm, 'weights_changed': changed})
            return metrics

        trainer.actor.update = checked_update
        return trainer

    with patch.object(run_consistency, 'build_trainer', side_effect=observed_build):
        run_consistency.run(config, args.output / 'consistency.json')
    if len(observations) != 2:
        raise AssertionError('Both optimizer steps must have observed nonzero updates')
    if int(os.environ.get('RANK', '0')) == 0:
        (args.output / 'updates.json').write_text(json.dumps({
            'scope': 'Synthetic alternating rewards; actual sampling, optimization and weight publication',
            'updates': observations}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
