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
"""Full logical parameter/gradient comparisons around the production Actor update."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, ContextManager, Iterable
from unittest.mock import patch

import torch
import torch.distributed as dist

from rl.workers.engine import optimizer as optimizer_engine


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _errors(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    norm = float(reference.norm())
    maximum = float(reference.abs().max())
    error = float(delta.norm())
    max_abs = float(delta.abs().max())
    return {
        'reference_norm': norm, 'actual_norm': float(actual.norm()),
        'relative_l2': error / max(norm, 1e-12), 'max_abs': max_abs,
        'difference_norm': error,
        'reference_max_abs': maximum, 'exact': torch.equal(actual, reference),
        'finite': bool(actual.isfinite().all() and reference.isfinite().all()),
        'elementwise_outliers': int((delta.abs() > 1e-5 + .05 * reference.abs()).sum()),
        # BF16 reductions can differ, but missing/duplicated/scaled gradients must fail.
        'passed': bool(actual.isfinite().all() and error <= .02 * norm + 1e-7
                       and max_abs <= .05 * maximum + 1e-7),
    }


def _combine_errors(parts: list[dict]) -> dict:
    """Combine disjoint slices without scanning large expert tensors twice."""
    norm = math.hypot(*(part['reference_norm'] for part in parts))
    error = math.hypot(*(part['difference_norm'] for part in parts))
    maximum = max(part['reference_max_abs'] for part in parts)
    max_abs = max(part['max_abs'] for part in parts)
    finite = all(part['finite'] for part in parts)
    return {
        'reference_norm': norm, 'actual_norm': math.hypot(*(part['actual_norm'] for part in parts)),
        'relative_l2': error / max(norm, 1e-12), 'max_abs': max_abs, 'difference_norm': error,
        'reference_max_abs': maximum, 'exact': all(part['exact'] for part in parts), 'finite': finite,
        'elementwise_outliers': sum(part['elementwise_outliers'] for part in parts),
        'passed': (finite and error <= .02 * norm + 1e-7 and max_abs <= .05 * maximum + 1e-7
                   and all(part['passed'] for part in parts)),
    }


class ParameterGradientComparison:
    """Observe synchronized pre-clip gradients without changing the optimizer path.

    All ranks reconstruct each logical tensor through its existing DTensor layout.
    Rank zero saves/compares every element, including individual expert slices.
    The ordinary optimizer still clips, steps and clears gradients exactly once.
    """

    def __init__(self, model: torch.nn.Module, output: Path, reference: Path | None,
                 device: torch.device) -> None:
        self.model = model
        self.output = output / 'parameters'
        self.reference = reference / 'parameters' if reference else None
        self.device = device
        self.rank = dist.get_rank()
        self.rows = []
        self.reference_rows = []
        self.current_digest = ''
        self.calls = 0
        self.original_clip = optimizer_engine.clip_grad_norm_
        self.parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        self.output.mkdir(parents=True, exist_ok=True)

    def _full(self, value: torch.Tensor) -> torch.Tensor:
        """Reconstruct a logical tensor and verify its digest on every rank."""
        # Match weight publication: stage offloaded shards on the collective device.
        staged = value.detach().to(device=self.device)
        tensor = staged.full_tensor() if hasattr(staged, 'full_tensor') else staged
        result = tensor.detach().cpu()
        self.current_digest = _digest(result)
        checksum = torch.tensor(list(bytes.fromhex(self.current_digest)), dtype=torch.uint8, device=self.device)
        expected = checksum.clone()
        dist.broadcast(expected, src=0)
        equal = torch.tensor(int(torch.equal(checksum, expected)), device=self.device)
        dist.all_reduce(equal, op=dist.ReduceOp.MIN)
        if not bool(equal):
            raise AssertionError('Reconstructed logical tensor differs between ranks')
        return result

    def _save_or_compare(self, index: int, phase: str, value: torch.Tensor) -> dict:
        # Large CPU snapshots are diagnostic-only; restore production thread settings.
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(8)
            return self._evaluate(index, phase, value)
        finally:
            torch.set_num_threads(threads)

    def _evaluate(self, index: int, phase: str, value: torch.Tensor) -> dict:
        """Save a reference tensor or compare the corresponding parameter phase."""
        print(f'CP parameter check: phase={phase} index={index} name={self.parameters[index][0]}', flush=True)
        path = f'{index:04d}-{phase}.pt'
        if self.reference is None:
            torch.save(value, self.output / path)
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': self.current_digest,
                    'finite': bool(value.isfinite().all())}
        if phase == 'initial':
            expected = self.reference_rows[index]['initial']
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': self.current_digest,
                    'passed': (self.current_digest == expected['sha256'] and list(value.shape) == expected['shape']
                               and str(value.dtype) == expected['dtype']),
                    'finite': bool(value.isfinite().all())}
        expected = torch.load(self.reference / path, map_location='cpu', weights_only=True)
        if value.shape != expected.shape:
            raise AssertionError(f'Parameter shape differs: actual={value.shape}, expected={expected.shape}')
        if phase == 'updated':
            actual_float, expected_float = value.float(), expected.float()
            finite = bool(actual_float.isfinite().all() and expected_float.isfinite().all())
            # Adam magnifies sign differences near zero; gradient acceptance is independent.
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': finite,
                    'max_abs': float((actual_float - expected_float).abs().max()),
                    'exact': torch.equal(value, expected),
                    'passed': finite and torch.allclose(actual_float, expected_float, atol=.002, rtol=.02)}
        if value.ndim == 3 and '.experts.' in self.rows[index]['name']:
            parts = [_errors(part, target) for part, target in zip(value, expected)]
            result = {**_combine_errors(parts), 'experts': parts}
        elif '.linear_qkv.' in self.rows[index]['name']:
            config = self.model.config
            groups, width = config.num_key_value_heads, config.head_dim
            sizes = (config.num_attention_heads // groups * width, width, width)
            actual_parts = value.reshape(groups, sum(sizes), *value.shape[1:]).split(sizes, dim=1)
            expected_parts = expected.reshape(groups, sum(sizes), *expected.shape[1:]).split(sizes, dim=1)
            parts = {name: _errors(actual, target) for name, actual, target in
                     zip(('q', 'k', 'v'), actual_parts, expected_parts)}
            result = {**_combine_errors(list(parts.values())), 'projections': parts}
        else:
            result = _errors(value, expected)
        result.update(shape=list(value.shape), dtype=str(value.dtype))
        if not result['passed']:
            torch.save(value, self.output / path)
        return result

    def initial(self) -> None:
        """Verify every initial weight and parameter name before comparing gradients."""
        if self.rank == 0 and self.reference:
            self.reference_rows = json.loads((self.reference / 'comparison.json').read_text())['parameters']
            names = [name for name, _ in self.parameters]
            expected_names = [row['name'] for row in self.reference_rows]
            if names != expected_names:
                raise AssertionError(f'Parameter names differ: actual={names}, expected={expected_names}')
        for index, (name, param) in enumerate(self.parameters):
            self.rows.append({'name': name, 'local_shape': list(param.to_local().shape)
                              if hasattr(param, 'to_local') else list(param.shape),
                              'layout': str(getattr(param, 'placements', None))})
            value = self._full(param)
            if self.rank == 0:
                self.rows[index]['initial'] = self._save_or_compare(index, 'initial', value)
        self._check('initial')

    def clip(self, parameters: Iterable[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Capture after hsdp_sync_stream/prepare_grads and before clipping."""
        self.calls += 1
        if self.calls != 1:
            raise AssertionError(f'Expected one optimizer update, got {self.calls}')
        for index, (name, param) in enumerate(self.parameters):
            main = getattr(param, 'main_param', param)
            grad = main.grad
            if grad is None:
                grad = getattr(param, 'main_grad', None)
            present = torch.tensor(int(grad is not None), device=self.device)
            dist.all_reduce(present, op=dist.ReduceOp.MIN)
            if not bool(present):
                raise AssertionError(f'Trainable parameter has no synchronized gradient: {name}')
            value = self._full(grad)
            if self.rank == 0:
                self.rows[index]['gradient'] = self._save_or_compare(index, 'gradient', value)
        self._check('gradient')
        return self.original_clip(parameters, *args, **kwargs)

    def observe(self) -> ContextManager:
        """Return a scoped hook on the existing production clipping boundary."""
        return patch.object(optimizer_engine, 'clip_grad_norm_', self.clip)

    def finish(self) -> None:
        """Compare post-step model weights and publish all parameter evidence."""
        if self.calls != 1:
            raise AssertionError(f'Expected one gradient capture, got {self.calls}')
        for index, (_, param) in enumerate(self.parameters):
            value = self._full(param)
            if self.rank == 0:
                self.rows[index]['updated'] = self._save_or_compare(index, 'updated', value)
        self._check('updated')

    def _check(self, phase: str) -> None:
        """Publish rank zero's parameter report and propagate failure to every rank."""
        passed = True
        if self.rank == 0:
            passed = all(row[phase].get('passed', row[phase]['finite']) for row in self.rows)
            report = {'phase': phase, 'passed': passed, 'optimizer_updates_observed': self.calls,
                      'thresholds': {'gradient_relative_l2': .02, 'gradient_max_relative_to_tensor_max': .05,
                                     'gradient_absolute_floor': 1e-7,
                                     'elementwise_outliers_diagnostic': {'atol': 1e-5, 'rtol': .05}},
                      'parameter_count': len(self.rows), 'parameters': self.rows}
            (self.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        flag = torch.tensor(int(passed), device=self.device)
        dist.broadcast(flag, src=0)
        if not bool(flag):
            raise AssertionError(f'Full parameter {phase} comparison failed; see {self.output}/comparison.json')
