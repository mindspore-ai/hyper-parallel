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
"""Distributed correctness worker for ScheduleMPipeTranspose (PP=2).

Launched per-rank by ``test_mpipe_transpose_dist.py`` via
``parallel_run`` / ``TorchCase`` (torchrun); the backend adapts to the device
(hccl on NPU, gloo on CPU) through ``init_backend(_DEVICE_TYPE)``.  Runs MPipe
Transpose with MB=4 (NT=2) for both the trainable-preprocess (T=2) and the
param-free dataload-only (T=0) paths, asserting that the per-stage parameter
gradients and the summed loss match a single-process reference built from the
same seed.
"""
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import PipelineStage, ScheduleMPipeTranspose
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

DIM = 8
NUM_TRANSPOSE_LAYERS = 2
MICRO_BATCH_NUM = 4
MICRO_SIZE = 2
SEED = 1234


class Preprocess(nn.Module):
    """Two-layer preprocess block (the transposed layers)."""

    def __init__(self) -> None:
        """Build the two linear preprocess layers."""
        super().__init__()
        self.l0 = nn.Linear(DIM, DIM)
        self.l1 = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two tanh-activated linear layers.

        Args:
            x (Tensor): The preprocess input tensor.
        """
        return torch.tanh(self.l1(torch.tanh(self.l0(x))))


class Body0(nn.Module):
    """Stage 0 body (the layers after the preprocess block)."""

    def __init__(self) -> None:
        """Build the stage-0 body linear layer."""
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the tanh-activated body layer.

        Args:
            hidden (Tensor): The hidden state from the preprocess block.
        """
        return torch.tanh(self.linear(hidden))


class LastStage(nn.Module):
    """Stage 1 (last): produces a scalar loss for the micro-batch."""

    def __init__(self) -> None:
        """Build the last-stage linear layer."""
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project the hidden state and reduce to a scalar loss.

        Args:
            hidden (Tensor): The hidden state from the stage-0 body.
        """
        return self.linear(hidden).pow(2).sum()


class _Identity(nn.Module):
    """Param-free identity preprocess for the dataload-only (T=0) scenario."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Args:
            x (Tensor): The preprocess input tensor (shipped without a parameter copy).
        """
        return x


def _build_modules(dataload_only=False):
    """Deterministically build (preprocess, body0, last) on ``_DEVICE_TYPE``.

    Identical across ranks and the reference (seeded on CPU, then moved to the
    device).  ``dataload_only`` swaps the trainable two-layer preprocess for a
    param-free identity, exercising the T=0 path (no broadcast / recompute).
    """
    torch.manual_seed(SEED)
    preprocess = _Identity() if dataload_only else Preprocess()
    device = torch.device(_DEVICE_TYPE)
    return preprocess.to(device), Body0().to(device), LastStage().to(device)


def _micro_inputs():
    """The MB deterministic micro-batch input slices (on ``_DEVICE_TYPE``)."""
    torch.manual_seed(SEED + 1)
    x = torch.randn(MICRO_BATCH_NUM * MICRO_SIZE, DIM)
    slices = [x[i * MICRO_SIZE:(i + 1) * MICRO_SIZE] for i in range(MICRO_BATCH_NUM)]
    return [to_device(s, _DEVICE_TYPE) for s in slices]


def _reference_grads(dataload_only=False, n_accum=1):
    """Single-process reference grads for preprocess / body0 / last, accumulated
    over ``n_accum`` identical passes (grad accumulation)."""
    preprocess, body0, last = _build_modules(dataload_only)
    total = None
    for _ in range(n_accum):
        for x in _micro_inputs():
            loss = last(body0(preprocess(x)))  # pylint: disable=not-callable
            total = loss if total is None else total + loss
    total.backward()
    return preprocess, body0, last, total.detach()


def _assert_grads_match(name, mod_a, mod_b):
    """Assert every parameter gradient of ``mod_a`` matches ``mod_b`` (the reference)."""
    for (pname, pa), (_, pb) in zip(mod_a.named_parameters(), mod_b.named_parameters()):
        assert pa.grad is not None, f"{name}.{pname} has no gradient in the MPipe run"
        diff = (pa.grad - pb.grad).abs().max().item()
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5, rtol=1e-4), \
            f"{name}.{pname} grad mismatch: max abs diff {diff}"


def _assert_loss_matches(losses, ref_total):
    """Assert the MPipe summed loss matches the single-process reference."""
    mpipe_total = torch.stack([loss.detach() for loss in losses]).sum()
    loss_diff = (mpipe_total - ref_total).abs().item()
    assert torch.allclose(mpipe_total, ref_total, atol=1e-5, rtol=1e-4), \
        f"summed loss mismatch: mpipe={mpipe_total.item()}, ref={ref_total.item()}, diff={loss_diff}"


def _run_case(num_transpose_layers, owner_backward=False, n_accum=1):
    """Run one MPipe Transpose schedule on this rank and check it vs the reference.

    ``num_transpose_layers == 0`` exercises the param-free dataload-only path;
    ``> 0`` the trainable-preprocess path -- centralized stage-0 backward by
    default, or owner-does-backward when ``owner_backward`` is set (each owner
    backprops its retained tower graph; tower grads are SUM-reduced to stage 0).
    ``n_accum > 1`` runs the schedule that many times before the grad check
    (gradient accumulation): the grads must accumulate, and owner-backward's
    GRAD_REDUCE must reduce only each run's contribution (not re-reduce earlier
    passes).
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    dataload_only = num_transpose_layers == 0
    preprocess, body0, last = _build_modules(dataload_only)
    device = torch.device(_DEVICE_TYPE)
    stage_module = body0 if rank == 0 else last
    stage = PipelineStage(stage_module, stage_index=rank, stage_num=world, device=device)
    schedule = ScheduleMPipeTranspose(
        [stage],
        micro_batch_num=MICRO_BATCH_NUM,
        preprocess_module=preprocess,
        num_transpose_layers=num_transpose_layers,
        owner_backward=owner_backward,
    )

    # Every rank reads the full batch; rank i uses its micro-batch i for the
    # transposed preprocess forward (matches the per-rank dataload convention).
    full_x = torch.cat(_micro_inputs(), dim=0)
    losses = None
    for _ in range(n_accum):  # gradient accumulation: grads add up across runs
        losses = schedule.run(full_x)

    ref_pre, ref_body0, ref_last, ref_total = _reference_grads(dataload_only, n_accum)
    # ``losses`` holds the last pass only, so the summed-loss check is meaningful
    # only without accumulation; the accumulated grads are the real assertion.
    check_loss = n_accum == 1

    if owner_backward:
        # Grads are SUM-reduced to every rank, so each replica ends with the
        # full reference gradient, including non-root.
        _assert_grads_match("preprocess", preprocess, ref_pre)
        if rank == 0:
            _assert_grads_match("body0", body0, ref_body0)
        else:
            _assert_grads_match("last", last, ref_last)
            if check_loss:
                _assert_loss_matches(losses, ref_total)
        return

    if rank == 0:
        # Preprocess gradients accumulate only on stage 0 (centralized backward).
        _assert_grads_match("preprocess", preprocess, ref_pre)
        _assert_grads_match("body0", body0, ref_body0)
    else:
        # The non-root preprocess copy is forward-only: it must carry no gradient.
        assert all(p.grad is None for p in preprocess.parameters()), \
            "non-root preprocess copy must not accumulate gradients (centralized on stage 0)"
        _assert_grads_match("last", last, ref_last)
        if check_loss:
            _assert_loss_matches(losses, ref_total)


def test_mpipe_transpose():
    """
    Feature: MPipe Transpose distributed execution (PP=2).
    Description: On each rank, run MPipe Transpose for the trainable-preprocess
        (T=2) and the param-free dataload-only (T=0) paths over a tiny model.
    Expectation: per-stage parameter gradients and the summed loss match a
        single-process reference built from the same seed.
    """
    init_backend(_DEVICE_TYPE)
    _run_case(NUM_TRANSPOSE_LAYERS)
    _run_case(0)
    print(f"[rank {dist.get_rank()}] MPipe Transpose distributed correctness OK")


def test_mpipe_transpose_owner_backward():
    """
    Feature: MPipe Transpose owner-does-backward (PP=2, trainable tower).
    Description: On each rank, run MPipe Transpose with owner_backward=True over
        the trainable-preprocess (T=2) tiny model -- owners backprop their retained
        tower graph and tower grads are SUM-reduced to stage 0.
    Expectation: every rank's preprocess (tower) gradient equals the single-process
        reference (the reduced full gradient), per-stage body grads match, and the
        summed loss matches.
    """
    init_backend(_DEVICE_TYPE)
    _run_case(NUM_TRANSPOSE_LAYERS, owner_backward=True)
    print(f"[rank {dist.get_rank()}] MPipe owner-backward distributed correctness OK")


def test_mpipe_transpose_owner_backward_accum():
    """
    Feature: MPipe owner-does-backward under gradient accumulation (PP=2).
    Description: Run owner_backward with 2 accumulation passes (2 schedule runs
        before the grad check); GRAD_REDUCE runs once per pass over the
        accumulating tower grad.
    Expectation: every rank's accumulated tower gradient equals the 2-pass
        single-process reference -- GRAD_REDUCE must reduce only each run's
        contribution, not re-reduce earlier passes.
    """
    init_backend(_DEVICE_TYPE)
    _run_case(NUM_TRANSPOSE_LAYERS, owner_backward=True, n_accum=2)
    print(f"[rank {dist.get_rank()}] MPipe owner-backward grad-accum correctness OK")


if __name__ == "__main__":
    test_mpipe_transpose()
    test_mpipe_transpose_owner_backward()
    test_mpipe_transpose_owner_backward_accum()
