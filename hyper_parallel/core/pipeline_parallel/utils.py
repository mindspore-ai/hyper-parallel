# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""pipeline parallel utils"""
from enum import Enum, auto


class MetaStepType(Enum):
    """Specify the enumeration type for MetaStep."""
    DATA_LOAD = auto()
    DATA_SEND = auto()
    DATA_RECV = auto()
    FWD = auto()
    BWD = auto()
    BWD_INPUT = auto()
    BWD_WEIGHT = auto()
    FWD_RECV = auto()
    FWD_SEND = auto()
    BWD_RECV = auto()
    BWD_SEND = auto()
    # Composite P2P: a contiguous run of FWD_SEND/FWD_RECV/BWD_SEND/BWD_RECV
    # coalesced by ``coalesce_p2p`` into one step whose ``sub_steps`` the runtime
    # groups by peer and issues as ``batch_isend_irecv`` (same-peer send+recv ->
    # duplex).  Produced under ``p2p_transport="batch"`` or ``"multi_stream"``.
    BATCH_SEND_RECV = auto()
    OVERLAP_F_B = auto()
    OVERLAP_B_F = auto()
    FSDP_UNSHARD = auto()
    FSDP_RESHARD = auto()
    FSDP_REDUCE_GRAD = auto()
    SWAP_LAUNCH_OFFLOAD = auto()
    SWAP_WAIT_OFFLOAD = auto()
    SWAP_LAUNCH_LOAD = auto()
    SWAP_WAIT_LOAD = auto()


class MetaStep:
    """
    Meta step of PipelineSchedule.
    An execution list composed of MetaStep can be constructed
    and fed into the PipelineSchedule for execution.

    Args:
        micro_index (int | None): The index of micro-batch.  ``None`` for
            composite types (``OVERLAP_F_B`` / ``OVERLAP_B_F``) whose real
            micro index lives in each ``sub_steps`` entry.
        type (MetaStepType): Specify the type of current step.
        stage_index (int | None): Stage index of current step.  ``None``
            for composite types; use ``sub_steps`` to get each direction's
            stage.
        sub_steps (tuple[MetaStep, MetaStep] | None): For composite types
            only: ``(fwd, bwd)`` for ``OVERLAP_F_B``, ``(bwd, fwd)`` for
            ``OVERLAP_B_F``.
        boundary_p2p (tuple[MetaStep, ...] | None): For ``OVERLAP_B_F`` under
            the ``"boundary"`` P2P transport only: P2P steps to issue at the
            fwd/bwd boundary inside the overlap (the forward's ``FWD_SEND``
            plus the next slot's recvs), hoisted out of the following gap by
            ``attach_fwd_boundary_p2p``.  Issued via
            :meth:`PipelineScheduleRuntime.exec_boundary_p2p`.
    """
    def __init__(self, micro_index, meta_type, stage_index, sub_steps=None,
                 boundary_p2p=None):
        self._type = meta_type
        self._micro_index = micro_index
        self._stage_index = stage_index
        self._sub_steps = sub_steps
        self._boundary_p2p = boundary_p2p

    @property
    def micro_index(self):
        """Return the micro-batch index of this step."""
        return self._micro_index

    @property
    def stage_index(self):
        """Return the stage index of this step."""
        return self._stage_index

    @property
    def type(self):
        """Return the MetaStepType of this step."""
        return self._type

    @property
    def sub_steps(self):
        """Return this step's sub-steps.

        ``(fwd, bwd)`` for OVERLAP_F_B, ``(bwd, fwd)`` for OVERLAP_B_F, or
        ``None`` for every other type.
        """
        return self._sub_steps

    @property
    def boundary_p2p(self):
        """P2P steps to issue at the overlap's fwd/bwd boundary, or ``None``."""
        return self._boundary_p2p

    def __eq__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return (self.type == value.type
                and self.micro_index == value.micro_index
                and self.stage_index == value.stage_index
                and self.sub_steps == value.sub_steps)

    def __ne__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return not self.__eq__(value)

    def __hash__(self):
        return hash((self.type, self.micro_index, self.stage_index))

    def __str__(self):
        if self.sub_steps:
            sub = ", ".join(str(s) for s in self.sub_steps)
            return (f"MetaStep(type={self.type}, micro_index={self.micro_index}, "
                    f"stage_index={self.stage_index}, sub_steps=[{sub}])")
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_str(step_str):
        """Parse a MetaStep from its string representation."""


class BatchDimSpec:
    """
    Specify the batch dimension of a Tensor.

    Args:
        batch_dim (int): batch dimension.
    """
    __slots__ = ("batch_dim",)

    def __init__(self, batch_dim):
        if not isinstance(batch_dim, int):
            raise TypeError(f"batch_dim must be int, but got type {type(batch_dim)}.")
        self.batch_dim = batch_dim

    def __repr__(self):
        return f"BatchDimSpec({self.batch_dim})"

    def __str__(self):
        return f"BatchDim(dim={self.batch_dim})"

    @staticmethod
    def from_tuple(batch_dims):
        """Create a tuple of BatchDimSpec from a tuple of batch dimensions."""
        if not isinstance(batch_dims, tuple):
            raise TypeError(f"batch_dims must be tuple, but got type {type(batch_dims)}.")
        return tuple(BatchDimSpec(dim) for dim in batch_dims)

    @staticmethod
    def from_dict(batch_dims):
        """Create a dict of BatchDimSpec from a dict mapping keys to batch dimensions."""
        if not isinstance(batch_dims, dict):
            raise TypeError(f"batch_dims must be dict, but got type {type(batch_dims)}.")
        return {k: BatchDimSpec(v) for k, v in batch_dims.items()}


class _RecvInfo:
    """
    Used for construct forward Receive operation and backward Send operation.

    ``requires_grad`` mirrors the forward tensor's ``requires_grad`` so
    pipeline code can skip backward send/recv for tensors that have no
    gradient, keeping the bwd-send count consistent with the peer's
    bwd-recv count.
    """

    def __init__(self, global_rank, buffer=None, requires_grad: bool = True):
        self._global_rank = global_rank
        self._buffer = buffer
        self._requires_grad = bool(requires_grad)

    @property
    def global_rank(self):
        """Return the global rank of the peer process."""
        return self._global_rank

    @property
    def buffer(self):
        """Return the receive/send buffer tensor."""
        return self._buffer

    @buffer.setter
    def buffer(self, val):
        """Set the receive/send buffer tensor."""
        self._buffer = val

    @property
    def requires_grad(self) -> bool:
        """Whether the corresponding forward tensor requires a gradient."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool) -> None:
        """Set whether the corresponding forward tensor requires a gradient."""
        self._requires_grad = bool(val)
