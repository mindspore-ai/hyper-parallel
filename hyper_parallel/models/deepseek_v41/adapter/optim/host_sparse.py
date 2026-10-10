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
"""One Trainer optimizer entry for dense parameters and CPU SparseAdam."""

import logging
import time
from typing import Any, Mapping

from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable

logger = logging.getLogger(__name__)


class HostSparseOptimizerCoordinator:
    """Keep sparse optimizer state out of the ordinary dense DCP payload."""

    def __init__(self, dense: Any, sparse: Any,
                 tables: Mapping[str, HostEngramTable]) -> None:
        """Bind dense and sparse leaves and expose one optimizer interface."""
        self.dense = dense
        self.sparse = sparse
        self.tables = tables
        self.state = "ACCUMULATING"
        self.optimizers_dict = dict(dense.optimizers_dict)
        if sparse is not None:
            if "engram_sparse" in self.optimizers_dict:
                raise ValueError("Dense optimizer already uses engram_sparse name")
            self.optimizers_dict["engram_sparse"] = sparse
        self.chained_optimizers = list(self.optimizers_dict.values())
        self.param_groups = [group for optimizer in self.chained_optimizers
                             for group in optimizer.param_groups]

    def step(self, closure: Any = None) -> Any:
        """Install sparse grads, update dense, then update CPU touched rows.

        Args:
            closure: Optional optimizer closure.
        """
        if self.state != "PREPARED":
            raise RuntimeError("Host sparse optimizer step requires PREPARED gradients")
        for table in self.tables.values():
            table.install_grad()
        result = self.dense.step(closure=closure)
        sparse_seconds = 0.0
        if self.sparse is not None:
            started = time.perf_counter()
            self.sparse.step()
            sparse_seconds = time.perf_counter() - started
        for fqn, table in self.tables.items():
            logger.info(
                "Host Engram %s: requests=%d unique_ids=%d forward_a2a_bytes=%d "
                "host_to_device_bytes=%d cpu_weight_moments_bytes=%d max_synced_nnz=%d "
                "coalesce_ms=%.3f sparse_adam_ms=%.3f",
                fqn, table.lookup_requests, table.lookup_unique_ids,
                table.forward_a2a_bytes, table.host_to_device_bytes,
                table.weight.numel() * 12, table.max_synced_nnz,
                table.coalesce_seconds * 1000.0, sparse_seconds * 1000.0,
            )
        self.state = "STEPPED"
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear dense and sparse gradients after a completed step.

        Args:
            set_to_none: Whether to clear gradients by assigning None.
        """
        self.dense.zero_grad(set_to_none=set_to_none)
        if self.sparse is not None:
            self.sparse.zero_grad(set_to_none=set_to_none)
        for table in self.tables.values():
            table.clear_step()
        self.state = "ACCUMULATING"

    def state_dict(self) -> dict[str, Any]:
        """Return only dense optimizer state for regular DCP."""
        return self.dense.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load only dense optimizer state; the sparse sidecar restores the rest.

        Args:
            state_dict: State dictionary to read or write.
        """
        self.dense.load_state_dict(state_dict)

    def drop_unpersisted_dense_state(self, persisted_keys: frozenset[str]) -> None:
        """Discard lazy moments absent from the saved dense optimizer state.

        Args:
            persisted_keys: DCP leaf keys present at the saved training step.
        """
        dense_raw = self.dense.optimizer
        named = dict(dense_raw.model.named_parameters())
        fqns_by_specificity = sorted(named, key=len, reverse=True)
        saved_fqns = set()
        for key in persisted_keys:
            if not key.startswith("state."):
                continue
            for fqn in fqns_by_specificity:
                if key.startswith(f"state.{fqn}."):
                    saved_fqns.add(fqn)
                    break
        optimizer_by_model_param = dense_raw.optimizer_param_by_model_param
        for fqn, model_param in named.items():
            if fqn in saved_fqns:
                continue
            optimizer_param = optimizer_by_model_param.get(model_param, model_param)
            for optimizer in self.dense.optimizers_dict.values():
                optimizer.state.pop(optimizer_param, None)

    def optimizer_for_dcp(self) -> Any:
        """Return the dense wrapper for lazy optimizer-state materialization."""
        return self.dense
