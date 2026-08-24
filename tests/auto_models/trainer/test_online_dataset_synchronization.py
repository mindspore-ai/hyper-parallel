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
"""Synchronization tests for Online Hugging Face Dataset construction."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from hyper_parallel.auto_models.components.distributed import infrastructure
from hyper_parallel.auto_models.components.datasets.llm import online_mapping_dataset
from hyper_parallel.auto_models.components.datasets.parallel import DatasetParallelContext


def test_online_mapping_download_synchronizes_shared_cache(monkeypatch) -> None:
    """Wait for the cache builder before other ranks reopen the Hub cache."""
    captured = {}
    expected_dataset = object()

    def _build_distributed_dataset(dataset_factory, parallel_context, *, barrier_needed):
        captured["parallel_context"] = parallel_context
        captured["barrier_needed"] = barrier_needed
        return expected_dataset

    monkeypatch.setattr(
        online_mapping_dataset,
        "build_distributed_dataset",
        _build_distributed_dataset,
    )
    parallel_context = DatasetParallelContext(distributed_enabled=True)

    dataset = online_mapping_dataset.build_online_mapping_dataset(
        data_path="unused",
        data_config={"hf_dataset_name": "Salesforce/wikitext"},
        parallel_context=parallel_context,
    )

    assert dataset is expected_dataset
    assert captured["parallel_context"] is not parallel_context
    assert isinstance(captured["parallel_context"].barrier, infrastructure.OnlineDatasetBarrier)
    assert captured["barrier_needed"] is True


def test_online_dataset_barrier_uses_ten_hour_gloo_monitored_barrier(monkeypatch) -> None:
    """Use a diagnostic long-timeout barrier without changing training collectives."""
    group = object()
    calls = []
    monkeypatch.setattr(
        infrastructure,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: True,
            get_world_size=lambda: 2,
            new_group=lambda **kwargs: calls.append(("new_group", kwargs)) or group,
            monitored_barrier=lambda **kwargs: calls.append(("monitored_barrier", kwargs)),
        ),
    )

    barrier = infrastructure.OnlineDatasetBarrier()
    barrier()

    expected_timeout = timedelta(hours=10)
    assert calls == [
        ("new_group", {"backend": "gloo", "timeout": expected_timeout}),
        (
            "monitored_barrier",
            {
                "group": group,
                "timeout": expected_timeout,
                "wait_all_ranks": True,
            },
        ),
    ]


def test_online_dataset_barrier_falls_back_when_gloo_is_unavailable(monkeypatch) -> None:
    """Retain the platform barrier on builds without Gloo support."""
    fallback_calls = []

    def _raise_gloo_error(**kwargs):
        del kwargs
        raise RuntimeError("Gloo is unavailable")

    monkeypatch.setattr(
        infrastructure,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: True,
            get_world_size=lambda: 2,
            new_group=_raise_gloo_error,
            barrier=lambda: fallback_calls.append(True),
        ),
    )

    barrier = infrastructure.OnlineDatasetBarrier()
    barrier()

    assert fallback_calls == [True]


@pytest.mark.parametrize(
    ("is_initialized", "world_size"),
    [(False, 1), (True, 1)],
)
def test_online_dataset_barrier_skips_non_distributed_runtime(
        is_initialized,
        world_size,
        monkeypatch,
) -> None:
    """Avoid auxiliary groups before distributed init or for one rank."""
    group_calls = []
    monkeypatch.setattr(
        infrastructure,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: is_initialized,
            get_world_size=lambda: world_size,
            new_group=lambda **kwargs: group_calls.append(kwargs),
        ),
    )

    infrastructure.OnlineDatasetBarrier()()

    assert group_calls == []


def test_online_dataset_barrier_propagates_monitored_timeout(monkeypatch) -> None:
    """Preserve missing-rank diagnostics instead of entering another barrier."""
    fallback_calls = []

    def _raise_timeout(**kwargs):
        del kwargs
        raise RuntimeError("rank 3 did not reach monitored barrier")

    monkeypatch.setattr(
        infrastructure,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: True,
            get_world_size=lambda: 4,
            new_group=lambda **kwargs: object(),
            monitored_barrier=_raise_timeout,
            barrier=lambda: fallback_calls.append(True),
        ),
    )

    with pytest.raises(RuntimeError, match="rank 3"):
        infrastructure.OnlineDatasetBarrier()()

    assert fallback_calls == []


def test_online_dataset_barrier_validates_timeout() -> None:
    """Reject non-positive Online Dataset synchronization timeouts."""
    with pytest.raises(ValueError, match="timeout must be positive"):
        infrastructure.OnlineDatasetBarrier(timeout=timedelta(0))
