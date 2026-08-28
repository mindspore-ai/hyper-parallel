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
"""Actor-to-rollout policy publication and synchronization lifecycle."""
import base64
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import pickle
from typing import Any, Callable, Mapping, Optional
from urllib import parse as urllib_parse
from hyper_parallel import get_platform
platform = get_platform()
POLICY_FINGERPRINT_ALGORITHM = "qwen_norms_f32_v3"
KEEP_SCHEDULER_PAUSED_TAG = "_hyper_keep_scheduler_paused"


@dataclass(frozen=True)
class PolicySnapshot:
    """One immutable publication of policy Actor weights."""
    version: int
    model_name: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("PolicySnapshot version must be non-negative")
        if not self.model_name:
            raise ValueError("PolicySnapshot model_name must be non-empty")


def synchronize_error(local_error: Optional[Exception], operation: str) -> None:
    """Make every training rank observe one local synchronization failure."""
    try:
        world_size = platform.get_world_size()
    except (RuntimeError, ValueError):
        world_size = 1
    if world_size <= 1:
        if local_error is not None:
            raise local_error
        return
    errors: list[Optional[str]] = [None] * world_size
    platform.all_gather_object(
        errors,
        None if local_error is None else str(local_error),
    )
    if any(error is not None for error in errors):
        raise RuntimeError(f"vLLM {operation} failed on at least one rank: {errors}")


def synchronized_call(operation: str, callback: Callable[[], Any]) -> Any:
    """Run one local operation and propagate its failure to every training rank."""
    result = None
    local_error = None
    try:
        result = callback()
    except Exception as error:  # pylint: disable=W0718
        local_error = error
    synchronize_error(local_error, operation)
    return result


def coordinator_call(operation: str, callback: Callable[[], Any]) -> Any:
    """Run one coordinator operation and return its result on every rank."""
    try:
        world_size = platform.get_world_size()
        rank = platform.get_rank()
    except (RuntimeError, ValueError):
        world_size = 1
        rank = 0
    if world_size <= 1:
        return callback()
    result = None
    local_error = None
    if rank == 0:
        try:
            result = callback()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
    synchronize_error(local_error, operation)
    gathered: list[Any] = [None] * world_size
    platform.all_gather_object(gathered, result if rank == 0 else None)
    return gathered[0]


def canonical_policy_weight_name(name: str) -> str:
    """Return the canonical Qwen3 Actor and rollout parameter name."""
    return name


def is_policy_fingerprint_weight(name: str) -> bool:
    """Return whether a stable replicated norm participates in verification."""
    canonical_name = canonical_policy_weight_name(name)
    return (
        canonical_name.startswith("model.")
        and canonical_name.endswith("norm.weight")
        and ".linear_attn." not in canonical_name
    )


def policy_fingerprint_header(name: str, shape: tuple[int, ...]) -> bytes:
    """Serialize one canonical parameter identity for deterministic hashing."""
    return json.dumps(
        [canonical_policy_weight_name(name), list(shape)],
        separators=(",", ":"),
    ).encode("utf-8")


def policy_tensor_fingerprint(
    name: str,
    shape: tuple[int, ...],
    values: bytes,
) -> tuple[str, str]:
    """Return one canonical parameter name and content digest."""
    canonical_name = canonical_policy_weight_name(name)
    digest = sha256()
    digest.update(policy_fingerprint_header(canonical_name, shape))
    digest.update(values)
    return canonical_name, digest.hexdigest()


def aggregate_policy_fingerprint(
    tensors: Mapping[str, str],
    value_count: int,
) -> dict[str, Any]:
    """Aggregate parameter digests without namespace-order ambiguity."""
    digest = sha256()
    for name, tensor_digest in sorted(tensors.items()):
        digest.update(
            json.dumps([name, tensor_digest], separators=(",", ":")).encode("utf-8")
        )
    return {
        "algorithm": POLICY_FINGERPRINT_ALGORITHM,
        "tensor_count": len(tensors),
        "value_count": value_count,
        "digest": digest.hexdigest(),
        "tensors": dict(sorted(tensors.items())),
    }


def policy_weight_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Hash replicated Qwen3 norm weights without copying the full policy."""
    tensor_digests = {}
    value_count = 0
    for name, tensor in sorted(state_dict.items(), key=lambda item: item[0]):
        if not is_policy_fingerprint_weight(name):
            continue
        values = platform.tensor_type_cast(
            tensor.detach().to("cpu").contiguous(),
            "float32",
        )
        canonical_name, tensor_digest = policy_tensor_fingerprint(
            name,
            tuple(values.shape),
            platform.tensor_to_numpy(values).tobytes(),
        )
        if canonical_name in tensor_digests:
            raise RuntimeError(
                f"Actor policy fingerprint has duplicate tensor {canonical_name!r}"
            )
        tensor_digests[canonical_name] = tensor_digest
        value_count += int(values.numel())
    if not tensor_digests:
        raise RuntimeError("Actor policy fingerprint found no language-model norm tensors")
    return aggregate_policy_fingerprint(tensor_digests, value_count)


def verify_policy_fingerprints(
    expected: Mapping[str, Any],
    actual: list[Mapping[str, Any]],
    expected_version: Optional[int] = None,
) -> None:
    """Require every rollout worker fingerprint to match the policy Actor."""
    if not actual:
        raise RuntimeError("vLLM policy fingerprint returned no worker results")
    fields = ("algorithm", "tensor_count", "value_count", "digest")
    version_mismatches = [
        result.get("version")
        for result in actual
        if expected_version is not None and result.get("version") != expected_version
    ]
    mismatches = [
        {field: result.get(field) for field in fields}
        for result in actual
        if any(result.get(field) != expected.get(field) for field in fields)
    ]
    if not mismatches and not version_mismatches:
        return
    expected_tensors = expected.get("tensors", {})
    tensor_mismatches = []
    for result in actual:
        actual_tensors = result.get("tensors", {})
        if actual_tensors != expected_tensors:
            tensor_mismatches.append(
                {
                    "rank": result.get("rank"),
                    "missing": sorted(set(expected_tensors) - set(actual_tensors)),
                    "unexpected": sorted(set(actual_tensors) - set(expected_tensors)),
                    "changed": sorted(
                        name
                        for name in set(expected_tensors) & set(actual_tensors)
                        if expected_tensors[name] != actual_tensors[name]
                    ),
                }
            )
    expected_summary = {field: expected.get(field) for field in fields}
    raise RuntimeError(
        "vLLM policy fingerprint mismatch: "
        f"expected={expected_summary}, actual={mismatches}, "
        f"expected_version={expected_version}, version_mismatches={version_mismatches}, "
        f"tensor_mismatches={tensor_mismatches}"
    )


class VLLMWeightSyncClientMixin:
    """Weight-control requests shared by the rollout-side vLLM HTTP client."""
    _base_url: str

    @property
    def is_server_owner(self) -> bool:
        """Return whether this trainer rank owns the connected server process."""
        return True

    def _request(
        self,
        method: str,
        route: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        """Return this rank-local rollout endpoint."""
        return self._base_url

    def get_world_size(self, base_url: Optional[str] = None) -> int:
        """Return the number of vLLM inference workers."""
        return int(
            self._request("GET", "get_world_size", base_url=base_url)["world_size"]
        )

    def init_weight_transfer(
        self,
        init_info: Mapping[str, Any],
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the server side of the stateless HCCL transfer group."""
        self._request(
            "POST",
            "init_weight_transfer_engine",
            {"init_info": dict(init_info)},
            timeout=180,
            base_url=base_url,
        )

    def pause(self) -> None:
        """Pause generation and invalidate request caches before transfer."""
        status = self._request("POST", "pause?mode=abort&clear_cache=true").get("status")
        if status != "paused":
            raise RuntimeError(f"vLLM /pause returned invalid status {status!r}")

    def is_paused(self) -> bool:
        """Return whether generation admission is closed."""
        value = self._request("GET", "is_paused").get("is_paused")
        if not isinstance(value, bool):
            raise RuntimeError("vLLM /is_paused did not return a boolean state")
        return value

    def sleep(self, level: int = 1, mode: str = "wait") -> None:
        """Drain generation and release tagged vLLM device memory."""
        self._request("POST", f"sleep?level={level}&mode={mode}")

    def wake_up(self, tags: tuple[str, ...]) -> None:
        """Restore executor memory without letting EngineCore resume scheduling."""
        query = urllib_parse.urlencode(
            [("tags", tag) for tag in (*tags, KEEP_SCHEDULER_PAUSED_TAG)]
        )
        self._request("POST", f"wake_up?{query}")

    def is_sleeping(self) -> bool:
        """Return the server's combined scheduler/device sleep state."""
        value = self._request("GET", "is_sleeping").get("is_sleeping")
        if not isinstance(value, bool):
            raise RuntimeError("vLLM /is_sleeping did not return a boolean state")
        return value

    def start_weight_update(self) -> None:
        """Start loading checkpoint-format Actor weights."""
        self._request("POST", "start_weight_update", {"is_checkpoint_format": True})

    def receive_weights(
        self,
        update_info: Mapping[str, Any],
        policy_version: int,
        base_url: Optional[str] = None,
    ) -> None:
        """Block until the server receives all HCCL weight buffers."""
        versioned_update = dict(update_info)
        versioned_update["_hyper_policy_version"] = policy_version
        self._request(
            "POST",
            "update_weights",
            {"update_info": versioned_update},
            timeout=600,
            base_url=base_url,
        )

    def receive_ipc_weights(
        self,
        base_url: str,
        update_info: Any,
        policy_version: int,
    ) -> None:
        """Send one merged NPU IPC handle set to a rollout replica."""
        update_fields = asdict(update_info)
        ipc_handles = update_fields.pop("ipc_handles")
        update_fields["ipc_handles_pickled"] = base64.b64encode(
            pickle.dumps(ipc_handles)
        ).decode("ascii")
        update_fields["_hyper_policy_version"] = policy_version
        self._request(
            "POST",
            "update_weights",
            {"update_info": update_fields},
            timeout=600,
            base_url=base_url,
        )

    def finish_weight_update(self) -> None:
        """Commit one completed Actor-to-rollout weight transfer."""
        self._request("POST", "finish_weight_update")

    def collective_rpc(
        self,
        method: str,
        kwargs: Optional[Mapping[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> list[Any]:
        """Invoke one registered worker method on every inference rank."""
        response = self._request(
            "POST",
            "collective_rpc",
            {"method": method, "kwargs": dict(kwargs or {})},
            base_url=base_url,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("vLLM collective RPC returned invalid worker results")
        return results

    def reset_prefix_cache(
        self,
        *,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> bool:
        """Invalidate cached prefixes after loading a new policy."""
        query = urllib_parse.urlencode(
            {
                "reset_running_requests": str(reset_running_requests).lower(),
                "reset_external": str(reset_connector).lower(),
            }
        )
        self._request("POST", f"reset_prefix_cache?{query}")
        return True

    def get_policy_weight_fingerprints(
        self,
        base_url: Optional[str] = None,
    ) -> list[Mapping[str, Any]]:
        """Return one post-transfer fingerprint per vLLM worker."""
        if base_url is None:
            results = self.collective_rpc("get_policy_weight_fingerprint")
        else:
            response = self._request(
                "POST",
                "collective_rpc",
                {
                    "method": "get_policy_weight_fingerprint",
                    "kwargs": {},
                },
                base_url=base_url,
            )
            results = response.get("results")
        if not isinstance(results, list) or not all(
            isinstance(result, Mapping) for result in results
        ):
            raise RuntimeError("vLLM policy fingerprint RPC returned invalid worker results")
        return results

    def verify_policy_weight_identity(
        self,
        expected_version: int,
        expected_fingerprint: Mapping[str, Any],
    ) -> None:
        """Require every internal-DP worker to match one expected identity."""
        self.collective_rpc(
            "verify_policy_weight_identity",
            {
                "expected_version": int(expected_version),
                "expected_fingerprint": dict(expected_fingerprint),
            },
        )

    def resume(self) -> None:
        """Resume rollout admission after a completed transfer."""
        status = self._request("POST", "resume").get("status")
        if status != "resumed":
            raise RuntimeError(f"vLLM /resume returned invalid status {status!r}")


class ActorRolloutWeightSync:
    """Move policy Actor weights into rollout and publish versions atomically."""

    def __init__(
        self,
        model_name: str,
        deployment: str,
        client_provider: Callable[[], Any],
        weight_transfer: Optional[Any],
    ) -> None:
        """Initialize one controller-owned policy publication transaction."""
        self._model_name = model_name
        self._deployment = deployment
        self._control_call = coordinator_call
        self._client_provider = client_provider
        self._weight_transfer = weight_transfer
        self._policy_version = 0
        self._policy_fingerprint: Optional[str] = None
        self._policy_identity: Optional[dict[str, Any]] = None
        self._policy_fingerprint_changed = False
        self._pending_policy_version: Optional[int] = None
        self._phase = "rollout"

    @property
    def policy_version(self) -> int:
        """Return the policy version admitted for generation."""
        return self._policy_version

    @property
    def policy_fingerprint(self) -> Optional[str]:
        """Return the fingerprint admitted for generation."""
        return self._policy_fingerprint

    @property
    def policy_fingerprint_changed(self) -> bool:
        """Return whether the last transfer changed the policy fingerprint."""
        return self._policy_fingerprint_changed

    @property
    def configured_strategy(self) -> Optional[str]:
        """Return the effective configured weight-transfer strategy."""
        return getattr(self._weight_transfer, "configured_strategy", None)

    @property
    def last_strategy(self) -> Optional[str]:
        """Return the strategy that completed the latest publication."""
        return getattr(self._weight_transfer, "last_strategy", None)

    @property
    def fallback_count(self) -> int:
        """Return the number of successful full-gather fallback publications."""
        return int(getattr(self._weight_transfer, "fallback_count", 0))

    @property
    def direct_success_count(self) -> int:
        """Return the number of successful direct-reshard publications."""
        return int(getattr(self._weight_transfer, "direct_success_count", 0))

    @property
    def phase(self) -> str:
        """Return the current residency and publication phase."""
        return self._phase

    def _capture_initial_policy_fingerprint(self, client: Any) -> Optional[Mapping[str, Any]]:
        """Capture and verify the policy fingerprint loaded at startup."""
        if (
            self._policy_fingerprint is not None
            or not isinstance(client, VLLMWeightSyncClientMixin)
        ):
            return None
        fingerprints = client.get_policy_weight_fingerprints()
        if not fingerprints:
            raise RuntimeError("vLLM initial policy fingerprint returned no worker results")
        verify_policy_fingerprints(
            fingerprints[0],
            fingerprints,
            expected_version=self._policy_version,
        )
        expected = fingerprints[0]
        client.verify_policy_weight_identity(self._policy_version, expected)
        self._policy_identity = dict(expected)
        self._policy_fingerprint = str(expected["digest"])
        return expected

    def generation_identity(self, client: Any) -> tuple[int, str]:
        """Verify and return the worker-owned identity serving the next request."""
        if self._phase != "rollout" or self._pending_policy_version is not None:
            raise RuntimeError(
                "Cannot generate from an unpublished rollout policy: "
                f"phase={self._phase!r}, pending={self._pending_policy_version}"
            )
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise RuntimeError("Consistency-profile generation requires the owned vLLM HTTP client")
        captured_identity = False
        if self._policy_identity is None:
            expected = self._control_call(
                "initial rollout identity",
                lambda: self._capture_initial_policy_fingerprint(client),
            )
            if not isinstance(expected, Mapping):
                raise RuntimeError("vLLM initial policy identity returned no coordinator result")
            self._policy_identity = dict(expected)
            self._policy_fingerprint = str(expected["digest"])
            captured_identity = True
        if not captured_identity:
            self._control_call(
                "generation rollout identity",
                lambda: client.verify_policy_weight_identity(
                    self._policy_version,
                    self._policy_identity,
                ),
            )
        return self._policy_version, self._policy_fingerprint

    @staticmethod
    def _server_owner_call(
        client: Any,
        operation: str,
        callback: Callable[[], Any],
    ) -> Any:
        """Run one mutating server operation exactly once on the coordinator."""
        del client
        return coordinator_call(operation, callback)

    def prepare_for_training(self) -> None:
        """Sleep colocated rollout before policy Actor training starts."""
        if self._deployment != "colocated":
            return
        if self._phase != "rollout":
            raise RuntimeError(f"Cannot prepare vLLM for training from phase {self._phase!r}")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        identity = self._control_call(
            "initial rollout identity",
            lambda: self._capture_initial_policy_fingerprint(client),
        )
        if isinstance(identity, Mapping):
            self._policy_identity = dict(identity)
            self._policy_fingerprint = str(identity["digest"])
        self._server_owner_call(
            client,
            "sleep before training",
            lambda: client.sleep(level=1, mode="wait"),
        )
        def verify_sleeping() -> None:
            """Require every rank's connected replica to be sleeping."""
            if not client.is_sleeping():
                raise RuntimeError("vLLM did not enter sleep mode before training")
        synchronized_call("sleep residency check", verify_sleeping)
        self._phase = "training"

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Transfer a newer Actor snapshot and stage its rollout version."""
        if snapshot.model_name != self._model_name:
            raise ValueError(
                f"Policy snapshot model mismatch: expected={self._model_name}, "
                f"received={snapshot.model_name}"
            )
        if snapshot.version <= self._policy_version:
            raise ValueError(
                "Policy snapshot version must increase: "
                f"current={self._policy_version}, received={snapshot.version}"
            )
        if self._weight_transfer is None:
            raise NotImplementedError(
                "vLLM iterative training requires a concrete WeightTransfer; "
                "the adapter will not acknowledge a new policy version without loading it"
            )
        if self._deployment == "colocated" and self._phase != "training":
            raise RuntimeError(f"Cannot refit colocated vLLM from phase {self._phase!r}")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        publish = getattr(self._weight_transfer, "publish", None)
        if not callable(publish):
            publish = getattr(self._weight_transfer, "transfer", None)
        if not callable(publish):
            publish = getattr(self._weight_transfer, "refit", None)
        if not callable(publish):
            raise ValueError(
                "The configured weight transfer must implement publish(), transfer(), or refit()"
            )
        transactional_client = isinstance(client, VLLMWeightSyncClientMixin)
        if transactional_client:
            self._pending_policy_version = snapshot.version
            self._phase = "refit"
        publish(client, snapshot)
        fingerprint = getattr(self._weight_transfer, "last_policy_fingerprint", None)
        if fingerprint is not None:
            self._policy_identity = dict(fingerprint)
            digest = str(fingerprint["digest"])
            self._policy_fingerprint_changed = (
                self._policy_fingerprint is None or digest != self._policy_fingerprint
            )
            self._policy_fingerprint = digest
        if self._deployment == "colocated":
            return
        if transactional_client:
            try:
                self._server_owner_call(
                    client,
                    "resume rollout admission",
                    client.resume,
                )
            except Exception as resume_error:
                try:
                    self._server_owner_call(
                        client,
                        "compensating rollout pause",
                        client.pause,
                    )
                except Exception as pause_error:
                    raise RuntimeError(
                        "vLLM rollout resume failed and compensating pause also failed: "
                        f"resume={resume_error!r}, pause={pause_error!r}"
                    ) from pause_error
                raise
            self._policy_version = snapshot.version
            self._pending_policy_version = None
            self._phase = "rollout"
        else:
            self._policy_version = snapshot.version

    def prepare_for_rollout(self) -> None:
        """Wake rollout memory and atomically expose a transferred policy."""
        if self._deployment != "colocated":
            return
        if self._phase not in ("training", "refit"):
            raise RuntimeError(f"Cannot prepare vLLM for rollout from phase {self._phase!r}")
        if self._phase == "refit" and self._pending_policy_version is None:
            raise RuntimeError("Colocated refit completed without a pending policy version")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        is_refit = self._phase == "refit"
        tags = ("kv_cache",) if is_refit else ("weights", "kv_cache")
        self._server_owner_call(
            client,
            "wake before rollout",
            lambda: client.wake_up(tags),
        )
        if is_refit:
            self._server_owner_call(
                client,
                "post-refit cache reset",
                client.pause,
            )
            def verify_post_refit_pause() -> None:
                """Require admission to remain closed after refit."""
                if not client.is_paused():
                    raise RuntimeError("vLLM did not remain paused after post-refit cache reset")
            synchronized_call("post-refit pause check", verify_post_refit_pause)
            expected_version = self._pending_policy_version
            def verify_pending_identity() -> None:
                """Verify worker-owned identity before opening admission."""
                expected_fingerprint = getattr(
                    self._weight_transfer,
                    "last_policy_fingerprint",
                    None,
                )
                if not isinstance(expected_fingerprint, Mapping):
                    raise RuntimeError("Shared refit did not expose an Actor policy fingerprint")
                client.verify_policy_weight_identity(
                    expected_version,
                    expected_fingerprint,
                )
            self._control_call("pending rollout identity", verify_pending_identity)
        try:
            self._server_owner_call(
                client,
                "resume rollout admission",
                client.resume,
            )
            def verify_rollout_residency() -> None:
                """Require both scheduler and device memory to be ready."""
                if client.is_paused() or client.is_sleeping():
                    raise RuntimeError("vLLM remained paused or sleeping after resume")
            self._control_call("rollout residency check", verify_rollout_residency)
        except Exception as resume_error:
            try:
                self._server_owner_call(
                    client,
                    "compensating rollout pause",
                    client.pause,
                )
            except Exception as pause_error:
                raise RuntimeError(
                    "vLLM rollout resume failed and compensating pause also failed: "
                    f"resume={resume_error!r}, pause={pause_error!r}"
                ) from pause_error
            raise
        if self._pending_policy_version is not None:
            self._policy_version = self._pending_policy_version
        self._pending_policy_version = None
        self._phase = "rollout"

    def close(self) -> None:
        """Release resources owned by the selected transfer implementation."""
        if self._weight_transfer is None:
            return
        close_transfer = getattr(self._weight_transfer, "close", None)
        if callable(close_transfer):
            close_transfer()
__all__ = [
    "ActorRolloutWeightSync",
    "KEEP_SCHEDULER_PAUSED_TAG",
    "POLICY_FINGERPRINT_ALGORITHM",
    "PolicySnapshot",
    "VLLMWeightSyncClientMixin",
    "aggregate_policy_fingerprint",
    "canonical_policy_weight_name",
    "coordinator_call",
    "is_policy_fingerprint_weight",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
    "policy_weight_fingerprint",
    "synchronized_call",
    "synchronize_error",
    "verify_policy_fingerprints",
]
