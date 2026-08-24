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
"""Trainer-owned parameter adaptation inside local compute regions."""

import functools
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Sequence

from hyper_parallel import DTensor


_LOCAL_COMPUTE_ACTIVE = ContextVar("trainer_local_compute_active", default=False)
_ADAPTED_MODULES = weakref.WeakSet()


@contextmanager
def local_compute_context() -> Iterator[None]:
    """Mark execution of a trainer-managed local computation."""
    token = _LOCAL_COMPUTE_ACTIVE.set(True)
    try:
        yield
    finally:
        _LOCAL_COMPUTE_ACTIVE.reset(token)


def install_local_compute_forward_adapters(
    module: Any,
    exclude: Sequence[str] = (),
) -> None:
    """Make descendant forwards expose their directly owned local parameters.

    A descendant FSDP pre-hook executes before its wrapped ``forward``, so the
    wrapper observes and temporarily unwraps the newly installed unsharded
    DTensor. Its restoration finishes before the FSDP post-hook. The same
    wrapper is a no-op when FSDP is disabled or execution is outside a local
    compute region.

    The root is intentionally omitted: the local-region skeleton already
    handles all parameters visible when its forward starts. Descendant
    installation is idempotent across overlapping local regions.

    Args:
        module: Root module of a trainer-managed local compute region.
        exclude: Relative descendant FQNs that remain DTensor dispatch islands.
    """
    excluded = tuple(name.rstrip(".") for name in exclude)
    descendants = tuple(module.named_modules())[1:]
    for relative_fqn, target in descendants:
        if any(
            relative_fqn == name or relative_fqn.startswith(name + ".")
            for name in excluded
        ):
            continue
        if target in _ADAPTED_MODULES:
            continue
        _wrap_module_forward(target)
        _ADAPTED_MODULES.add(target)


def _wrap_module_forward(module: Any) -> None:
    """Wrap one descendant module without changing its public forward contract."""
    original_forward = module.forward

    @functools.wraps(original_forward)
    def local_param_forward(*args: Any, **kwargs: Any) -> Any:
        """Run forward with directly owned DTensor parameters unwrapped locally."""
        if not _LOCAL_COMPUTE_ACTIVE.get():
            return original_forward(*args, **kwargs)
        saved = []
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(parameter, DTensor):
                saved.append((name, parameter))
                module._parameters[name] = parameter.to_local()  # pylint: disable=protected-access
        try:
            return original_forward(*args, **kwargs)
        finally:
            for name, parameter in saved:
                module._parameters[name] = parameter  # pylint: disable=protected-access

    module.forward = local_param_forward
