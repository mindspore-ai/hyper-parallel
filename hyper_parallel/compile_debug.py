# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Debug helpers for dumping ``torch.compile`` FX graphs."""
import io
import itertools
import logging
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import torch
from torch.fx import GraphModule

logger = logging.getLogger(__name__)

_GRAPH_COUNTER = itertools.count()
_BackendCompiler = Callable[[GraphModule, tuple[Any, ...]], Callable[..., Any]]


def maybe_wrap_compile_backend(backend: Any, name: str) -> Any:
    """Wrap a ``torch.compile`` backend with FX graph dumping when enabled.

    Enable this by setting ``HYPER_COMPILE_DUMP_GRAPH=1``. The wrapped backend
    first writes the Dynamo/AOT FX graph to disk, then delegates compilation to
    the original backend. It can be passed to either ``torch.compile`` or
    ``nn.Module.compile``.

    Args:
        backend: Backend accepted by ``torch.compile``. Common values are
            ``"inductor"``, ``"aot_eager"``, or a custom backend callable.
        name: Human-readable graph family name used in dump filenames.

    Returns:
        The original backend when dumping is disabled, otherwise a backend
        callable that dumps graphs before delegating to ``backend``.
    """
    if not _env_flag_enabled("HYPER_COMPILE_DUMP_GRAPH"):
        return backend

    compiler = _resolve_backend_compiler(backend)

    def _dump_backend(gm: GraphModule, example_inputs: tuple[Any, ...]) -> Callable[..., Any]:
        if _should_dump_current_rank():
            _dump_graph_module(gm, example_inputs, name)
        return compiler(gm, example_inputs)

    return _dump_backend


def _env_flag_enabled(name: str) -> bool:
    """Return whether an environment flag is enabled."""
    return os.environ.get(name, "0").lower() in ("1", "true", "on", "yes")


def _resolve_backend_compiler(backend: Any) -> _BackendCompiler:
    """Resolve a string backend into the callable expected by Dynamo."""
    if callable(backend):
        return backend
    if not isinstance(backend, str):
        raise ValueError(f"compile backend must be a string or callable, got {type(backend)!r}")
    try:
        return torch._dynamo.lookup_backend(backend)  # pylint: disable=protected-access
    except Exception as error:
        raise ValueError(f"failed to resolve torch.compile backend {backend!r}") from error


def _dump_graph_module(gm: GraphModule, example_inputs: tuple[Any, ...], name: str) -> None:
    """Write one FX graph module and its input metadata to disk."""
    graph_id = next(_GRAPH_COUNTER)
    dump_dir = _rank_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{graph_id:04d}_{_sanitize_name(name)}"
    _write_text(dump_dir / f"{stem}.py", gm.code)
    _write_text(dump_dir / f"{stem}.graph.txt", str(gm.graph))
    _write_text(dump_dir / f"{stem}.inputs.txt", _format_example_inputs(example_inputs))
    readable = _format_readable_graph(gm)
    if readable:
        _write_text(dump_dir / f"{stem}.readable.txt", readable)

    _log_info_rank0("Dumped torch.compile FX graph to %s", dump_dir / f"{stem}.py")


def _log_info_rank0(message: str, *args: Any) -> None:
    """Log with rank0 helper when available, otherwise use regular info."""
    log_fn = getattr(logger, "info_rank0", logger.info)
    log_fn(message, *args)


def _rank_dump_dir() -> Path:
    """Return the per-rank graph dump directory."""
    dump_root = Path(os.environ.get("HYPER_COMPILE_DUMP_DIR", Path.cwd() / "compile_dumps"))
    return dump_root / f"rank_{_current_rank()}"


def _should_dump_current_rank() -> bool:
    """Return whether the current distributed rank should dump graphs."""
    ranks = os.environ.get("HYPER_COMPILE_DUMP_RANKS", "0").strip().lower()
    if ranks in ("*", "all"):
        return True
    allowed_ranks = {rank.strip() for rank in ranks.split(",") if rank.strip()}
    return str(_current_rank()) in allowed_ranks


def _current_rank() -> int:
    """Best-effort global rank lookup without importing distributed packages."""
    for name in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                logger.warning("Ignoring invalid rank env %s=%r", name, value)
    return 0


def _sanitize_name(name: str) -> str:
    """Normalize a graph family name for use in filenames."""
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._")
    return sanitized or "compile_graph"


def _format_example_inputs(example_inputs: tuple[Any, ...]) -> str:
    """Format graph example input metadata without synchronizing tensor data."""
    lines = []
    for index, value in enumerate(example_inputs):
        if isinstance(value, torch.Tensor):
            lines.append(
                f"input[{index}]: Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device}, requires_grad={value.requires_grad})"
            )
        else:
            lines.append(f"input[{index}]: {type(value).__name__}")
    return "\n".join(lines) + "\n"


def _format_readable_graph(gm: GraphModule) -> str:
    """Return ``GraphModule.print_readable`` output when available."""
    if not hasattr(gm, "print_readable"):
        return ""
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = gm.print_readable()
    if isinstance(result, str):
        stream.write(result)
    return stream.getvalue()


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to ``path``."""
    path.write_text(content, encoding="utf-8")
