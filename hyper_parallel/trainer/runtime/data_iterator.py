# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
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

"""Trainer-side dataloader iteration helpers.

``BackgroundPrefetcher`` and ``HyperIter`` are split out of the former
``auto_models/trainer/base.py`` in stage 7 (05 §15.11 step 3); class names and
checkpointing semantics are retained.
"""

__all__ = [
    "BackgroundPrefetcher",
    "HyperIter",
]

import logging
import queue
import threading
from typing import Any, Dict


logger = logging.getLogger(__name__)


class BackgroundPrefetcher:
    """
    Prefetches batches from a dataloader in a background thread to overlap data loading
    with GPU computation. Synchronizes dataloader state for correct checkpointing.
    """

    def __init__(self, dataloader: Any, maxsize: int = 1) -> None:
        """Start the background worker prefetching from ``dataloader``."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.original_state_dict = getattr(dataloader, "state_dict", None)
        self.current_state = None
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self) -> None:
        """Prefetch data and capture dataloader state in a background thread."""
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.iterator)
                except StopIteration:
                    self._put_result((StopIteration, None))
                    break

                # A stop request cannot interrupt next(), so discard the item once
                # that call returns instead of retaining another batch and state.
                if self.stop_event.is_set():
                    break

                # Ensure we capture the state so that subsequent dataloader advances
                # don't mutate the captured state in-place. The underlying dataloader's
                # state_dict() should handle deepcopying if necessary.
                state = self.original_state_dict() if self.original_state_dict else None
                if not self._put_result((item, state)):
                    break
        # The worker must transfer any producer failure back to the training thread.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._put_result((exc, None))
        finally:
            # A timed-out stop cannot cancel next(), so the worker must release
            # its own references when that call eventually returns.
            if self.stop_event.is_set():
                self._cleanup_after_stop()

    def _put_result(self, result: tuple[Any, Any]) -> bool:
        """Put a worker result while allowing a stop request to cancel the write."""
        while not self.stop_event.is_set():
            try:
                self.queue.put(result, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _drain_queue(self) -> None:
        """Release every result currently retained by the prefetch queue."""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def _cleanup_after_stop(self) -> None:
        """Release queued results and live dataloader references after stopping."""
        self._drain_queue()
        self.iterator = None
        self.dataloader = None
        self.original_state_dict = None
        # The final checkpoint may be collected after stop(), so keep the
        # consumed-batch snapshot while releasing live dataloader references.

    def __iter__(self) -> "BackgroundPrefetcher":
        """Return this prefetcher as its own iterator."""
        return self

    def __next__(self) -> Any:
        """Return the next prefetched item, re-raising worker failures."""
        res = self.queue.get()
        if isinstance(res, tuple) and len(res) == 2:
            item, state = res
            if item is StopIteration:
                raise StopIteration
            if isinstance(item, Exception):
                raise item
            self.current_state = state
            return item
        if res is StopIteration:
            raise StopIteration
        if isinstance(res, Exception):
            raise res
        return res

    def state_dict(self) -> Dict[str, Any]:
        """Return the dataloader state captured alongside the current item."""
        if self.current_state is not None:
            return self.current_state
        if self.original_state_dict:
            return self.original_state_dict()
        return {}

    def stop(self, timeout: float | None = 5.0) -> bool:
        """Stop the background worker and release its dataloader references.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits until the worker exits.

        Returns:
            Whether the worker terminated within the requested timeout.
        """
        self.stop_event.set()
        self._drain_queue()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            logger.warning("BackgroundPrefetcher worker thread did not terminate within timeout.")
            return False

        self._cleanup_after_stop()
        return True


class HyperIter:
    """
    A unified iterator wrapper that handles both standard iteration and background prefetching.
    """

    def __init__(self, dataloader: Any, use_background_prefetcher: bool = False, maxsize: int = 1) -> None:
        """Wrap ``dataloader`` in either direct iteration or background prefetching."""
        self.dataloader = dataloader
        self.use_background_prefetcher = use_background_prefetcher
        if use_background_prefetcher:
            self.iterator = BackgroundPrefetcher(dataloader, maxsize=maxsize)
        else:
            self.iterator = iter(dataloader)

    def __iter__(self) -> "HyperIter":
        """Return this wrapper as its own iterator."""
        return self

    def __next__(self) -> Any:
        """Return the next batch from the underlying iterator."""
        return next(self.iterator)

    def stop(self, timeout: float | None = 5.0) -> bool:
        """Stop the background prefetch worker when one is active.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits until the worker exits.

        Returns:
            Whether the worker is stopped, or ``True`` when prefetching is disabled.
        """
        if self.use_background_prefetcher and hasattr(self.iterator, "stop"):
            return self.iterator.stop(timeout=timeout)
        return True

    def state_dict(self) -> Dict[str, Any]:
        """Return the underlying dataloader or prefetcher state for checkpointing."""
        if self.use_background_prefetcher and hasattr(self.iterator, "state_dict"):
            return self.iterator.state_dict()
        if hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}
