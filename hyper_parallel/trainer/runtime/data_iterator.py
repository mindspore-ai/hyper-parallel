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
``auto_models/trainer/base.py`` in stage 7 (05 §15.11 step 3); class names,
signatures and checkpointing semantics are unchanged.
"""

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

    def _worker(self):
        """Prefetch data and capture dataloader state in a background thread."""
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.iterator)
                except StopIteration:
                    self.queue.put((StopIteration, None))
                    break

                # Ensure we capture the state so that subsequent dataloader advances
                # don't mutate the captured state in-place. The underlying dataloader's
                # state_dict() should handle deepcopying if necessary.
                state = self.original_state_dict() if self.original_state_dict else None
                self.queue.put((item, state))
        # The worker must transfer any producer failure back to the training thread.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.queue.put((exc, None))

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

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker and wait up to ``timeout`` seconds."""
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning("BackgroundPrefetcher worker thread did not terminate within timeout.")


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

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background prefetch worker when one is active."""
        if self.use_background_prefetcher and hasattr(self.iterator, "stop"):
            self.iterator.stop(timeout=timeout)

    def state_dict(self) -> Dict[str, Any]:
        """Return the underlying dataloader or prefetcher state for checkpointing."""
        if self.use_background_prefetcher and hasattr(self.iterator, "state_dict"):
            return self.iterator.state_dict()
        if hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}


__all__ = [
    "BackgroundPrefetcher",
    "HyperIter",
]
