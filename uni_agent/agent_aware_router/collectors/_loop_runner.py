# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Private event-loop ownership for asynchronous collectors."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable

from .transport.base import Transport

logger = logging.getLogger(__name__)


class _LoopRunner:
    """Own one transport's loop, thread, subscription and pending cleanup."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self.thread: threading.Thread | None = None
        self.subscription_future: Future | None = None
        self.cleanup_future: Future | None = None
        self._terminal_failure: str | None = None

    def _ensure_startable(self) -> None:
        if self._terminal_failure is not None:
            raise RuntimeError(self._terminal_failure)
        if self.thread is not None or self.loop.is_running():
            raise RuntimeError("stop the collector before starting it again")
        if not self.loop.is_closed() and any(not task.done() for task in asyncio.all_tasks(self.loop)):
            raise RuntimeError("finish collector cleanup before starting it again")

    def start(self) -> bool:
        self._ensure_startable()
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        self.loop.call_soon(ready.set)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return ready.wait(5)

    def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        self.subscription_future = asyncio.run_coroutine_threadsafe(self._transport.subscribe(handler), self.loop)
        self.subscription_future.add_done_callback(self._log_subscription_result)

    def _log_subscription_result(self, completed: Future) -> None:
        if completed is not self.subscription_future or completed.cancelled():
            return
        error = completed.exception()
        if error is not None:
            logger.error(
                "Collector subscription failed for %s",
                type(self._transport).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )

    def get_status(self) -> dict:
        """Describe subscription activity; endpoint readiness is a separate fact."""
        future, thread, cleanup = self.subscription_future, self.thread, self.cleanup_future
        error = self._terminal_failure
        if error is None and future is not None and future.done() and not future.cancelled():
            exception = future.exception()
            if exception is not None:
                error = f"{type(exception).__name__}: {exception}"
        return {
            "active": bool(
                future is not None
                and not future.done()
                and thread is not None
                and thread.is_alive()
                and cleanup is None
            ),
            "error": error,
        }

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            if not any(not task.done() for task in asyncio.all_tasks(self.loop)):
                self.loop.close()

    async def _cancel_and_drain(self) -> None:
        try:
            self._transport.stop()
        finally:
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() if not task.done() and task is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _stop_after_cleanup(self, completed: Future) -> None:
        # This callback runs after the result has crossed into the caller's Future.
        self.loop.call_soon_threadsafe(self.loop.stop)

    def stop(self, timeout: float) -> None:
        if self._terminal_failure is not None:
            return
        if self.cleanup_future is None:
            if self.loop.is_running():
                self.cleanup_future = asyncio.run_coroutine_threadsafe(self._cancel_and_drain(), self.loop)
                self.cleanup_future.add_done_callback(self._stop_after_cleanup)
            elif self.loop.is_closed():
                return
            else:
                self._transport.stop()
                if self.thread is not None:
                    self.loop.call_soon_threadsafe(self.loop.stop)

        cleanup_error: Exception | None = None
        if self.cleanup_future is not None:
            if (
                not self.cleanup_future.done()
                and self.thread is not None
                and self.thread.ident is not None
                and not self.thread.is_alive()
            ):
                self.thread.join()
                if self.loop.is_running():
                    raise RuntimeError("collector event loop is still running after its owner exited")
                if not self.loop.is_closed():
                    # Terminate the failed loop; pending tasks have not completed cleanup.
                    self.loop.close()
                self._terminal_failure = "collector event loop exited before cleanup completed; replace the collector"
                self.thread = None
                self.subscription_future = None
                self.cleanup_future = None
                raise RuntimeError(self._terminal_failure)
            try:
                self.cleanup_future.result(timeout=timeout)
            except FutureTimeoutError as exc:
                if not self.cleanup_future.done():
                    raise TimeoutError("collector cleanup timed out") from exc
                cleanup_error = self.cleanup_future.exception()
            except Exception as exc:
                cleanup_error = exc

        self._join_and_close(timeout)
        self.thread = None
        self.subscription_future = None
        self.cleanup_future = None
        if cleanup_error is not None:
            raise RuntimeError("collector cleanup did not finish") from cleanup_error

    def _join_and_close(self, timeout: float) -> None:
        if self.thread is not None:
            self.thread.join(timeout=min(10, timeout))
            if self.thread.is_alive():
                raise TimeoutError("collector event loop did not stop")
        if self.loop.is_running():
            raise TimeoutError("collector event loop is still running")
        if not self.loop.is_closed():
            if any(not task.done() for task in asyncio.all_tasks(self.loop)):
                raise RuntimeError("collector event loop still has pending tasks")
            self.loop.close()
