import asyncio
import concurrent.futures
import functools
import inspect
import time
from contextlib import contextmanager


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    """Accumulate the elapsed wall time of the ``with`` block into ``timing_raw[name]``.

    Drop-in replacement for ``verl.utils.profiler.simple_timer`` -- same
    signature and accumulation semantics, but uses ``time.perf_counter``
    directly so we don't have to drag in ``codetiming`` (or ``verl``)
    just to time a block of code.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timing_raw[name] = timing_raw.get(name, 0.0) + elapsed


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def auto_await(func):
    """Auto await a coroutine function.

    Handles three cases:
    1. When the decorated function is called with await: returns the coroutine
       so the caller can await it.
    2. When called directly and there is no running event loop: runs the
       coroutine with asyncio.run() and returns the result.
    3. When called directly and the event loop is already running: runs the
       coroutine (e.g. in a thread pool to avoid deadlock) and returns the result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)

        if not inspect.iscoroutine(coro):
            return coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Case 1: No running loop -> run with asyncio.run()
        if loop is None:
            return asyncio.run(coro)

        # Case 2: Running loop -> return coro if caller will await
        caller_frame = inspect.currentframe()
        if caller_frame is not None:
            caller_frame = caller_frame.f_back
        caller_is_async = caller_frame is not None and (caller_frame.f_code.co_flags & inspect.CO_COROUTINE) != 0
        if caller_is_async:
            return coro

        # Case 3: Running loop -> run coro in thread pool
        # (cannot block the loop thread without deadlock)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return wrapper
