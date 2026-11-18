"""Opt-in asynchronous diagnostics for subprocess regression tests."""

import asyncio
import gc

import pytest


@pytest.fixture
def no_unhandled_asyncio_errors(monkeypatch):
    original_run = asyncio.run
    diagnostics = []

    def checked_run(coro, **kwargs):
        async def checked():
            asyncio.get_running_loop().set_exception_handler(lambda loop, context: diagnostics.append(context))
            try:
                return await coro
            finally:
                gc.collect()
                await asyncio.sleep(0)

        result = original_run(checked(), **kwargs)
        gc.collect()
        assert not diagnostics, diagnostics
        return result

    monkeypatch.setattr(asyncio, "run", checked_run)
    yield
    gc.collect()
    assert not diagnostics, diagnostics
