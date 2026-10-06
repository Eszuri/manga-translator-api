"""Serialize synchronous image/GPU work without blocking the HTTP event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json

from app.core.request_limits import current_image_request
from app.core.config import settings


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-worker")
_decode_executor = ThreadPoolExecutor(
    max_workers=settings.MAX_IMAGE_REQUESTS, thread_name_prefix="image-decode"
)


async def run_image_task(function, *args, **kwargs):
    return await _run_native_task(_executor, function, *args, **kwargs)


async def run_decode_task(function, *args, **kwargs):
    return await _run_native_task(_decode_executor, function, *args, **kwargs)


async def _run_native_task(executor, function, *args, **kwargs):
    lease = current_image_request.get()
    if lease is not None:
        lease.start_task()
    try:
        future = executor.submit(partial(function, *args, **kwargs))
    except BaseException:
        if lease is not None:
            lease.finish_task()
        raise
    if lease is not None:
        future.add_done_callback(lambda _: lease.finish_task())
    return await asyncio.wrap_future(future)


async def stream_with_heartbeats(events, interval=10.0):
    """Keep clients informed while a stage runs or waits for the GPU worker."""
    pending = None
    try:
        while True:
            pending = asyncio.create_task(anext(events))
            while True:
                done, _ = await asyncio.wait({pending}, timeout=interval)
                if done:
                    break
                yield json.dumps({"stage": "heartbeat"}) + "\n"
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield event
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await events.aclose()
