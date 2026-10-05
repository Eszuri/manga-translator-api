"""Serialize synchronous image/GPU work without blocking the HTTP event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json


# A single worker prevents concurrent Run calls on shared DirectML sessions.
# Cancelling an HTTP request cannot release the worker while inference still runs.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-worker")


async def run_image_task(function, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(
        _executor, partial(function, *args, **kwargs)
    )


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
