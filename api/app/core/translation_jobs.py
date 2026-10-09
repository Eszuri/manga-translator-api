"""Cooperative cancellation for the single-process image pipeline."""

import asyncio
from collections import OrderedDict
import time

from fastapi import HTTPException

from app.core.image_worker import run_image_task
from app.core.request_limits import current_image_request


JOB_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class JobCancelled(Exception):
    pass


class TranslationJob:
    def __init__(self, job_id):
        self.job_id = job_id
        self.claimed = False
        self.cancel_requested = False
        self.status = None
        self.settled = False
        self.finished_at = None
        self._async_task = None
        self._lease = None

    def claim(self):
        if self.claimed:
            raise HTTPException(status_code=409, detail="This translation job ID has already been used.")
        self.claimed = True
        self.status = None
        self.settled = False
        self.finished_at = None
        self._lease = current_image_request.get()
        if self._lease is not None:
            loop = asyncio.get_running_loop()
            self._lease.add_release_callback(lambda: loop.call_soon_threadsafe(self._settle))

    def request_cancel(self):
        if self.cancel_requested or self.status in ("done", "error"):
            return
        self.cancel_requested = True
        if self._async_task is not None:
            self._async_task.cancel()
        if not self.claimed:
            # Remember an upload/queued request that has not reached its handler.
            self.finish("cancelled")

    def check_cancelled(self):
        if self.cancel_requested:
            raise JobCancelled()

    async def run_native(self, function, *args, **kwargs):
        self.check_cancelled()
        try:
            result = await run_image_task(function, *args, **kwargs)
        except Exception:
            self.check_cancelled()
            raise
        # Never cancel the native worker; its return is the safe stage boundary.
        self.check_cancelled()
        return result

    async def run_async(self, function, *args, **kwargs):
        self.check_cancelled()
        task = asyncio.create_task(function(*args, **kwargs))
        self._async_task = task
        try:
            result = await task
        except asyncio.CancelledError:
            self.check_cancelled()
            raise
        except Exception:
            self.check_cancelled()
            raise
        finally:
            self._async_task = None
        self.check_cancelled()
        return result

    def finish(self, status):
        if self.status is None:
            self.status = status
        if self._lease is None:
            self._settle()

    def _settle(self):
        # Called only after the request and every native task release their lease.
        if self.status is None:
            self.status = "cancelled" if self.cancel_requested else "error"
        self.settled = True
        self.finished_at = time.monotonic()

    def snapshot(self):
        status = self.status or ("cancelling" if self.cancel_requested else "running")
        if status == "cancelled" and not self.settled:
            status = "cancelling"
        return {"job_id": self.job_id, "status": status, "settled": self.settled}


class TranslationJobs:
    def __init__(self, max_entries=1024, retention_seconds=600):
        self._jobs = OrderedDict()
        self.max_entries = max_entries
        self.retention_seconds = retention_seconds

    def _get_or_create(self, job_id):
        now = time.monotonic()
        for key, job in list(self._jobs.items()):
            if job.settled and now - job.finished_at >= self.retention_seconds:
                del self._jobs[key]
        if job_id in self._jobs:
            return self._jobs[job_id]
        if len(self._jobs) >= self.max_entries:
            # Never evict live work or cancellation tombstones: an in-flight
            # upload must still observe a cancellation accepted for its ID.
            raise HTTPException(status_code=503, detail="Translation job registry is full. Retry later.")
        job = TranslationJob(job_id)
        self._jobs[job_id] = job
        return job

    def register(self, job_id=None):
        # Legacy callers remain cancellable on disconnect and need no registry ID.
        job = self._get_or_create(job_id) if job_id is not None else TranslationJob(None)
        job.claim()
        return job

    def cancel(self, job_id):
        job = self._get_or_create(job_id)
        job.request_cancel()
        return job.snapshot()


translation_jobs = TranslationJobs()
