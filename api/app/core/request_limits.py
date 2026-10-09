from contextvars import ContextVar
import asyncio
from threading import Lock

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class ImageRequestLease:

    def __init__(self, release):
        self._release = release
        self._lock = Lock()
        self._tasks = 0
        self._finished = False
        self._released = False
        self._release_callbacks = []

    def add_release_callback(self, callback):
        with self._lock:
            if not self._released:
                self._release_callbacks.append(callback)
                return
        callback()

    def start_task(self):
        with self._lock:
            self._tasks += 1

    def finish_task(self):
        with self._lock:
            self._tasks -= 1
            self._release_if_finished()

    def finish_request(self):
        with self._lock:
            self._finished = True
            self._release_if_finished()

    def _release_if_finished(self):
        if self._finished and self._tasks == 0 and not self._released:
            self._released = True
            self._release()
            for callback in self._release_callbacks:
                callback()
            self._release_callbacks.clear()


current_image_request: ContextVar[ImageRequestLease | None] = ContextVar(
    "current_image_request", default=None
)


class ImageRequestLimitMiddleware:
    def __init__(self, app: ASGIApp, max_requests: int, api_prefix: str, max_upload_bytes: int):
        self.app = app
        self.max_requests = max_requests
        # Allow room for multipart boundaries and the small pipeline form fields.
        self.max_body_bytes = max_upload_bytes + 64 * 1024
        self._lock = Lock()
        self._active = 0
        # Admission is bounded; execution is serial in the single-worker server.
        self._processing_slot = asyncio.Semaphore(1)
        self._paths = {
            f"{api_prefix}/{endpoint}" for endpoint in (
                "detect/bubbles", "ocr/recognize", "ocr/recognize-crop",
                "translate/page", "translate/inpaint-page", "translate/inpaint-stream",
            )
        }

    def _release(self):
        with self._lock:
            self._active -= 1

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"].rstrip("/") not in self._paths):
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                oversized = int(value) > self.max_body_bytes
            except ValueError:
                continue
            if oversized:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Image request body exceeds the configured size limit."},
                )
                await response(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    # FastAPI preserves HTTPException raised while parsing a form.
                    # Reject this chunk before the multipart parser can spool it.
                    raise HTTPException(
                        status_code=413,
                        detail="Image request body exceeds the configured size limit.",
                    )
            return message

        with self._lock:
            accepted = self._active < self.max_requests
            if accepted:
                self._active += 1

        if not accepted:
            response = JSONResponse(
                status_code=503,
                content={"detail": "Image processing capacity is full. Retry after an active request finishes."},
                headers={"Retry-After": "3"},
            )
            await response(scope, receive, send)
            return

        try:
            await asyncio.wait_for(self._processing_slot.acquire(), timeout=300)
        except TimeoutError:
            self._release()
            await JSONResponse(
                status_code=503,
                content={"detail": "Timed out waiting in the image processing queue."},
                headers={"Retry-After": "3"},
            )(scope, receive, send)
            return
        except BaseException:
            self._release()
            raise

        loop = asyncio.get_running_loop()

        def release_processing_slot():
            self._release()
            # Native GPU tasks may finish after the client disconnects.
            loop.call_soon_threadsafe(self._processing_slot.release)

        lease = ImageRequestLease(release_processing_slot)
        token = current_image_request.set(lease)
        try:
            await self.app(scope, limited_receive, send)
        finally:
            current_image_request.reset(token)
            lease.finish_request()
