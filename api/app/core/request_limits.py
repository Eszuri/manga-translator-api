from contextvars import ContextVar
from threading import Lock

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class ImageRequestLease:

    def __init__(self, release):
        self._release = release
        self._lock = Lock()
        self._tasks = 0
        self._finished = False
        self._released = False

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


current_image_request: ContextVar[ImageRequestLease | None] = ContextVar(
    "current_image_request", default=None
)


class ImageRequestLimitMiddleware:
    def __init__(self, app: ASGIApp, max_requests: int, api_prefix: str):
        self.app = app
        self.max_requests = max_requests
        self._lock = Lock()
        self._active = 0
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

        lease = ImageRequestLease(self._release)
        token = current_image_request.set(lease)
        try:
            await self.app(scope, receive, send)
        finally:
            current_image_request.reset(token)
            lease.finish_request()
