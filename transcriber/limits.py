"""Request-size limit enforced at the ASGI boundary, before the framework parses the upload."""
from __future__ import annotations

import json
import logging

from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("transcriber.api")


class MaxBodySizeMiddleware:
    """Pure ASGI wrapper placed OUTSIDE the FastAPI app (see api.py), so it does not
    depend on how FastAPI/Starlette turn exceptions into responses.

    - Content-Length present and too large -> 413 sent immediately; the app never runs.
    - Content-Length missing (chunked) or understated -> bytes are counted as they arrive.
      The moment the limit is passed, this wrapper sends the 413 itself, tells the app the
      client disconnected (so it stops reading), drops anything the app tries to send
      afterwards, and swallows the resulting error. At most ~max_bytes reaches the
      framework's upload spool.
    - Edge case: if a route has ALREADY started its response before the limit is hit, a
      413 can no longer be sent (the status line is gone). The limiter then still stops
      reading, and drops the rest of the app's response, so the server closes the
      connection with an incomplete response - the client sees a failed request, never a
      response that looks complete. Routes that never read the body are not affected:
      the limiter only counts bytes the app actually pulls.
    A reverse proxy limit (e.g. nginx client_max_body_size) should sit in front as well.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > self.max_bytes
            except ValueError:
                return await self._reject(send, 400, "Invalid Content-Length")
            if too_big:
                return await self._reject(send, 413, "Request too large")

        received = 0
        rejected = False
        response_started = False

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:                       # we already answered with 413; ignore the app
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    if not response_started:
                        await self._reject(send, 413, "Request too large")
                    else:
                        log.warning("Request body passed %d bytes after %s %s had already started "
                                    "its response; closing the connection with an incomplete response",
                                    self.max_bytes, scope.get("method"), scope.get("path"))
                    return {"type": "http.disconnect"}   # makes the app stop reading the body
            return message

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected:                   # a real error unrelated to the size limit
                raise

    @staticmethod
    async def _reject(send: Send, status: int, detail: str) -> None:
        """Send a complete JSON error response directly over ASGI."""
        log.warning("Rejected request before the app: %d %s", status, detail)
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
