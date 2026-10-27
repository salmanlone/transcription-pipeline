"""Exceptions that decide the HTTP status. Anything else that escapes is a server bug (500)."""


class InvalidInputError(ValueError):
    """The caller's file or options are bad (API: 422). Kept separate from plain ValueError so a
    ValueError raised by a bug in our code or a library is reported as a server error, not blamed
    on the client."""


class ModelBusyError(RuntimeError):
    """All model slots are in use by other requests; the caller should retry later (HTTP 503)."""
