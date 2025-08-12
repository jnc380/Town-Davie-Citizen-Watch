# Lazy ASGI application loader for Vercel
from typing import Any, Awaitable, Callable

_cached_app = None


def _load_app():
    global _cached_app
    if _cached_app is None:
        from capstone.hybrid_rag_system import app as real_app
        _cached_app = real_app
    return _cached_app


async def app(scope: dict, receive: Callable[[], Awaitable[dict]], send: Callable[[dict], Awaitable[None]]):
    application = _load_app()
    await application(scope, receive, send) 