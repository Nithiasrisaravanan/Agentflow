"""
app/main.py — FastAPI application factory.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.cache import close_redis, get_redis
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("AgentFlow starting", env=settings.app_env, model=settings.openai_model)
    # Eagerly init Redis on startup
    try:
        await get_redis()
    except Exception as exc:
        logger.warning("Redis unavailable at startup (continuing anyway)", error=str(exc))
    yield
    await close_redis()
    logger.info("AgentFlow shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AgentFlow",
        description=(
            "An LLM-powered task-automation agent built with FastAPI, "
            "OpenAI function calling, async orchestration, and Redis caching."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"service": "AgentFlow", "docs": "/docs"})

    return app


app = create_app()
