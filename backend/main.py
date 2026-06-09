"""
IntelliVoice — FastAPI Application Entry Point

Sets up the FastAPI app with middleware, routes, and lifecycle events.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from config import get_settings, setup_logging
from config.logging_config import get_logger
from backend.core.gpu_manager import GPUManager
from backend.core.pipeline import AudioPipeline
from backend.api.routes import audio_ws, chat, health

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("starting_intellivoice", env=settings.app_env)

    # Initialize GPU manager
    gpu_manager = GPUManager()
    app.state.gpu_manager = gpu_manager
    gpu_manager.log_gpu_info()

    # Initialize the audio pipeline
    pipeline = AudioPipeline(gpu_manager=gpu_manager)
    app.state.pipeline = pipeline

    # Load all pipeline models (VAD, noisereduce, ASR, LLM, TTS)
    await pipeline.initialize()

    logger.info("intellivoice_ready", port=settings.port)

    yield

    # Shutdown
    logger.info("shutting_down")
    await pipeline.shutdown()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Multilingual Real-Time Speech-to-Speech AI Assistant",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(audio_ws.router, tags=["Audio WebSocket"])
    app.include_router(chat.router, tags=["Chat"])

    # Serve frontend static files
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/app/index.html")

        app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info("serving_frontend", path=str(frontend_dir))

    return app


# Create the app instance
app = create_app()
