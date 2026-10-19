"""Application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraudlens import __version__
from fraudlens.api.routes import etl, health, models, predictions, schedules
from fraudlens.monitoring.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("fraudlens.api")
    logger.info("api_starting", extra={"version": __version__})
    yield
    logger.info("api_stopping")


app = FastAPI(
    title="FraudLens",
    description="Transaction risk detection, ETL and monitoring API.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(etl.router)
app.include_router(schedules.router)
app.include_router(models.router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logging.getLogger("fraudlens.api").info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the detail, return a safe and concise response."""
    logging.getLogger("fraudlens.api").exception("unhandled_exception", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
