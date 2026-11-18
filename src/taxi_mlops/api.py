"""HTTP service for pinned model releases and causal historical replay."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from taxi_mlops.release import (
    HistoryUnavailable,
    checked_prediction,
    forecast_features,
    load_release,
)

logger = logging.getLogger("uvicorn.error")


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone_id: int = Field(strict=True, gt=0)
    forecast_origin: AwareDatetime

    @field_validator("forecast_origin", mode="before")
    @classmethod
    def timestamp_format(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("Use an ISO timestamp with an explicit UTC offset")
        return value

    @field_validator("forecast_origin")
    @classmethod
    def aligned_hour(cls, value):
        value = value.astimezone(UTC)
        if value.minute or value.second or value.microsecond:
            raise ValueError("forecast_origin must align with a UTC hour")
        return value


class ForecastResponse(BaseModel):
    request_id: str
    zone_id: int
    target_start: AwareDatetime
    target_end: AwareDatetime
    expected_pickups: float = Field(ge=0, allow_inf_nan=False)
    model_version: str
    release_id: str


def create_app(release_dir: str | Path | None = None) -> FastAPI:
    """Load one local release per application startup."""
    directory = Path(
        release_dir or os.getenv("TAXI_RELEASE_DIR", "models/final-serving")
    ).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.release = None
        app.state.model = None
        app.state.ready = False
        try:
            bundle = load_release(directory)
        except Exception:
            logger.exception("Release unavailable at %s", directory)
        else:
            app.state.release = bundle
            app.state.model = bundle.model
            app.state.ready = True
            logger.info("Loaded release %s", bundle.manifest["release_id"])
        try:
            yield
        finally:
            app.state.ready = False
            app.state.release = None
            app.state.model = None

    app = FastAPI(title="NYC taxi demand: historical replay", lifespan=lifespan)
    app.state.release = None
    app.state.model = None
    app.state.ready = False

    @app.get("/health", responses={503: {"description": "Not ready"}})
    def health() -> dict[str, str]:
        if not app.state.ready:
            raise HTTPException(
                status_code=503, detail="Forecasting resources are not ready"
            )
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info() -> dict:
        bundle = app.state.release
        if bundle is None:
            raise HTTPException(status_code=503, detail="No model release is loaded")
        return {
            key: bundle.manifest[key]
            for key in (
                "model_name",
                "model_version",
                "run_id",
                "release_id",
                "git_commit",
                "git_dirty",
                "history_policy",
            )
        }

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        bundle = app.state.release
        if bundle is None:
            raise HTTPException(
                status_code=503, detail="Forecasting resources are not ready"
            )
        if request.zone_id not in bundle.zones:
            raise HTTPException(status_code=422, detail="Unsupported pickup zone")
        try:
            inputs = forecast_features(
                bundle.history,
                bundle.params,
                bundle.zones,
                request.zone_id,
                request.forecast_origin,
            )
        except HistoryUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        try:
            prediction = checked_prediction(bundle.model, inputs)
        except Exception as error:
            logger.exception("Model inference failed")
            raise HTTPException(
                status_code=500, detail="Model inference failed"
            ) from error
        response = ForecastResponse(
            request_id=uuid4().hex,
            zone_id=request.zone_id,
            target_start=request.forecast_origin,
            target_end=request.forecast_origin + timedelta(hours=1),
            expected_pickups=prediction,
            model_version=bundle.manifest["model_version"],
            release_id=bundle.manifest["release_id"],
        )
        logger.info(
            "Forecast request_id=%s zone_id=%s origin=%s release_id=%s",
            response.request_id,
            response.zone_id,
            response.target_start.isoformat(),
            response.release_id,
        )
        return response

    return app
