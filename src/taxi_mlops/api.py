"""HTTP entry point for the taxi forecasting service."""

from fastapi import FastAPI, HTTPException


def create_app() -> FastAPI:
    """Create a separate application instance for serving or testing."""
    app = FastAPI(title="NYC taxi demand")
    app.state.ready = False

    @app.get("/health", responses={503: {"description": "Not ready"}})
    def health() -> dict[str, str]:
        if not app.state.ready:
            raise HTTPException(
                status_code=503,
                detail="Forecasting resources are not ready",
            )
        return {"status": "ready"}

    return app
