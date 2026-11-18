# 0006. Serving API contract

Status: Proposed

## Context

The methodology requests three endpoints, while ADR-0001 specifies
different paths. We need one explicit contract for implementation,
integration tests, and the demonstration.

## Decision

Use these endpoints for the initial service:

- POST /forecast
- GET /health
- GET /model-info

POST /forecast accepts:
- zone_id: an integer from the frozen supported-zone list.
- forecast_origin: a timezone-aware, hour-aligned timestamp.

The service constructs features from complete observations available
by forecast_origin, using the shared feature implementation.

A successful forecast returns:
- request_id
- zone_id
- target_start
- target_end
- expected_pickups
- model_version
- release_id

expected_pickups must be finite and nonnegative; it may be fractional.

GET /health reports readiness:
- 200 when the model and history source are usable.
- 503 when required resources are unavailable.

Each forecast separately checks its requested history window.

GET /model-info reports the loaded model name, numeric version,
source run ID, and release ID. It returns 503 if no model is loaded.

Error behavior:
- 422: invalid request, unsupported zone, or invalid timestamp.
- 503: model or required historical observations are unavailable.
- 500: unexpected inference failure or invalid model output.

Missing or incomplete observations must never become artificial zeros.
Load an exact model version once; changing an alias alone must not
change the running model.

## Alternatives considered

Retain ADR-0001's endpoint paths: choose the methodology's three
requested paths for the initial implementation.

Accept caller-provided lag features: choose server-built features
to keep preprocessing controlled and consistent.

## Consequences

Integration tests will verify this contract.
Health indicates readiness, not forecasting accuracy.
This proposal replaces only ADR-0001's endpoint-path decision;
the existing forecasting and model-identity rules remain applicable.