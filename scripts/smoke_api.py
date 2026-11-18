"""Check a running API over HTTP; reusable for local and container smoke tests."""

import argparse
import json
import math
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--expected-release",
        type=Path,
        help="Compare the served identity with this release.json",
    )
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"status": "ready"}
        info = client.get("/model-info")
        info.raise_for_status()
        identity = info.json()
        if args.expected_release is not None:
            expected = json.loads(args.expected_release.read_text())
            assert identity["release_id"] == expected["release_id"], (
                "The API is serving a different release"
            )
            assert identity["git_commit"] == expected["git_commit"]
            assert identity["git_dirty"] == expected["git_dirty"]
        request = {"zone_id": 237, "forecast_origin": "2020-02-20T15:00:00Z"}
        forecast = client.post("/forecast", json=request)
        forecast.raise_for_status()
        prediction = forecast.json()
        assert (
            math.isfinite(prediction["expected_pickups"])
            and prediction["expected_pickups"] >= 0
        )
        assert prediction["model_version"] == identity["model_version"]
        assert prediction["release_id"] == identity["release_id"]
        invalid_zone = client.post("/forecast", json={**request, "zone_id": 999})
        assert invalid_zone.status_code == 422
        invalid_time = client.post(
            "/forecast", json={**request, "forecast_origin": "2020-02-20T15:00:00"}
        )
        assert invalid_time.status_code == 422
        missing_history = client.post(
            "/forecast", json={**request, "forecast_origin": "2020-01-01T05:00:00Z"}
        )
        assert missing_history.status_code == 503
    print(
        json.dumps(
            {
                "result": "PASS",
                "model_info": identity,
                "forecast": prediction,
                "checks": {
                    "health": 200,
                    "forecast": 200,
                    "unsupported_zone": 422,
                    "naive_timestamp": 422,
                    "missing_history": 503,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
