"""Select the supported zones by training-period pickup volume, then freeze them.

Per the forecast contract: zone selection uses the training period only, and is frozen
before calibration/test are touched. Run once; re-run only if the training data changes.

Usage:
    uv run python scripts/select_zones.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taxi_mlops.config import load_params

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    params = load_params()
    n_zones = params["data"]["n_zones"]

    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    totals = train.groupby("zone_id").pickups.sum().sort_values(ascending=False)
    selected = totals.head(n_zones)

    zones = pd.read_csv(ROOT / "data" / "external" / "taxi_zone_lookup.csv").set_index("LocationID")
    out = [
        {
            "zone_id": int(zid),
            "pickups_train": int(count),
            "name": zones.loc[zid, "Zone"] if zid in zones.index else None,
            "borough": zones.loc[zid, "Borough"] if zid in zones.index else None,
        }
        for zid, count in selected.items()
    ]

    config_dir = ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    out_path = config_dir / "selected_zones.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"froze {len(out)} zones to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
