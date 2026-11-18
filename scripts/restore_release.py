"""Restore an explicitly selected retained deployment to an explicitly selected service."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rollback_demo import ROOT, restore_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--project", required=True, help="Compose project to recreate")
    parser.add_argument(
        "--port", type=int, required=True, help="Host port for the restored API"
    )
    args = parser.parse_args()
    restored = restore_deployment(args.record.resolve(), args.project, args.port)
    report = {
        "result": "PASS",
        "wall_time_utc": datetime.now(UTC).isoformat(),
        "record": str(args.record.resolve()),
        "project": args.project,
        "port": args.port,
        **restored,
    }
    directory = ROOT / "reports" / "restores"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6] + ".json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": "PASS",
                "evidence": str(output),
                "model_info": restored["model_info"],
                "recovery_seconds_wall": restored["recovery_seconds_wall"],
                "recovery_within_120_seconds": restored["recovery_within_120_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
