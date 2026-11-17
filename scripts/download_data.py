"""Download TLC trip-record data and the zone lookup, with a frozen hash manifest.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --force            # re-download existing files
    uv run python scripts/download_data.py --accept-change     # allow an upstream byte change
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
MANIFEST_PATH = ROOT / "data" / "manifest.json"


def load_params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"skip (already downloaded): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.replace(dest)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def previous_hash(existing_manifest: dict, rel_path: str) -> str | None:
    for entry in existing_manifest.get("files", []):
        if entry["file"] == rel_path:
            return entry["sha256"]
    return None


def build_entry(url: str, dest: Path, citation: str) -> dict:
    return {
        "url": url,
        "file": relative(dest),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "citation": citation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the file already exists",
    )
    parser.add_argument(
        "--accept-change",
        action="store_true",
        help="allow writing a manifest entry whose hash differs from the previous one",
    )
    args = parser.parse_args()

    params = load_params()
    data_cfg = params["data"]
    source = data_cfg["source"]
    taxi_type = data_cfg["taxi_type"]
    base_url = source["base_url"]
    citation = source["citation"]

    existing_manifest = (
        json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    )

    targets = [
        (
            f"{base_url}/{source['zone_lookup_path']}",
            EXTERNAL_DIR / "taxi_zone_lookup.csv",
        )
    ]
    for month in data_cfg["months"]:
        trip_path = source["trip_data_path"].format(taxi_type=taxi_type, month=month)
        targets.append(
            (
                f"{base_url}/{trip_path}",
                RAW_DIR / f"{taxi_type}_tripdata_{month}.parquet",
            )
        )

    entries = []
    for url, dest in targets:
        download(url, dest, args.force)
        entry = build_entry(url, dest, citation)
        prior = previous_hash(existing_manifest, entry["file"])
        if prior is not None and prior != entry["sha256"] and not args.accept_change:
            raise SystemExit(
                f"hash changed for {entry['file']}: expected {prior}, got {entry['sha256']}. "
                "Re-run with --accept-change if this is an intentional upstream update."
            )
        entries.append(entry)

    manifest = {
        "citation": citation,
        "info_page": source["info_page"],
        "files": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written to {relative(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
