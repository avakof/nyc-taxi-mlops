"""Reproduction must preserve source identity when downloaded bytes are unchanged."""

import json
import sys

import download_data
import yaml
from download_data import preserve_snapshot_timestamp


def test_unchanged_snapshot_preserves_its_timestamp():
    previous = {
        "file": "data/raw/example.parquet",
        "sha256": "same",
        "size_bytes": 5,
        "url": "https://example.test/data",
        "citation": "test",
        "downloaded_at": "first-download",
    }
    entry = {**previous, "downloaded_at": "later-download"}
    assert preserve_snapshot_timestamp(entry, {"files": [previous]}) == previous


def test_changed_bytes_keep_new_timestamp():
    previous = {
        "file": "data/raw/example.parquet",
        "sha256": "old",
        "downloaded_at": "first-download",
    }
    entry = {**previous, "sha256": "new", "downloaded_at": "later-download"}
    assert preserve_snapshot_timestamp(entry, {"files": [previous]}) == entry


def test_repeated_download_preserves_manifest_bytes(tmp_path, monkeypatch):
    params = {
        "data": {
            "taxi_type": "yellow",
            "months": ["2020-01"],
            "source": {
                "base_url": "https://example.test",
                "zone_lookup_path": "zones.csv",
                "trip_data_path": "{taxi_type}_{month}.parquet",
                "citation": "synthetic test source",
                "info_page": "https://example.test/info",
            },
        }
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params))
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(download_data, "ROOT", tmp_path)
    monkeypatch.setattr(download_data, "PARAMS_PATH", params_path)
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(download_data, "EXTERNAL_DIR", tmp_path / "external")
    monkeypatch.setattr(download_data, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(sys, "argv", ["download_data.py"])

    def fake_download(url, dest, force):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists():
            dest.write_bytes(dest.name.encode())

    monkeypatch.setattr(download_data, "download", fake_download)
    download_data.main()
    before = manifest.read_bytes()
    download_data.main()
    assert manifest.read_bytes() == before
    assert len(json.loads(before)["files"]) == 2
