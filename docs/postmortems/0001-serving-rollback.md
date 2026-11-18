# Incident: injected serving-code failure and verified rollback

Scenario: isolated bad-code deployment drill `20261118T095307Z-3c11e0`. Related decision: [ADR-0007](../adr/0007-release-rollback.md). Recorded by Ali's executed drill on 2026-11-18.

## Impact

The deliberately faulty image returned HTTP 500 for the forecast probe in a separate Compose project on port 8002. The normal service on port 8001 was not recreated. One known zone/hour was probed: zone 237, target `[2020-02-20T15:00:00Z, 2020-02-20T16:00:00Z)`. This simulation timestamp is distinct from execution wall time. The injected wrapper affects forecast requests generally; the measured sample contains one probe per phase.

## Timeline (UTC wall clock)

| Event | UTC wall time | Health | Forecast | Probe latency |
| --- | --- | ---: | ---: | ---: |
| Healthy baseline | 2026-11-18T09:53:36.776788+00:00 | 200 | 200 | 64.06 ms |
| Failure detected | 2026-11-18T09:53:57.330619+00:00 | 200 | 500 | 4.87 ms |
| Recovery verified | 2026-11-18T09:54:18.806032+00:00 | 200 | 200 | 35.55 ms |

Measured rollback duration: **21.47 wall-clock seconds**, below the provisional 120-second target for this run. These individual latency samples are not a latency percentile or general availability claim.

## Root cause and detection

A derived image deliberately installed an HTTP wrapper returning `Injected deployment failure` for `/forecast`. The original model and history were still loadable, so resource health stayed 200. The forecast probe detected the functional failure. This was a software deployment fault, not observed demand drift.

## Decision and response

Restore the retained image and compatible bundle, recreate without rebuilding/pulling, verify actual image/platform identity and `/model-info`, then repeat the known forecast. No retraining or Registry alias change was needed. Forecasts returned **586.0 pickups** both before and after recovery. History bytes were identical; observations were not rewound.

## Artifact identities

- Model: `nyc-taxi-demand-forecaster`, version `1`.
- Source run: `d231eecba6544db5ae5b511841b5eb17`.
- Good/restored image: `sha256:19f4ddc35f37cfabb349436493011252b72594842138a30cac3ea7d6bbfa94ef`.
- Faulty image: `sha256:e371025042b400d9cfbe9735fdf8ce0bfac5a3c70cfea52b447e87151fa5c0cb`.
- Good/restored release: `sha256:7b7bd49f39934658e7e15b9664eb9a9cd1134e4a1cca664b29b408b95687ad8d`.
- Faulty release: `sha256:8570d0ae5fc7fc41e9a597edd73594c4fadb37dc9314518b544af17886012cce`.
- Unchanged bundled history SHA-256: `4a78426c6b4b04db2c7e54d32a169ef230c1c76a08d8d649e10564dc1d39ae72`.
- Prototype Git identity: `bce6d07df67075254f8ab71e1082789f7483b5a2`, `git_dirty=true`. The final delivery bootstrap requires a clean commit; this earlier measured drill is retained as recorded evidence.

Full local evidence: `reports/rollback/20261118T095307Z-3c11e0/result.json` and its `known-good-deployment.json`. Generated binaries and full reports are ignored; the tracked scripts reproduce the drill with new run IDs and measured timings.

## Corrective actions and acceptance checks

| Action | Acceptance check |
| --- | --- |
| Include functional forecasts in deployment verification | Smoke rejects a 500 even when `/health` is 200 |
| Retain immutable image and compatible bundle | Restore exact image/platform/release identity |
| Verify a known result after recovery | Original and recovered prediction both equal 586.0 |
| Freeze final delivery provenance | Bootstrap runs from a clean Git checkout and reports `git_dirty=false` |
| Keep demand-drift conclusions separate | Julia's drift/retraining incident supplies its own detection and quality evidence |
