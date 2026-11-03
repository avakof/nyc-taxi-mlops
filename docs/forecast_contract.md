## Objective

Predict the expected number of yellow-taxi pickups in a supported
NYC pickup zone during the next hour.

## Forecast timing

T is both the forecast issuance time and the target-hour start.

The target is the pickup count during [T, T + 1 hour).

Intervals include their start and exclude their end.

## Eligible observations

A historical count may be used only when:
- Its interval ends at or before T.
- It is available at or before T.
- Its source interval is marked complete.

Historical replay assumes completed hourly counts become available
at hour closure. This is a simulation assumption, not a live TLC feed.

## Initial features

Historical pickup counts at lags of 1, 2, 3, 24, and 168 elapsed hours.

Rolling means over preceding 3, 24, and 168 completed hours.

Zone identity and New York calendar features.

Features must never include the target-hour count or later observations.

## Time representation

Use UTC for interval identity and elapsed-hour calculations.

Use America/New_York for local calendar features.

Test daylight-saving transitions explicitly.

## Missing observations

Zero means a complete hour with no accepted pickups.

Missing or incomplete data must not automatically become zero.

The initial service rejects forecasts requiring unavailable history.

## Supported zones

Select 20 zones using training-period pickup volume only.

Freeze the selected zone IDs before validation and testing.

## Initial chronological splits

Split by target-hour start, consistently across all zones.

The following boundaries are New York local midnight.
Start dates are inclusive; end dates are exclusive.

Warmup:      2020-01-01 to 2020-01-08

Training:    2020-01-08 to 2020-02-03

Calibration: 2020-02-03 to 2020-02-17

Test:        2020-02-17 to 2020-03-01

Later replay:2020-03-01 to 2020-05-01

## Evaluation timing

A forecast may be scored only after its target interval closes
and its complete observed count becomes available.

Missing labels must be reported through coverage metrics.

## Example

At 2020-02-20 10:00 America/New_York:
- Predict pickups during 10:00–11:00.
- The 09:00–10:00 count is eligible if complete and available.
- The 10:00–11:00 count is unavailable at issuance.
- Score the forecast at or after 11:00 when its label is available.