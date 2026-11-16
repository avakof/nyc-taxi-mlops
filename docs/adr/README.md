# Architecture decision records

`0001-initial-architecture.md` is one combined ADR covering the baseline design, one section per area (data/features, split, model, serving, packaging, monitoring, retraining/rollback intent). This is the accepted variant for a system's starting design: normally an ADR is one-decision-per-file, but bundling the initial baseline into one doc avoids ten near-empty files for decisions made together, in the same sitting, by the same two people.

Every decision made **after** the baseline gets its own numbered file (`0002-short-title.md`, ...), immutable once accepted — never edit an old ADR's decision, write a new one that supersedes it. This applies especially to drift thresholds, retraining-gate tuning, and anything decided in response to an actual incident. Use [`template.md`](template.md). Status moves from `Proposed` → `Accepted` (or `Superseded by 000N`).

An ADR documents *why*, not incident timelines. When a decision's consequences later surface as an operational incident (drift fires, a release gets rolled back), write that up as a [post-mortem](../postmortems/template.md) and link it back from the relevant ADR's Consequences section.

For the presentation, the ADR set is the live-demo backbone: walk through the numbered decisions in order, call out tradeoffs, and close with a "what we'd do with more time" entry in the last one.
