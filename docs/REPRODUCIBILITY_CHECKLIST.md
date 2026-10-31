# Release Checks

The release is limited to experiments in the revised manuscript. The original source/results were read without modification during packaging. The old `v1.0.0` Git tag is retained.

| Check | Scope |
| --- | --- |
| Numerical input identity | Group-aware and session-stratified sources have identical canonical arrays |
| Metadata | Aligned anonymous sessions, site labels and group assignments; no public reverse mapping |
| Input/feature identity | Included526 and post506 slices audited; all4600 light16 vectors recomputed exactly |
| Splits | Outer/inner session disjointness, group-aware domain disjointness, each session tested once per seed |
| HPO | All7200 trial records and100 configuration choices independently ranked |
| Thresholds/results | All700 final-run thresholds reproduced from validation only; all test metrics recalculated |
| Intervals | Five-seed statistics and2000-draw conditional domain intervals recalculated against references |
| Paired result | Group-aware ResNet1D-minus-TCN seed/domain intervals recalculated |
| Latency | Original three-repeat session/checkpoint summaries reaggregated, not remeasured from withheld captures |
| Original checkpoints | 28 split/model/condition combinations replayed on8inputs each, including all10families and paper variants |
| Training smoke | Small TCN, ROCKET-style and ResNet1D+light16 fits exercised without full retraining |
| Privacy | Public text/archives scanned for source paths, addresses, original site identifiers and excluded raw files |
| Integrity | Per-data-file SHA-256 and release-wide SHA-256 manifests |

See `validation_checks.json` and `results/tables/verification.json` for recorded outcomes. Checkpoint replay uses private locally retained weights; these weights are not publicly distributed. Public users can run unit tests and full result reaggregation, or retrain checkpoints themselves. Neither these checks nor new group-aware splits eliminate the cohort's historical exploratory reuse. Numerical agreement with recorded values is distinct from independent replication on new data.
