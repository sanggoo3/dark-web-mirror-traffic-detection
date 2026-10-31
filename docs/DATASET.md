# Dataset: direction526-20261101

This is the corrected protocol-filter cohort used for the revised manuscript's 526-input performance results. It is not the historical `v1.0.0` post-handshake dataset. Exact file fingerprints are in `data/DATASET_VERSION.json`; the original internal canonical numeric archive SHA-256 is `d0af74cb9714dcac39e2248402097ec17f33977e4c2c250e3f1b10a65b2e2cbb`. The released archive has different key names/packaging and its own fingerprint; do not confuse the two.

## Cohort and Labels

| Label | Class | Sessions | Site labels |
| --- | --- | ---: | ---: |
| 0 | Clear web | 2300 | 23 |
| 1 | Dark mirror web | 2300 | 23 |
| Total | | 4600 | 46 |

Each site label has 100 sessions. Aliases mapping to the same registrable domain share one anonymous domain ID. Group-aware splits hold all sessions from a domain together. The anonymous IDs preserve the verified original assignments; spelling and URL-to-domain reverse mappings are deliberately not public. All arrays and CSVs use the same zero-based `index`. `session_id` is a public pseudonym, not a filename or capture identifier.

## Array Schema

Load with `numpy.load(path, allow_pickle=False)`.

`data/processed/direction_sequences.npz`:

| Key | Shape | Type / meaning |
| --- | --- | --- |
| `prefix` | (4600, 1276) | int8; first eligible directions within 20 seconds, right padded with 0 |
| `relative_times_s` | (4600, 1276) | float64; seconds relative to first captured frame; NaN at padding |
| `window_counts` | (4600, 4) | int64; eligible counts at actual <=1, <=3, <=5, <=10 seconds |
| `retained_count` | (4600,) | int32; eligible count over the entire <=20-second session, including beyond prefix |
| `y` | (4600,) | int64; 0 Clear / 1 Dark mirror |
| `groups` | (4600,) | Unicode anonymous domain IDs |

`light_statistics_526.npz` contains `light`, shape (4600,16), float32. It is redundant with the first 526 directions and relative times and is released for exact feature checking. **Relative times and these features are auxiliary inputs only for the explicitly identified fusion experiment; the baseline uses only `prefix`.** No absolute capture timestamps are released.

`full20s_direction_descriptives.csv` adds whole-session direction-run counts for the paper's descriptive traffic-volume comparison. Whole-session counts and run counts are **not baseline/fusion classifier inputs**. The prefix alone cannot reconstruct run counts beyond position 1276.

## Input Construction

```python
baseline = prefix[:, :526]                  # includes first 20; length 526
post_handshake = prefix[:, 20:526]           # excludes once; length 506
reduction = prefix[:, :300]                  # includes first 20; length 300
handshake_only = zeros((4600, 526))          # put prefix[:, :20] at positions 0:20
```

The minimum eligible 20-second count is 532, so no primary 526-input sample is padded. Longer 905/1276 inputs have class-dependent padding incidence; zero positions can reveal available traffic length. Fixed-length truncation reduces whole-session volume dependence but does not guarantee volume invariance or removal of all confounding.

CNN/TCN/Transformer inputs are float32 `(N,1,L)`; GRU/BiGRU/LSTM receive the same values reshaped to `(N,L,1)`. Reshaping introduces no new information. No packet size, URL, hostname or payload is a baseline feature.

## Metadata and Splits

- `sessions.csv`: row index, anonymous session/site/domain IDs, class and eligible 20-second count.
- `sites_and_domain_groups.csv`: pseudonymous site-to-group assignments, including aliases.
- `data/splits/{group_aware,session_stratified}/split_assignments.csv`: exact ordered outer `fit`, `validation`, `test` indices.
- `inner_split_assignments.csv`: exact ordered inner `fit`/`validation` indices, all inside the outer fit subset.
- `latency_sample_assignments.csv`: fixed representative input indices used for reported timing.

The supplied assignments, rather than a regenerated split on renamed domain strings, are the authoritative evaluation partitions. A renamed string can alter sorting/tie decisions in a group splitter. All learning seeds reuse these fixed partitions. Session-stratified splitting deliberately permits the same domain in fit and test; it measures new sessions from previously observed sites, unlike the group-aware protocol.

## Privacy and Limitations

The release is pseudonymized: original URLs/domains, IPs, capture dates, raw paths, payloads, PCAP files and private mapping tables are excluded. Stable group membership, labels, counts, relative timings and model probabilities remain by design. These can support traffic fingerprinting or linkage if combined with external information; pseudonymization is not a claim of guaranteed irreversible anonymity. Do not attempt re-identification. No raw-PCAP collection or security-sensitive browsing logs are necessary to run the released learning experiments.

Table 3(a)'s historical statistics must be corrected using this version; see [MANUSCRIPT_CORRECTIONS](MANUSCRIPT_CORRECTIONS.md). The fixed lengths are pre-existing design constants, not newly estimated training-fold medians.
