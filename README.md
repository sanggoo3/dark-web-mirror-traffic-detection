# Dark Web Mirror Traffic Detection

Data and code supporting **Detecting Dark Web Mirror Sites on the ClearNET: An Encrypted Traffic Classification Approach** (revised Korean manuscript).

**Use release [`v2.0.1`](https://github.com/sanggoo3/dark-web-mirror-traffic-detection/releases/tag/v2.0.1) with dataset `direction526-20261101`.** This is the reproducibility version for the revised manuscript.

## Availability

Processed, pseudonymized direction sequences, auxiliary relative timing data, split assignments, model configurations and recorded numerical results are public here. **Raw PCAP/PCAPNG files are not public for privacy and security reasons.** Original URLs, domain strings, IP addresses, capture dates and local source paths are not included. Anonymous site/domain relationships are retained to audit group separation. See [DATASET](docs/DATASET.md), [PREPROCESSING](docs/PREPROCESSING.md) and [availability statements](docs/AVAILABILITY_ko.md).

The primary input is the **first 526 eligible packet directions within a 20-second session, including the initial 20 packets**. Values are +1 client-to-server, -1 server-to-client, 0 right padding. Length 506 is only the ablation slice `q[20:526]`. Classification is not restricted to the first three seconds. **Table 3(a) contains historical statistics that differ from the corrected final data**; corrected numbers are supplied in [MANUSCRIPT_CORRECTIONS](docs/MANUSCRIPT_CORRECTIONS.md). Fixed comparison lengths remain 137/526/905/1276.

## Contents

| Location | Contents |
| --- | --- |
| `data/processed/` | Compressed numeric arrays; no raw network payload |
| `data/metadata/` | Anonymous session/site/group assignments and timing sample membership |
| `data/splits/` | Exact outer and inner assignments for both five-fold protocols |
| `configs/` | All 12-candidate grids, 100 fold/model winners, feature scalers and provenance |
| `src/darkmirror_repro/` | Models, tuning/training, locked test evaluation, preprocessing, statistics and timing |
| `results/recorded/` | 7,200 HPO trials, 700 final runs, validation/test probabilities and timed calls |
| `results/tables/` | Recalculated manuscript tables and Figure 3 data/plot |
| `docs/` | Protocol, limitations, corrections, checks and availability wording |

Only experiments reported in this manuscript are included. Historical fixed holdout, masked/shuffled controls, statistics-only and whole-session volume-fusion experiments are excluded. See [EXPERIMENTS](docs/EXPERIMENTS.md) for the exact table-to-command mapping.

For a Korean walkthrough of the modules and execution flow, see [CODE_STRUCTURE_ko](docs/CODE_STRUCTURE_ko.md).

## Install

Python 3.12.7 was used. Create a separate environment. For recorded CUDA 12.1 training, install PyTorch from its [official historical wheel index](https://docs.pytorch.org/get-started/previous-versions/):

```console
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

PyTorch CPU is sufficient for verification and recorded-result reaggregation. The reference training entry point requires CUDA. Run commands from the repository root; keep the repository data beside the editable package. A different data location can be supplied using `python -m darkmirror_repro --root PATH ...`.

## Verify and Recalculate

```console
python -m darkmirror_repro verify
python -m unittest discover -s tests -v
python -m darkmirror_repro reproduce --out artifacts/reproduced
```

`verify` checks the versioned SHA-256 manifest, array encoding, padding and split disjointness. `reproduce` independently recalculates metrics, validation thresholds, HPO winners, seed intervals, 2,000-replicate conditional domain intervals, timing summaries and Figure 3. It asserts agreement with the released reference numbers. It **does not retrain**. `--skip-bootstrap` provides a faster audit without claiming to recalculate domain intervals.

## Retrain

```console
python -m darkmirror_repro tune --out artifacts/rerun
python -m darkmirror_repro train --out artifacts/rerun
python -m darkmirror_repro evaluate --out artifacts/rerun
python -m darkmirror_repro summarize --out artifacts/rerun
```

This executes 7,200 inner-HPO fits and 700 final fits, followed by batch-one test evaluation. For retraining with the already recorded validation-selected settings, skip HPO using `train --recorded-choices`. Commands support `--split`, `--model`, `--fold` filters; training also supports `--seed`. Such a filtered run is a smoke/subset experiment, not a complete paper rerun. Outputs go to ignored `artifacts/`; released results are never overwritten. No model checkpoint weights are distributed; train them locally.

Numerical equality after retraining is not guaranteed across GPU, driver or library versions. The validation checks include replay of the original checkpoints for all released model/condition combinations, but not a new full 7,900-fit rerun. The cohort was inspected during prior exploratory development; group separation prevents shared-domain leakage within the supplied splits, not all possible historical development bias or proof of external generalization.

## Processing Time

Recorded timing can be reaggregated publicly. New end-to-end PCAP preprocessing measurements require separately held captures and client-IP metadata, so this part cannot be independently repeated from processed sequences alone. [LATENCY](docs/LATENCY.md) specifies the exact timing boundaries and private-input benchmark command. Reported processing excludes packet collection wait and model loading; it is not live wall-clock detection delay.

## Version and Licensing

`data/DATASET_VERSION.json` identifies every data file by SHA-256; `release_manifest.json` identifies the entire snapshot (excluding itself). Git tag `v2.0.1` pins source and dataset together. Cite this release rather than the changing default branch.

Code: [MIT](LICENSE). Data, results, metadata, configurations and documentation: [CC BY-NC 4.0](DATA_LICENSE.md). Cite the paper title and the exact repository tag/dataset version; publication DOI is not supplied until assigned.
