# Detecting Dark Web Mirror Sites on the ClearNET

This repository is the reproducibility package for the paper:

**Detecting Dark Web Mirror Sites on the ClearNET: An Encrypted Traffic Classification Approach**

It contains the processed direction-only sequence data, code, and final result tables needed to reproduce the paper's experimental tables. Raw `pcap` or `pcapng` captures are intentionally not included; the shared dataset starts after protocol filtering and packet-direction encoding.

The released data are anonymized. Site/group identifiers, sample file identifiers, and client IP values in the metadata and `.npz` files have been replaced with synthetic IDs such as `clear_site_01`, `mirror_site_01`, `clear_sample_0001`, and `CLIENT_IP_REDACTED`. The anonymization preserves labels, sample counts, group counts, sequence values, and all table-reproduction behavior.

## Scope

The paper studies binary session-level classification between ordinary Clear Web access and Dark Web Mirror site access in a ClearNET browser environment. The model input excludes payloads, URLs, page content, packet sizes, and timing. Each sample is represented only by packet direction:

| Value | Meaning |
|---:|---|
| `+1` | client-to-server packet |
| `-1` | server-to-client packet |
| `0` | padding |

Labels follow the paper definition:

| Label | Meaning |
|---:|---|
| `0` | Clear Web |
| `1` | Dark Web Mirror site |

## Repository Layout

```text
data/processed/                  processed direction-only sequence data
docs/                            dataset, experiment, and reproducibility notes
results/paper_tables/            final paper-ready tables
results/experiment_outputs/      backing CSV files for window and ablation results
results/model_outputs/           backing CSV files for model comparison
results/tcn_tuning/              TCN tuning outputs
results/resnet1d_tuning/         ResNet1D tuning outputs
scripts/                         runnable entry-point scripts
src/darkmirror_repro/            reusable reproduction code
```

## Final Input Condition

The final model-comparison condition uses the 3-second median-based sequence length from the paper:

- read the first 526 packet directions from each session
- remove the first 20 packets as the handshake/connection-initialization segment
- use the remaining fixed-length direction-only sequence
- final input length: `seq_len=506`

Main final data file:

```text
data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz
```

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `X_dir_cnn` | `(n, 1, L)` | Conv1D, TCN, ResNet1D-style input |
| `X_dir_rnn` | `(n, L, 1)` | GRU, BiGRU, LSTM-style input |
| `y` | `(n,)` | binary labels |
| `groups` | `(n,)` | anonymized URL/site group IDs |
| `file_names` | `(n,)` | anonymized sample IDs |

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

The original final runs used Python 3.12.7, PyTorch 2.5.1+cu121, CUDA 12.1, and an NVIDIA GeForce RTX 3060 Ti. See `environment_versions.json`.

## Reproduce Tables

Run from the repository root:

```bash
python scripts/reproduce_tables.py
```

Regenerated tables are written to:

```text
results/reproduced_tables/
```

The script rebuilds the paper-ready CSV and Markdown tables from the packaged processed data and backing result files.

## Inspect Data

```bash
python scripts/inspect_dataset.py
```

## Optional Retraining and Tuning

Reference retraining:

```bash
python scripts/train_reference_model.py --model tcn
python scripts/train_reference_model.py --model resnet1d
```

Hyperparameter tuning reruns under the final `seq_len=506` post-handshake condition:

```bash
python scripts/tune_tcn.py
python scripts/tune_resnet1d.py
```

Retraining can produce small numerical differences across PyTorch, CUDA, and GPU environments.

## Interpretation Notes

The final experiments use a sample-level stratified split with seed `42`. The same URL group can appear in train, validation, and test splits, so the results should be interpreted as sample-level session classification performance rather than unseen-URL generalization.

TCN achieved the highest ROC-AUC under the final sequence condition. ResNet1D is also retained as a final candidate because the paper evaluates detection performance together with end-to-end processing time.
