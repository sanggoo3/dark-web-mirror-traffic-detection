# Dark Web Mirror Traffic Detection

Minimal data and code for **Detecting Dark Web Mirror Sites on the ClearNET: An Encrypted Traffic Classification Approach**.

Release: [v2.0.0](https://github.com/sanggoo3/dark-web-mirror-traffic-detection/releases/tag/v2.0.0). Dataset: `direction526-20261101`. Exact file SHA-256 values are in `release_manifest.json`.

## Files

| Path | Purpose |
| --- | --- |
| `data/processed/` | Direction sequences and same-prefix light16 features |
| `data/metadata/` | Anonymous session/site/group assignments, descriptive counts, timing sample indices |
| `data/splits/` | Exact outer/inner indices for group-aware and session-stratified evaluation |
| `configs/experiment.json` | Candidate grids, training, threshold, bootstrap and timing settings |
| `configs/selected_configurations.json` | Recorded validation-selected configuration for each fold/model |
| `src/darkmirror_repro/` | Models, tuning/training, evaluation, statistics and preprocessing/timing code |

No archived predictions, duplicate result tables, plots, checkpoints or auxiliary documentation are distributed. New outputs are written to ignored `artifacts/`.

## Data and Input

4,600 sessions; labels0=Clear web,1=Dark mirror web; 2,300 sessions per class. `sessions.csv` maps each zero-based array index to anonymous session/site/domain IDs. Original URLs/domains, IPs, capture dates, paths and raw PCAP/PCAPNG are not public for privacy/security reasons. Anonymous group relationships are retained for split auditing; pseudonymization does not guarantee irreversible anonymity.

Load NPZ files with `numpy.load(..., allow_pickle=False)`. `direction_sequences.npz` contains `prefix`(4600,1276) int8, `relative_times_s`(4600,1276) float64, `window_counts`(4600,4), `retained_count`(4600,), `y` and `groups`. Directions are+1 client-to-server/-1 server-to-client/0 right padding; padded times are NaN. Window counts refer to actual <=1/3/5/10seconds. All rows share the same metadata index.

Primary input: first526 eligible directions within20seconds, **including initial20**. Post-handshake control: `q[20:526]`, length506, removed once. The cohort has no padding at526. Fixed comparison lengths137/526/905/1276 are pre-existing design constants, not time-cropped classification inputs. Corrected packet-count statistics differ from historical Table3(a): 3s mean833.4770; 10s median1275, while the fixed experimental length1276 remains unchanged. `descriptives` recalculates these statistics and class-specific padding incidence.

Direction-only models use no time/count/payload features. `light_statistics_526.npz` contains the redundant float32 `light`(4600,16) features for the explicit fusion experiment only. `features.py` defines their exact formulas from the same526 directions/relative times. Fit-only StandardScaler is used for fusion. `full20s_direction_descriptives.csv` contains descriptive whole-session counts, not classifier inputs.

## Install and Run

Recorded environment: Python3.12.7, PyTorch2.5.1+cu121, CUDA12.1, Windows, RTX3060Ti8GB. Use a separate environment and run from this repository:

```console
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m darkmirror_repro verify
python -m darkmirror_repro descriptives
python -m darkmirror_repro tune --out artifacts/rerun
python -m darkmirror_repro train --out artifacts/rerun
python -m darkmirror_repro evaluate --out artifacts/rerun
python -m darkmirror_repro summarize --out artifacts/rerun
```

Training requires CUDA. Verification/descriptives do not. To skip HPO and retrain the recorded settings, use `train --recorded-choices`. `tune`/`train` accept `--split`, `--model`, `--fold`; `train` also accepts `--seed`. Full summaries require all700 final evaluations. This minimal package requires retraining to regenerate performance tables; it does not reaggregate archived paper predictions. Retrained results can differ across hardware/software.

## Evaluation Protocol

Both protocols use fixed outer5-fold partitions and inner3-fold tuning on fit only. Group-aware assignments hold domains together; session-stratified assignments allow known domains across partitions. Outer split seed42, reserved validation seed42+fold, inner split seed142+fold. Use the supplied ordered indices, not regenerated splits on renamed domain strings. No test input enters tuning, scaler fitting, early stopping or threshold selection.

Ten families: TCN,ResNet1D,InceptionTimeLite,1D-CNN,FCN-1D,GRU,BiGRU,LSTM,Transformer Encoder,ROCKET-style. Each family has12candidates, evaluated using pilot seeds11/23 and3innerfolds. Rank by validation ROC-AUC mean descending, SD ascending, configuration ID ascending. Final seeds11/23/71/101/202. Main evaluation includes500fits; TCN length/handshake comparisons150; TCN/ResNet1D light16 fusion50. Total7,200HPO+700finalfits. Auxiliary comparisons reuse each fold's526-selected configuration.

Adam/BCEWithLogitsLoss; at most50epochs, minimum20, patience10 on validation ROC-AUC; best checkpoint restored, gradient clipping1. Threshold0.10..0.90 step0.01, validation F1 maximum, then Accuracy maximum, then lower threshold. Freeze decisions before batch1test evaluation. RNNs receive the same values reshaped(N,L,1); other neural models receive(N,1,L). TCN is symmetric-padded, not strictly causal; InceptionTimeLite and ROCKET-style are implementation-specific, not canonical InceptionTime/ROCKET.

First average5folds per seed, then report mean/sample SD across5seedmeans. Seed95%CI uses t(df4). Conditional domain bootstrap uses2,000draws, seed20261022; trained models fixed, no retraining variation. Group-aware weights are drawn within held-out folds; session-stratified weights are shared across folds. Paired differences use matched seeds/domain draws. Historical exploratory cohort reuse limits claims of independent external validation.

## Optional PCAP Timing

With separately authorized private captures, provide an external CSV with `index,source_path,client_ip` covering the timing samples. Do not commit it. Install TShark, then run:

```console
python -m darkmirror_repro benchmark --out artifacts/rerun --private-manifest PRIVATE_CSV --tshark TSHARK_PATH
```

Reference filter: `(tcp or tls or quic) and not (dns or icmp or icmpv6)`; outerTCP or QUIC/UDP only, known client IPv4, capture order, <=20seconds. Fast PCAPNG parser candidates are corpus-specific, not general QUIC dissection; every supplied prefix is checked against TShark/public data before timing.

Batch1,50GPUwarmups,3repeats/input; continuous file-read/parse/packing/H2D/forward/D2H/decision/finalsync. Initial sync, model loading, packet collection wait, assertions and output I/O excluded; warm caches. Average repeats per session/checkpoint before mean/SD. Preprocess/Inference/Total boundaries are in `latency.py`. New end-to-end times require private captures and cannot be independently reproduced from public processed sequences alone.

## Availability and License

Processed sequences, anonymous assignment/split metadata and experimental code are public at the v2.0.0 release above. Raw captures are withheld. `release_manifest.json` pins the exact dataset files. Code: [MIT](LICENSE). Data/configurations/README/manifest: [CC BY-NC4.0](DATA_LICENSE.md). Cite the paper and this release URL.
