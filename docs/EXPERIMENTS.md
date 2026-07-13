# Experiments

This document summarizes the experimental procedure represented by the shared data and code. It follows the paper workflow: direction-only sequence construction, initial-window analysis, handshake ablation, sequence-length reduction, model comparison, and final TCN/ResNet1D hyperparameter selection.

## Split

The final experiments use sample-level stratified random splits with seed `42`.

| Split | Samples | Class 0 | Class 1 | URL groups |
|---|---:|---:|---:|---:|
| Subtrain | 3312 | 1656 | 1656 | 46 |
| Validation | 368 | 184 | 184 | 46 |
| Test | 920 | 460 | 460 | 46 |

Because the split is sample-level, repeated sessions from the same URL can appear in subtrain, validation, and test. The reported values therefore measure sample-level session classification, not unseen-URL generalization.

The released dataset uses anonymized group and sample identifiers. Original URL/site names, original capture file names, and client IP values are not required for the experiments and are not included. The anonymized identifiers preserve the same 46 URL/site groups and the same sample-level split behavior.

## Preprocessing

For each session, the original preprocessing:

1. keeps packets whose protocol belongs to the target web encrypted-traffic protocol set;
2. encodes client-to-server packets as `+1`;
3. encodes server-to-client packets as `-1`;
4. truncates sequences longer than the target length;
5. pads shorter sequences with `0`.

The final post-handshake input removes the first 20 packet directions from the 3-second median-based length of 526, yielding `seq_len=506`.

## Metrics

The paper reports Accuracy, Precision, Recall, F1-score, and ROC-AUC. Dark Web Mirror site is the positive class. ROC-AUC is used as the main ranking metric because it evaluates class separability across thresholds, while Accuracy, Precision, Recall, and F1-score depend on the selected decision threshold.

## Training Procedure

Final deep-learning runs used:

| Item | Value |
|---|---|
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Maximum epochs | 50 |
| Early stopping | validation ROC-AUC |
| Patience | 10 epochs |
| Seed | 42 |
| Threshold selection | validation F1-score |
| Threshold search range | 0.10 to 0.90, step 0.01 |

## Experiment-to-Table Mapping

| Paper table in this package | Purpose | Main condition |
|---|---|---|
| Table 3 | Initial time-window performance | fixed sequence lengths from 1 s, 3 s, 5 s, and 10 s median packet counts |
| Table 5 | Handshake ablation | include, exclude, and handshake-only sequence variants |
| Table 6 | Sequence length reduction | post-handshake `seq_len=506` versus `seq_len=300` |
| Table 7 | Model performance comparison | final `seq_len=506` post-handshake input |
| Table 8 | TCN and ResNet1D hyperparameter settings and tuning summary | final `seq_len=506` post-handshake input |

## Candidate Models

The model comparison evaluates sequence classifiers suitable for fixed-length direction-only inputs:

| Family | Models |
|---|---|
| Convolution-based | ResNet1D, TCN, InceptionTime, 1D-CNN, FCN-1D, ROCKET-style |
| Recurrent-based | GRU, BiGRU, LSTM |
| Attention-based | Transformer Encoder |

All models receive the same direction-only sequence values. Architecture-specific array shapes are provided in each `.npz` file as `X_dir_cnn` and `X_dir_rnn`.

## Final Candidate Selection

Under the final `seq_len=506` post-handshake condition, TCN achieves the highest ROC-AUC in the packaged final tables. ResNet1D is retained as the other final candidate because the paper jointly considers detection performance and processing time.

Final same-condition hyperparameter summaries:

| Model | Selected candidate | Channels | Kernel size(s) | Batch size | Threshold |
|---|---|---|---|---:|---:|
| TCN | `tcn_grid_07` | `(32, 64, 128)` | `5` | 64 | 0.9000 |
| ResNet1D | `R10` | `(64, 128, 128)` | `(8, 5, 3)` | 128 | 0.5400 |

Both models use learning rate `0.0010` and weight decay `0.0001`.

## Reproduction Commands

Regenerate final tables:

```bash
python scripts/reproduce_tables.py
```

Inspect dataset shape, labels, and groups:

```bash
python scripts/inspect_dataset.py
```

Optional retraining:

```bash
python scripts/train_reference_model.py --model tcn
python scripts/train_reference_model.py --model resnet1d
```

Optional tuning reruns:

```bash
python scripts/tune_tcn.py
python scripts/tune_resnet1d.py
```
