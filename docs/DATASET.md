# Dataset

This repository contains the processed data used for reproducing the paper experiments. Raw packet captures are not shared. The included files begin after protocol filtering and direction-only sequence construction.

The shared data have been anonymized for public release. Original URL/site names, original capture file names, and client IP values are not included. They are replaced with synthetic identifiers while preserving labels, sequence arrays, sample counts, group counts, and reproducibility of the reported tables.

## Paper Dataset Summary

The paper collected 4,600 web-access sessions:

| Class | URLs | Repetitions per URL | Session duration | Sessions |
|---|---:|---:|---:|---:|
| Clear Web | 23 | 100 | 20 s | 2,300 |
| Dark Web Mirror site | 23 | 100 | 20 s | 2,300 |
| Total | 46 | 100 | 20 s | 4,600 |

The binary label is assigned by URL type and applied to every repeated session from that URL.

| Label | Meaning |
|---:|---|
| `0` | Clear Web |
| `1` | Dark Web Mirror site |

## Direction Encoding

Only packet transmission direction is retained:

| Value | Meaning |
|---:|---|
| `+1` | client-to-server |
| `-1` | server-to-client |
| `0` | padding |

Payloads, URL strings, page content, packet sizes, and timing values are not included in the model input.

## Protocol Filtering

The original preprocessing retained packets associated with web encrypted-traffic flows: TCP, TLS, and QUIC. DNS, ICMP, and UDP packets not identified as QUIC were excluded before direction encoding.

## File Schema

Each `.npz` dataset contains:

| Key | Shape | Description |
|---|---|---|
| `X_dir_cnn` | `(n, 1, L)` | input format for Conv1D-style models, including TCN and ResNet1D |
| `X_dir_rnn` | `(n, L, 1)` | input format for recurrent models |
| `y` | `(n,)` | binary labels |
| `groups` | `(n,)` | anonymized URL/site group IDs, e.g., `clear_site_01`, `mirror_site_01` |
| `file_names` | `(n,)` | anonymized sample IDs, e.g., `clear_sample_0001`, `mirror_sample_0001` |

Metadata CSV files are sanitized and do not contain local raw-capture paths. The `client_ip` field is redacted as `CLIENT_IP_REDACTED`.

## Data-to-Experiment Mapping

| Paper experiment | Included data |
|---|---|
| Initial time-window comparison | `data/processed/initial_time_windows/1s_seq137.npz`, `3s_seq526.npz`, `5s_seq905.npz`, `10s_seq1276.npz` |
| Handshake ablation | `data/processed/handshake_ablation/seq506_posthandshake.npz`, `seq506_handshake_included.npz`, `seq506_handshake_only.npz`, `seq506_handshake_masked.npz` |
| Sequence length reduction | `data/processed/handshake_ablation/seq506_posthandshake.npz`, `seq300_posthandshake.npz` |
| Model comparison | `data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz` |
| TCN and ResNet1D tuning | `data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz` |

## Final Input

The final model-comparison and tuning input is:

```text
data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz
```

It follows this construction:

1. Select the first 526 packet directions from the session.
2. Treat the first 20 packets as the handshake/connection-initialization segment.
3. Remove those 20 packets.
4. Use the remaining `506` packet-direction symbols as the fixed-length input.

This is equivalent to `direction[20:526]`.

Therefore, `seq_len=506` always refers to the post-handshake length after removing the initial 20 packets from the 526-direction baseline. It does not mean that 20 packets are removed again from an already 506-long sequence.
