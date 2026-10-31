# Corrections Required for the Revised Manuscript

## Table 3(a): Corrected Cohort

The submitted PDF mixes historical packet-count descriptives with performance evaluated on the corrected protocol-filter cohort. The public release uses the corrected cohort throughout. Use these Table3(a) values:

| Initial window (s) | Minimum | Maximum | Mean | Median | Predefined classification length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 395 | 139.3161 | 137.0000 | 137 |
| 3 | 197 | 4715 | 833.4770 | 526.0000 | 526 |
| 5 | 357 | 8359 | 1310.5078 | 904.5000 | 905 |
| 10 | 529 | 10727 | 1929.5050 | 1275.0000 | 1276 |

For example, the PDF gives 3s mean631.6737, whereas the final corrected data give833.4770. The10s median is1275, not1276. The historical rows are retained separately as `results/recorded/table3a_reported_in_submitted_pdf.csv`, **not** as descriptions of the released data.

The predefined classification lengths137/526/905/1276 remain unchanged by explicit author decision. Do not call1276 the newly computed10s median or silently replace it with1275: the reported models were trained/evaluated using1276. This descriptive correction does not require changing the existing performance results or conducting a new1275 experiment.

## Input and Timing Wording

- Classification at lengths137/526/905/1276 uses packet prefixes from the20-second session, not strictly time-cropped1/3/5/10-second traffic. Time labels refer to the historical rationale for selecting fixed lengths only.
- The primary526 input **includes** the initial20 packets. Any pipeline/figure caption saying the main input excludes them should be corrected. Only the506 control excludes them once.
- The initial20-packet "handshake" is an operational early-packet window, not proof that exactly20 packets delimit every protocol handshake.
- Current Table4(b) and group-aware timings both come from continuous file-to-decision measurement. Component means add algebraically, but this does not mean total time was synthesized from independently timed passes. Align prose accordingly.
- Table4(c) combines HPO selection AUC (inner pilot validation; SD across five outer folds) with timing of final checkpoints on reserved validation inputs. These are separate stages, not one common repeated-measurement population.
- Domain bootstrap intervals are conditional on already trained models and do not include retraining variability. Distinguish them from five-seed t intervals.

The publisher's PDF is not distributed or modified by this repository. These notes identify which manuscript text/statistics should be updated to match the public, versioned evidence.
