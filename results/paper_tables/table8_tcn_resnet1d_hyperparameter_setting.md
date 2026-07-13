# Table 8. Final Hyperparameter Settings for TCN and ResNet1D

| Parameter          | TCN                                                          | ResNet1D                                                     |
|:-------------------|:-------------------------------------------------------------|:-------------------------------------------------------------|
| Input              | first 526 directions -> remove first 20 -> final seq_len=506 | first 526 directions -> remove first 20 -> final seq_len=506 |
| Handshake removal  | first 20 packets                                             | first 20 packets                                             |
| Block structure    | 3 TCN blocks                                                 | 3 residual blocks                                            |
| Channels           | (32, 64, 128)                                                | (64, 128, 128)                                               |
| Kernel size(s)     | 5                                                            | (8, 5, 3) per block                                          |
| Dilation           | (1, 2, 4)                                                    | -                                                            |
| Dropout            | 0.1000                                                       | 0.0000                                                       |
| Learning rate      | 0.0010                                                       | 0.0010                                                       |
| Weight decay       | 0.0001                                                       | 0.0001                                                       |
| Batch size         | 64                                                           | 128                                                          |
| Decision threshold | 0.9000                                                       | 0.5400                                                       |

Note: Both TCN and ResNet1D hyperparameters were selected using the same final input: first 526 directions, remove first 20 handshake packets, final seq_len=506. Decision thresholds were selected by validation F1-score.
