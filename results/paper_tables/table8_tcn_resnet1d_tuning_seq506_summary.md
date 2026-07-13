# Table 8. TCN and ResNet1D Hyperparameter Optimization Results

| Model    | Selected candidate   | Input condition                                              | Channels       | Kernel size(s)   | Dropout   | Learning rate   | Weight decay   | Batch size   | Threshold   | Val ROC-AUC   | Accuracy   | Precision   | Recall   | F1-score   | ROC-AUC   |
|:---------|:---------------------|:-------------------------------------------------------------|:---------------|:-----------------|:----------|:----------------|:---------------|:-------------|:------------|:--------------|:-----------|:------------|:---------|:-----------|:----------|
| TCN      | tcn_grid_07          | first 526 directions -> remove first 20 -> final seq_len=506 | (32, 64, 128)  | 5                | 0.1000    | 0.0010          | 0.0001         | 64           | 0.9000      | 0.9912        | 0.9315     | 0.9671      | 0.8935   | 0.9288     | 0.9824    |
| ResNet1D | R10                  | first 526 directions -> remove first 20 -> final seq_len=506 | (64, 128, 128) | (8, 5, 3)        | 0.0000    | 0.0010          | 0.0001         | 128          | 0.5400      | 0.9900        | 0.9293     | 0.9647      | 0.8913   | 0.9266     | 0.9818    |

Note: Both models use the same final input condition: first 526 directions, remove first 20 handshake packets, final seq_len=506.
