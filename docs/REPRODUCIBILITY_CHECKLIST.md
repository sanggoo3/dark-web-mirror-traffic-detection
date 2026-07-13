# Reproducibility Checklist

## Included

- Processed direction-only sequence datasets used by the paper experiments
- Sanitized and anonymized metadata without local raw-capture paths
- Synthetic site IDs, sample IDs, and redacted client IP values
- Final paper-ready tables in CSV and Markdown
- Backing result CSV files for initial-window, ablation, sequence-length, timing, and model-comparison tables
- TCN hyperparameter tuning outputs
- ResNet1D hyperparameter tuning outputs
- Scripts for table regeneration, dataset inspection, reference retraining, and tuning reruns
- Environment version record
- SHA-256 file manifest

## Not Included

- Raw `pcap` or `pcapng` captures
- Local absolute raw-capture paths
- Original URL/site names and original capture file names
- Browser automation and traffic-collection tooling
- Payload, URL-content, page-content, packet-size, or timing features
- Intermediate files not needed for table reproduction
- Figure assets

## Final Experimental Settings

| Item | Value |
|---|---|
| Task | binary session-level Clear Web vs Dark Web Mirror site classification |
| Positive class | Dark Web Mirror site |
| Input feature | direction-only sequence |
| Direction values | `+1`, `-1`, `0` padding |
| Final sequence construction | first 526 packet directions, then remove first 20 handshake packets |
| Final length | `seq_len=506` after removal, not before removal |
| Split | sample-level stratified split |
| Random seed | 42 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Early stopping | validation ROC-AUC, patience 10 |
| Threshold selection | validation F1-score |
| Main model-selection metric | ROC-AUC |
| Final candidates | TCN and ResNet1D |

## Verification

Run these commands from the repository root:

```bash
python scripts/inspect_dataset.py
python scripts/reproduce_tables.py
```

After `reproduce_tables.py`, compare `results/reproduced_tables/` with `results/paper_tables/`. Small retraining differences can occur only when rerunning training or tuning, not when regenerating packaged tables.
