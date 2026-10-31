# Manuscript Experiment Protocol

## Table Mapping

| Manuscript item | Released experiment / output in `results/tables/` |
| --- | --- |
| Table 1 | Cohort counts, `table1_dataset.csv` |
| Table 2 | Recorded environment, `table2_environment.json` |
| Table 3(a) | Actual time-window packet counts, **corrected** `table3a_corrected_packet_counts.csv` |
| Table 3(b), Figure 3 | TCN session-stratified first 137/526/905/1276 directions from 20-second sessions; `table3b_tcn_fixed_lengths.csv`, `figure3_*` |
| Table 4(a) | All ten models, session-stratified, included 526; `table4a_session_models.csv` |
| Table 4(b) | Session-stratified singleton PCAPNG test timing; `table4b_session_test_time.csv` |
| Table 4(c) | TCN/ResNet1D HPO selection AUC and final-checkpoint validation timing; `table4c_validation_selection_time.csv` |
| Table 5(a) | TCN included526 / post506 / first20 padded526, session-stratified; `table5a_handshake.csv` |
| Table 5(b) | TCN included526 vs included300, session-stratified; `table5b_reduction.csv` |
| Table 6 | TCN/ResNet1D same-prefix direction + light16 vs direction, session-stratified; `table6_fusion16.csv`, `latency_summary.csv` |
| Table 7 | All ten models, group-aware included526, including seed/domain intervals; `table7_*`, `group_aware_full_statistics.csv` |
| Table 8 | Session-stratified main-model ROC-AUC seed intervals; `table8_auc_intervals.csv` |
| Group-aware timing paragraph | TCN/ResNet1D included526 test rows of `latency_summary.csv` |
| Paired ResNet1D/TCN paragraph | `group_aware_paired_resnet_minus_tcn.csv` |
| Full-session descriptive paragraph | `full20s_class_descriptives.csv`; separate from the classifier input |

Raw PCAP collection, paper diagrams, reference bibliography and Word-building utilities are not learning experiments and are not included. No fixed holdout, masked/shuffled control, statistics-only MLP, volume-only MLP or 17-feature whole-volume fusion is exposed in this snapshot.

## Two Five-Fold Protocols

Group-aware outer splits use StratifiedGroupKFold(5, shuffle=True, random_state=42) over the original registrable domains. Within each outer training pool, four class-stratified domains are reserved for final validation using train_test_split(test_size=4, random_state=42+fold). Inner tuning uses StratifiedGroupKFold(3, shuffle=True, random_state=142+fold) on the remaining outer fit subset only. Original domain spelling is not public, so use the supplied exact indices.

Session-stratified outer splits use StratifiedKFold(5, shuffle=True, random_state=42). Reserve 10% of the outer training pool with StratifiedShuffleSplit(random_state=42+fold). Inner tuning uses StratifiedKFold(3, shuffle=True, random_state=142+fold) on fit only. Each outer fold has 3312 fit / 368 validation / 920 test sessions. Group-aware counts vary with whole-group membership; `verify` reports them. Sessions do not overlap within any split; group-aware domain IDs do not overlap either. Final outer validation and test sessions never enter inner tuning.

## Tuning and Training

Each of ten families has 12 predeclared candidate settings in `configs/experiment.json`. For each split/fold/family, compare every candidate over three inner folds and pilot seeds 11/23 (72 fits). Rank by mean inner validation ROC-AUC descending, sample SD ascending, then configuration ID ascending. The selected settings are different per fold; do not present one global test-selected optimum. Family ranking, where used, is based on these validation scores, with SD/name tie-breaks, never test performance.

Two split protocols x five outer folds x ten families x twelve candidates x three inner folds x two pilot seeds = **7,200 HPO fits**. After selection, train each selected model using final seeds **11,23,71,101,202**. Baselines: 500 fits. Additional paper-only TCN length/handshake comparisons: 150. Same-prefix light16 fusion: 50. Total final fits: **700**. The 506 control reuses each fold's 526-selected backbone settings rather than independently retuning every ablation.

Neural training uses Adam, BCEWithLogitsLoss, gradient clipping at 1, up to 50 epochs, minimum 20 epochs and patience 10 for validation ROC-AUC non-improvement. Strictly greater AUC updates the best checkpoint. Learning rate, weight decay, architecture, dropout and batch size come from the selected configuration. Batch order uses a generator seeded with seed+1009. Random, NumPy and PyTorch seeds are set. CUDA matmul TF32=False, cuDNN benchmark=False; neural cuDNN TF32=True, deterministic except TCN. CPU threads=2. Prediction during training uses batch 128; reported final test probabilities use **batch 1** from the frozen checkpoint.

TCN here is the actual symmetric-padded residual dilated 1D implementation, not a strictly causal TCN. InceptionTimeLite is a lightweight implementation. ROCKET-style is a custom random undilated convolution feature extractor with lengths7/9/11/15, maximum and PPV features, training-only StandardScaler and LogisticRegression(lbfgs,max_iter=3000). It is **not canonical ROCKET**. Its convolutions explicitly use FP32/cuDNN TF32=False for consistent feature generation and singleton replay; convergence failure raises an error.

## Threshold and Fusion

Select threshold from the reserved final validation probabilities, using 0.10..0.90 at increments of 0.01, maximum F1, then maximum Accuracy, then the lower threshold. Save the configuration/checkpoint/threshold lock before evaluating any test input. Test labels are never used for threshold selection. Published validation probabilities are the original batch128 values used for these decisions; their role differs from singleton test and latency records.

Fusion computes the 16 features in `features.py` from exactly the same first 526 directions/timestamps. Fit StandardScaler on outer fit sessions only; reuse it on validation/test. Concatenate the scaled features after the direction backbone representation and train the enlarged head/backbone from scratch with the fold's direction-selected configuration. There is no separate fusion HPO. No whole-session packet count is added.

Features: log1p span(ms); log1p IAT mean, population SD, median, p90, p99, maximum(ms); IAT burstiness `(SD-mean)/(SD+mean)` (0 if denominator0); outgoing fraction; adjacent direction-switch fraction; log1p direction-run count, run-length mean, population SD, median, p90, maximum. There are 525 IATs; negative gaps are clipped to zero without reordering packets. Quantiles use NumPy's default linear method. Feature arithmetic is float64, final vector float32.

## Statistical Reporting

For each seed, average the five outer-fold metrics with equal fold weights. Report mean and sample SD across the five seed means, not SD across 25 independent observations. Seed 95% CI is mean +/- t(0.975,df=4)*SD/sqrt(5), clipped to[0,1] for metrics. Pilot seeds also occur in final training; seed CIs are conditional descriptive repeatability estimates, not wholly independent new tuning replications.

Domain intervals use 2,000 class-stratified cluster bootstrap draws with seed20261022 and fixed trained predictions. For group-aware evaluation, domain multiplicities are drawn within each held-out fold. For session-stratified evaluation, use the **same class/domain multiplicities in every fold** because domains recur across folds. Compute each replicate's fold/seed averages and percentile2.5/97.5 boundaries. These intervals exclude retraining variability and are conditional on this cohort, models and evaluation partition. Paired ResNet1D-minus-TCN comparisons use matched seed values and the same domain draws, with unclipped difference CIs.

The highest test mean is a descriptive outcome, not evidence that the family was chosen on test. Group-aware validation reduces shared-domain leakage but does not erase historical exploratory reuse of the dataset or establish deployment/external validity. No new full retraining was performed while packaging this release; see `validation_checks.json` for the exact packaging tests.
