from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import METRIC_COLUMNS, paper_round, repo_root, write_table


MODEL_ORDER = [
    "TCN",
    "ResNet1D",
    "InceptionTime",
    "1D-CNN",
    "FCN-1D",
    "ROCKET-style",
    "GRU",
    "BiGRU",
    "Transformer Encoder",
    "LSTM",
]


def best_tcn_metrics(root: Path) -> dict:
    df = pd.read_csv(root / "results" / "tcn_tuning" / "tcn_hyperparameter_tuning_results.csv")
    row = df.sort_values(["val_ROC-AUC", "val_F1-score", "ROC-AUC"], ascending=[False, False, False]).iloc[0]
    return {key: row[key] for key in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}


def best_resnet_metrics(root: Path) -> dict:
    df = pd.read_csv(root / "results" / "resnet1d_tuning" / "resnet1d_hyperparameter_tuning_results.csv")
    row = df.sort_values(["val_ROC-AUC", "val_F1-score", "ROC-AUC"], ascending=[False, False, False]).iloc[0]
    return {key: row[key] for key in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}


def read_one(path: Path) -> dict:
    return pd.read_csv(path, encoding="utf-8-sig").iloc[0].to_dict()


def build_table3(root: Path) -> pd.DataFrame:
    exp = root / "results" / "experiment_outputs" / "table3_time_windows"
    rows = []
    for label, seq_len in [("1s", 137), ("3s", 526), ("5s", 905), ("10s", 1276)]:
        row = read_one(exp / f"table3_{label}_seq{seq_len}_result.csv")
        rows.append({"Initial time window": label, "seq_len": seq_len, **{m: row[m] for m in METRIC_COLUMNS}})
    return paper_round(pd.DataFrame(rows))


def build_table5(root: Path) -> pd.DataFrame:
    table3 = root / "results" / "experiment_outputs" / "table3_time_windows"
    abl = root / "results" / "experiment_outputs" / "handshake_ablation"
    tcn_metrics = best_tcn_metrics(root)
    sources = [
        ("Handshake included", 526, table3 / "table3_3s_seq526_result.csv"),
        ("Post-handshake sequence", 506, None),
        ("Handshake masked", 506, abl / "table_mask_first20_result.csv"),
        ("Handshake only", 506, abl / "table_only_first20_result.csv"),
    ]
    rows = []
    for condition, seq_len, path in sources:
        row = tcn_metrics if path is None else read_one(path)
        rows.append(
            {
                "Handshake processing condition": condition,
                "seq_len": seq_len,
                **{m: row[m] for m in METRIC_COLUMNS},
            }
        )
    return paper_round(pd.DataFrame(rows))


def build_table6(root: Path) -> pd.DataFrame:
    abl = root / "results" / "experiment_outputs" / "handshake_ablation"
    tcn_metrics = best_tcn_metrics(root)
    sources = [
        (506, None),
        (300, abl / "table_after_first20_seq300_result.csv"),
    ]
    rows = []
    for seq_len, path in sources:
        row = tcn_metrics if path is None else read_one(path)
        rows.append({"seq_len": seq_len, **{m: row[m] for m in METRIC_COLUMNS}})
    return paper_round(pd.DataFrame(rows))


def build_table7(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "results" / "model_outputs" / "pcapng_seq506_all_model_performance_time.csv")
    out = df[["Model", *METRIC_COLUMNS]].copy()
    for model, metrics in [("TCN", best_tcn_metrics(root)), ("ResNet1D", best_resnet_metrics(root))]:
        mask = out["Model"].eq(model)
        for col, value in metrics.items():
            out.loc[mask, col] = value
    out["_order"] = out["Model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    out = out.sort_values("_order").drop(columns="_order")
    return paper_round(out)


def build_timing_table(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "results" / "model_outputs" / "pcapng_seq506_all_model_performance_time.csv")
    out = pd.DataFrame(
        {
            "Model": df["Model"],
            "Preprocess (ms)": df["pcapng_preprocess_mean_ms"],
            "Inference (ms)": df["inference_mean_ms"],
            "Total (ms)": df["pcap_to_prediction_mean_ms"],
        }
    )
    out["_order"] = out["Model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    out = out.sort_values("_order").drop(columns="_order")
    return paper_round(out)


def build_table8() -> pd.DataFrame:
    rows = [
        ("Input", "pcapng seq_len=506 post-handshake direction-only sequence", "pcapng seq_len=506 post-handshake direction-only sequence"),
        ("Handshake removal", "first 20 packets", "first 20 packets"),
        ("Block structure", "3 TCN blocks", "3 residual blocks"),
        ("Channels", "(32, 64, 128)", "(64, 128, 128)"),
        ("Kernel size(s)", "5", "(8, 5, 3) per block"),
        ("Dilation", "(1, 2, 4)", "-"),
        ("Dropout", "0.1000", "0.0000"),
        ("Learning rate", "0.0010", "0.0010"),
        ("Weight decay", "0.0001", "0.0001"),
        ("Batch size", "64", "128"),
        ("Decision threshold", "0.9000", "0.5400"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "TCN", "ResNet1D"])


def build_table8_tuning_seq506(root: Path) -> pd.DataFrame:
    tcn = pd.read_csv(root / "results" / "tcn_tuning" / "tcn_hyperparameter_tuning_results.csv")
    res = pd.read_csv(root / "results" / "resnet1d_tuning" / "resnet1d_hyperparameter_tuning_results.csv")
    tcn_best = tcn.sort_values(["val_ROC-AUC", "val_F1-score", "ROC-AUC"], ascending=[False, False, False]).iloc[0]
    res_best = res.sort_values(["val_ROC-AUC", "val_F1-score", "ROC-AUC"], ascending=[False, False, False]).iloc[0]
    out = pd.DataFrame(
        [
            {
                "Model": "TCN",
                "Selected candidate": tcn_best["candidate_id"],
                "Input condition": "seq_len=506 post-handshake direction-only sequence",
                "Channels": tcn_best["channels"],
                "Kernel size(s)": str(int(tcn_best["kernel_size"])),
                "Dropout": tcn_best["dropout"],
                "Learning rate": tcn_best["lr"],
                "Weight decay": tcn_best["weight_decay"],
                "Batch size": int(tcn_best["batch_size"]),
                "Threshold": tcn_best["threshold"],
                "Val ROC-AUC": tcn_best["val_ROC-AUC"],
                "Accuracy": tcn_best["Accuracy"],
                "Precision": tcn_best["Precision"],
                "Recall": tcn_best["Recall"],
                "F1-score": tcn_best["F1-score"],
                "ROC-AUC": tcn_best["ROC-AUC"],
            },
            {
                "Model": "ResNet1D",
                "Selected candidate": res_best["candidate_id"],
                "Input condition": "seq_len=506 post-handshake direction-only sequence",
                "Channels": res_best["channels"],
                "Kernel size(s)": res_best["kernels"],
                "Dropout": res_best["dropout"],
                "Learning rate": res_best["learning_rate"],
                "Weight decay": res_best["weight_decay"],
                "Batch size": int(res_best["batch_size"]),
                "Threshold": res_best["Threshold"],
                "Val ROC-AUC": res_best["val_ROC-AUC"],
                "Accuracy": res_best["Accuracy"],
                "Precision": res_best["Precision"],
                "Recall": res_best["Recall"],
                "F1-score": res_best["F1-score"],
                "ROC-AUC": res_best["ROC-AUC"],
            },
        ]
    )
    return paper_round(out)


def main() -> None:
    root = repo_root()
    out_dir = root / "results" / "reproduced_tables"
    write_table(build_table3(root), out_dir, "table3_initial_time_window", "Table 3. Classification Performance by Initial Time Window")
    write_table(build_table5(root), out_dir, "table5_handshake_ablation", "Table 5. Handshake Ablation Results")
    write_table(build_table6(root), out_dir, "table6_sequence_length_reduction", "Table 6. Sequence Length Reduction Results")
    write_table(build_table7(root), out_dir, "table7_model_performance_comparison", "Table 7. Model Performance Comparison")
    write_table(
        build_timing_table(root),
        out_dir,
        "table7_all_model_pcapng_time_compact_ms",
        "Table 7. All Model pcapng Time",
        "Total time is Preprocess plus Inference for one test sample.",
    )
    write_table(
        build_table8(),
        out_dir,
        "table8_tcn_resnet1d_hyperparameter_setting",
        "Table 8. Final Hyperparameter Settings for TCN and ResNet1D",
        "Both TCN and ResNet1D hyperparameters were selected using the same final seq_len=506 post-handshake direction-only sequence. Decision thresholds were selected by validation F1-score.",
    )
    write_table(
        build_table8_tuning_seq506(root),
        out_dir,
        "table8_tcn_resnet1d_tuning_seq506_summary",
        "Table 8. TCN and ResNet1D Hyperparameter Optimization Results",
        "Both models use the same final input condition: seq_len=506 post-handshake direction-only sequence.",
    )
    print(f"[done] reproduced tables: {out_dir}")


if __name__ == "__main__":
    main()
