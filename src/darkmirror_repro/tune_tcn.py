from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import binary_metrics, paper_round, repo_root, seed_everything, select_threshold, stratified_splits
from .models import DirectionTCN
from .train_reference_models import predict_proba, train_model


def candidate_grid() -> list[dict]:
    rows = []
    idx = 1
    for channels, kernel_size, dropout in itertools.product(
        [(16, 32, 64), (32, 64, 128)],
        [3, 5],
        [0.1, 0.2],
    ):
        rows.append(
            {
                "candidate_id": f"tcn_grid_{idx:02d}",
                "channels": channels,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "fc_hidden": 64,
                "batch_size": 64,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "max_epochs": 50,
                "patience": 10,
            }
        )
        idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TCN hyperparameter grid on the final seq_len=506 input.")
    parser.add_argument(
        "--data",
        default="data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz",
        help="Path relative to repository root or an absolute path.",
    )
    parser.add_argument("--out-dir", default="results/reproduced_tcn_tuning")
    args = parser.parse_args()

    root = repo_root()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(data_path, allow_pickle=True)
    x = data["X_dir_cnn"].astype(np.float32)
    y = data["y"].astype(int)

    train_idx, test_idx, subtrain_idx, val_idx = stratified_splits(y)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_subtrain = x_train[subtrain_idx]
    y_subtrain = y_train[subtrain_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    rows = []
    for candidate in candidate_grid():
        seed_everything()
        model = DirectionTCN(
            channels=candidate["channels"],
            kernel_size=candidate["kernel_size"],
            dropout=candidate["dropout"],
            fc_hidden=candidate["fc_hidden"],
        )
        params = {
            "batch_size": candidate["batch_size"],
            "lr": candidate["lr"],
            "weight_decay": candidate["weight_decay"],
        }
        start = time.time()
        model, best_auc, best_epoch = train_model(
            model,
            x_subtrain,
            y_subtrain,
            x_val,
            y_val,
            params,
            device,
            max_epochs=candidate["max_epochs"],
            patience=candidate["patience"],
        )
        train_seconds = time.time() - start
        val_prob = predict_proba(model, x_val, device)
        test_prob = predict_proba(model, x_test, device)
        threshold, threshold_df = select_threshold(y_val, val_prob)
        val_metrics = binary_metrics(y_val, val_prob, threshold)
        test_metrics = binary_metrics(y_test, test_prob, threshold)
        threshold_df.to_csv(out_dir / f"{candidate['candidate_id']}_val_threshold_search.csv", index=False)

        row = {
            **candidate,
            "channels": str(candidate["channels"]),
            "threshold": threshold,
            "val_Accuracy": val_metrics["Accuracy"],
            "val_Precision": val_metrics["Precision"],
            "val_Recall": val_metrics["Recall"],
            "val_F1-score": val_metrics["F1-score"],
            "val_ROC-AUC": val_metrics["ROC-AUC"],
            "Accuracy": test_metrics["Accuracy"],
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "F1-score": test_metrics["F1-score"],
            "ROC-AUC": test_metrics["ROC-AUC"],
            "best_val_auc_from_training": float(best_auc),
            "best_epoch": int(best_epoch),
            "train_seconds": float(train_seconds),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    df = pd.DataFrame(rows).sort_values(["val_ROC-AUC", "val_F1-score", "ROC-AUC"], ascending=[False, False, False])
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(out_dir / "tcn_hyperparameter_tuning_results.csv", index=False, encoding="utf-8-sig")
    paper_cols = [
        "rank",
        "candidate_id",
        "channels",
        "kernel_size",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "threshold",
        "val_ROC-AUC",
        "ROC-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]
    paper = paper_round(df[paper_cols])
    paper.to_csv(out_dir / "tcn_hyperparameter_tuning_results_paper.csv", index=False, encoding="utf-8-sig")
    (out_dir / "tcn_hyperparameter_tuning_results_paper.md").write_text(
        "\n".join(["# TCN Hyperparameter Tuning Results", "", paper.to_markdown(index=False, disable_numparse=True), ""]),
        encoding="utf-8",
    )
    print(f"[done] TCN tuning results: {out_dir}")


if __name__ == "__main__":
    main()
