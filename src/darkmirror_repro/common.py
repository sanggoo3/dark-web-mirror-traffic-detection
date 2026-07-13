from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.1
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.01), 2)
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_splits(y: np.ndarray, seed: int = RANDOM_STATE):
    idx = np.arange(len(y))
    train_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(idx, y)
    )
    y_train = y[train_idx]
    train_inner_idx = np.arange(len(y_train))
    subtrain_idx, val_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE_WITHIN_TRAIN, random_state=seed).split(
            train_inner_idx, y_train
        )
    )
    return train_idx, test_idx, subtrain_idx, val_idx


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        rows.append({"Threshold": float(threshold), **binary_metrics(y_true, y_prob, float(threshold))})
    df = pd.DataFrame(rows)
    best = df.sort_values(["F1-score", "Accuracy", "Threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["Threshold"]), df


def paper_round(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rounded_cols = METRIC_COLUMNS + [
        "Threshold",
        "threshold",
        "val_Accuracy",
        "val_Precision",
        "val_Recall",
        "val_F1-score",
        "val_ROC-AUC",
        "Val ROC-AUC",
        "Dropout",
        "Learning rate",
        "Weight decay",
        "Preprocess (ms)",
        "Inference (ms)",
        "Total (ms)",
    ]
    for col in rounded_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda value: f"{float(value):.4f}")
    return out


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, title: str, note: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    lines = [f"# {title}", "", df.to_markdown(index=False, disable_numparse=True), ""]
    if note:
        lines.extend([f"Note: {note}", ""])
    (out_dir / f"{stem}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
