from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .common import binary_metrics, repo_root, seed_everything, select_threshold, stratified_splits
from .models import build_model


def predict_proba(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def train_model(model, x_train, y_train, x_val, y_val, params, device, max_epochs=50, patience=10):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=max(params["batch_size"], 256),
        shuffle=False,
    )
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best_state = None
    best_epoch = 0
    best_auc = -np.inf
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_probs = []
        val_true = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_true.append(yb.numpy())
        val_prob = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_auc = binary_metrics(val_true, val_prob, threshold=0.5)["ROC-AUC"]

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reference TCN or ResNet1D on the final direction-only data.")
    parser.add_argument("--model", choices=["tcn", "resnet1d"], default="tcn")
    parser.add_argument(
        "--data",
        default="data/processed/final_seq506_posthandshake/direction_seq506_posthandshake.npz",
        help="Path relative to repository root or an absolute path.",
    )
    parser.add_argument("--out-dir", default="results/reproduced_training")
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

    model, params = build_model(args.model)
    start = time.time()
    model, best_auc, best_epoch = train_model(model, x_subtrain, y_subtrain, x_val, y_val, params, device)
    train_seconds = time.time() - start

    val_prob = predict_proba(model, x_val, device)
    test_prob = predict_proba(model, x_test, device)
    threshold, threshold_df = select_threshold(y_val, val_prob)
    metrics = binary_metrics(y_test, test_prob, threshold)

    row = {
        "model": args.model,
        "threshold": threshold,
        **metrics,
        "best_val_auc_from_training": float(best_auc),
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_seconds),
        "device": str(device),
    }
    pd.DataFrame([row]).to_csv(out_dir / f"{args.model}_reference_result.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / f"{args.model}_validation_threshold_search.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{args.model}_reference_result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

