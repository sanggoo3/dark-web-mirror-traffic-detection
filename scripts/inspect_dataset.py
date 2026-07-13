from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def describe_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    y = data["y"]
    groups = pd.Series(data["groups"].astype(str))
    print(f"\n{path}")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  class0={int((y == 0).sum())}, class1={int((y == 1).sum())}, urls={groups.nunique()}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "data" / "processed").rglob("*.npz")):
        describe_npz(path)


if __name__ == "__main__":
    main()

