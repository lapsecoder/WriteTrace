"""Grouped, stratified train/val/test splitting (leakage-safe, deterministic)."""
from __future__ import annotations

import numpy as np
import pandas as pd


RARE_STRATUM_MIN = 10  # strata rarer than this are coarsened to "<label>|rare"


def make_stratum(label_col: pd.Series, stratum_col: pd.Series) -> pd.Series:
    """Combined stratification key; rare strata coarsened so SGKF stays valid."""
    raw = label_col.astype(str) + "|" + stratum_col.astype(str)
    counts = raw.value_counts()
    rare = counts[counts < RARE_STRATUM_MIN].index
    return raw.where(~raw.isin(rare), label_col.astype(str) + "|rare")


def grouped_split(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    stratum_col: str = "stratum",
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> pd.Series:
    """Assign each row to train/val/test.

    Guarantees (tested):
    - no group_id ever appears in more than one split;
    - every row is assigned;
    - deterministic for a fixed seed and row order;
    - stratification approximates fractions per stratum.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, fractions
    n = len(df)
    if n == 0:
        return pd.Series(dtype=str)
    from sklearn.model_selection import StratifiedGroupKFold  # lazy: keeps spawn
    # workers light (no scipy/sklearn import in child processes)
    n_splits = max(5, int(round(1.0 / min(fractions))))
    y = df[stratum_col].to_numpy()
    groups = df[group_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(n, -1, dtype=np.int64)
    for k, (_tr, te) in enumerate(sgkf.split(np.zeros(n), y, groups)):
        fold[te] = k
    assert (fold >= 0).all(), "unassigned rows"

    # Map folds -> split names. Fractions are multiples of 1/n_splits when possible;
    # with n_splits=5 and (0.6,0.2,0.2): folds [0,1,2]->train, {3}->val, {4}->test.
    acc, boundaries = 0.0, []
    for f in fractions:
        acc += f
        boundaries.append(acc)
    names = np.empty(n, dtype=object)
    fold_order = list(range(n_splits))
    # deterministic fold -> split assignment honouring cumulative fractions
    cumulative, fi = 0.0, 0
    split_of_fold: dict[int, str] = {}
    labels_seq = ["train", "val", "test"]
    for si, f in enumerate(fractions):
        take = int(round(f * n_splits))
        for _ in range(take):
            if fi < n_splits:
                split_of_fold[fold_order[fi]] = labels_seq[si]
                fi += 1
    for k in range(n_splits):
        names[fold == k] = split_of_fold.get(k, "train")
    return pd.Series(names, index=df.index, name="split")
