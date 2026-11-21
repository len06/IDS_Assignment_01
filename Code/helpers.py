"""Helper functions for the speed-trap assignment (given to the students, do not edit).

Contents
* plot style (``BLUE``, ``BLUE_RAMP``) - applied on import
* Question 4: ``time_splits``, ``to_long``, ``check_no_leakage``, ``region_neighbours``, ``day_auc``, ``evaluate``
* Question 5: ``lag_pairs``, ``stratified_permutations``
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent / "data"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 110,
})

TEST_START = pd.Timestamp("2025-10-01")  # test set = season 2025/26
MIN_HISTORY = 20  # the first measured days have too little history and are dropped


# --------------------------------------------------------------------------- Question 4: data handling
def to_long(frames: dict[str, pd.DataFrame | pd.Series], hits: pd.DataFrame) -> pd.DataFrame:
    """Stack wide feature tables to one row per (measured day, region) and add label and split.

    ``frames`` maps a feature name to either a day x region DataFrame (same index/columns as ``hits``)
    or a Series indexed by day (day-level feature, repeated for every region).
    The first ``MIN_HISTORY`` measured days are dropped. Adds ``hit`` (0/1) and ``split`` (train/test).
    """
    n_days, n_regions = hits.shape
    out = {"Datum": np.repeat(hits.index.to_numpy(), n_regions),
           "region": np.tile(hits.columns.to_numpy(), n_days),
           "hit": hits.to_numpy().ravel().astype(int)}
    for name, f in frames.items():
        if isinstance(f, pd.Series):
            out[name] = np.repeat(f.reindex(hits.index).to_numpy(), n_regions)
        else:
            out[name] = f.reindex(index=hits.index, columns=hits.columns).to_numpy().ravel()
    df = pd.DataFrame(out)
    df = df[df["Datum"] >= hits.index[MIN_HISTORY]].copy()
    df["split"] = np.where(df["Datum"] < TEST_START, "train", "test")
    return df.reset_index(drop=True)


def check_no_leakage(feature_fn, hits: pd.DataFrame, n_traps: pd.Series, cut: int = 400) -> None:
    """Truncation test: features of the first ``cut`` days must not change when later days are removed.

    ``feature_fn(hits, n_traps)`` has to return a dict of wide tables / Series as passed to ``to_long``.
    """
    full = feature_fn(hits, n_traps)
    part = feature_fn(hits.iloc[:cut], n_traps.iloc[:cut])
    assert full.keys() == part.keys(), "the set of features must not depend on the data"
    for name, f in full.items():
        a, b = f.iloc[:cut].to_numpy(), part[name].to_numpy()
        try:
            same = np.allclose(a.astype(float), b.astype(float), equal_nan=True)
        except ValueError:  # categorical (text) feature
            same = np.array_equal(a.astype(str), b.astype(str))
        if not same:
            raise AssertionError(f"feature '{name}' changes when future days are removed -> it uses future information")
    print(f"OK: none of the {len(full)} features uses information from later days.")


def time_splits(train: pd.DataFrame, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window cross-validation folds for ``GridSearchCV(cv=...)``.

    The training days are cut into ``n_folds + 1`` consecutive blocks; fold i trains on blocks 0..i-1
    and validates on block i, so the validation data always lie *after* the training data.
    Pass the result as ``cv=time_splits(train)``.
    """
    blocks = np.array_split(np.sort(train["Datum"].unique()), n_folds + 1)
    return [(np.flatnonzero(train["Datum"].isin(np.concatenate(blocks[:i]))),
             np.flatnonzero(train["Datum"].isin(blocks[i]))) for i in range(1, n_folds + 1)]


def region_neighbours(streets: pd.DataFrame, regions: pd.Series, n: int = 2) -> dict[int, list[int]]:
    """For every region the ``n`` regions whose centres (mean middle node of their streets) are closest."""
    centres = streets[["x", "y"]].groupby(regions).mean()
    dist = np.hypot(*(centres.to_numpy()[:, None, :] - centres.to_numpy()[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(dist, np.inf)
    return {r: [int(centres.index[j]) for j in np.argsort(dist[i])[:n]] for i, r in enumerate(centres.index)}


# --------------------------------------------------------------------------- Question 4: metrics
def day_auc(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """ROC AUC of the ranking of the regions *within* each day (the "where" metric).

    Days on which all or none of the regions have a trap are skipped. Everything that shifts all
    regions of a day alike (weekday, how many traps there are that day) cancels out.
    """
    d = df.assign(p=p)
    return d.groupby("Datum").apply(lambda g: roc_auc_score(g["hit"], g["p"]) if 0 < g["hit"].sum() < len(g) else np.nan,
                                   include_groups=False).dropna()


def evaluate(test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Table of log-loss, pooled ROC AUC and mean within-day AUC for each model."""
    rows = {name: {"log-loss": log_loss(test["hit"], p), "ROC AUC": roc_auc_score(test["hit"], p), "day AUC": day_auc(test, p).mean()}
            for name, p in predictions.items()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- Question 5
def lag_pairs(days: pd.DatetimeIndex, lags) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (a, b) of measured days that are exactly ``lag`` calendar days apart, for each lag in ``lags``.

    Returns two integer arrays ``a`` (earlier day) and ``b`` (later day) of positions in ``days``.
    Days without any trap are not part of ``days``, so a pair only exists if both days were measured.
    """
    pos = pd.Series(np.arange(len(days)), index=days)
    a_all, b_all = [], []
    for lag in lags:
        later = days + pd.Timedelta(days=int(lag))
        ok = later.isin(days)
        a_all.append(pos.to_numpy()[ok])
        b_all.append(pos.loc[later[ok]].to_numpy())
    return np.concatenate(a_all), np.concatenate(b_all)


def stratified_permutations(stratum: np.ndarray, n: int, rng: np.random.Generator):
    """Yield ``n`` random permutations ``p`` of ``0..len(stratum)-1`` with ``stratum[p[i]] == stratum[i]``.

    Use ``M[p]`` to assign to every date a *different* day's set of streets from the same stratum.
    """
    base = np.lexsort((np.arange(len(stratum)), stratum))
    for _ in range(n):
        p = np.empty(len(stratum), dtype=int)
        p[base] = np.lexsort((rng.random(len(stratum)), stratum))
        yield p
