"""Evaluation protocol: a locked 20 % test set + 5-fold CV on the remaining development set.

Both splits are grouped by `group_id` (linked patients never end up in different subsets) and
stratified by murmur label x campaign.

Usage:  python -m src.splits     (writes splits/splits.csv)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data import load_patients

N_FOLDS = 5


def make_splits(patients, n_folds=N_FOLDS, seed=config.SEED):
    strata = (patients["murmur"] + "_" + patients["campaign"]).to_numpy()
    groups = patients["group_id"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)   # 1/5 = 20 % test
    dev_idx, test_idx = next(outer.split(np.zeros(len(patients)), strata, groups))

    out = patients[["patient_id", "group_id", "murmur", "campaign"]].copy()
    out["split"] = "dev"
    out.loc[out.index[test_idx], "split"] = "test"
    out["fold"] = -1

    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(np.zeros(len(dev_idx)), strata[dev_idx], groups[dev_idx])):
        out.loc[out.index[dev_idx[val_idx]], "fold"] = k
    check_splits(out)
    return out


def check_splits(splits):
    """Every group must live in exactly one subset (test, or one single CV fold)."""
    where = splits["split"] + splits["fold"].astype(str)
    n_places = where.groupby(splits["group_id"]).nunique()
    assert (n_places == 1).all(), f"groups split across subsets: {n_places[n_places > 1].index.tolist()}"
    assert (splits.loc[splits["split"] == "dev", "fold"] >= 0).all()


def load_splits():
    return pd.read_csv(config.SPLITS_CSV)


def proportions(splits):
    key = splits["split"].where(splits["split"] == "test", "fold " + splits["fold"].astype(str))
    tab = pd.crosstab(key, splits["murmur"])[config.CLASSES]
    tab["n"] = tab.sum(axis=1)
    for c in config.CLASSES:
        tab[f"%{c}"] = (100 * tab[c] / tab["n"]).round(1)
    return tab


if __name__ == "__main__":
    splits = make_splits(load_patients())
    config.SPLITS_CSV.parent.mkdir(exist_ok=True)
    splits.to_csv(config.SPLITS_CSV, index=False)
    print(proportions(splits))
    print(f"saved {config.SPLITS_CSV}")
