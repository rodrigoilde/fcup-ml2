"""Patient-level evaluation protocol.

* 140 patients have an `Additional ID` pointing to another patient of the public set: they are the
  same subject, recorded in both campaigns (CC2014 and CC2015). We merge them into one *subject
  group*, so a subject never appears in two subsets.
* 20% of the subject groups form a hold-out **test** set, used only once at the very end.
* The remaining 80% (**dev**) is split into 5 folds: in iteration k, fold k is the validation set
  (early stopping, threshold tuning, model selection) and the other 4 folds are the training set.

Both splits are stratified by the murmur label, so every subset has similar class proportions.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import SEED, SPLITS_FILE

N_FOLDS = 5
TEST_FRACTION = 0.2


def subject_groups(patients):
    """Map each patient id to a subject group id (smallest patient id of the linked component)."""
    parent = {pid: pid for pid in patients["patient_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pid, other in zip(patients["patient_id"], patients["additional_id"]):
        if pd.notna(other) and int(other) in parent:
            a, b = find(pid), find(int(other))
            parent[max(a, b)] = min(a, b)
    return patients["patient_id"].map(find)


def make_splits(patients, n_folds=N_FOLDS, test_fraction=TEST_FRACTION, seed=SEED):
    """Return a DataFrame: patient_id, subject_group, murmur, subset ('test'/'dev'), fold (-1 for test)."""
    df = patients[["patient_id", "murmur"]].copy()
    df["subject_group"] = subject_groups(patients).values

    outer = StratifiedGroupKFold(n_splits=round(1 / test_fraction), shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, df["murmur"], df["subject_group"]))
    df["subset"] = "dev"
    df.loc[df.index[test_idx], "subset"] = "test"
    df["fold"] = -1

    dev = df.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(dev, dev["murmur"], dev["subject_group"])):
        df.loc[dev.index[val_idx], "fold"] = k
    return df[["patient_id", "subject_group", "murmur", "subset", "fold"]]


def save_splits(splits, path=SPLITS_FILE):
    splits.to_csv(path, index=False)


def load_splits(path=SPLITS_FILE):
    return pd.read_csv(path)


def fold_patients(splits, k):
    """Patient ids of the training and validation sets of dev iteration k."""
    dev = splits[splits["subset"] == "dev"]
    return (dev.loc[dev["fold"] != k, "patient_id"].to_numpy(),
            dev.loc[dev["fold"] == k, "patient_id"].to_numpy())


def test_patients(splits):
    return splits.loc[splits["subset"] == "test", "patient_id"].to_numpy()
