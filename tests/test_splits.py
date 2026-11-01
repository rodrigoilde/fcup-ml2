import pytest

from src.config import SPLITS_FILE
from src.data import load_dataset
from src.splits import load_splits, make_splits, subject_groups


@pytest.fixture(scope="module")
def patients():
    return load_dataset()[0]


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_versioned_split_is_reproducible(patients, splits):
    fresh = make_splits(patients).sort_values("patient_id").reset_index(drop=True)
    saved = splits.sort_values("patient_id").reset_index(drop=True)
    assert (fresh[["patient_id", "subset", "fold"]] == saved[["patient_id", "subset", "fold"]]).all().all()


def test_every_patient_once(patients, splits):
    assert len(splits) == len(patients) == splits["patient_id"].nunique() == 942


def test_additional_id_pairs_are_grouped(patients):
    groups = dict(zip(patients["patient_id"], subject_groups(patients)))
    pairs = patients.dropna(subset=["additional_id"])
    assert len(pairs) == 140
    assert all(groups[a] == groups[int(b)] for a, b in zip(pairs["patient_id"], pairs["additional_id"]))


def test_subject_in_a_single_subset_and_fold(splits):
    # a subject group must never be split between test/dev or between two dev folds
    assert (splits.groupby("subject_group")[["subset", "fold"]].nunique() == 1).all().all()


def test_test_set_size_and_folds(splits):
    assert splits["subset"].value_counts()["test"] == pytest.approx(0.2 * 942, abs=10)
    assert set(splits.loc[splits["subset"] == "dev", "fold"]) == {0, 1, 2, 3, 4}
    assert (splits.loc[splits["subset"] == "test", "fold"] == -1).all()


def test_stratification(splits):
    overall = splits["murmur"].value_counts(normalize=True)
    for _, part in splits.groupby(["subset", "fold"]):
        props = part["murmur"].value_counts(normalize=True)
        assert ((props - overall).abs() < 0.03).all()
