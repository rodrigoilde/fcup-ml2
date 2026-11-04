import numpy as np
import pytest

from src.metrics import confusion_matrix_pua, weighted_accuracy, per_class_recall
from src.data import load_patients, recording_label
from src.splits import make_splits, check_splits


def test_confusion_matrix_rows_are_predictions():
    m = confusion_matrix_pua(["Present"], ["Absent"])
    assert m[2, 0] == 1 and m.sum() == 1          # predicted Absent (row 2), true Present (column 0)


def test_weighted_accuracy_hand_computed():
    y_true = ["Present", "Present", "Unknown", "Absent", "Absent", "Absent"]
    y_pred = ["Present", "Absent", "Unknown", "Absent", "Present", "Absent"]
    # numerator 5*1 + 3*1 + 1*2 = 10; denominator 5*2 + 3*1 + 1*3 = 16
    assert weighted_accuracy(y_true, y_pred) == pytest.approx(10 / 16)


def test_weighted_accuracy_matches_statement_formula():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 500)
    y_pred = rng.integers(0, 3, 500)
    m = confusion_matrix_pua(y_true, y_pred)
    (pp, pu, pa), (up, uu, ua), (ap, au, aa) = m
    expected = (5 * pp + 3 * uu + aa) / (5 * (pp + up + ap) + 3 * (pu + uu + au) + (pa + ua + aa))
    assert weighted_accuracy(y_true, y_pred) == pytest.approx(expected)


def test_trivial_classifiers_on_public_set():
    y = load_patients()["murmur"].tolist()
    n = len(y)
    assert weighted_accuracy(y, ["Absent"] * n) == pytest.approx(695 / 1794)
    assert weighted_accuracy(y, ["Present"] * n) == pytest.approx(895 / 1794)
    assert weighted_accuracy(y, y) == 1.0
    assert per_class_recall(y, ["Present"] * n)["Present"] == 1.0


def test_recording_label_is_location_aware():
    assert recording_label("Present", "AV+TV", "TV") == "Present"
    assert recording_label("Present", "AV+TV", "MV") == "Absent"
    assert recording_label("Unknown", np.nan, "MV") == "Unknown"
    assert recording_label("Absent", np.nan, "AV") == "Absent"


def test_linked_patients_share_a_group():
    p = load_patients()
    linked = p.dropna(subset=["additional_id"])
    gid = dict(zip(p["patient_id"], p["group_id"]))
    assert all(gid[a] == gid[b] for a, b in zip(linked["patient_id"], linked["additional_id"]))


def test_splits_do_not_break_groups_and_are_reproducible():
    p = load_patients()
    s1, s2 = make_splits(p), make_splits(p)
    check_splits(s1)
    assert s1.equals(s2)
    assert 0.17 < (s1["split"] == "test").mean() < 0.23


def test_patient_aggregation_and_decisions():
    import pandas as pd
    from src.aggregate import patient_probs, decide_argmax, decide_bayes
    rec = pd.DataFrame({"patient_id": [1, 1, 2], "p_Present": [0.9, 0.1, 0.2],
                        "p_Unknown": [0.05, 0.1, 0.1], "p_Absent": [0.05, 0.8, 0.7]})
    mean = patient_probs(rec, "mean")
    assert mean.loc[1, "p_Present"] == pytest.approx(0.5)
    mx = patient_probs(rec, "max_present")
    assert mx.loc[1, "p_Present"] > mean.loc[1, "p_Present"]
    assert np.allclose(mx.sum(axis=1), 1.0)
    # patient 2: argmax says Absent, but 5 * 0.2 = 1.0 > 1 * 0.7, so the Bayes rule says Present
    assert decide_argmax(mean.loc[[2]].to_numpy())[0] == "Absent"
    assert decide_bayes(mean.loc[[2]].to_numpy())[0] == "Present"
