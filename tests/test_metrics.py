import numpy as np
import pytest

from src.metrics import confusion_matrix, summary, weighted_accuracy, weighted_accuracy_from_cm

# Class counts of the public CirCor set (Present, Unknown, Absent)
N_P, N_U, N_A = 179, 68, 695
Y_ALL = ["Present"] * N_P + ["Unknown"] * N_U + ["Absent"] * N_A


def test_perfect_prediction():
    y = ["Present", "Unknown", "Absent", "Absent"]
    assert weighted_accuracy(y, y) == 1.0


def test_confusion_matrix_layout():
    # rows = predicted, columns = true (Table 1 of the project statement)
    A = confusion_matrix(["Present", "Absent"], ["Absent", "Absent"])
    assert A[2, 0] == 1 and A[2, 2] == 1 and A.sum() == 2


def test_hand_computed_example():
    y_true = ["Present", "Present", "Unknown", "Absent", "Absent"]
    y_pred = ["Present", "Absent", "Absent", "Absent", "Present"]
    # numerator 5*1 + 3*0 + 1*1 = 6 ; denominator 5*2 + 3*1 + 1*2 = 15
    assert weighted_accuracy(y_true, y_pred) == pytest.approx(6 / 15)


def test_formula_of_the_statement():
    A = np.array([[10, 2, 7], [3, 4, 5], [6, 1, 50]])   # A[i, j]: predicted i, true j
    num = 5 * A[0, 0] + 3 * A[1, 1] + A[2, 2]
    den = 5 * A[:, 0].sum() + 3 * A[:, 1].sum() + A[:, 2].sum()
    assert weighted_accuracy_from_cm(A) == pytest.approx(num / den)


@pytest.mark.parametrize("label, expected", [("Present", 895 / 1794), ("Absent", 695 / 1794),
                                             ("Unknown", 204 / 1794)])
def test_trivial_classifiers(label, expected):
    assert weighted_accuracy(Y_ALL, [label] * len(Y_ALL)) == pytest.approx(expected)


def test_integer_labels_match_strings():
    y_true, y_pred = [0, 1, 2, 2], [0, 2, 2, 0]
    names = np.array(["Present", "Unknown", "Absent"])
    assert weighted_accuracy(y_true, y_pred) == weighted_accuracy(names[y_true], names[y_pred])


def test_summary_recalls():
    s = summary(["Present", "Unknown", "Absent", "Absent"], ["Present", "Absent", "Absent", "Present"])
    assert s["recall_Present"] == 1.0 and s["recall_Unknown"] == 0.0 and s["recall_Absent"] == 0.5
