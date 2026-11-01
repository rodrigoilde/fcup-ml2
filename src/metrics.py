"""Patient-level evaluation: weighted accuracy of the PhysioNet 2022 murmur task and extra metrics.

The weighted accuracy follows `compute_weighted_accuracy` of the official evaluation code
(https://github.com/physionetchallenges/evaluation-2022): with the confusion matrix
A[i, j] = number of patients predicted as class i whose true class is j, and the class weights
w = (5, 3, 1) for (Present, Unknown, Absent),

    WA = sum_j w_j A[j, j] / sum_ij w_j A[i, j]

i.e. the accuracy obtained when each Present patient counts 5 times and each Unknown patient 3 times.
"""

import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_TO_IDX, CLASS_WEIGHTS_WA, CLASSES


def _to_idx(y):
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        return np.array([CLASS_TO_IDX[v] for v in y])
    return y.astype(int)


def confusion_matrix(y_true, y_pred):
    """3x3 matrix, rows = predicted class, columns = true class (order Present, Unknown, Absent)."""
    A = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(_to_idx(y_true), _to_idx(y_pred)):
        A[p, t] += 1
    return A


def weighted_accuracy_from_cm(A):
    W = np.tile(CLASS_WEIGHTS_WA, (len(CLASSES), 1))      # weight of the TRUE class (column)
    B = W * A
    return float(np.trace(B) / B.sum()) if B.sum() > 0 else float("nan")


def weighted_accuracy(y_true, y_pred):
    return weighted_accuracy_from_cm(confusion_matrix(y_true, y_pred))


def summary(y_true, y_pred):
    """Weighted accuracy (main metric) plus additional metrics: accuracy, per-class recall and F1."""
    A = confusion_matrix(y_true, y_pred)
    out = {"weighted_accuracy": weighted_accuracy_from_cm(A), "accuracy": float(np.trace(A) / A.sum())}
    f1s = []
    for i, c in enumerate(CLASSES):
        tp, n_true, n_pred = A[i, i], A[:, i].sum(), A[i, :].sum()
        recall = tp / n_true if n_true else float("nan")
        precision = tp / n_pred if n_pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[f"recall_{c}"] = float(recall)
        out[f"f1_{c}"] = float(f1)
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def plot_confusion_matrix(A, title="", ax=None):
    """Plot a confusion matrix laid out as Table 1 of the project statement."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(A, cmap="Blues")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, int(A[i, j]), ha="center", va="center",
                    color="white" if A[i, j] > A.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted")
    ax.set_title(title or f"WA = {weighted_accuracy_from_cm(A):.3f}")
    return ax
