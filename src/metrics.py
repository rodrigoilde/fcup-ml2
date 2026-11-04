"""Patient-level evaluation, following Section 3.4 of the project statement.

Convention (Table 1 of the statement, and the official Challenge code): rows = PREDICTED class,
columns = TRUE class, class order Present, Unknown, Absent. Note that sklearn uses the transpose.
"""
import numpy as np
import matplotlib.pyplot as plt

from .config import CLASSES, CLASS_WEIGHTS_METRIC


def _to_idx(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels
    return np.array([CLASSES.index(l) for l in labels])


def confusion_matrix_pua(y_true, y_pred):
    """m[i, j] = number of patients predicted as class i whose true class is j."""
    t, p = _to_idx(y_true), _to_idx(y_pred)
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(m, (p, t), 1)
    return m


def weighted_accuracy(y_true, y_pred, weights=CLASS_WEIGHTS_METRIC):
    """WA = (5 m_PP + 3 m_UU + m_AA) / (5 n_P + 3 n_U + n_A)."""
    m = confusion_matrix_pua(y_true, y_pred)
    w = np.asarray(weights, dtype=float)
    return float((w * np.diag(m)).sum() / (w * m.sum(axis=0)).sum())


def per_class_recall(y_true, y_pred):
    m = confusion_matrix_pua(y_true, y_pred)
    n_true = m.sum(axis=0)
    return {c: (m[i, i] / n_true[i] if n_true[i] else np.nan) for i, c in enumerate(CLASSES)}


def macro_f1(y_true, y_pred):
    m = confusion_matrix_pua(y_true, y_pred)
    f1 = []
    for i in range(len(CLASSES)):
        tp, n_pred, n_true = m[i, i], m[i, :].sum(), m[:, i].sum()
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1))


def report(y_true, y_pred):
    """Main metric plus the additional ones, as a flat dict."""
    out = {"WA": weighted_accuracy(y_true, y_pred)}
    out.update({f"recall_{c}": r for c, r in per_class_recall(y_true, y_pred).items()})
    out["macro_F1"] = macro_f1(y_true, y_pred)
    return out


def plot_confusion(m, title=None, ax=None):
    ax = ax or plt.gca()
    ax.imshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("true"); ax.set_ylabel("predicted")
    if title:
        ax.set_title(title)
    return ax
