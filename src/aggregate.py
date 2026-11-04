"""From recording-level probabilities to one decision per patient."""
import numpy as np
import pandas as pd

from .config import CLASSES, CLASS_WEIGHTS_METRIC

PROB_COLS = [f"p_{c}" for c in CLASSES]


def patient_probs(rec_probs, aggregation="mean"):
    """rec_probs: one row per recording with `patient_id` and p_Present/p_Unknown/p_Absent.

    - "mean":        average each probability over the patient's recordings;
    - "max_present": a patient has a murmur if it is heard at ANY location, so P(Present) is the max over
                     recordings; P(Unknown) and P(Absent) are averaged; the result is renormalized.
    """
    g = rec_probs.groupby("patient_id")
    p = g[PROB_COLS].mean()
    if aggregation == "mean":
        pass
    elif aggregation == "max_present":
        p["p_Present"] = g["p_Present"].max()
        p = p.div(p.sum(axis=1), axis=0)
    else:
        raise ValueError(aggregation)
    return p


def decide_argmax(probs):
    return np.array(CLASSES)[np.argmax(np.asarray(probs), axis=1)]


def decide_bayes(probs, weights=CLASS_WEIGHTS_METRIC):
    """Bayes decision for the weighted accuracy: a correct prediction of class c is worth w_c, so the
    expected gain of predicting c is w_c * p_c (the denominator of WA does not depend on predictions)."""
    return np.array(CLASSES)[np.argmax(np.asarray(probs) * np.asarray(weights), axis=1)]


DECISIONS = {"argmax": decide_argmax, "bayes": decide_bayes}


def predict_patients(rec_probs, aggregation, decision):
    """-> DataFrame indexed by patient_id with the aggregated probabilities and the predicted label."""
    p = patient_probs(rec_probs, aggregation)
    p["pred"] = DECISIONS[decision](p[PROB_COLS].to_numpy())
    return p
