"""From window-level probabilities to exactly one prediction per patient.

1. window -> recording: average of the softmax probabilities of the windows of a recording.
2. recording -> patient, hierarchical rule (as in the PhysioNet 2022 challenge baselines):
       Present  if max over recordings of P(Present) > tau_present
       Unknown  elif max over recordings of P(Unknown) > tau_unknown
       Absent   otherwise
   A murmur may be audible at only one auscultation location (34 of the 179 Present patients), so
   one confident recording is enough: averaging over locations would dilute it.
The thresholds are chosen on validation data to maximise the weighted accuracy.
"""

import numpy as np
import pandas as pd

from .config import CLASSES
from .metrics import weighted_accuracy

PROB_COLS = [f"p_{c}" for c in CLASSES]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def windows_to_recordings(probs, rec_ids, recordings):
    """Average window probabilities per recording -> recordings DataFrame with p_Present, p_Unknown, p_Absent."""
    df = pd.DataFrame(probs, columns=PROB_COLS)
    df["recording_id"] = rec_ids
    rec = df.groupby("recording_id", sort=False)[PROB_COLS].mean().reset_index()
    return recordings[["recording_id", "patient_id", "location", "murmur"]].merge(rec, on="recording_id")


def recordings_to_patients(rec_probs):
    """One row per patient with the max and mean of each class probability over its recordings."""
    g = rec_probs.groupby("patient_id")
    out = g[PROB_COLS].max().add_suffix("_max").join(g[PROB_COLS].mean().add_suffix("_mean"))
    out["murmur"] = g["murmur"].first()
    if "fold" in rec_probs:
        out["fold"] = g["fold"].first()
    return out.reset_index()


def predict_hierarchical(patients, tau_present=0.5, tau_unknown=0.5):
    p, u = patients["p_Present_max"].to_numpy(), patients["p_Unknown_max"].to_numpy()
    return np.where(p > tau_present, "Present", np.where(u > tau_unknown, "Unknown", "Absent"))


def predict_argmax_mean(patients):
    """Simpler alternative: class with the highest probability averaged over all recordings."""
    mean = patients[[f"{c}_mean" for c in PROB_COLS]].to_numpy()
    return np.array(CLASSES)[mean.argmax(axis=1)]


def tune_thresholds(patients, grid=THRESHOLD_GRID):
    """Grid search of (tau_present, tau_unknown) maximising the weighted accuracy on `patients`."""
    best = (0.5, 0.5, -1.0)
    for tp in grid:
        for tu in grid:
            wa = weighted_accuracy(patients["murmur"], predict_hierarchical(patients, tp, tu))
            if wa > best[2]:
                best = (float(tp), float(tu), wa)
    return best


def cross_fitted_predictions(patients_oof):
    """Predict each validation fold with thresholds tuned on the OTHER folds' validation predictions.

    Tuning the thresholds on the same patients we report on would inflate the cross-validation
    score; this way the reported per-fold WA never uses its own labels for threshold selection.
    `patients_oof` must have a `fold` column. Returns (predictions, {fold: (tau_p, tau_u)})."""
    preds = pd.Series(index=patients_oof.index, dtype=object)
    taus = {}
    for k in sorted(patients_oof["fold"].unique()):
        other, this = patients_oof["fold"] != k, patients_oof["fold"] == k
        if other.sum() == 0:                              # only one fold available: no cross-fitting
            tp, tu = 0.5, 0.5
        else:
            tp, tu, _ = tune_thresholds(patients_oof[other])
        taus[int(k)] = (tp, tu)
        preds[this] = predict_hierarchical(patients_oof[this], tp, tu)
    return preds.to_numpy(), taus


RULES = ["argmax_mean", "hierarchical_fixed", "hierarchical_tuned"]


def evaluate_cv(rec_oof):
    """Patient-level evaluation of cross-validation predictions with three aggregation rules:

    * argmax_mean: class with the highest mean probability over the patient's recordings;
    * hierarchical_fixed: hierarchical rule with tau_present = tau_unknown = 0.5;
    * hierarchical_tuned: hierarchical rule with thresholds cross-fitted on the other folds.

    Returns (patients with one prediction column per rule, per-fold table of weighted accuracies)."""
    pats = recordings_to_patients(rec_oof)
    pats["pred_argmax_mean"] = predict_argmax_mean(pats)
    pats["pred_hierarchical_fixed"] = predict_hierarchical(pats)
    pats["pred_hierarchical_tuned"], taus = cross_fitted_predictions(pats)
    rows = []
    for k, part in pats.groupby("fold"):
        row = {"fold": int(k), "patients": len(part)}
        for rule in RULES:
            row[f"WA_{rule}"] = weighted_accuracy(part["murmur"], part[f"pred_{rule}"])
        row["tau_present"], row["tau_unknown"] = taus[int(k)]
        rows.append(row)
    return pats, pd.DataFrame(rows)
