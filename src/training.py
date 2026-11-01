"""Training loop, early stopping on the patient-level weighted accuracy, and cross-validation runner."""

import json
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .aggregation import predict_hierarchical, recordings_to_patients, windows_to_recordings
from .config import CLASS_TO_IDX, CLASSES, MODELS_DIR, RESULTS_DIR, SEED
from .features import Standardizer, stack_windows
from .metrics import weighted_accuracy
from .splits import fold_patients


def window_labels(recordings, rec_ids, mode="patient"):
    """Integer label of each window.

    mode="patient": every window inherits the label of its patient (default).
    mode="refined": windows of a Present patient recorded at a location where the murmur was NOT
        heard (see `Murmur locations`) are labelled Absent, which reduces label noise."""
    rec = recordings.set_index("recording_id").loc[rec_ids]
    labels = rec["murmur"].to_numpy()
    if mode == "refined":
        labels = np.where((labels == "Present") & ~rec["murmur_audible_here"].to_numpy(), "Absent", labels)
    return np.array([CLASS_TO_IDX[l] for l in labels])


class PatientWAEarlyStopping(keras.callbacks.Callback):
    """After each epoch, predict the validation windows, aggregate them per patient and compute the
    weighted accuracy (fixed thresholds 0.5). Keeps the weights of the best epoch and stops after
    `patience` epochs without improvement. Early stopping thus follows the metric we are evaluated
    on, rather than the window-level loss (windows are not what we are asked to classify)."""

    def __init__(self, X_val, rec_ids_val, recordings_val, patience=8):
        super().__init__()
        self.X_val, self.rec_ids_val, self.recordings_val = X_val, rec_ids_val, recordings_val
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.best_wa, self.best_epoch, self.best_weights, self.wait = -1.0, 0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val, batch_size=512, verbose=0)
        patients = recordings_to_patients(windows_to_recordings(probs, self.rec_ids_val, self.recordings_val))
        wa = weighted_accuracy(patients["murmur"], predict_hierarchical(patients))
        logs["val_patient_wa"] = wa                    # recorded by the History callback
        if wa > self.best_wa:
            self.best_wa, self.best_epoch, self.wait = wa, epoch, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_model(build_fn, X_train, y_train, X_val, y_val, rec_ids_val, recordings_val,
                epochs=40, batch_size=64, lr=1e-3, patience=8, clipnorm=None, verbose=2):
    """Build, compile and train a model from scratch. Returns (model, history dict)."""
    model = build_fn(input_shape=X_train.shape[1:])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    # 'balanced' class weights: n_samples / (n_classes * n_samples_of_class), so each class
    # contributes the same total weight to the loss despite Absent being ~74% of the windows.
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_train)
    early = PatientWAEarlyStopping(X_val, rec_ids_val, recordings_val, patience)
    t0 = time.time()
    h = model.fit(X_train, keras.utils.to_categorical(y_train, len(CLASSES)),
                  validation_data=(X_val, keras.utils.to_categorical(y_val, len(CLASSES))),
                  epochs=epochs, batch_size=batch_size, class_weight=dict(enumerate(cw)),
                  callbacks=[early], verbose=verbose)
    history = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    history.update(best_epoch=early.best_epoch + 1, best_val_patient_wa=early.best_wa,
                   seconds=time.time() - t0, params=model.count_params(),
                   class_weight=[float(c) for c in cw])
    return model, history


def run_cv(name, build_fn, feats, recordings, splits, folds=range(5), std_axis=(0, 2),
           add_channel=False, label_mode="patient", retrain=False, **train_kwargs):
    """Train/validate `build_fn` on the dev folds and return the validation predictions per recording.

    For each fold k: train on the other dev folds, standardise with training statistics, early-stop
    on fold k, then predict fold k. Models go to models/<name>_fold<k>.keras and histories to
    results/history_<name>.json. With retrain=False, previously saved results are loaded instead."""
    oof_path = RESULTS_DIR / f"oof_recordings_{name}.csv"
    hist_path = RESULTS_DIR / f"history_{name}.json"
    folds = list(folds)
    if not retrain and oof_path.exists() and hist_path.exists():
        oof, histories = pd.read_csv(oof_path), json.loads(hist_path.read_text())
        if set(oof["fold"]) >= set(folds):
            print(f"Loaded saved results of '{name}' (set RETRAIN=True to train again).")
            return oof[oof["fold"].isin(folds)].reset_index(drop=True), {int(k): v for k, v in histories.items()}

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    oof, histories = [], {}
    for k in folds:
        train_ids, val_ids = fold_patients(splits, k)
        rec_tr = recordings[recordings["patient_id"].isin(train_ids)]
        rec_va = recordings[recordings["patient_id"].isin(val_ids)]
        X_tr, rid_tr = stack_windows(rec_tr, feats)
        X_va, rid_va = stack_windows(rec_va, feats)
        y_tr, y_va = window_labels(recordings, rid_tr, label_mode), window_labels(recordings, rid_va, "patient")
        std = Standardizer(std_axis).fit(X_tr)
        np.savez(MODELS_DIR / f"{name}_fold{k}_std.npz", mean=std.mean, std=std.std)
        X_tr, X_va = std.transform(X_tr), std.transform(X_va)
        if add_channel:
            X_tr, X_va = X_tr[..., None], X_va[..., None]
        print(f"[{name}] fold {k}: {len(train_ids)} train patients ({len(X_tr)} windows), "
              f"{len(val_ids)} val patients ({len(X_va)} windows)")

        keras.utils.set_random_seed(SEED + k)
        model, hist = train_model(build_fn, X_tr, y_tr, X_va, y_va, rid_va, rec_va, **train_kwargs)
        model.save(MODELS_DIR / f"{name}_fold{k}.keras")
        histories[k] = hist
        print(f"[{name}] fold {k}: best epoch {hist['best_epoch']}, val patient WA "
              f"{hist['best_val_patient_wa']:.3f}, {hist['seconds'] / 60:.1f} min")

        rec_probs = windows_to_recordings(model.predict(X_va, batch_size=512, verbose=0), rid_va, rec_va)
        rec_probs["fold"] = k
        oof.append(rec_probs)
        del X_tr, X_va
        keras.backend.clear_session()

    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(oof_path, index=False)
    hist_path.write_text(json.dumps({str(k): v for k, v in histories.items()}))
    return oof, histories


def fold_standardizer(name, k, feats, recordings, splits, std_axis):
    """Standardizer of fold k: saved statistics if available, otherwise refitted on its training folds."""
    path = MODELS_DIR / f"{name}_fold{k}_std.npz"
    std = Standardizer(std_axis)
    if path.exists():
        with np.load(path) as f:
            std.mean, std.std = f["mean"], f["std"]
    else:
        train_ids, _ = fold_patients(splits, k)
        std.fit(stack_windows(recordings[recordings["patient_id"].isin(train_ids)], feats)[0])
        np.savez(path, mean=std.mean, std=std.std)
    return std


def predict_with_fold_models(name, feats, recordings, splits, target, folds=range(5), std_axis=(0, 2),
                             add_channel=False):
    """Ensemble of the fold models: average their window probabilities on new recordings `target`
    (test set or hidden set; `feats` must contain their features) -> probabilities per recording."""
    X_t, rid_t = stack_windows(target, feats)
    probs = np.zeros((len(X_t), len(CLASSES)))
    for k in folds:
        X = fold_standardizer(name, k, feats, recordings, splits, std_axis).transform(X_t)
        model = keras.models.load_model(MODELS_DIR / f"{name}_fold{k}.keras")
        probs += model.predict(X[..., None] if add_channel else X, batch_size=512, verbose=0) / len(folds)
    return windows_to_recordings(probs, rid_t, target)


def plot_histories(histories, title=""):
    """Training curves of every fold: loss, window accuracy and validation patient-level WA."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for k, h in histories.items():
        c = f"C{int(k)}"
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], "--", color=c, alpha=0.7)
        axes[0].plot(ep, h["val_loss"], "-", color=c, label=f"fold {k}")
        axes[1].plot(ep, h["accuracy"], "--", color=c, alpha=0.7)
        axes[1].plot(ep, h["val_accuracy"], "-", color=c)
        axes[2].plot(ep, h["val_patient_wa"], "-", color=c)
        axes[2].plot(h["best_epoch"], h["best_val_patient_wa"], "o", color=c, markeredgecolor="black")
    axes[0].set_title("Loss (dashed: train, solid: val)")
    axes[1].set_title("Window accuracy (dashed: train, solid: val)")
    axes[2].set_title("Validation patient-level WA (dot: best epoch)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    return fig
