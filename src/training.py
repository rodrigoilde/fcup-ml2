"""Training utilities: crop datasets, recording-level inference, patient-level early stopping, CV loop."""
import time

import numpy as np
import pandas as pd
import keras

from . import config
from .aggregate import PROB_COLS, predict_patients
from .metrics import weighted_accuracy
from .preprocessing import sliding_windows

MODELS_DIR = config.RESULTS_DIR / "models"


def class_weights(labels, n_classes=3):
    """Inverse-frequency weights, normalized so that the mean weight over the samples is 1."""
    counts = np.bincount(labels, minlength=n_classes)
    return len(labels) / (n_classes * np.maximum(counts, 1))


class CropDataset(keras.utils.PyDataset):
    """Each epoch draws one random crop of `win` samples/frames from every recording.

    arrays[name] is a (T, C) array (CNN) or a (frames, states) tuple (multi-task RNN).
    With deterministic=True the central crop is used and nothing is shuffled (for validation curves).
    """

    def __init__(self, arrays, names, labels, win, batch_size=32, sample_weights=None, augment=False,
                 multitask=False, deterministic=False, seed=config.SEED, **kwargs):
        super().__init__(**kwargs)
        self.arrays, self.names, self.labels = arrays, list(names), np.asarray(labels)
        self.win, self.batch_size, self.sample_weights = win, batch_size, sample_weights
        self.augment, self.multitask, self.deterministic = augment, multitask, deterministic
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.names))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.names) / self.batch_size))

    def on_epoch_end(self):
        if not self.deterministic:
            self.rng.shuffle(self.order)

    def _crop(self, a):
        n = len(a[0]) if self.multitask else len(a)
        start = (n - self.win) // 2 if self.deterministic else self.rng.integers(0, n - self.win + 1)
        if self.multitask:
            return a[0][start:start + self.win], a[1][start:start + self.win]
        return a[start:start + self.win], None

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        xs, ss = [], []
        for j in idx:
            x, s = self._crop(self.arrays[self.names[j]])
            x = x.astype(np.float32)
            if self.augment:
                # random gain and polarity inversion; no time reversal (it would swap S1 and S2)
                x = x * self.rng.uniform(0.8, 1.25) * self.rng.choice([-1.0, 1.0])
            xs.append(x)
            ss.append(s)
        x, y = np.stack(xs), self.labels[idx]
        if self.multitask:
            y = {"murmur": y, "seg": np.stack(ss).astype(np.int32)}
        if self.sample_weights is None:
            return x, y
        w = self.sample_weights[idx]
        if self.multitask:
            w = {"murmur": w, "seg": np.ones((len(idx), self.win), dtype=np.float32)}
        return x, y, w


def predict_recordings(model, arrays, names, win, hop=None, multitask=False, max_windows=1024):
    """Sliding windows over each recording, window probabilities averaged per recording."""
    hop = hop or win // 2
    names = list(names)
    probs = np.zeros((len(names), 3), dtype=np.float32)
    buf, owner = [], []

    def flush():
        out = model.predict(np.concatenate(buf).astype(np.float32), batch_size=256, verbose=0)
        out = out["murmur"] if multitask else out
        own = np.concatenate(owner)
        for r in np.unique(own):
            probs[r] = out[own == r].mean(axis=0)
        buf.clear(); owner.clear()

    for r, name in enumerate(names):
        x = arrays[name][0] if multitask else arrays[name]
        w = sliding_windows(x, win, hop)
        buf.append(w); owner.append(np.full(len(w), r))
        if sum(len(b) for b in buf) >= max_windows:
            flush()
    if buf:
        flush()
    return pd.DataFrame(probs, columns=PROB_COLS).assign(recording=names)


class PatientWACallback(keras.callbacks.Callback):
    """Computes the patient-level weighted accuracy on the validation fold at the end of each epoch and
    logs it as `val_patient_wa`, so that EarlyStopping can monitor the metric we are evaluated on."""

    def __init__(self, arrays, val_recordings, patient_labels, win, aggregation, decision, multitask=False):
        super().__init__()
        self.arrays, self.val = arrays, val_recordings
        self.patient_labels = patient_labels
        self.win, self.aggregation, self.decision, self.multitask = win, aggregation, decision, multitask

    def on_epoch_end(self, epoch, logs=None):
        rp = predict_recordings(self.model, self.arrays, self.val["recording"], self.win, multitask=self.multitask)
        rp["patient_id"] = self.val["patient_id"].to_numpy()
        pp = predict_patients(rp, self.aggregation, self.decision)
        logs["val_patient_wa"] = weighted_accuracy(self.patient_labels.loc[pp.index], pp["pred"])


def fold_data(recordings, splits, fold):
    """Training / validation recordings of one CV fold (the test patients are never touched)."""
    dev = splits[splits["split"] == "dev"]
    val_ids = set(dev.loc[dev["fold"] == fold, "patient_id"])
    train_ids = set(dev.loc[dev["fold"] != fold, "patient_id"])
    tr = recordings[recordings["patient_id"].isin(train_ids)].reset_index(drop=True)
    va = recordings[recordings["patient_id"].isin(val_ids)].reset_index(drop=True)
    return tr, va


def run_cv(build_fn, arrays, recordings, splits, variant, win, folds=range(5), epochs=60, patience=10,
           batch_size=32, lr=1e-3, multitask=False, loss_weights=None, save_models=True):
    """K-fold CV on the development set. Returns (out-of-fold recording probabilities, histories).

    variant: dict(name, class_weighted: bool, aggregation: "mean"|"max_present", decision: "argmax"|"bayes")
    """
    from .config import CLASS_TO_IDX
    patient_labels = splits.set_index("patient_id")["murmur"]
    oof, histories = [], {}
    for k in folds:
        keras.utils.set_random_seed(config.SEED + k)
        tr, va = fold_data(recordings, splits, k)
        y_tr = tr["rec_label"].map(CLASS_TO_IDX).to_numpy()
        y_va = va["rec_label"].map(CLASS_TO_IDX).to_numpy()
        sw = class_weights(y_tr)[y_tr].astype(np.float32) if variant["class_weighted"] else None

        train_ds = CropDataset(arrays, tr["recording"], y_tr, win, batch_size, sample_weights=sw,
                               augment=not multitask, multitask=multitask, seed=config.SEED + k)
        val_ds = CropDataset(arrays, va["recording"], y_va, win, batch_size, multitask=multitask,
                             deterministic=True)

        model = build_fn()
        if multitask:
            loss = {"murmur": keras.losses.SparseCategoricalCrossentropy(),
                    "seg": keras.losses.SparseCategoricalCrossentropy()}
            metrics = {"murmur": ["accuracy"], "seg": ["accuracy"]}
        else:
            loss, metrics = keras.losses.SparseCategoricalCrossentropy(), ["accuracy"]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics,
                      loss_weights=loss_weights)

        callbacks = [
            PatientWACallback(arrays, va, patient_labels, win, variant["aggregation"], variant["decision"], multitask),
            keras.callbacks.EarlyStopping(monitor="val_patient_wa", mode="max", patience=patience,
                                          restore_best_weights=True),
        ]
        t0 = time.time()
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
        histories[k] = h.history
        wa = h.history["val_patient_wa"]
        print(f"{variant['name']} fold {k}: best val patient WA {max(wa):.3f} at epoch {int(np.argmax(wa)) + 1}"
              f" / {len(wa)} epochs | {time.time() - t0:.0f}s")

        rp = predict_recordings(model, arrays, va["recording"], win, multitask=multitask)
        rp = rp.assign(fold=k, patient_id=va["patient_id"].to_numpy(), location=va["location"].to_numpy(),
                       rec_label=va["rec_label"].to_numpy())
        oof.append(rp)
        if save_models:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODELS_DIR / f"{variant['name']}_fold{k}.keras")
    return pd.concat(oof, ignore_index=True), histories
