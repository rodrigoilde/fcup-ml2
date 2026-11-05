import time

import numpy as np
import pandas as pd
import keras


def sliding_windows(x, win, hop):
    """Janelas com sobreposição que cobrem a gravação toda. A última é encostada ao fim para não perdermos o final."""
    starts = list(range(0, len(x) - win + 1, hop))
    if starts[-1] != len(x) - win:
        starts.append(len(x) - win)
    return np.stack([x[s:s + win] for s in starts])


def class_weights(labels, n_classes):
    """Pesos inversamente proporcionais à frequência de cada classe, normalizados para a média por amostra dar 1."""
    counts = np.bincount(labels, minlength=n_classes)
    return len(labels) / (n_classes * np.maximum(counts, 1))


class CropDataset(keras.utils.PyDataset):
    """Em cada época tira um crop aleatório de win amostras (ou frames) de cada gravação. Serve para a CNN, em que
    arrays[name] é um array (T, C), e para a RNN multitarefa, em que é um tuplo (frames, estados).

    Com augment=True também mudamos o ganho e a polaridade ao acaso. Não invertemos o sinal no tempo porque isso
    trocava o S1 com o S2. Com deterministic=True usa sempre o crop central e não baralha (para as curvas de
    validação)."""

    def __init__(self, arrays, names, labels, win, batch_size=32, sample_weights=None, augment=False,
                 multitask=False, deterministic=False, seed=0, **kwargs):
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


def predict_recordings(model, arrays, names, win, classes, hop=None, multitask=False, max_windows=1024):
    """Passa janelas deslizantes por cada gravação e faz a média das probabilidades das janelas. Devolve uma linha
    por gravação com as colunas p_<classe>. As janelas de várias gravações vão juntas para o mesmo predict, que
    assim é bem mais rápido."""
    hop = hop or win // 2
    names = list(names)
    probs = np.zeros((len(names), len(classes)), dtype=np.float32)
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
    return pd.DataFrame(probs, columns=[f"p_{c}" for c in classes]).assign(recording=names)


class PatientWACallback(keras.callbacks.Callback):
    """No fim de cada época prevê as gravações do fold de validação, passa-as ao patient_score (que agrega por
    paciente, decide e calcula a WA) e guarda o resultado como val_patient_wa. Assim o EarlyStopping pode olhar
    para a métrica com que somos avaliados, em vez da val_loss."""

    def __init__(self, arrays, val_recordings, win, classes, patient_score, multitask=False):
        super().__init__()
        self.arrays, self.val, self.win, self.classes = arrays, val_recordings, win, classes
        self.patient_score, self.multitask = patient_score, multitask

    def on_epoch_end(self, epoch, logs=None):
        rp = predict_recordings(self.model, self.arrays, self.val["recording"], self.win, self.classes,
                                multitask=self.multitask)
        rp["patient_id"] = self.val["patient_id"].to_numpy()
        logs["val_patient_wa"] = self.patient_score(rp)


def fold_data(recordings, splits, fold):
    """Gravações de treino e de validação de um fold. Os pacientes de teste nunca entram aqui."""
    dev = splits[splits["split"] == "dev"]
    val_ids = set(dev.loc[dev["fold"] == fold, "patient_id"])
    train_ids = set(dev.loc[dev["fold"] != fold, "patient_id"])
    tr = recordings[recordings["patient_id"].isin(train_ids)].reset_index(drop=True)
    va = recordings[recordings["patient_id"].isin(val_ids)].reset_index(drop=True)
    return tr, va


def run_cv(build_fn, arrays, recordings, splits, name, win, classes, patient_score, class_weighted=False,
           folds=range(5), epochs=60, patience=10, batch_size=32, lr=1e-3, multitask=False, loss_weights=None,
           seed=42, models_dir=None):
    """CV nos folds de desenvolvimento. Em cada fold treina um modelo novo com early stopping no patient_score e
    prevê as gravações de validação. Devolve as probabilidades out-of-fold de cada gravação e o histórico de cada
    fold. Se models_dir não for None, guarda lá o modelo de cada fold."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    oof, histories = [], {}
    for k in folds:
        keras.utils.set_random_seed(seed + k)
        tr, va = fold_data(recordings, splits, k)
        y_tr = tr["rec_label"].map(class_to_idx).to_numpy()
        y_va = va["rec_label"].map(class_to_idx).to_numpy()
        sw = class_weights(y_tr, len(classes))[y_tr].astype(np.float32) if class_weighted else None

        train_ds = CropDataset(arrays, tr["recording"], y_tr, win, batch_size, sample_weights=sw,
                               augment=not multitask, multitask=multitask, seed=seed + k)
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
            PatientWACallback(arrays, va, win, classes, patient_score, multitask),
            keras.callbacks.EarlyStopping(monitor="val_patient_wa", mode="max", patience=patience,
                                          restore_best_weights=True),
        ]
        t0 = time.time()
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
        histories[k] = h.history
        wa = h.history["val_patient_wa"]
        print(f"{name} fold {k}: melhor WA de validação {max(wa):.3f} na época {int(np.argmax(wa)) + 1}"
              f" de {len(wa)} | {time.time() - t0:.0f}s")

        rp = predict_recordings(model, arrays, va["recording"], win, classes, multitask=multitask)
        rp = rp.assign(fold=k, patient_id=va["patient_id"].to_numpy(), location=va["location"].to_numpy(),
                       rec_label=va["rec_label"].to_numpy())
        oof.append(rp)
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            model.save(models_dir / f"{name}_fold{k}.keras")
    return pd.concat(oof, ignore_index=True), histories
