"""Features for each architecture, computed per window and cached on disk.

* 2D CNN: log-mel spectrogram of each 4 s window -> (n_mels, frames) = (32, 251).
* RNN: sequence of 13 MFCC + delta + delta-delta per frame -> (frames, 39) = (126, 39).
"""

import hashlib
import json

import librosa
import numpy as np

from .config import BANDPASS, CACHE_DIR, HOP_SEC, MEL, MFCC, SR, WINDOW_SEC
from .preprocessing import make_windows, preprocess_recording


def logmel(windows, params=MEL):
    """(n, samples) -> (n, n_mels, frames) log-mel spectrograms in dB."""
    S = librosa.feature.melspectrogram(y=windows, sr=SR, power=2.0, **params)
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)


def mfcc_sequence(windows, params=MFCC):
    """(n, samples) -> (n, frames, 3 * n_mfcc) MFCC, delta and delta-delta, time first (RNN input)."""
    m = librosa.feature.mfcc(y=windows, sr=SR, **params)                   # (n, n_mfcc, frames)
    feats = np.concatenate([m, librosa.feature.delta(m, order=1), librosa.feature.delta(m, order=2)], axis=1)
    return np.transpose(feats, (0, 2, 1)).astype(np.float32)


FEATURES = {"logmel": (logmel, MEL), "mfcc": (mfcc_sequence, MFCC)}


def _cache_path(kind, band, window_sec, hop_sec):
    params = dict(kind=kind, band=band, window=window_sec, hop=hop_sec, sr=SR, **FEATURES[kind][1])
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{kind}_{key}.npz"


def compute_features(recordings, kind, band=BANDPASS, window_sec=WINDOW_SEC, hop_sec=HOP_SEC,
                     use_cache=True, verbose=True):
    """Return {recording_id: array (n_windows, ...)} for every recording of the DataFrame.

    Results are cached in cache/<kind>_<hash>.npz; the hash changes with any parameter, so a stale
    cache is never reused after changing the pre-processing."""
    fn, _ = FEATURES[kind]
    path = _cache_path(kind, band, window_sec, hop_sec)
    cached = {}
    if use_cache and path.exists():
        with np.load(path) as f:
            cached = {k: f[k] for k in f.files}
    missing = [r for r in recordings.itertuples() if r.recording_id not in cached]
    for i, r in enumerate(missing):
        x = preprocess_recording(r.wav_path, band)
        cached[r.recording_id] = fn(make_windows(x, SR, window_sec, hop_sec)).astype(np.float16)
        if verbose and (i + 1) % 500 == 0:
            print(f"  {kind}: {i + 1}/{len(missing)} recordings")
    if missing and use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, **cached)
    return {rid: cached[rid] for rid in recordings["recording_id"]}


def stack_windows(recordings, feats):
    """Stack the windows of the given recordings -> X (N, ...) float32 and the recording id of each window."""
    X = np.concatenate([feats[rid] for rid in recordings["recording_id"]]).astype(np.float32)
    rec_ids = np.concatenate([[rid] * len(feats[rid]) for rid in recordings["recording_id"]])
    return X, rec_ids


class Standardizer:
    """Per-feature z-score fitted on the training windows only (never on validation/test data).

    `axis` lists the axes that are averaged: for log-mel (N, mels, frames) we use (0, 2), one
    mean/std per mel band; for MFCC sequences (N, frames, coefs) we use (0, 1), one per coefficient."""

    def __init__(self, axis):
        self.axis = axis

    def fit(self, X):
        self.mean = X.mean(axis=self.axis, keepdims=True)
        self.std = X.std(axis=self.axis, keepdims=True) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)
