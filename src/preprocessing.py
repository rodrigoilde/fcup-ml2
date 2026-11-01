"""Signal pre-processing: loading, band-pass filtering, normalisation and fixed-length windows."""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from .config import BANDPASS, HOP_SEC, SR, WINDOW_SEC


def load_wav(path):
    """Read a 16-bit WAV file as float32 in [-1, 1]. Returns (signal, sampling rate)."""
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    return x.astype(np.float32), sr


def bandpass(x, sr=SR, band=BANDPASS, order=4):
    """Zero-phase Butterworth band-pass filter (forward-backward, so S1/S2 are not shifted in time)."""
    sos = butter(order, band, btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def normalize(x):
    """Z-score: zero mean, unit variance (removes differences in recording gain between stethoscopes)."""
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def preprocess_recording(path, band=BANDPASS):
    x, sr = load_wav(path)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr} Hz")
    return normalize(bandpass(x, sr, band))


def make_windows(x, sr=SR, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    """Split a signal into overlapping windows of fixed length -> array (n_windows, window_len).

    A signal shorter than one window is zero-padded at the end (does not happen in the public set,
    where the shortest recording lasts 5.2 s, but may happen on new data)."""
    win, hop = int(window_sec * sr), int(hop_sec * sr)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    starts = range(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts])
