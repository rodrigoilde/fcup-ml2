"""Raw audio -> multi-band waveform for the 1D CNN."""
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import config

CLIP = 20.0   # robust z-score limit: removes friction spikes and flattens the tallest S1/S2 peaks (<2 % of samples)


def load_wav(path):
    sr, x = wavfile.read(path)
    assert sr == config.SR_RAW, f"{path}: unexpected sampling rate {sr}"
    return x.astype(np.float32) / 32768.0


def downsample(x):
    """4 kHz -> 2 kHz (resample_poly applies the anti-aliasing filter)."""
    return signal.resample_poly(x, 1, config.SR_RAW // config.SR).astype(np.float32)


@lru_cache(maxsize=None)
def _band_filters():
    return [signal.butter(4, band, btype="bandpass", fs=config.SR, output="sos") for band in config.BANDS]


def filterbank(x):
    """(T,) signal at 2 kHz -> (T, n_bands). Zero-phase filtering keeps S1/S2 aligned across bands."""
    return np.stack([signal.sosfiltfilt(sos, x) for sos in _band_filters()], axis=-1).astype(np.float32)


def robust_normalize(x):
    """Per-channel robust z-score (median / MAD), then clip the rare spikes."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    return np.clip((x - med) / (mad + 1e-8), -CLIP, CLIP).astype(np.float32)


def preprocess_cnn(path):
    """Full CNN pre-processing of one recording: (T, 5) float32 at 2 kHz."""
    return robust_normalize(filterbank(downsample(load_wav(path))))


def load_cnn_inputs(recordings, cache=True):
    """{recording name: (T, 5) float16 array}, cached on disk under cache/cnn/."""
    out = {}
    cache_dir = config.CACHE_DIR / "cnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, path in zip(recordings["recording"], recordings["wav"]):
        f = cache_dir / f"{name}.npy"
        if cache and f.exists():
            out[name] = np.load(f)
        else:
            out[name] = preprocess_cnn(path).astype(np.float16)
            if cache:
                np.save(f, out[name])
    return out


def random_crop(x, win, rng):
    start = rng.integers(0, len(x) - win + 1)
    return x[start:start + win]


def sliding_windows(x, win=config.WIN, hop=config.HOP):
    """Overlapping windows covering the whole recording (the last one is aligned to the end)."""
    starts = list(range(0, len(x) - win + 1, hop))
    if starts[-1] != len(x) - win:
        starts.append(len(x) - win)
    return np.stack([x[s:s + win] for s in starts])
