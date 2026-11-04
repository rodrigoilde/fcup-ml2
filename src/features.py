"""Log-mel frame sequences and frame-level heart-sound states for the RNN."""
import numpy as np
import librosa

from . import config
from .data import read_tsv
from .preprocessing import load_wav, downsample

HOP = int(config.MEL_HOP_SECONDS * config.SR)        # 40 samples = 20 ms
WIN = int(config.MEL_WIN_SECONDS * config.SR)        # 100 samples = 50 ms
N_FFT = 256                                          # zero-padded FFT: 7.8 Hz bins, no empty mel filters


def logmel(x):
    """(T,) signal at 2 kHz -> (n_frames, N_MELS) log-mel, standardized per band over time."""
    mel = librosa.feature.melspectrogram(y=x, sr=config.SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                                         n_mels=config.N_MELS, fmin=config.MEL_FMIN, fmax=config.MEL_FMAX)
    z = np.log(mel + 1e-8).T
    return ((z - z.mean(axis=0)) / (z.std(axis=0) + 1e-6)).astype(np.float32)


def frame_labels(tsv_path, n_frames):
    """State (0..4) at the centre of each frame; frames outside the annotated span are 0."""
    seg = read_tsv(tsv_path)
    t = np.arange(n_frames) * config.MEL_HOP_SECONDS          # librosa frames are centred at i * hop
    idx = np.searchsorted(seg[:, 0], t, side="right") - 1
    inside = (idx >= 0) & (t < seg[np.clip(idx, 0, None), 1])
    return np.where(inside, seg[np.clip(idx, 0, None), 2], 0).astype(np.int32)


def load_rnn_inputs(recordings, cache=True):
    """{recording name: (log-mel (n_frames, 32) float16, states (n_frames,) int8)}, cached under cache/rnn/."""
    out = {}
    cache_dir = config.CACHE_DIR / "rnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, wav, tsv in zip(recordings["recording"], recordings["wav"], recordings["tsv"]):
        f = cache_dir / f"{name}.npz"
        if cache and f.exists():
            d = np.load(f)
            out[name] = (d["x"], d["s"])
            continue
        x = logmel(downsample(load_wav(wav)))
        s = frame_labels(tsv, len(x))
        out[name] = (x.astype(np.float16), s.astype(np.int8))
        if cache:
            np.savez(f, x=out[name][0], s=out[name][1])
    return out
