"""Project-wide constants: paths, signal parameters, class order and seed."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The dataset is not part of the repository. By default we expect it next to the repo folder;
# set the environment variable CIRCOR_DATA_DIR to point somewhere else.
DATA_DIR = Path(os.environ.get(
    "CIRCOR_DATA_DIR",
    REPO_ROOT.parent / "the-circor-digiscope-phonocardiogram-dataset-1.0.3" / "training_data",
))

SPLITS_FILE = REPO_ROOT / "splits" / "splits.csv"
CACHE_DIR = REPO_ROOT / "cache"          # extracted features (not versioned)
MODELS_DIR = REPO_ROOT / "models"        # trained weights, one file per model and fold
RESULTS_DIR = REPO_ROOT / "results"      # metrics tables and figures

SEED = 42

# Classes in the order used by the PhysioNet 2022 evaluation code (and by Table 1 of the project statement).
CLASSES = ["Present", "Unknown", "Absent"]
CLASS_WEIGHTS_WA = [5, 3, 1]             # weights of the weighted accuracy metric
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]

# Signal pre-processing
SR = 4000                                # all recordings are sampled at 4 kHz
BANDPASS = (25.0, 400.0)                 # Hz, heart sounds and most murmur energy
WINDOW_SEC = 4.0                         # fixed-length window fed to the networks
HOP_SEC = 2.0                            # 50% overlap between consecutive windows

# 2D CNN input: log-mel spectrogram of each window -> (N_MELS, frames)
MEL = dict(n_fft=256, hop_length=64, n_mels=32, fmin=25.0, fmax=800.0)

# RNN input: sequence of MFCC + delta + delta-delta vectors -> (frames, 3 * N_MFCC)
MFCC = dict(n_mfcc=13, n_fft=256, hop_length=128, n_mels=32, fmin=25.0, fmax=800.0)
