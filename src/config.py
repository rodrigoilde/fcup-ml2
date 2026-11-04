"""Project-wide constants and paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The dataset is not part of the repository. Override with the CIRCOR_DIR environment variable.
DATA_DIR = Path(os.environ.get(
    "CIRCOR_DIR", ROOT.parent / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"))
RECORDINGS_DIR = DATA_DIR / "training_data"
PATIENTS_CSV = DATA_DIR / "training_data.csv"

SPLITS_CSV = ROOT / "splits" / "splits.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = ROOT / "cache"

SEED = 42

# Class order follows Table 1 of the project statement.
CLASSES = ["Present", "Unknown", "Absent"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_WEIGHTS_METRIC = (5.0, 3.0, 1.0)      # weights of the official weighted accuracy

LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]

# Signal processing
SR_RAW = 4000
SR = 2000                                    # after 1:2 downsampling (Nyquist 1 kHz)
BANDS = [(25, 45), (45, 80), (80, 200), (200, 400), (400, 800)]   # filter bank for the 1D CNN (Hz)

# CNN windows
WIN_SECONDS = 4.0
WIN = int(WIN_SECONDS * SR)                  # 8000 samples
HOP = WIN // 2                               # 50 % overlap at inference

# Log-mel frames for the RNN
MEL_WIN_SECONDS = 0.050
MEL_HOP_SECONDS = 0.020
N_MELS = 32
MEL_FMIN, MEL_FMAX = 25, 800
RNN_WIN = 250                                # 5 s crops of 20 ms frames (shortest recording is 5.2 s)

# Heart-sound states in the .tsv files
STATES = ["unannotated", "S1", "systole", "S2", "diastole"]
