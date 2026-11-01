"""Read the CirCor patient files into a patient table and a recording table.

Each patient has a text file `<id>.txt`:

    13918 4 4000                                  <- patient id, number of recordings, sampling rate
    AV 13918_AV.hea 13918_AV.wav 13918_AV.tsv     <- one line per recording
    ...
    #Age: Child                                   <- metadata and labels
    #Murmur: Present
    ...

We parse these files (instead of `training_data.csv`) so that the same code also reads the hidden
test set, whose files have the same layout but no labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

# "#Key: value" fields of the patient file -> column name in the patient table
_FIELDS = {
    "Age": "age", "Sex": "sex", "Height": "height", "Weight": "weight",
    "Pregnancy status": "pregnant", "Murmur": "murmur", "Murmur locations": "murmur_locations",
    "Most audible location": "most_audible_location", "Systolic murmur timing": "systolic_timing",
    "Systolic murmur grading": "systolic_grading", "Diastolic murmur timing": "diastolic_timing",
    "Outcome": "outcome",
    "Campaign": "campaign", "Additional ID": "additional_id",
}


def _parse_value(text):
    text = text.strip()
    return np.nan if text in ("", "nan") else text


def parse_patient_file(path):
    """Return (patient dict, list of recording dicts) for one `<id>.txt` file."""
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    pid, n_rec, sr = lines[0].split()[:3]
    patient = {"patient_id": int(pid), "n_recordings": int(n_rec), "sr": int(sr)}

    recordings = []
    for line in lines[1:1 + int(n_rec)]:
        parts = line.split()
        location, wav = parts[0], parts[2]
        tsv = parts[3] if len(parts) > 3 else None
        recordings.append({
            "patient_id": int(pid),
            "recording_id": Path(wav).stem,                 # e.g. 13918_AV, or 49748_AV_1 for repeated locations
            "location": location,
            "wav_path": str(path.parent / wav),
            "tsv_path": str(path.parent / tsv) if tsv else None,
        })

    for line in lines[1 + int(n_rec):]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            if key.strip() in _FIELDS:
                patient[_FIELDS[key.strip()]] = _parse_value(value)
    return patient, recordings


def load_dataset(data_dir=DATA_DIR):
    """Read every patient file of `data_dir`.

    Returns
    -------
    patients : DataFrame, one row per patient (labels are NaN if the files have none).
    recordings : DataFrame, one row per recording, with the patient label and a flag telling whether
        the murmur was heard at that location (`murmur_audible_here`, only True for Present patients).
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No patient .txt files found in {data_dir}. Set CIRCOR_DATA_DIR.")
    patients, recordings = [], []
    for f in files:
        p, r = parse_patient_file(f)
        patients.append(p)
        recordings.extend(r)

    patients = pd.DataFrame(patients)
    for col in _FIELDS.values():
        if col not in patients:
            patients[col] = np.nan
    patients["height"] = pd.to_numeric(patients["height"], errors="coerce")
    patients["weight"] = pd.to_numeric(patients["weight"], errors="coerce")
    patients["additional_id"] = pd.to_numeric(patients["additional_id"], errors="coerce").astype("Int64")
    patients["murmur_locations"] = patients["murmur_locations"].apply(
        lambda s: s.split("+") if isinstance(s, str) else [])

    recordings = pd.DataFrame(recordings).merge(
        patients[["patient_id", "murmur", "murmur_locations"]], on="patient_id", how="left")
    recordings["murmur_audible_here"] = [
        m == "Present" and loc in locs
        for m, loc, locs in zip(recordings["murmur"], recordings["location"], recordings["murmur_locations"])
    ]
    recordings = recordings.drop(columns="murmur_locations")
    return patients, recordings


def load_segmentation(tsv_path):
    """Read a segmentation file: DataFrame with start, end (seconds) and state
    (0 = unannotated, 1 = S1, 2 = systole, 3 = S2, 4 = diastole)."""
    return pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "state"])
