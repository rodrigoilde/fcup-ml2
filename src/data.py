"""Loading the CirCor patient table and the recording index."""
import numpy as np
import pandas as pd

from . import config


def _link_groups(patient_ids, additional_ids):
    """Union-find over (Patient ID, Additional ID): the same child recorded in both campaigns
    appears under two Patient IDs that point to each other. Returns {patient_id: group_id}."""
    parent = {p: p for p in patient_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, a in zip(patient_ids, additional_ids):
        if a in parent:
            ra, rp = find(a), find(p)
            if ra != rp:
                parent[max(ra, rp)] = min(ra, rp)
    return {p: find(p) for p in patient_ids}


def load_patients():
    """One row per patient, with the murmur label and a `group_id` shared by linked patients."""
    df = pd.read_csv(config.PATIENTS_CSV, dtype={"Patient ID": int})
    df = df.rename(columns={"Patient ID": "patient_id", "Recording locations:": "recording_locations",
                            "Murmur": "murmur", "Murmur locations": "murmur_locations",
                            "Campaign": "campaign", "Additional ID": "additional_id", "Age": "age"})
    df["additional_id"] = pd.to_numeric(df["additional_id"], errors="coerce").astype("Int64")
    groups = _link_groups(df["patient_id"].tolist(),
                          [int(a) if pd.notna(a) else None for a in df["additional_id"]])
    df["group_id"] = df["patient_id"].map(groups)
    return df


def recording_label(patient_murmur, murmur_locations, location):
    """Location-aware recording label: a recording of a Present patient is Present only if its
    location is listed in `Murmur locations`; elsewhere the murmur was not heard (Absent)."""
    if patient_murmur != "Present":
        return patient_murmur
    heard = set(str(murmur_locations).split("+"))
    return "Present" if location in heard else "Absent"


def load_recordings(patients=None):
    """One row per .wav file, read from the per-patient <id>.txt files."""
    if patients is None:
        patients = load_patients()
    rows = []
    for p in patients.itertuples(index=False):
        lines = (config.RECORDINGS_DIR / f"{p.patient_id}.txt").read_text().splitlines()
        n_rec = int(lines[0].split()[1])
        for line in lines[1:1 + n_rec]:
            location, hea, wav, tsv = line.split()
            n_samples = int((config.RECORDINGS_DIR / hea).read_text().split()[3])
            rows.append(dict(
                patient_id=p.patient_id, group_id=p.group_id, recording=wav[:-4], location=location,
                wav=str(config.RECORDINGS_DIR / wav), tsv=str(config.RECORDINGS_DIR / tsv),
                duration=n_samples / config.SR_RAW, patient_label=p.murmur,
                rec_label=recording_label(p.murmur, p.murmur_locations, location),
                campaign=p.campaign))
    return pd.DataFrame(rows)


def read_tsv(path):
    """Segmentation file -> array of (start_s, end_s, state), state in 0..4 (0 = unannotated)."""
    return np.loadtxt(path, ndmin=2)


def unannotated_fraction(tsv_path, duration):
    """Fraction of the recording not covered by S1/systole/S2/diastole annotations."""
    seg = read_tsv(tsv_path)
    annotated = sum(end - start for start, end, state in seg if state != 0)
    return 1.0 - annotated / duration
