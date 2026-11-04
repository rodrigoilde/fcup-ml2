# Heart murmur detection from phonocardiograms (CirCor DigiScope)

Machine Learning II (CC3043), FCUP, 2026/2027 — group project.

For each **patient** we predict whether a heart murmur is *Present*, *Absent* or *Unknown* from their auscultation
recordings. Models are evaluated with the weighted accuracy of the PhysioNet Challenge 2022 (weights 5 / 3 / 1).

We implement two of the three classifiers asked for in the statement:

- a **1D CNN** on a multi-band version of the waveform (5 band-pass channels, 25–800 Hz);
- a **multi-task bidirectional GRU** on log-mel frames, with an auxiliary head that predicts the heart-sound state
  (S1 / systole / S2 / diastole / not annotated) of every frame.

**The deliverable is [`main.ipynb`](main.ipynb).** It runs the whole pipeline in order (exploration → protocol →
pre-processing → CNN → RNN) and calls the modules in `src/`.

> **Status — checkpoint of 5 November 2026.** Data exploration, evaluation protocol, metric, pre-processing for both
> models and a cross-validated CNN baseline are done. The RNN runs end to end (smoke run on one fold) but has not been
> cross-validated or tuned yet. The test set has not been used.

## Setup

```bash
conda activate ml2                 # or any Python 3.11 environment
pip install -r requirements.txt
```

The dataset is **not** included. By default the code expects the public CirCor data (v1.0.3) next to this repository:

```
../the-circor-digiscope-phonocardiogram-dataset-1.0.3/
    training_data.csv
    training_data/   (<patient>.txt, <patient>_<loc>.wav/.hea/.tsv)
```

Set the `CIRCOR_DIR` environment variable to point to another location.

## How to run

```bash
pytest tests                       # metric, labels, splits, aggregation (8 tests)
jupyter notebook main.ipynb        # run all cells in order
```

`splits/splits.csv` is included. It is deterministic and can be re-created with `python -m src.splits`.

Training switches in the setup cell of `main.ipynb`:

| Switch | `False` (default) | `True` |
|---|---|---|
| `RETRAIN_CNN` | reuse the out-of-fold predictions in `results/` (trains if they are missing) | 5-fold CV of two configurations, ~35 min on a laptop CPU |
| `RETRAIN_RNN` | reuse the smoke-run results (trains if they are missing) | smoke run on fold 0, ~3 min |

With saved results the notebook runs in a couple of minutes. Pre-processed arrays are cached in `cache/` and trained models are
saved in `results/models/` (both git-ignored and re-created automatically).

## Evaluation protocol

- **Grouped by subject, not only by patient ID.** 70 children were recorded in both campaigns and appear under two
  Patient IDs (linked by `Additional ID`). Linked IDs share a `group_id` and always fall in the same subset.
- **Locked test set**: 20 % of the patients (189), stratified by murmur × campaign. Not used so far; it will be used
  once, at the end.
- **5-fold cross-validation** on the remaining 753 development patients, for every model decision. The audio analyses
  that inform the pre-processing also use the development set only.
- One prediction per patient; patient-level confusion matrix with rows = predicted, columns = true (Table 1 of the
  statement).

## Results so far (development set, 5-fold CV, out-of-fold, patient level)

| Model | Loss | Aggregation / decision | WA (mean ± std over folds) |
|---|---|---|---|
| always *Present* (reference) | – | – | 0.499 |
| 1D CNN, E1 | class-weighted CE | mean / argmax | 0.779 ± 0.025 |
| 1D CNN, E2 | plain CE | max-Present / Bayes rule for WA | **0.821 ± 0.025** |
| BiGRU multi-task (smoke run) | CE + 0.5 · segmentation CE | max-Present / Bayes | 0.806 (fold 0 only, not comparable) |

These are validation numbers: the early-stopping epoch is chosen on the same fold, so they are optimistic. Confusion
matrices, additional metrics and their interpretation are in `main.ipynb`, Sections 5.5–5.8.

## Repository structure

```
main.ipynb           the deliverable notebook (executed, with outputs)
src/
  config.py          paths and constants
  data.py            patient table (+ linked groups), recording index, location-aware recording labels
  splits.py          test split + CV folds (python -m src.splits)
  metrics.py         weighted accuracy, confusion matrix, additional metrics
  preprocessing.py   2 kHz resampling, filter bank, robust normalization, windows (CNN)
  features.py        log-mel frames and frame-level heart-sound states (RNN)
  models.py          build_cnn1d, build_bigru_mt
  aggregate.py       recording -> patient aggregation and decision rules
  training.py        crop datasets, patient-level early stopping, CV loop
tests/               unit tests
splits/splits.csv    the fixed split
results/             out-of-fold predictions, CV summaries, figures
docs/                planning and checkpoint notes (Portuguese, internal)
```
