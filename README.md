# Heart murmur detection from phonocardiograms

Machine Learning II project (CC3043, DCC/FCUP, 2026/2027).

We predict, **for each patient**, whether a heart murmur is *Present*, *Absent* or *Unknown*, using the heart sound recordings of the public part of the [CirCor DigiScope dataset](https://physionet.org/content/circor-heart-sound/1.0.3/) (942 patients, 3163 recordings). This is the murmur task of the [PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/1.0.0/), evaluated with its **weighted accuracy**.

We implement two classifiers:

- a **2D CNN** on log-mel spectrograms of 4 s windows;
- an **RNN** (2-layer bidirectional GRU) on MFCC sequences of the same windows.

Window predictions are averaged per recording, and the recordings of a patient are combined with a hierarchical rule (max over recordings, thresholds tuned on validation data), giving one prediction per patient.

## Repository layout

```
main.ipynb          the deliverable: data exploration, protocol, features, CNN, RNN, results
src/                reusable code imported by the notebook
  config.py           paths, signal parameters, class order, seed
  data.py             reads the patient .txt files -> patient and recording tables
  splits.py           subject-grouped, stratified test split + 5 dev folds
  preprocessing.py    band-pass filter, normalisation, fixed-length windows
  features.py         log-mel (CNN) and MFCC sequences (RNN), cached in cache/
  models.py           build_cnn(), build_rnn()
  training.py         training loop, early stopping on patient-level WA, cross-validation, ensembles
  aggregation.py      windows -> recordings -> one prediction per patient
  metrics.py          weighted accuracy (official definition), confusion matrix, extra metrics
splits/splits.csv   the fixed split used by everybody (patient_id, subject_group, subset, fold)
models/             trained models (one per architecture and fold) and their standardisation statistics
results/            cross-validation predictions, training histories, summary tables, figures
tests/              checks of the metric and of the split
docs/checkpoint.md  status and talking points for the checkpoint
```

## Setup

1. **Environment** (Python 3.11):

   ```bash
   conda env create -f environment.yml
   conda activate ml2
   ```

   or, with an existing Python 3.11: `pip install -r requirements.txt`.

2. **Dataset** (not included in the repository). Download it from PhysioNet and place it next to the repository folder, so that this path exists:

   ```
   ../the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/
   ```

   To use another location, set an environment variable before starting Jupyter:
   `export CIRCOR_DATA_DIR=/path/to/training_data`.

## Running

Open `main.ipynb` with the `ml2` environment as kernel (in VS Code, select the `ml2` interpreter; in Jupyter, `python -m ipykernel install --user --name ml2` once) and run all cells. Or, from the terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Flags in the first code cell:

| Flag | Default | Meaning |
|---|---|---|
| `RETRAIN` | `False` | `False` loads the saved models and results from `models/` and `results/`, so the whole notebook runs in a few minutes. `True` trains everything again (about 2–3 h on a laptop CPU). Missing results are always trained. |
| `FOLDS_TO_RUN` | `range(5)` | dev folds used for cross-validation |
| `EVALUATE_TEST` | `False` | the hold-out test set stays locked until the final model is chosen |

The first run extracts the features into `cache/` (about 30 s); later runs reuse them.

Run the tests with `pytest tests/`.

## Evaluation protocol

- The split is done **by subject**. The 70 subjects recorded in both campaigns (two patient IDs linked by `Additional ID`) are kept in the same subset.
- **Test set:** 20% of the subjects, stratified by murmur label. It is never used for any decision, and is evaluated once at the end.
- **Dev set:** the remaining 80%, divided into 5 stratified folds. In iteration *k*, fold *k* is used for validation (early stopping, threshold tuning, model selection) and the other four for training. We report the mean ± standard deviation over the 5 folds.
- **Main metric:** patient-level weighted accuracy (weights 5/3/1 for Present/Unknown/Absent). Predicting *Present* for everyone gives a WA of about 0.50, so a useful model has to beat that.

## Current results (5-fold cross-validation on the dev set)

_Pending: training in progress._

## Team

| Member | Main responsibility |
|---|---|
| _name_ | data, evaluation protocol, aggregation |
| _name_ | CNN |
| _name_ | RNN |
