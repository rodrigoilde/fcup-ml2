# Checkpoint — 5 November 2026

Status of the project and talking points for the checkpoint conversation. Everything listed here is implemented in `src/` and shown in `main.ipynb`.

## 1. Status at a glance

| Step of the statement | Status |
|---|---|
| Data exploration | done (`main.ipynb`, Section 1) |
| Evaluation protocol (patient/subject split, test set, 5-fold CV) | done, fixed in `splits/splits.csv`, checked by `tests/test_splits.py` |
| Weighted accuracy (official definition) | done, `src/metrics.py`, checked by `tests/test_metrics.py` |
| Pre-processing and features (CNN and RNN) | done, first configuration |
| CNN: architecture, training, patient-level results | **first baseline trained (5 folds)** |
| RNN: architecture, training, patient-level results | **first baseline trained (5 folds)** |
| Hyperparameter tuning, data augmentation, label refinement | next |
| Final test evaluation, hidden-set predictions, slides | end of November |

## 2. Main decisions and why

| Decision | Why |
|---|---|
| Split **by subject**, merging the 140 patient IDs that are the same 70 people (`Additional ID`, both campaigns) | otherwise one person can be in training and test → optimistic results |
| **20% hold-out test** (locked) + **5-fold CV** on the remaining 80%, stratified | only 68 Unknown patients: one validation set would hold about 14 of them, too noisy to decide anything; the test set must never influence a decision |
| Band-pass **25–400 Hz**, z-score per recording | removes movement/breathing (< 25 Hz) and broadband noise (> 400 Hz, highest in Unknown recordings); removes gain differences |
| **4 s windows, 2 s hop** | recordings last 5–65 s; 4 s holds several heartbeats; about 31 000 training examples instead of 3163 |
| CNN on **log-mel** (32 × 251) | the murmur is a local time–frequency pattern between S1 and S2 → 2D convolutions |
| RNN on **MFCC + Δ + ΔΔ** (126 × 39) | compact vector per frame; sequence of S1–systole–S2–diastole; half the time steps → faster, more stable BPTT |
| Small networks (about 100k parameters), GAP in the CNN, BN, dropout, L2 | about 600 training patients per fold → overfitting is the main risk |
| **Balanced class weights** in the cross-entropy | 74% Absent; without weights the network predicts Absent |
| **Early stopping on the patient-level WA** of the validation fold | we are evaluated per patient with the WA, not per window with the loss |
| Window → recording: **mean**; recording → patient: **max + hierarchical thresholds** | a murmur may be heard at a single location (34 of the 179 Present patients); one confident recording must be enough |
| Thresholds tuned on validation, **cross-fitted** (fold *k* uses thresholds tuned on the other folds) | tuning and reporting on the same patients would inflate the score |
| Keras/TensorFlow on CPU | same tools as the practical classes; tensorflow-metal gives wrong results on Apple Silicon |

## 3. Preliminary results (5-fold CV, dev set, patient level)

Weighted accuracy, mean ± standard deviation over the 5 validation folds (first configuration, no tuning yet):

| Aggregation rule | CNN (98k params) | RNN (115k params) |
|---|---|---|
| argmax of the mean probability | 0.708 ± 0.067 | 0.745 ± 0.053 |
| hierarchical, thresholds 0.5 | **0.748 ± 0.068** | **0.786 ± 0.023** |
| hierarchical, cross-fitted thresholds | 0.742 ± 0.084 | 0.779 ± 0.029 |
| *reference: always Present* | *0.499* | *0.499* |

Additional metrics (5 folds pooled, cross-fitted thresholds): recall Present 0.84 / 0.85, recall Unknown 0.22 / 0.30, recall Absent 0.77 / 0.84, macro-F1 0.58 / 0.63 (CNN / RNN). Confusion matrices: `results/figures/comparison_confusion.png`.

What we learned from these first results (to show at the checkpoint):

- **Both models beat the trivial bar by far**, and the RNN is better and more stable than the CNN. The difference is still within the CNN's fold-to-fold variation.
- **Max over recordings > mean** for both models (+0.04): this confirms the EDA finding that the murmur is often heard at only one location.
- **The networks detect the murmur where it is heard:** median P(Present) is 0.88–0.93 for those recordings, 0.09–0.10 for Absent patients, and about 0.25 for recordings of Present patients taken where the murmur is not heard. That last group is the label noise → refined labels.
- **Unknown is the hard class** (recall 0.22–0.30). Its errors are partly cheap (Absent predicted Unknown costs weight 1).
- **CNN training is unstable** (validation accuracy jumps between epochs). **The RNN overfits** (training accuracy 0.98; validation loss rises after epochs 6–14).
- **Tuning the thresholds did not help:** early stopping already picks the epoch that suits threshold 0.5.
- **Honest caveat:** the epoch is chosen on the same fold that we report on. Choosing it by validation loss instead gives 0.711 (CNN) and 0.750 (RNN), so the CV numbers above are about 0.04 optimistic. The locked test set will give the unbiased estimate.

## 4. Next steps (until 30 November)

1. **Training stability:** a lower learning rate or a schedule (`ReduceLROnPlateau`), a larger batch, recurrent dropout in the GRU, and early stopping on an inner validation split (removes the optimism of the CV estimate).
2. **Labels:** test `label_mode="refined"`, where windows of Present patients recorded where the murmur is not heard are labelled Absent.
3. **Data augmentation:** random crops in time, random gain, additive noise, SpecAugment (time/frequency masks).
4. **Hyperparameters**, always compared with the mean CV WA: band-pass cut-offs, window length, number of mel bands/MFCCs, depth and width, dropout, class weights that follow the metric (5/3/1).
5. **Final evaluation:** ensemble of the 5 fold models, thresholds tuned on all validation folds, one evaluation on the test set.
6. **Bonus:** run the same ensemble on the hidden set (`HIDDEN_DIR` in Section 7 of the notebook) → `results/hidden_predictions.csv`.
7. **Slides** (10 minutes) and self-assessment form.

## 5. Proposed division of work

| Member | Responsibility |
|---|---|
| A — _name_ | data, evaluation protocol, aggregation and thresholds, final evaluation, hidden-set predictions |
| B — _name_ | CNN: tuning, augmentation, analysis of errors |
| C — _name_ | RNN: tuning, alternative recurrent units (LSTM), analysis of errors |

Everyone writes their part of the slides. Before the presentation, each member must be able to explain all the parts (see below).

## 6. Questions everyone should be able to answer

- **Why is "always Present" already at WA ≈ 0.50?** Present patients weigh 5 each: 5·179 / (5·179 + 3·68 + 695) ≈ 0.50 on the whole public set. "Always Absent" gets 74% accuracy but WA 0.39.
- **Why not split by recording?** The recordings of one patient are very similar, so the model would be tested on patients it has already heard. Splitting by patient ID alone is not enough either, because 70 people have two IDs.
- **Where is the test set used?** Nowhere until the end. Early stopping, thresholds and model choices only use the validation folds.
- **How many inputs per patient?** Variable: 1 to 6 recordings, each with about 10 windows. Windows are averaged per recording, and the recordings are combined with max + thresholds.
- **Why max and not mean over recordings?** The murmur may be audible at only one valve; the mean would dilute it.
- **How are the class weights computed?** `n_windows / (3 · n_windows_of_class)` on the training windows of each fold: about 1.8 (Present), 6.8 (Unknown) and 0.44 (Absent).
- **Why standardise with training statistics only?** Using validation/test statistics would leak information from those sets.
- **Why GAP in the CNN?** One value per feature map ("is the pattern somewhere in the window"). It needs far fewer parameters than Flatten + Dense and is more robust to where the murmur falls in the window.
- **Why a GRU, why bidirectional, why clip gradients?** Gates avoid vanishing gradients over 126 steps (with fewer parameters than an LSTM); the whole window is available, so the future context helps; clipping prevents exploding gradients in BPTT.
- **Why is the validation curve so noisy?** About 150 validation patients per fold (about 29 Present and 11 Unknown): one Present patient more or less changes the WA by about 0.02. On top of that, a high learning rate and batch normalisation make the window-level validation accuracy jump between epochs (Section 3).
