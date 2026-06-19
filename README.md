# Decision Transformer for Sleep Apnea Detection from ECG
 
This repository contains a complete pipeline for sequential sleep apnea detection from single-lead ECG signals using an offline reinforcement learning approach via a Decision Transformer (DT).
 
## Project Overview
 
Instead of treating sleep apnea detection as a static classification task, this project frames it as an offline sequential decision-making problem. The model acts as a clinical agent that evaluates **60-minute sliding contextual windows** of patient ECG-derived features.
 
By utilizing **Return-to-Go (RTG) Conditioning**, the model's clinical sensitivity and specificity can be dynamically steered at inference time without modifying the underlying network weights or threshold parameters.
 
### Core Engineering Details
 
* **Feature Extraction — Two-Layer Pipeline:**
  * **R-peak / QRS detection:** `pyECGdeli` (`ECGdeli.QRS_detection`), a Python port of the KIT-IBT ECGdeli wavelet-based detector, with a NeuroKit2 Pan-Tompkins fallback if no peaks are returned.
  * **Morphological delineation:** NeuroKit2's discrete wavelet transform delineator (`method="dwt"`), which implements the same Laguna-wavelet approach as ECGdeli's P/T-wave detection stages. This produces ECGdeli-equivalent fiducial points (P-on/peak/off, Q, R, S, T-on/peak/off) and derived features: PR interval, QT interval, QTc (Bazett), QRS duration, P-wave duration, ST level, and P/T/Q/S amplitudes (15 `morph_*` features total).
  * **HRV time-domain features** via `neurokit2.hrv_time`.
  * **Apnea-band spectral power** (0.008–0.036 Hz fraction of total RR-interval PSD via Welch's periodogram) and **EDR variance** (ECG-derived respiration, from R-amplitude modulation), each computed over a 5-minute sliding context.
* **Scaling:** `RobustScaler` (median/IQR-based), fit once on the training set and reused for test data — chosen over z-score normalization for robustness to ECG outlier artifacts.
* **Data Augmentation:** Training data is augmented with three trajectory types generated from a single `LogisticRegression` baseline classifier's predicted probabilities:
  * **Perfect Agent** — predictions equal ground-truth labels (oracle upper bound).
  * **Cautious Agent** — alarms whenever predicted apnea probability ≥ 0.20 (high sensitivity, more false alarms).
  * **Careless Agent** — alarms only when predicted apnea probability ≥ 0.80 (low sensitivity, more missed events).
* **Clinical Reward Matrix** (defined once in `data_prep.py`'s `compute_clinical_reward`, and imported by `test.py` so training and evaluation always share identical reward semantics):
  | Prediction | Ground Truth | Reward |
  |:---:|:---:|:---:|
  | Apnea | Apnea | **+10.0** (True Positive) |
  | Normal | Normal | **+1.0** (True Negative) |
  | Apnea | Normal | **−1.0** (False Positive) |
  | Normal | Apnea | **−10.0** (False Negative) |
* **Loss Function:** `FocalLoss` (γ = 2.0) with square-root-dampened class weights (`sqrt(N / class_count)`), not plain cross-entropy — down-weights easy/majority examples and focuses gradient on hard, minority-class (apnea) predictions.
* **Patient Isolation:** The evaluation loop automatically flushes the Transformer's context history buffers — and resets the target RTG — whenever a new patient boundary is reached (per-patient timestep resets to 0), preventing inter-patient data leakage.
---
 
## What Changed From the Original Pipeline
 
This codebase was substantially revised partway through development. If you're comparing against an earlier version of this repo or earlier write-ups, note the following:
 
**Removed / replaced:**
* ~~89 fixed `neurokit2`/`scipy`-only features~~ → feature count is now dynamic (HRV time-domain + 15 morphology + 2 spectral/EDR), locked in automatically per run via `data/feature_columns.npy`.
* ~~Z-score normalization~~ → `RobustScaler`.
* ~~Plain class-weighted cross-entropy~~ → `FocalLoss`.
* ~~Random-noise synthetic agents (+15%/+30% randomly flipped labels)~~ → probability-threshold agents derived from an actual trained classifier.
* ~~Reward matrix with `0.0` for correct classifications~~ → corrected to `+10.0` (TP) / `+1.0` (TN), since the original `0.0` baseline gave the Decision Transformer no positive signal to reinforce correct behavior.
* ~~20-minute context window~~ → 60-minute context window.
* ~~Inline, duplicated reward logic in the evaluation script~~ → single shared `compute_clinical_reward` function imported by both preprocessing and evaluation.
**Added:**
* `ECGdeli` (via `pyECGdeli`) integration for wavelet-based QRS detection, replacing a pure NeuroKit2-only peak detection path.
* Full P/QRS/T morphological feature extraction (ECGdeli-equivalent), giving the model access to interval and amplitude features beyond HRV alone.
* A fallback path to NeuroKit2's Pan-Tompkins detector if `pyECGdeli` returns no peaks for a given recording.
* Windows-safe `DataLoader` configuration in `train.py` (`num_workers=0` on Windows to avoid `spawn`-related multiprocessing hangs).
**Still planned / not yet implemented:**
* Foundation-model-based ECG representations as state inputs (currently handcrafted features only).
* Extension to heterogeneous physiological streams (multivariate vitals beyond single-lead ECG).
* Multi-horizon event sensitivity and demographic/subgroup robustness analysis.
---
 
## Installation & Setup
 
1. **Clone this repository** to your local computer.
2. **Clone `pyECGdeli`** into the project root (required by `data_prep.py`):
```bash
   git clone https://github.com/NPilia/pyECGdeli.git
```
3. **Install Dependencies:** Ensure you have Python 3.8+ and install the required signal-processing and deep-learning packages:
```bash
   pip install torch numpy scipy pandas wfdb neurokit2 scikit-learn joblib
```
4. **Download Data:** Because the processed data files are too large for GitHub, you may either:
   * Download the processed `data/` folder from the cloud link below and place it directly in the root directory of this project:
     * **Processed Data Link:** `https://drive.google.com/drive/folders/14Ua93MEvPOmxf3_4jQ6BFd6imagkPVks?usp=sharing`
     * ⚠️ **Note:** this link was generated for the *previous* feature set. If you re-run preprocessing with the current `data_prep.py`, your local `data/` folder will have a different (larger) feature dimensionality than this archive. Don't mix outputs from the old and new pipeline versions in the same `data/` directory.
   * **Or** place the raw `apnea-ecg-database-1.0.0` folder in the root directory and run preprocessing from scratch (see below). If using the official test set, confirm your `x01.apn`–`x35.apn` files contain real per-minute annotations and not placeholder labels — some distributions require a separate answer-key download.
---
 
## Pipeline Execution Order
 
To reproduce the preprocessing, training, and RTG-conditioned clinical rollouts, run the scripts in the following exact order:
 
### 1. Pre-Processing (`data_prep.py` & `prep_test_data.py`)
Extracts ECGdeli QRS peaks, ECGdeli-equivalent morphological features, HRV features, and spectral/EDR features from the raw PhysioNet ECG records, then generates the multi-agent augmented training trajectories.
```bash
python data_prep.py
python prep_test_data.py
```
*Output: Generates processed `.npz` arrays, `feature_columns.npy`, `robust_scaler.pkl`, and `rtg_scale.npy` inside a local `data/` directory. Run `data_prep.py` first — `prep_test_data.py` depends on the column list and scaler it produces.*
 
### 2. Dataset Definition (`dataset.py`)
Contains the `DecisionTransformerDataset` class. Handles NaN/Inf cleaning, builds patient-boundary-safe sliding context windows for training, and is feature-dimension agnostic (adapts automatically to however many columns preprocessing produces). *(Does not need to be run directly.)*
 
### 3. Model Architecture (`model.py`)
Defines the causal Transformer architecture, state/action/RTG embedding layers, and the attention-based prediction head. *(Does not need to be run directly.)*
 
### 4. Model Training (`train.py`)
Trains the network for 10 epochs using AdamW, gradient clipping, a linear warmup/decay LR schedule, and Focal Loss with square-root-dampened class weights.
```bash
python train.py
```
*Output: Saves the optimized weights to `decision_transformer_weights.pth`.*
 
### 5. RTG-Conditioned Evaluation (`test.py`)
Executes autoregressive clinical rollouts on the unseen test set, sweeping across a range of raw target RTG values to demonstrate inference-time control over the sensitivity/specificity tradeoff.
```bash
python test.py
```
 
---
 
## RTG Sweep Reference
 
`test.py` currently sweeps these **raw** target RTG values (internally divided by the fixed `rtg_scale` saved during preprocessing, then clamped to `[-1, 1]` before being fed to the model):
 
```python
test_raw_targets = [5000.0, 3000.0, 1000.0, 0.0, -1000.0, -3000.0]
```
 
> ⚠️ The persona table below (Hyper-Vigilant / Standard Expert / Cautious / Passive) describes the *intended* behavior pattern from an earlier experimental run with a different reward scale, model checkpoint, and test set size. The specific sensitivity/specificity numbers have **not been reverified against the current pipeline** (new features, new reward matrix, new RTG scale) and should be treated as illustrative only. Re-run `test.py` after training on the current pipeline and replace this table with your actual results — the relative *direction* (higher RTG → higher sensitivity, lower specificity) should hold, but the exact percentages and RTG breakpoints will differ.
 
| Target RTG Prompt | Clinical Persona (expected direction) | Behavioral Style |
| :---: | :--- | :--- |
| **High (e.g. `+5000`)** | Hyper-Vigilant | Aggressively hunts for apnea; expect higher sensitivity, lower specificity. |
| **Mid (e.g. `0` to `+1000`)** | Standard Expert | Balanced tradeoff; expect the best joint sensitivity/specificity. |
| **Low (e.g. `-1000`)** | Cautious / Hesitant | More passive; expect higher specificity, lower sensitivity. |
| **Very low (e.g. `-3000`)** | Passive Baseline | Defaults heavily toward predicting "Healthy"; expect highest specificity, lowest sensitivity. |
 
**TODO before sharing results with faculty:** replace the table above with the actual `Sensitivity` / `Specificity` printout from your own `test.py` run, and confirm the real test-set minute count (the previous "17,206 minutes" figure has not been reverified against the current preprocessing pipeline's feature set).