# Decision Transformer for Sleep Apnea Detection from ECG

This repository contains a complete pipeline for sequential sleep apnea detection from single-lead ECG signals using an offline reinforcement learning approach via a Decision Transformer (DT).

## Project Overview

Instead of treating sleep apnea detection as a static classification task, this project frames it as an offline sequential decision-making problem. The model acts as a clinical agent that evaluates 20-minute sliding contextual windows of patient ECG data. 

By utilizing **Static Persona Conditioning** (Return-to-Go / RTG prompting), the model's clinical sensitivity and specificity can be dynamically steered at inference time without modifying the underlying network weights or threshold parameters.

### Core Engineering Details
* **Feature Extraction:** Extracts 86 physiological features per minute using `neurokit2` and `scipy`, including the foundational **Apnea Band (0.008 - 0.036 Hz)** (equivalent to the 0.5 to 2.2 cycles per minute spectral power fraction defined in PhysioNet literature) via Welch's periodogram.
* **Data Augmentation:** The training data is augmented with off-policy trajectories from an *Expert Agent* (ground-truth labels), a *Cautious Agent* (+15% random False Positives), and a *Careless Agent* (+30% random False Negatives) to expose the Transformer to varying degrees of clinical error.
* **Clinical Reward Matrix:** Programmed to mirror clinical risks, penalizing False Positives at `-1.0` and severely penalizing False Negatives (missed apnea) at `-10.0`. Perfect classifications yield a reward of `0.0`.
* **Patient Isolation:** The evaluation loop automatically flushes the Transformer's context history buffers whenever a new patient boundary is reached (timestamp resets to 0), preventing inter-patient data leakage.

---

## Installation & Setup

1. **Clone this repository** to your local computer.
2. **Install Dependencies:** Ensure you have Python 3.8+ and install the required signal-processing and deep-learning packages:
   ```bash
   pip install torch numpy scipy pandas wfdb neurokit2
   ```
3. **Download Data:** Because the processed data files are too large for GitHub, please download the processed `data/` folder from the cloud link below and place it directly in the root directory of this project:
   * **Processed Data Link:** `[INSERT YOUR GOOGLE DRIVE / CLOUD LINK HERE]`
   * *Alternatively, place the raw `apnea-ecg-database-1.0.0` folder in the root directory if you wish to run preprocessing from scratch.*

---

## Pipeline Execution Order

To reproduce the preprocessing, training, and multi-persona clinical rollouts, run the scripts in the following exact order:

### 1. Pre-Processing (`data_prep.py` & `prep_test_data.py`)
Extracts the 86 physiological features from the raw PhysioNet ECG records and generates the multi-persona augmented training trajectories.
```bash
python data_prep.py
python prep_test_data.py
```
*Output: Generates processed `.npz` arrays inside a local `data/` directory.*

### 2. Dataset Definition (`dataset.py`)
This script contains the `DecisionTransformerDataset` class. It manages sliding window tokenization, handles missing values/NaN cleaning, and applies consistent Z-score normalization across training and test splits. *(Does not need to be run directly).*

### 3. Model Architecture (`model.py`)
Defines the Causal Transformer architecture, state/action/RTG embedding layers, and the attention-based prediction head. *(Does not need to be run directly).*

### 4. Model Training (`train.py`)
Trains the network for 10 epochs using an AdamW optimizer, gradient clipping, and a square-root dampened class-weighted Cross-Entropy loss function to counter class imbalance.
```bash
python train.py
```
*Output: Saves the optimized weights to `decision_transformer_weights.pth` (~3.3 MB).*

### 5. Multi-Persona Evaluation (`test.py`)
Executes clinical rollouts on the unseen test set (17,206 minutes) using Static Persona Conditioning.
```bash
python test.py
```

---

## Persona Reference Guide

When running `test.py`, the model's diagnostic behavior shifts dramatically depending on the static `initial_target_rtg` prompt provided to the network:

| Target RTG Prompt | Clinical Persona | Sensitivity (Catching Apnea) | Specificity (Avoiding Alarms) | Behavioral Style |
| :---: | :--- | :---: | :---: | :--- |
| **`+0.15`** | Hyper-Vigilant | **82.72%** | 76.81% | Aggressively hunts for apnea; ideal for high-risk first-stage screenings. Extrapolates along linear embeddings. |
| **`0.00`** | Standard Expert | **75.60%** | **80.72%** | Balanced clinician aiming for zero penalties. Highest overall diagnostic performance. |
| **`-0.20`** | Cautious / Hesitant | 54.20% | 89.41% | More passive; stops guessing unless certain, suppressing false alarms. |
| **`-0.50`** | Passive Baseline | 39.98% | **93.46%** | Defaults heavily to the majority class (Healthy/Control). |

---

## Recommended `.gitignore` Configuration
To keep your GitHub repository clean and lightweight, ensure a `.gitignore` file is present in the root directory containing:
```text
.venv/
__pycache__/
*.pyc
apnea-ecg-database-1.0.0/
apnea-ecg-database-1.0.0.zip
data/
*.npz
*.dat
*.hea
temp.py
```