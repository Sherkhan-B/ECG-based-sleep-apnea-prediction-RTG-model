"""
data_prep.py — ECG Apnea Feature Extraction Pipeline
======================================================
Decision Transformer for Clinical Event Prediction (SURA 2025)

ECGdeli integration strategy
------------------------------
The *authoritative* ECGdeli toolkit (KIT-IBT/ECGdeli) is a MATLAB library.
Its Python port (NPilia/pyECGdeli) is work-in-progress and has only published
QRS/R-peak detection — it does NOT yet expose P-wave or T-wave delineation.

Therefore we use a two-layer strategy that matches the professor's intent:

  Layer 1 — QRS detection:
    pyECGdeli.ECGdeli.QRS_detection()  ← same wavelet algorithm as the MATLAB
    toolbox, validated on the QT Database (mean error −2.00 ± 3.85 samples).

  Layer 2 — Full P/QRS/T delineation:
    neurokit2.ecg_delineate(method="dwt")  ← discrete wavelet transform method
    that implements the same Laguna/Martinez wavelet approach used inside
    ECGdeli's T_Detection.m and P_Detection.m.  This gives us the complete
    fiducial point table (P-on, P-peak, P-off, Q, R, S, T-peak, T-off) that
    the MATLAB toolbox would produce.

ECGdeli FPT column reference (MATLAB, 1-indexed; Python port is 0-indexed):
  col 1  Pon      col 2  Ppeak    col 3  Poff
  col 4  QRSon    col 5  Q        col 6  R
  col 7  S        col 8  QRSoff   col 9  Ton
  col 10 Tpeak   col 11 Toff

Feature groups produced per minute-window
------------------------------------------
  A. HRV time-domain  (NeuroKit2 hrv_time)
  B. ECG morphology   (ECGdeli-equivalent: PR/QT/ST intervals + amplitudes)
  C. Apnea-band power spectral feature (0.008–0.036 Hz)
  D. EDR variance     (ECG-derived respiration from R-amplitude modulation)
"""

import os
import sys

# ── pyECGdeli path injection ──────────────────────────────────────────────────
# Clone with:  git clone https://github.com/NPilia/pyECGdeli.git
# The repo contains a single file: ECGdeli.py  (QRS_detection lives there)
local_deli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyECGdeli'))
if local_deli_path not in sys.path:
    sys.path.append(local_deli_path)

import ECGdeli  # from pyECGdeli — provides QRS_detection (R-peaks via wavelet)

import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import warnings
import joblib
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import scipy.signal as signal
import scipy.interpolate as interp
import scipy.integrate as integrate

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  ECGdeli-equivalent morphological feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_morphology_features(ecg_chunk: np.ndarray, r_peaks_rel: np.ndarray, fs: int) -> dict:
    """
    Compute ECGdeli-equivalent morphological features from a single ECG chunk.

    Uses NeuroKit2's DWT delineator — which implements the same Laguna wavelet
    method as ECGdeli's P_Detection.m and T_Detection.m — to locate P-on,
    P-peak, P-off, Q, R, S, T-peak, T-off for every beat in the chunk.

    Parameters
    ----------
    ecg_chunk     : cleaned ECG signal (1-D, one minute window)
    r_peaks_rel   : R-peak sample indices relative to ecg_chunk start
    fs            : sampling frequency in Hz

    Returns
    -------
    dict of scalar morphological features, or empty dict on failure
    """
    feats = {}
    try:
        if len(r_peaks_rel) < 3:
            return feats

        # NeuroKit2 delineation — DWT method mirrors ECGdeli's wavelet pipeline
        _, waves = nk.ecg_delineate(
            ecg_chunk,
            r_peaks_rel,
            sampling_rate=fs,
            method="dwt",   # discrete wavelet transform — same as ECGdeli
            show=False
        )

        def _safe_median(arr):
            """Return median of a list, ignoring NaN/None entries."""
            vals = [v for v in arr if v is not None and not np.isnan(float(v))]
            return float(np.median(vals)) if vals else np.nan

        def _safe_mean(arr):
            vals = [v for v in arr if v is not None and not np.isnan(float(v))]
            return float(np.mean(vals)) if vals else np.nan

        r_arr = np.array(r_peaks_rel, dtype=float)

        # ── Interval features (in seconds) ───────────────────────────────────
        # PR interval  =  R − P-peak
        p_peaks  = np.array(waves.get("ECG_P_Peaks",   [np.nan] * len(r_arr)), dtype=float)
        t_peaks  = np.array(waves.get("ECG_T_Peaks",   [np.nan] * len(r_arr)), dtype=float)
        q_peaks  = np.array(waves.get("ECG_Q_Peaks",   [np.nan] * len(r_arr)), dtype=float)
        s_peaks  = np.array(waves.get("ECG_S_Peaks",   [np.nan] * len(r_arr)), dtype=float)
        p_onsets = np.array(waves.get("ECG_P_Onsets",  [np.nan] * len(r_arr)), dtype=float)
        p_offsets= np.array(waves.get("ECG_P_Offsets", [np.nan] * len(r_arr)), dtype=float)
        t_offsets= np.array(waves.get("ECG_T_Offsets", [np.nan] * len(r_arr)), dtype=float)
        r_onsets = np.array(waves.get("ECG_R_Onsets",  [np.nan] * len(r_arr)), dtype=float)  # QRS onset

        pr_intervals = (r_arr - p_peaks) / fs          # seconds
        qt_intervals = (t_offsets - r_onsets) / fs
        qrs_durations= (s_peaks - q_peaks) / fs
        p_durations  = (p_offsets - p_onsets) / fs

        # RR → heart rate
        rr_sec = np.diff(r_arr) / fs
        rr_sec = rr_sec[rr_sec > 0]  # guard against bad peaks

        feats["morph_pr_mean"]    = _safe_mean(pr_intervals)
        feats["morph_pr_std"]     = float(np.nanstd(pr_intervals))
        feats["morph_qt_mean"]    = _safe_mean(qt_intervals)
        feats["morph_qt_std"]     = float(np.nanstd(qt_intervals))
        feats["morph_qrs_mean"]   = _safe_mean(qrs_durations)
        feats["morph_qrs_std"]    = float(np.nanstd(qrs_durations))
        feats["morph_p_dur_mean"] = _safe_mean(p_durations)

        # QTc (Bazett correction): QTc = QT / sqrt(RR)
        if len(rr_sec) > 0 and not np.isnan(feats["morph_qt_mean"]):
            mean_rr = float(np.mean(rr_sec))
            feats["morph_qtc_bazett"] = feats["morph_qt_mean"] / np.sqrt(mean_rr) if mean_rr > 0 else np.nan
        else:
            feats["morph_qtc_bazett"] = np.nan

        # ── Amplitude features (normalised by R-amplitude) ───────────────────
        def _amp(indices):
            """Mean amplitude at given sample indices in ecg_chunk."""
            idx = [int(i) for i in indices if not np.isnan(i) and 0 <= int(i) < len(ecg_chunk)]
            return float(np.mean(ecg_chunk[idx])) if idx else np.nan

        r_amp  = _amp(r_arr)
        p_amp  = _amp(p_peaks)
        t_amp  = _amp(t_peaks)
        q_amp  = _amp(q_peaks)
        s_amp  = _amp(s_peaks)

        feats["morph_r_amplitude"]   = r_amp
        feats["morph_p_amplitude"]   = p_amp
        feats["morph_t_amplitude"]   = t_amp
        feats["morph_p_r_ratio"]     = (p_amp / r_amp) if (r_amp and not np.isnan(r_amp) and r_amp != 0) else np.nan
        feats["morph_t_r_ratio"]     = (t_amp / r_amp) if (r_amp and not np.isnan(r_amp) and r_amp != 0) else np.nan
        feats["morph_q_amplitude"]   = q_amp
        feats["morph_s_amplitude"]   = s_amp

        # ST elevation/depression: mean amplitude 60–80 ms after J-point (S-peak)
        st_amps = []
        for sp in s_peaks:
            if np.isnan(sp):
                continue
            j60  = int(sp + 0.060 * fs)
            j80  = int(sp + 0.080 * fs)
            if j80 < len(ecg_chunk):
                st_amps.append(float(np.mean(ecg_chunk[j60:j80])))
        feats["morph_st_level"] = float(np.mean(st_amps)) if st_amps else np.nan

        # P-wave morphology: std of P amplitudes (biphasic P → high std)
        p_amp_vals = [ecg_chunk[int(p)] for p in p_peaks if not np.isnan(p) and 0 <= int(p) < len(ecg_chunk)]
        feats["morph_p_amp_std"] = float(np.std(p_amp_vals)) if len(p_amp_vals) > 1 else np.nan

    except Exception as e:
        # Soft failure — return whatever partial features were filled
        pass

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-patient worker (runs in a subprocess via ProcessPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_patient(patient_args):
    """Worker function to process ONE patient. Takes a tuple of (patient_id, is_train)."""
    patient_id, is_train = patient_args
    print(f"--- Starting Patient '{patient_id}' ---")

    states  = []   # one feature dict per minute (or None placeholder)
    actions = []

    seed = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.default_rng(seed)

    failed_minutes   = 0
    failure_reasons  = {}

    # Cohort group from patient ID prefix
    cohort_map = {'a': 'Apnea Group (A)', 'c': 'Control Group (C)',
                  'b': 'Borderline Group (B)'}
    cohort = cohort_map.get(patient_id[0], 'Test Cohort (X)')
    cohort_counts = {'Apnea Group (A)': 0, 'Control Group (C)': 0,
                     'Borderline Group (B)': 0, 'Test Cohort (X)': 0}

    local_data_dir = 'apnea-ecg-database-1.0.0'
    file_path = os.path.join(local_data_dir, patient_id)

    try:
        record     = wfdb.rdrecord(file_path)
        annotation = wfdb.rdann(file_path, 'apn')

        ecg_signal  = record.p_signal[:, 0]
        fs          = record.fs
        labels      = annotation.symbol

        # ── Layer 0: NeuroKit2 cleaning ───────────────────────────────────────
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)

        # ── Layer 1: pyECGdeli QRS detection (wavelet, same as MATLAB ECGdeli) ─
        # Requires column-vector shape (N, 1) to match the MATLAB API contract
        ecg_col = cleaned_ecg.reshape(-1, 1)
        t0 = time.time()
        deli_output = ECGdeli.QRS_detection(ecg_col, fs)
        print(f"  [{patient_id}] ECGdeli QRS detection: {time.time() - t0:.1f}s")

        # Extract R-peak sample indices from the FPT table
        # pyECGdeli FPT is 0-indexed; R-peak is column index 5 (same as MATLAB col 6)
        all_r_peaks = np.array([], dtype=int)
        if deli_output is not None:
            if isinstance(deli_output, tuple) and len(deli_output) > 0:
                fpt_table = deli_output[0]
                if len(fpt_table) > 0:
                    try:
                        fpt_array = np.array(fpt_table, dtype=float)
                        peaks = fpt_array[:, 5]          # col 5 = R-peak (0-indexed)
                        all_r_peaks = peaks[~np.isnan(peaks)].astype(int)
                    except (ValueError, TypeError):
                        extracted = []
                        for row in fpt_table:
                            if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 5:
                                val = row[5]
                                if isinstance(val, (int, float)) and not np.isnan(val):
                                    extracted.append(int(val))
                        all_r_peaks = np.unique(extracted).astype(int)

        # Fallback: if pyECGdeli produced no peaks, use NeuroKit2 Pan-Tompkins
        if len(all_r_peaks) == 0:
            print(f"  [{patient_id}] WARNING: ECGdeli returned no R-peaks — falling back to NeuroKit2")
            _, ecg_info = nk.ecg_process(cleaned_ecg, sampling_rate=fs)
            all_r_peaks = ecg_info["ECG_R_Peaks"]

        # ─────────────────────────────────────────────────────────────────────
        #  Minute-level loop
        # ─────────────────────────────────────────────────────────────────────
        for i in range(len(labels)):
            cohort_counts[cohort] += 1

            action = 1 if labels[i] == 'A' else 0
            actions.append(action)

            # ── Window augmentation ──────────────────────────────────────────
            # Training: random 0–5 s offset to decorrelate identical epochs
            # Validation: fixed 2.5 s offset (deterministic, centred)
            if is_train:
                offset = rng.integers(0, int(5 * fs))
            else:
                offset = int(2.5 * fs)

            start_idx = int(i * 60 * fs) + offset
            end_idx   = start_idx + int(55 * fs)

            feature_dict = None
            try:
                # ── R-peaks in this minute window ────────────────────────────
                chunk_peaks     = all_r_peaks[(all_r_peaks >= start_idx) & (all_r_peaks < end_idx)]
                chunk_peaks_rel = chunk_peaks - start_idx   # relative indices for delineation

                if len(chunk_peaks_rel) > 5:
                    # ── A. HRV time-domain features (NeuroKit2) ──────────────
                    hrv      = nk.hrv_time(chunk_peaks_rel, sampling_rate=fs, show=False)
                    hrv_dict = hrv.iloc[0].to_dict()

                    # ── B. ECGdeli morphology features (DWT delineation) ─────
                    ecg_chunk = cleaned_ecg[start_idx:end_idx]
                    morph_feats = extract_morphology_features(ecg_chunk, chunk_peaks_rel, fs)
                    hrv_dict.update(morph_feats)

                    # ── C. Apnea-band spectral power (5-minute context window) ─
                    # Uses a 5-minute sliding context so apnea-epoch PSD is
                    # more stable than from a single 55-second chunk
                    window_start = max(0, i - 4) * 60 * fs
                    window_peaks = all_r_peaks[(all_r_peaks >= window_start) & (all_r_peaks < end_idx)]

                    apnea_band_ratio = 0.0
                    if len(window_peaks) > 10:
                        rr_intervals = np.diff(window_peaks) / fs
                        rr_times     = window_peaks[1:] / fs

                        f_interp = interp.interp1d(
                            rr_times, rr_intervals, kind='cubic', fill_value="extrapolate"
                        )
                        time_grid  = np.arange(rr_times[0], rr_times[-1], 1 / 4.0)
                        rr_interp  = f_interp(time_grid)

                        safe_nperseg = min(256, len(rr_interp))
                        freqs, psd   = signal.welch(rr_interp, fs=4.0, nperseg=safe_nperseg)

                        # 0.008–0.036 Hz is the autonomic signature of sleep apnea
                        band_mask        = (freqs >= 0.008) & (freqs <= 0.036)
                        apnea_power      = integrate.trapezoid(psd[band_mask], freqs[band_mask])
                        total_power      = integrate.trapezoid(psd, freqs)
                        apnea_band_ratio = apnea_power / total_power if total_power > 0 else 0.0

                    # ── D. EDR variance (ECG-derived respiration) ────────────
                    # R-amplitude modulation across the context window encodes
                    # respiratory effort; apnea events suppress this modulation
                    edr_variance = 0.0
                    if len(window_peaks) > 10:
                        r_amplitudes = cleaned_ecg[window_peaks]
                        edr_variance = float(np.var(r_amplitudes))

                    hrv_dict['apnea_band_ratio'] = float(apnea_band_ratio)
                    hrv_dict['edr_variance']      = float(edr_variance)

                    # Sanitise any residual NaN/Inf values
                    for k, v in hrv_dict.items():
                        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            hrv_dict[k] = 0.0

                    feature_dict = hrv_dict

            except Exception as e:
                feature_dict = None
                failed_minutes += 1
                err_type = type(e).__name__
                failure_reasons[err_type] = failure_reasons.get(err_type, 0) + 1

            states.append(feature_dict)

        # Surface per-patient failure statistics
        if failed_minutes > 0:
            print(
                f"  [{patient_id}] {failed_minutes}/{len(labels)} minutes failed "
                f"feature extraction: {failure_reasons}"
            )

        # ── Impute failed minutes with a zero vector ──────────────────────────
        template_keys = next((list(s.keys()) for s in states if s is not None), None)

        if template_keys is None:
            print(f"Failed to extract any features for patient {patient_id}; skipping.")
            return [], np.array([]), cohort_counts

        zero_template = {k: 0.0 for k in template_keys}
        states = [s if s is not None else dict(zero_template) for s in states]

        print(f"+++ Finished '{patient_id}' | {len(states)} minutes contributed +++")
        return states, np.array(actions), cohort_counts

    except FileNotFoundError:
        print(f"Failed to find local files for patient {patient_id}.")
        return [], np.array([]), {}
    except Exception as e:
        print(f"Failed to process patient {patient_id}: {e}")
        return [], np.array([]), {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def download_and_extract_features(patient_list, is_train=True):
    """Distributes patients across CPU cores and aggregates results."""
    os.makedirs('data', exist_ok=True)

    all_states     = []
    all_actions    = []
    all_timesteps  = []
    episode_lengths = []
    total_cohort   = {
        'Apnea Group (A)': 0, 'Control Group (C)': 0,
        'Borderline Group (B)': 0, 'Test Cohort (X)': 0
    }

    max_workers  = 12
    patient_args = [(pid, is_train) for pid in patient_list]
    print(f"\nBooting {max_workers} CPU cores for parallel processing...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_patient, arg): arg[0] for arg in patient_args}
        for future in as_completed(futures):
            states, actions, cohort_counts = future.result()
            for k in total_cohort:
                if k in cohort_counts:
                    total_cohort[k] += cohort_counts[k]
            if len(states) > 0:
                all_states.extend(states)
                all_actions.extend(actions)
                all_timesteps.extend(np.arange(len(actions)))
                episode_lengths.append(len(actions))

    print("\n--- Final Dataset Patient Cohort Distribution ---")
    for name, count in total_cohort.items():
        if count > 0:
            print(f"  {name}: {count} minutes")
    print("-------------------------------------------------")

    df_states = pd.DataFrame(all_states).fillna(0.0)

    # Lock / load master column order
    columns_map_path = 'data/feature_columns.npy'
    if is_train:
        master_cols = sorted(df_states.columns)
        np.save(columns_map_path, master_cols)
        print(f"Locked {len(master_cols)} feature columns "
              f"({sum(1 for c in master_cols if c.startswith('morph_'))} morphological).")
    else:
        master_cols = np.load(columns_map_path, allow_pickle=True).tolist()
        print(f"Loaded {len(master_cols)} master feature columns.")

    df_states = df_states.reindex(columns=master_cols, fill_value=0.0)

    return (
        df_states.to_numpy(),
        np.array(all_actions),
        np.array(all_timesteps),
        episode_lengths,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RTG computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_clinical_reward(pred: int, truth: int) -> float:
    """
    Single source of truth for the asymmetric clinical reward structure.

    Imported by both data_prep.py (to build training RTGs) and test.py
    (to update RTG during autoregressive rollout), so the model is always
    evaluated under the exact same reward semantics it was trained on.

    True Positive  +10   (correctly alarming on apnea)
    True Negative  + 1   (correctly quiet in normal minute)
    False Positive − 1   (unnecessary alarm — alarm fatigue)
    False Negative −10   (missed apnea — safety-critical failure)
    """
    if   pred == 1 and truth == 1: return  10.0
    elif pred == 0 and truth == 0: return   1.0
    elif pred == 1 and truth == 0: return  -1.0
    else:                           return -10.0  # pred == 0 and truth == 1


def calculate_rtg_and_rewards(agent_actions, true_labels, episode_lengths):
    """
    Builds the backward-cumulative RTG signal for an offline RL trajectory.

    RTG is computed backwards WITHIN each patient episode so that a high
    target RTG at inference time steers the model toward sensitive behaviour,
    while a lower target RTG selects more conservative prediction.
    """
    rewards = np.zeros(len(agent_actions), dtype=np.float32)
    for i, (pred, truth) in enumerate(zip(agent_actions, true_labels)):
        rewards[i] = compute_clinical_reward(pred, truth)

    rtg = np.zeros_like(rewards)
    idx = 0
    for length in episode_lengths:
        current_rtg = 0.0
        for j in reversed(range(idx, idx + length)):
            current_rtg  = rewards[j] + current_rtg
            rtg[j]       = current_rtg
        idx += length

    assert idx == len(agent_actions), (
        f"episode_lengths sum {idx} ≠ {len(agent_actions)} actions — "
        "patient boundaries don't match data."
    )
    return rtg


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic agent augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_agents(states, true_labels, timesteps, episode_lengths):
    """
    Creates three behavioural trajectories from a single logistic-regression
    baseline, giving the Decision Transformer RTG diversity to learn from:

      Perfect Agent    — oracle upper bound (RTG always high)
      Cautious Agent   — threshold 0.20 → high sensitivity, more FP
      Careless Agent   — threshold 0.80 → low sensitivity, more FN

    Labels (true_labels) are never modified; only the predicted action
    sequence changes so that RTG reflects three distinct operating points.
    """
    print("\nTraining baseline classifier for synthetic agent generation...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(states, true_labels)
    apnea_prob = clf.predict_proba(states)[:, 1]

    agents = {
        "Perfect":  true_labels.copy(),
        "Cautious": (apnea_prob >= 0.20).astype(int),
        "Careless": (apnea_prob >= 0.80).astype(int),
    }

    aug_states, aug_actions, aug_rtgs, aug_timesteps = [], [], [], []
    for name, acts in agents.items():
        rtgs = calculate_rtg_and_rewards(acts, true_labels, episode_lengths)
        aug_states.append(states)
        aug_actions.append(acts)
        aug_rtgs.append(rtgs)
        aug_timesteps.append(timesteps)
        print(f"  {name:8s} agent | mean RTG {rtgs.mean():.1f} | "
              f"sensitivity {acts[true_labels==1].mean():.2f}")

    return (
        np.vstack(aug_states),
        np.concatenate(aug_actions),
        np.concatenate(aug_rtgs),
        np.concatenate(aug_timesteps),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    apnea_patients      = [f'a{i:02d}' for i in range(1, 21)]
    control_patients    = [f'c{i:02d}' for i in range(1, 11)]
    borderline_patients = [f'b{i:02d}' for i in range(1, 6)]
    train_patients      = apnea_patients + control_patients + borderline_patients

    raw_states, true_labels, raw_timesteps, episode_lengths = \
        download_and_extract_features(train_patients, is_train=True)

    # ── Robust scaling ────────────────────────────────────────────────────────
    print("\nFitting RobustScaler...")
    scaler = RobustScaler()
    scaled_states = scaler.fit_transform(raw_states)
    joblib.dump(scaler, 'data/robust_scaler.pkl')

    # ── Synthetic agent augmentation ─────────────────────────────────────────
    aug_states, aug_actions, aug_rtgs, aug_timesteps = generate_synthetic_agents(
        scaled_states, true_labels, raw_timesteps, episode_lengths
    )

    # ── RTG normalisation ─────────────────────────────────────────────────────
    # Divide by (10 × longest episode) so RTG ∈ [−1, 1] roughly.
    # At inference: specify RTG_target = 1.0 for aggressive detection,
    #               RTG_target ≈ 0.1 for conservative behaviour.
    print("\nNormalising RTGs...")
    max_ep   = max(episode_lengths)
    rtg_scale = 10.0 * max_ep
    aug_rtgs /= rtg_scale
    print(f"  RTG scale:  {rtg_scale:.1f}  (longest episode: {max_ep} min)")
    print(f"  RTG range:  [{aug_rtgs.min():.3f}, {aug_rtgs.max():.3f}]")
    np.save('data/rtg_scale.npy', np.array([rtg_scale]))

    # Episode lengths for the 3× augmented dataset
    episode_lengths_aug = episode_lengths * 3

    save_path = 'data/processed_train_dataset.npz'
    np.savez(
        save_path,
        states         = aug_states,
        actions        = aug_actions,
        rtgs           = aug_rtgs,
        timesteps      = aug_timesteps,
        episode_lengths= np.array(episode_lengths_aug),
    )

    print("\n--- Pipeline Complete ---")
    print(f"Saved: {save_path}")
    print(f"States shape:      {aug_states.shape}")
    feature_cols = np.load('data/feature_columns.npy', allow_pickle=True).tolist()
    morph_cols   = [c for c in feature_cols if c.startswith('morph_')]
    print(f"Total features:    {aug_states.shape[1]}")
    print(f"  HRV features:    {aug_states.shape[1] - len(morph_cols) - 2}  (from hrv_time)")
    print(f"  Morphology:      {len(morph_cols)}  (ECGdeli-equivalent via DWT delineation)")
    print(f"  Spectral/EDR:    2  (apnea_band_ratio, edr_variance)")