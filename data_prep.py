import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd  
import os
import warnings
import joblib 
from concurrent.futures import ProcessPoolExecutor, as_completed

import scipy.signal as signal
import scipy.interpolate as interp
import scipy.integrate as integrate

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore") 

def process_single_patient(patient_args):
    """Worker function to process ONE patient. Takes a tuple of (patient_id, is_train)."""
    patient_id, is_train = patient_args
    print(f"--- Starting Patient '{patient_id}' ---")
    states = [] 
    actions = []
    
    # Determine the cohort group from the patient ID prefix
    if patient_id.startswith('a'):
        cohort = 'Apnea Group (A)'
    elif patient_id.startswith('c'):
        cohort = 'Control Group (C)'
    elif patient_id.startswith('b'):
        cohort = 'Borderline Group (B)'
    else:
        cohort = 'Test Cohort (X)'
        
    cohort_counts = {'Apnea Group (A)': 0, 'Control Group (C)': 0, 'Borderline Group (B)': 0, 'Test Cohort (X)': 0}
    
    local_data_dir = 'apnea-ecg-database-1.0.0'
    file_path = os.path.join(local_data_dir, patient_id)
    
    try:
        record = wfdb.rdrecord(file_path)
        annotation = wfdb.rdann(file_path, 'apn')

        ecg_signal = record.p_signal[:, 0]
        fs = record.fs  
        labels = annotation.symbol 

        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs)
        all_r_peaks = info['ECG_R_Peaks']

        for i in range(len(labels)):
            # Tally every available minute to track true database distribution
            cohort_counts[cohort] += 1
                
            # Exclude borderline patient data from training vectors
            if cohort == 'Borderline Group (B)':
                continue 

            # Offline 55-second Window Augmentation
            if is_train:
                offset = np.random.randint(0, int(5 * fs))
            else:
                offset = int(2.5 * fs)
                
            start_idx = int(i * 60 * fs) + offset
            end_idx = start_idx + int(55 * fs)
            
            try:
                chunk_peaks = all_r_peaks[(all_r_peaks >= start_idx) & (all_r_peaks < end_idx)]
                chunk_peaks_relative = chunk_peaks - start_idx
                
                if len(chunk_peaks_relative) > 5:
                    hrv = nk.hrv(chunk_peaks_relative, sampling_rate=fs)
                    hrv_dict = hrv.iloc[0].to_dict()
                else:
                    continue 

                window_start_idx = max(0, i - 4) * 60 * fs
                window_peaks = all_r_peaks[(all_r_peaks >= window_start_idx) & (all_r_peaks < end_idx)]
                
                if len(window_peaks) > 10:
                    rr_intervals = np.diff(window_peaks) / fs
                    rr_times = window_peaks[1:] / fs
                    
                    f_interp = interp.interp1d(rr_times, rr_intervals, kind='cubic', fill_value="extrapolate")
                    time_grid = np.arange(rr_times[0], rr_times[-1], 1/4.0)
                    rr_interp = f_interp(time_grid)
                    
                    safe_nperseg = min(256, len(rr_interp))
                    
                    freqs, psd = signal.welch(rr_interp, fs=4.0, nperseg=safe_nperseg)
                    band_mask = (freqs >= 0.008) & (freqs <= 0.036)
                    
                    apnea_power = integrate.trapezoid(psd[band_mask], freqs[band_mask])
                    total_power = integrate.trapezoid(psd, freqs)
                    apnea_band_ratio = apnea_power / total_power if total_power > 0 else 0.0
                    
                    r_amplitudes = cleaned_ecg[window_peaks]
                    edr_variance = np.var(r_amplitudes)
                else:
                    apnea_band_ratio = 0.0
                    edr_variance = 0.0
                
                hrv_dict['apnea_band_ratio'] = float(apnea_band_ratio)
                hrv_dict['edr_variance'] = float(edr_variance)
                
                for k, v in hrv_dict.items():
                    if np.isnan(v) or np.isinf(v):
                        hrv_dict[k] = 0.0
                
                action = 1 if labels[i] == 'A' else 0
                
                states.append(hrv_dict)
                actions.append(action)
                
            except Exception:
                pass 
                
        print(f"+++ Finished '{patient_id}' | Contributed {len(states)} minutes to dataset +++")
        return states, np.array(actions), cohort_counts
        
    except FileNotFoundError:
        print(f"Failed to find local files for patient {patient_id}.")
        return [], np.array([]), {}
    except Exception as e:
        print(f"Failed to process patient {patient_id}: {e}")
        return [], np.array([]), {}


def download_and_extract_features(patient_list, is_train=True):
    """Distributes patients across CPU cores and tracks class distributions."""
    if not os.path.exists('data'):
        os.makedirs('data')
        
    all_states = [] 
    all_actions = []  
    all_timesteps = [] 
    total_cohort_distribution = {'Apnea Group (A)': 0, 'Control Group (C)': 0, 'Borderline Group (B)': 0, 'Test Cohort (X)': 0}

    max_workers = max(1, os.cpu_count() - 1)
    print(f"\nBooting up {max_workers} CPU cores for parallel processing...")

    patient_args = [(pid, is_train) for pid in patient_list]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_patient, arg): arg[0] for arg in patient_args}
        
        for future in as_completed(futures):
            states, actions, cohort_counts = future.result()
            
            # Aggregate cohort statistics
            for k in total_cohort_distribution:
                if k in cohort_counts:
                    total_cohort_distribution[k] += cohort_counts[k]

            if len(states) > 0:
                all_states.extend(states)
                all_actions.extend(actions)
                all_timesteps.extend(np.arange(len(actions)))

    print("\n--- Final Dataset Patient Cohort Distribution ---")
    for cohort_name, count in total_cohort_distribution.items():
        if count > 0:
            print(f"{cohort_name}: {count} total minutes")
    print("------------------------------------------------")

    # 1. Turn into DataFrame
    df_states = pd.DataFrame(all_states)
    
    # 2. Fix internal NaNs immediately
    df_states = df_states.fillna(0.0)
    
    # 3. Establish or load master columns map in the correct sequence
    columns_map_path = 'data/feature_columns.npy'
    if is_train:
        master_cols = sorted(df_states.columns)
        np.save(columns_map_path, master_cols)
        print(f"Locked in {len(master_cols)} master feature definitions.")
    else:
        master_cols = np.load(columns_map_path, allow_pickle=True).tolist()
        print(f"Loaded {len(master_cols)} master feature definitions.")
        
    # 4. Align columns and handle any completely missing feature columns
    df_states = df_states.reindex(columns=master_cols, fill_value=0.0)
    
    raw_states = df_states.to_numpy()
    raw_actions = np.array(all_actions)
    raw_timesteps = np.array(all_timesteps)

    return raw_states, raw_actions, raw_timesteps


def calculate_rtg_and_rewards(agent_actions, true_labels):
    """Calculates cumulative RTG using positive rewards and negative penalties."""
    rewards = np.zeros_like(agent_actions, dtype=np.float32)
    for i in range(len(agent_actions)):
        pred = agent_actions[i]
        truth = true_labels[i]
        
        # New Reward Structure
        if pred == 1 and truth == 1:
            rewards[i] = 10.0   # True Positive
        elif pred == 0 and truth == 0:
            rewards[i] = 1.0    # True Negative
        elif pred == 1 and truth == 0:
            rewards[i] = -1.0   # False Positive penalty
        elif pred == 0 and truth == 1:
            rewards[i] = -10.0  # False Negative penalty
            
    rtg = np.zeros_like(rewards)
    current_rtg = 0.0
    for i in reversed(range(len(rewards))):
        current_rtg = rewards[i] + current_rtg
        rtg[i] = current_rtg
    return rtg


def generate_synthetic_agents(states, true_labels, timesteps):
    """Uses a baseline model thresholding to create behavioral diversity without corrupting labels."""
    print("\nTraining Baseline Classifier to generate synthetic behaviors...")
    
    # Train a fast baseline model on the scaled states
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(states, true_labels)
    apnea_probabilities = clf.predict_proba(states)[:, 1]

    aug_states = []
    aug_actions = []
    aug_rtgs = []
    aug_timesteps = [] 

    # 1. The Perfect Agent
    perfect_actions = true_labels.copy()
    perfect_rtgs = calculate_rtg_and_rewards(perfect_actions, true_labels)
    aug_states.append(states)
    aug_actions.append(perfect_actions)
    aug_rtgs.append(perfect_rtgs)
    aug_timesteps.append(timesteps) 

    # 2. The Cautious Agent (Low threshold to alarm -> More False Positives)
    # If there's even a 20% chance of Apnea, ring the alarm.
    cautious_actions = (apnea_probabilities >= 0.20).astype(int)
    cautious_rtgs = calculate_rtg_and_rewards(cautious_actions, true_labels)
    aug_states.append(states)
    aug_actions.append(cautious_actions)
    aug_rtgs.append(cautious_rtgs)
    aug_timesteps.append(timesteps) 

    # 3. The Careless Agent (High threshold to alarm -> More False Negatives)
    # Require 80% certainty of Apnea to ring the alarm.
    careless_actions = (apnea_probabilities >= 0.80).astype(int)
    careless_rtgs = calculate_rtg_and_rewards(careless_actions, true_labels)
    aug_states.append(states)
    aug_actions.append(careless_actions)
    aug_rtgs.append(careless_rtgs)
    aug_timesteps.append(timesteps) 

    return np.vstack(aug_states), np.concatenate(aug_actions), np.concatenate(aug_rtgs), np.concatenate(aug_timesteps)


if __name__ == "__main__":
    apnea_patients = [f'a{i:02d}' for i in range(1, 21)]
    control_patients = [f'c{i:02d}' for i in range(1, 11)]
    borderline_patients = [f'b{i:02d}' for i in range(1, 6)]
    
    # Combine all 35 training records
    train_patients = apnea_patients + control_patients + borderline_patients
    
    raw_states, true_labels, raw_timesteps = download_and_extract_features(train_patients, is_train=True)
    
    # Apply Robust Scaling to replace standard z-score normalization
    print("\nFitting RobustScaler to features...")
    scaler = RobustScaler()
    scaled_states = scaler.fit_transform(raw_states)
    joblib.dump(scaler, 'data/robust_scaler.pkl')
    
    # Augment dataset with synthetic agents using the scaled states
    aug_states, aug_actions, aug_rtgs, aug_timesteps = generate_synthetic_agents(
        scaled_states, true_labels, raw_timesteps
    )
    
    # Normalize RTGs to [-1.0, 1.0] ---
    print("\nNormalizing RTGs to [-1, 1] scale...")
    rtg_max = np.max(aug_rtgs)
    rtg_min = np.min(aug_rtgs)
    aug_rtgs = 2 * ((aug_rtgs - rtg_min) / (rtg_max - rtg_min)) - 1
    print(f"RTG Bounds Normalized -> Original Min: {rtg_min:.1f} | Original Max: {rtg_max:.1f}")
    
    save_path = 'data/processed_train_dataset.npz'
    np.savez(
        save_path, 
        states=aug_states, 
        actions=aug_actions, 
        rtgs=aug_rtgs, 
        timesteps=aug_timesteps
    )
    
    print("\n--- Pipeline Complete ---")
    print(f"Data saved successfully to: {save_path}")
    print(f"Final States Shape: {aug_states.shape}")