import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd  # <-- Added for robust feature alignment
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import scipy.signal as signal
import scipy.interpolate as interp
import scipy.integrate as integrate

warnings.filterwarnings("ignore") 

def process_single_patient(patient_id):
    """Worker function to process ONE patient from local files. Runs on a separate CPU core."""
    print(f"--- Starting Patient '{patient_id}' ---")
    states = []  # Will hold feature dictionaries
    actions = []
    
    local_data_dir = 'apnea-ecg-database-1.0.0'
    file_path = os.path.join(local_data_dir, patient_id)
    
    try:
        record = wfdb.rdrecord(file_path)
        annotation = wfdb.rdann(file_path, 'apn')

        ecg_signal = record.p_signal[:, 0]
        fs = record.fs  
        labels = annotation.symbol 

        # Clean and get peaks once
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs)
        all_r_peaks = info['ECG_R_Peaks']

        for i in range(len(labels)):
            start_idx = i * 60 * fs
            end_idx = (i + 1) * 60 * fs
            
            try:
                # Isolate the R-peaks for just this 1-minute chunk
                chunk_peaks = all_r_peaks[(all_r_peaks >= start_idx) & (all_r_peaks < end_idx)]
                chunk_peaks_relative = chunk_peaks - start_idx
                
                if len(chunk_peaks_relative) > 5:
                    hrv = nk.hrv(chunk_peaks_relative, sampling_rate=fs)
                    # Convert to dictionary to keep explicit feature names intact
                    hrv_dict = hrv.iloc[0].to_dict()
                else:
                    continue 

                # The 5-Minute Lookback Window
                window_start_idx = max(0, i - 4) * 60 * fs
                window_peaks = all_r_peaks[(all_r_peaks >= window_start_idx) & (all_r_peaks < end_idx)]
                
                if len(window_peaks) > 10:
                    rr_intervals = np.diff(window_peaks) / fs
                    rr_times = window_peaks[1:] / fs
                    
                    f_interp = interp.interp1d(rr_times, rr_intervals, kind='cubic', fill_value="extrapolate")
                    time_grid = np.arange(rr_times[0], rr_times[-1], 1/4.0)
                    rr_interp = f_interp(time_grid)
                    
                    # Dynamically bound nperseg so it never exceeds the signal length
                    safe_nperseg = min(256, len(rr_interp))
                    
                    # The Apnea Band (0.008 - 0.036 Hz)
                    freqs, psd = signal.welch(rr_interp, fs=4.0, nperseg=safe_nperseg)
                    band_mask = (freqs >= 0.008) & (freqs <= 0.036)
                    
                    # Use explicit scipy.integrate.trapezoid to avoid NumPy removal errors
                    apnea_power = integrate.trapezoid(psd[band_mask], freqs[band_mask])
                    total_power = integrate.trapezoid(psd, freqs)
                    apnea_band_ratio = apnea_power / total_power if total_power > 0 else 0.0
                    
                    # Amplitude Variance
                    r_amplitudes = cleaned_ecg[window_peaks]
                    edr_variance = np.var(r_amplitudes)
                else:
                    apnea_band_ratio = 0.0
                    edr_variance = 0.0
                
                # Append engineering features directly into the dictionary
                hrv_dict['apnea_band_ratio'] = float(apnea_band_ratio)
                hrv_dict['edr_variance'] = float(edr_variance)
                
                # Clean up any potential NaN/Inf edge cases inside values
                for k, v in hrv_dict.items():
                    if np.isnan(v) or np.isinf(v):
                        hrv_dict[k] = 0.0
                
                action = 1 if labels[i] == 'A' else 0
                
                states.append(hrv_dict)
                actions.append(action)
                
            except Exception:
                pass # Safe to skip genuinely corrupt single minutes
                
        print(f"+++ Finished Patient '{patient_id}' +++")
        return states, np.array(actions)
        
    except FileNotFoundError:
        print(f"Failed to find local files for patient {patient_id}.")
        return [], np.array([])
    except Exception as e:
        print(f"Failed to process patient {patient_id}: {e}")
        return [], np.array([])

def download_and_extract_features(patient_list):
    """Distributes patients across all available CPU cores and aligns features."""
    if not os.path.exists('data'):
        os.makedirs('data')
        
    all_states = []   # Will be a combined list of dictionaries
    all_actions = []  
    all_timesteps = [] 

    max_workers = max(1, os.cpu_count() - 1)
    print(f"\nBooting up {max_workers} CPU cores for parallel processing...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_patient, pid): pid for pid in patient_list}
        
        for future in as_completed(futures):
            states, actions = future.result()
            if len(states) > 0:
                all_states.extend(states)
                all_actions.extend(actions)
                all_timesteps.extend(np.arange(len(actions)))

    # Safe DataFrame stacking and cross-cohort column alignment ---
    print("\nAligning features across the entire cohort...")
    df_states = pd.DataFrame(all_states)
    
    # Save/Load a master column map to guarantee Train/Test shape consistency
    columns_map_path = 'data/feature_columns.npy'
    if not os.path.exists(columns_map_path):
        # Training Mode: Establish the alphabetical master feature order
        master_cols = sorted(df_states.columns)
        np.save(columns_map_path, master_cols)
        print(f"Locked in {len(master_cols)} master feature definitions to disk.")
    else:
        # Testing Mode: Force matching column order and structure
        master_cols = np.load(columns_map_path, allow_pickle=True).tolist()
        print(f"Loaded {len(master_cols)} master feature definitions for alignment.")
        
    # Reindex fills completely missing features with 0.0 and aligns everything perfectly
    df_states = df_states.reindex(columns=master_cols, fill_value=0.0)
    
    raw_states = df_states.to_numpy()
    raw_actions = np.array(all_actions)
    raw_timesteps = np.array(all_timesteps)

    return raw_states, raw_actions, raw_timesteps


def calculate_rtg_and_rewards(agent_actions, true_labels):
    """Calculates asymmetric RTG based on the agent's performance."""
    rewards = np.zeros_like(agent_actions, dtype=np.float32)
    for i in range(len(agent_actions)):
        pred = agent_actions[i]
        truth = true_labels[i]
        
        if pred == truth:
            rewards[i] = 0.0
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
    """Generates perfect, cautious, and careless trajectories."""
    print("\nGenerating synthetic trajectories for the entire cohort...")
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

    # 2. The Cautious Agent 
    cautious_actions = true_labels.copy()
    normal_indices = np.where(true_labels == 0)[0]
    fp_count = int(0.15 * len(normal_indices))
    if fp_count > 0:
        fp_indices = np.random.choice(normal_indices, fp_count, replace=False)
        cautious_actions[fp_indices] = 1
    cautious_rtgs = calculate_rtg_and_rewards(cautious_actions, true_labels)
    aug_states.append(states)
    aug_actions.append(cautious_actions)
    aug_rtgs.append(cautious_rtgs)
    aug_timesteps.append(timesteps) 

    # 3. The Careless Agent
    careless_actions = true_labels.copy()
    apnea_indices = np.where(true_labels == 1)[0]
    fn_count = int(0.30 * len(apnea_indices))
    if fn_count > 0:
        fn_indices = np.random.choice(apnea_indices, fn_count, replace=False)
        careless_actions[fn_indices] = 0
    careless_rtgs = calculate_rtg_and_rewards(careless_actions, true_labels)
    aug_states.append(states)
    aug_actions.append(careless_actions)
    aug_rtgs.append(careless_rtgs)
    aug_timesteps.append(timesteps) 

    return np.vstack(aug_states), np.concatenate(aug_actions), np.concatenate(aug_rtgs), np.concatenate(aug_timesteps)


if __name__ == "__main__":
    # Define the 30 training patients: a01-a20 (Apnea) and c01-c10 (Control)
    apnea_patients = [f'a{i:02d}' for i in range(1, 21)]
    control_patients = [f'c{i:02d}' for i in range(1, 11)]
    train_patients = apnea_patients + control_patients
    
    # Extract raw features and the new raw timesteps for all 30 patients
    raw_states, true_labels, raw_timesteps = download_and_extract_features(train_patients)
    
    # Augment dataset with synthetic agents (passing and unpacking the timesteps)
    aug_states, aug_actions, aug_rtgs, aug_timesteps = generate_synthetic_agents(
        raw_states, true_labels, raw_timesteps
    )
    
    # Save everything to disk including the clean timesteps
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
    print(f"Final Actions Shape: {aug_actions.shape}")
    print(f"Final Timesteps Shape: {aug_timesteps.shape}")