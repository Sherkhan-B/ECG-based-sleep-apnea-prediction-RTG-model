import numpy as np
import joblib
import os
from data_prep import download_and_extract_features

if __name__ == "__main__":
    test_patients = [f'x{i:02d}' for i in range(1, 36)]
    
    print("Starting Test Dataset Extraction...")
    
    # Pass is_train=False to ensure it uses a consistent center crop
    raw_states, true_labels, raw_timesteps = download_and_extract_features(test_patients, is_train=False)
    
    # Load and apply the RobustScaler fitted on the training set
    scaler_path = 'data/robust_scaler.pkl'
    if os.path.exists(scaler_path):
        print("Loading RobustScaler and transforming test features...")
        scaler = joblib.load(scaler_path)
        scaled_states = scaler.transform(raw_states)
    else:
        raise FileNotFoundError("robust_scaler.pkl not found! Please run train_prep.py first.")

    dummy_rtgs = np.zeros_like(true_labels, dtype=float)
    
    save_path = 'data/processed_test_dataset.npz'
    
    np.savez(
        save_path, 
        states=scaled_states,  # Save the SCALED states
        actions=true_labels, 
        rtgs=dummy_rtgs,
        timesteps=raw_timesteps
    )
    
    print(f"\nTest Data saved successfully to: {save_path}")
    print(f"Total pure evaluation minutes: {len(scaled_states)}")