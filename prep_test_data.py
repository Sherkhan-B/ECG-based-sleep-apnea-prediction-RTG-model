import numpy as np
from data_prep import download_and_extract_features

if __name__ == "__main__":
    # Test on completely unseen patients (PhysioNet Apnea-ECG test set x01-x35)
    test_patients = [f'x{i:02d}' for i in range(1, 36)]
    
    print("Starting Test Dataset Extraction...")
    
    # Catch all 3 returned arrays including the true timesteps
    raw_states, true_labels, raw_timesteps = download_and_extract_features(test_patients)
    
    # Create dummy RTGs just so the dataset class doesn't crash during loading
    # (These will be ignored during the actual testing loop)
    dummy_rtgs = np.zeros_like(true_labels, dtype=float)
    
    # Save the raw reality to the test file
    save_path = 'data/processed_test_dataset.npz'
    
    # Include timesteps in the saved .npz file
    np.savez(
        save_path, 
        states=raw_states, 
        actions=true_labels, 
        rtgs=dummy_rtgs,
        timesteps=raw_timesteps
    )
    
    print(f"\nTest Data saved successfully to: {save_path}")
    print(f"Total pure evaluation minutes: {len(raw_states)}")