import numpy as np
import os

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

print("Reading training data to extract Return-to-Go (RTG) boundaries...")
train_data = np.load('data/processed_train_dataset.npz')
raw_rtgs = train_data['rtgs']

# Calculate the absolute minimum and maximum raw scores achieved during training
rtg_min = float(np.min(raw_rtgs))
rtg_max = float(np.max(raw_rtgs))

# Save them to disk where test.py expects them
np.save('data/rtg_bounds.npy', np.array([rtg_min, rtg_max]))

print("\n--- Success ---")
print(f"Saved RTG Bounds -> Min: {rtg_min} | Max: {rtg_max}")
print("Saved to 'data/rtg_bounds.npy'. You can now run test.py safely!")