import torch
from torch.utils.data import Dataset
import numpy as np
import os

class DecisionTransformerDataset(Dataset):
    def __init__(self, data_path='data/processed_train_dataset.npz', context_len=20):
        self.context_len = context_len
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {data_path}. Did you run data_prep.py first?")
            
        np_data = np.load(data_path)
        
        # Load pre-scaled data directly from your prep pipeline
        raw_states = np_data['states']
        raw_rtgs = np_data['rtgs']
        
        # Replaces NaNs with 0.0, and any infinity values with 0.0 as a final safeguard
        clean_states = np.nan_to_num(raw_states, nan=0.0, posinf=0.0, neginf=0.0)
        clean_rtgs = np.nan_to_num(raw_rtgs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert arrays to PyTorch Tensors
        self.states = torch.tensor(clean_states, dtype=torch.float32)
        self.actions = torch.tensor(np_data['actions'], dtype=torch.long)
        self.rtgs = torch.tensor(clean_rtgs, dtype=torch.float32).unsqueeze(-1) 
        
        # Load true resetting episodic timesteps from data_prep
        if 'timesteps' in np_data:
            self.timesteps = torch.tensor(np_data['timesteps'], dtype=torch.long)
        else:
            print("WARNING: 'timesteps' key missing from npz. Creating standard range.")
            self.timesteps = torch.arange(len(self.states), dtype=torch.long)
            
        # --- NEW: Build Valid Indices to Prevent Patient Boundary Crossing ---
        print("Calculating safe trajectory windows (preventing patient crossover)...")
        valid_indices = []
        total_possible_steps = len(self.states) - context_len
        
        # We only allow a starting index if the timesteps strictly increase for the whole context window
        for i in range(total_possible_steps):
            window_timesteps = self.timesteps[i : i + context_len]
            # Check if window_timesteps is strictly sequential (no sudden resets to 0)
            if torch.all(window_timesteps[1:] == window_timesteps[:-1] + 1):
                valid_indices.append(i)
                
        self.valid_indices = valid_indices
        print(f"Dataset ready: Built {len(self.valid_indices)} safe sliding windows.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Fetch the actual safe starting index
        safe_idx = self.valid_indices[idx]
        
        # Slice exactly context_len steps knowing it stays within one patient
        s = self.states[safe_idx : safe_idx + self.context_len]
        a = self.actions[safe_idx : safe_idx + self.context_len]
        r = self.rtgs[safe_idx : safe_idx + self.context_len]
        t = self.timesteps[safe_idx : safe_idx + self.context_len]
        
        return s, a, r, t