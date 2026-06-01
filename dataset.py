import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class DecisionTransformerDataset(Dataset):
    def __init__(self, data_path='data/processed_train_dataset.npz', context_len=20, stats_path='data/train_stats.npz'):
        self.context_len = context_len
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {data_path}. Did you run data_prep.py first?")
            
        np_data = np.load(data_path)
        raw_states = np_data['states']
        raw_rtgs = np_data['rtgs']
        
        # Replaces NaNs with 0.0, and any infinity values with 0.0
        raw_states = np.nan_to_num(raw_states, nan=0.0, posinf=0.0, neginf=0.0)
        raw_rtgs = np.nan_to_num(raw_rtgs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Consistent Z-Score Normalization across Train/Test
        if 'train' in data_path:
            # If training, calculate metrics and lock them to disk
            state_mean = np.mean(raw_states, axis=0)
            state_std = np.std(raw_states, axis=0) + 1e-8
            np.savez(stats_path, mean=state_mean, std=state_std)
            print("Calculated and saved training normalization constants.")
        else:
            # If testing, force-load the training constants to prevent data shift
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                state_mean = stats['mean']
                state_std = stats['std']
                print("Successfully loaded training normalization constants for test evaluation.")
            else:
                print("WARNING: train_stats.npz not found! Defaulting to local stats (Not recommended for evaluation).")
                state_mean = np.mean(raw_states, axis=0)
                state_std = np.std(raw_states, axis=0) + 1e-8

        norm_states = (raw_states - state_mean) / state_std
        
        # Normalize RTGs (Scale to roughly [-1.0, 0.0])
        rtg_max = np.max(np.abs(raw_rtgs)) + 1e-8
        norm_rtgs = raw_rtgs / rtg_max 
        
        # Convert arrays to PyTorch Tensors
        self.states = torch.tensor(norm_states, dtype=torch.float32)
        self.actions = torch.tensor(np_data['actions'], dtype=torch.long)
        self.rtgs = torch.tensor(norm_rtgs, dtype=torch.float32).unsqueeze(-1) 
        
        # Load true resetting episodic timesteps from data_prep
        if 'timesteps' in np_data:
            self.timesteps = torch.tensor(np_data['timesteps'], dtype=torch.long)
        else:
            print("WARNING: 'timesteps' key missing from npz. Creating standard range.")
            self.timesteps = torch.arange(len(self.states), dtype=torch.long)
            
        self.dataset_length = len(self.states) - context_len

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        s = self.states[idx : idx + self.context_len]
        a = self.actions[idx : idx + self.context_len]
        r = self.rtgs[idx : idx + self.context_len]
        t = self.timesteps[idx : idx + self.context_len]
        
        return s, a, r, t