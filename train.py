import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from dataset import DecisionTransformerDataset
from model import DecisionTransformer

def set_seed(seed=42):
    """Locks down all sources of randomness for deterministic training loops."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # Guarantee reproducibility 
    set_seed(42)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Data
    print("Loading dataset...")
    context_window = 20
    dataset = DecisionTransformerDataset(data_path='data/processed_train_dataset.npz', context_len=context_window)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model
    state_dim = dataset.states.shape[1] 
    
    # Set max_ep_len to the actual single-patient trajectory limit
    max_len = 2000
    
    model = DecisionTransformer(state_dim=state_dim, max_ep_len=max_len).to(device)
    
    # Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Calculate and apply Dampened Class Weights 
    # Read raw action array directly from the dataset to compute distribution
    labels = dataset.actions.numpy() if isinstance(dataset.actions, torch.Tensor) else dataset.actions
    count_healthy = np.sum(labels == 0)
    count_apnea = np.sum(labels == 1)
    total_samples = len(labels)
    
    # Square root dampening prevents the pendulum from over-correcting too hard
    weight_healthy = np.sqrt(total_samples / count_healthy)
    weight_apnea = np.sqrt(total_samples / count_apnea)
    
    class_weights = torch.tensor([weight_healthy, weight_apnea], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Loss weights applied -> Healthy: {weight_healthy:.4f} | Apnea: {weight_apnea:.4f}")
    
    # Training Loop
    epochs = 10
    print("\n--- Starting Training ---")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (states, actions, rtgs, timesteps) in enumerate(dataloader):
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)
            
            optimizer.zero_grad()
            logits = model(states, actions, rtgs, timesteps)
            
            # Reshape tokens for CrossEntropy Loss evaluation
            logits = logits.reshape(-1, 2)
            targets = actions.reshape(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            # Clip gradients to prevent stability explosions in Transformer attention layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")
        
    # Save the trained model weights
    save_path = 'decision_transformer_weights.pth'
    torch.save(model.state_dict(), save_path)
    print("\n--- Training Complete ---")
    print(f"Model weights successfully saved to '{save_path}'")

if __name__ == "__main__":
    train()