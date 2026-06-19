import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import platform

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

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    Down-weights easy examples and focuses on hard-to-predict minority classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Alpha is applied separately in forward() after pt is computed
        self.ce_loss = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha as a separate per-sample multiplier
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    context_window = 60
    dataset = DecisionTransformerDataset(data_path='data/processed_train_dataset.npz', context_len=context_window)
    
    # FIX: num_workers > 0 uses 'spawn' (not 'fork') for multiprocessing on
    # Windows. Each worker re-imports this module and re-pickles the full
    # Dataset object (including the potentially large valid_indices list
    # built in __init__), which reliably causes hangs / silent worker
    # crashes / BrokenPipeError on Windows. Forcing num_workers=0 there
    # keeps data loading single-process and avoids this entirely.
    # On Linux/Mac (which use 'fork'), num_workers=4 is safe and faster.
    is_windows = platform.system() == "Windows"
    safe_num_workers = 0 if is_windows else 4
    if is_windows:
        print("Detected Windows — setting num_workers=0 to avoid multiprocessing "
              "spawn issues with DataLoader (hangs / BrokenPipeError).")

    dataloader = DataLoader(
        dataset, 
        batch_size=16,      
        shuffle=True, 
        num_workers=safe_num_workers,
        pin_memory=True       
    )
    
    state_dim = dataset.states.shape[1] 
    max_len = 2000
    
    model = DecisionTransformer(state_dim=state_dim, max_ep_len=max_len).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    total_steps = 10 * len(dataloader) # epochs * batches
    warmup_steps = int(0.1 * total_steps) # 10% of training used for warmup

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    labels = dataset.actions.numpy() if isinstance(dataset.actions, torch.Tensor) else dataset.actions
    count_healthy = np.sum(labels == 0)
    count_apnea = np.sum(labels == 1)
    total_samples = len(labels)
    
    weight_healthy = np.sqrt(total_samples / count_healthy)
    weight_apnea = np.sqrt(total_samples / count_apnea)
    
    class_weights = torch.tensor([weight_healthy, weight_apnea], dtype=torch.float).to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    print(f"Focal Loss Alpha Weights -> Healthy: {weight_healthy:.4f} | Apnea: {weight_apnea:.4f}")
    
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
            
            logits = logits.reshape(-1, 2)
            targets = actions.reshape(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            scheduler.step() # Step the learning rate scheduler every batch
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
    save_path = 'decision_transformer_weights.pth'
    torch.save(model.state_dict(), save_path)
    print("\n--- Training Complete ---")
    print(f"Model weights successfully saved to '{save_path}'")

if __name__ == "__main__":
    train()