import torch
import numpy as np
import random
import os
from dataset import DecisionTransformerDataset
from model import DecisionTransformer

def set_seed(seed=42):
    """Ensures deterministic, reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_rollout(model, dataset, initial_target_raw_rtg, rtg_min, rtg_max, device, context_len=20):
    model.eval()
    
    eval_length = len(dataset.states)
    tp, fp, fn, tn = 0, 0, 0, 0
    
    print(f"\nRunning full clinical rollout with Raw Initial Target = {initial_target_raw_rtg}...")
    
    states_hist = torch.zeros((1, context_len, model.state_dim), device=device)
    actions_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
    rtgs_hist = torch.zeros((1, context_len, 1), device=device)
    timesteps_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
    
    has_real_timesteps = hasattr(dataset, 'timesteps') or 'timesteps' in dataset.__dict__
    
    # Track the raw score for the current patient
    current_raw_rtg = initial_target_raw_rtg
    
    with torch.no_grad():
        for t in range(eval_length):
            if t > 0 and t % 5000 == 0:
                print(f"  ...processed {t}/{eval_length} minutes")
                
            if has_real_timesteps:
                safe_t = int(dataset.timesteps[t])
                is_new_patient = (safe_t == 0 and t > 0)
            else:
                safe_t = t % 450
                is_new_patient = (t % 450 == 0 and t > 0)
                
            if is_new_patient:
                # Reset histories AND reset the target score for the new patient
                states_hist = torch.zeros((1, context_len, model.state_dim), device=device)
                actions_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
                rtgs_hist = torch.zeros((1, context_len, 1), device=device)
                timesteps_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
                current_raw_rtg = initial_target_raw_rtg 
            
            safe_t = min(safe_t, model.embed_timestep.num_embeddings - 1)
            
            current_state = dataset.states[t].to(device)
            true_label = dataset.actions[t].item()
            
            # Convert raw tracked score to the [-1, 1] scale for the model
            normalized_rtg = 2 * ((current_raw_rtg - rtg_min) / (rtg_max - rtg_min)) - 1
            normalized_rtg = max(-1.0, min(1.0, normalized_rtg)) # Clamp to prevent embedding blowouts
            
            states_hist = torch.cat([states_hist[:, 1:, :], current_state.unsqueeze(0).unsqueeze(0)], dim=1)
            rtgs_hist = torch.cat([rtgs_hist[:, 1:, :], torch.tensor([[[normalized_rtg]]], device=device)], dim=1)
            timesteps_hist = torch.cat([timesteps_hist[:, 1:], torch.tensor([[safe_t]], device=device)], dim=1)
            
            logits = model(states_hist, actions_hist, rtgs_hist, timesteps_hist)
            next_action_logits = logits[0, -1, :] 
            predicted_action = torch.argmax(next_action_logits).item()
            
            actions_hist = torch.cat([actions_hist[:, 1:], torch.tensor([[predicted_action]], device=device)], dim=1)
            
            # --- CALCULATE IMMEDIATE REWARD & DECREMENT ---
            if predicted_action == 1 and true_label == 1:
                immediate_reward = 10.0  
                tp += 1
            elif predicted_action == 0 and true_label == 0:
                immediate_reward = 1.0   
                tn += 1
            elif predicted_action == 1 and true_label == 0:
                immediate_reward = -1.0  
                fp += 1 
            else:
                immediate_reward = -10.0 
                fn += 1 

            current_raw_rtg -= immediate_reward

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"Results -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Sensitivity (Catching Apnea): {sensitivity:.2%}")
    print(f"Specificity (Avoiding False Alarms): {specificity:.2%}")
    return sensitivity, specificity


if __name__ == "__main__":
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading test dataset and normalization bounds...")
    test_dataset = DecisionTransformerDataset(data_path='data/processed_test_dataset.npz')
    state_dim = test_dataset.states.shape[1]
    
    # Load the RTG bounds saved during data prep
    if os.path.exists('data/rtg_bounds.npy'):
        rtg_bounds = np.load('data/rtg_bounds.npy')
        rtg_min, rtg_max = float(rtg_bounds[0]), float(rtg_bounds[1])
    else:
        raise FileNotFoundError("rtg_bounds.npy missing! Please run data_prep.py first.")
    
    model = DecisionTransformer(state_dim=state_dim, max_ep_len=2000).to(device)
    model.load_state_dict(torch.load('decision_transformer_weights.pth', map_location=device, weights_only=True))
    print("Model weights loaded successfully.")
    
    # Test using RAW scores (e.g., target accumulating 3000 positive reward points)
    # The script will handle shrinking these down to the [-1, 1] scale for the model.
    test_raw_targets = [5000.0, 3000.0, 1000.0, 0.0, -1000.0, -3000.0]
    
    for target in test_raw_targets:
        run_rollout(model, test_dataset, initial_target_raw_rtg=target, rtg_min=rtg_min, rtg_max=rtg_max, device=device)