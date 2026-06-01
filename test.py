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

def run_rollout(model, dataset, initial_target_rtg, device, context_len=20):
    model.eval()
    
    eval_length = len(dataset.states)
    tp, fp, fn, tn = 0, 0, 0, 0
    
    print(f"\nRunning full clinical rollout with Static Persona RTG = {initial_target_rtg}...")
    
    # Initialize the very first sequence context
    states_hist = torch.zeros((1, context_len, model.state_dim), device=device)
    actions_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
    rtgs_hist = torch.zeros((1, context_len, 1), device=device)
    timesteps_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
    
    # Check if timesteps are explicitly packed in the dataset object
    has_real_timesteps = hasattr(dataset, 'timesteps') or 'timesteps' in dataset.__dict__
    
    with torch.no_grad():
        for t in range(eval_length):
            if t > 0 and t % 5000 == 0:
                print(f"  ...processed {t}/{eval_length} minutes")
                
            # Detect exact clinical boundaries using real timestamps
            # If a patient's clock drops back to 0, reset the Transformer's memory context
            if has_real_timesteps:
                safe_t = int(dataset.timesteps[t])
                is_new_patient = (safe_t == 0 and t > 0)
            else:
                # Fallback estimation if array is missing
                safe_t = t % 450
                is_new_patient = (t % 450 == 0 and t > 0)
                
            if is_new_patient:
                states_hist = torch.zeros((1, context_len, model.state_dim), device=device)
                actions_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
                rtgs_hist = torch.zeros((1, context_len, 1), device=device)
                timesteps_hist = torch.zeros((1, context_len), dtype=torch.long, device=device)
            
            # Enforce embedding safety ceiling by looking directly up the layer's capacity
            safe_t = min(safe_t, model.embed_timestep.num_embeddings - 1)
            
            current_state = dataset.states[t].to(device)
            true_label = dataset.actions[t].item()
            
            # Append current step items to history windows
            states_hist = torch.cat([states_hist[:, 1:, :], current_state.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Instead of a wandering current_rtg, consistently append our targeted persona score.
            rtgs_hist = torch.cat([rtgs_hist[:, 1:, :], torch.tensor([[[initial_target_rtg]]], device=device)], dim=1)
            timesteps_hist = torch.cat([timesteps_hist[:, 1:], torch.tensor([[safe_t]], device=device)], dim=1)
            
            # Predict next action
            logits = model(states_hist, actions_hist, rtgs_hist, timesteps_hist)
            next_action_logits = logits[0, -1, :] 
            predicted_action = torch.argmax(next_action_logits).item()
            
            # Update action history with what we actually chose
            actions_hist = torch.cat([actions_hist[:, 1:], torch.tensor([[predicted_action]], device=device)], dim=1)
            
            # Calculate clinical performance metrics
            if predicted_action == true_label:
                if true_label == 1: tp += 1
                else: tn += 1
            elif predicted_action == 1 and true_label == 0:
                fp += 1 # False Positive
            elif predicted_action == 0 and true_label == 1:
                fn += 1 # False Negative

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"Results -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Sensitivity (Catching Apnea): {sensitivity:.2%}")
    print(f"Specificity (Avoiding False Alarms): {specificity:.2%}")
    return sensitivity, specificity


if __name__ == "__main__":
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading test dataset...")
    test_dataset = DecisionTransformerDataset(data_path='data/processed_test_dataset.npz')
    state_dim = test_dataset.states.shape[1]
    
    # Initialize and load model
    model = DecisionTransformer(state_dim=state_dim, max_ep_len=2000).to(device)
    model.load_state_dict(torch.load('decision_transformer_weights.pth', map_location=device, weights_only=True))
    print("Model weights loaded successfully.")
    
    # Evaluate under different clean, normalized persona variants (Strictly bounded between -1.0 and 0.0)
    for i in [100.00, 1.00, 0.15, 0.00, -0.05, -0.10, -0.20, -0.30, -0.40, -0.50, -1.00]:
        run_rollout(model, test_dataset, initial_target_rtg=i, device=device)