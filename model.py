import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=128, max_ep_len=5000, n_layers=3, n_heads=4):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Project raw dimensions to uniform embedding size 'hidden_dim'
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Embedding(action_dim, hidden_dim)
        
        # Native PyTorch Transformer Encoder configured for Causal Masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Predictor Head
        self.predict_action = nn.Linear(hidden_dim, action_dim)

    def _generate_causal_mask(self, sz, device):
        # Generates an upper-triangular matrix of -inf to prevent looking ahead
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len, _ = states.shape
        device = states.device
        
        # Project each raw stream into hidden_dim vectors
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # Add temporal positioning
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Interleave sequences -> [R, S, A, R, S, A...]
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_dim)
        
        # Apply causal mask and process
        causal_mask = self._generate_causal_mask(3 * seq_len, device)
        transformer_outputs = self.transformer(stacked_inputs, mask=causal_mask)
        
        # Extract state features (located at index 1 of every triplet)
        triplet_outputs = transformer_outputs.reshape(batch_size, seq_len, 3, self.hidden_dim)
        state_features = triplet_outputs[:, :, 1, :]
        
        # Predict actions from state representations
        action_logits = self.predict_action(state_features)
        
        return action_logits

if __name__ == "__main__":
    print("Testing Decision Transformer Forward Pass...")
    
    # Create dummy data matching our DataLoader shapes
    batch_size = 32
    seq_len = 20
    state_dim = 89 # 87 From NeuroKit + 2 From Leading Paper
    
    dummy_states = torch.randn(batch_size, seq_len, state_dim)
    dummy_actions = torch.randint(0, 2, (batch_size, seq_len))
    dummy_rtgs = torch.randn(batch_size, seq_len, 1)
    dummy_timesteps = torch.randint(0, 100, (batch_size, seq_len))
    
    # Initialize model
    model = DecisionTransformer(state_dim=state_dim)
    
    # Pass dummy data through the model
    logits = model(dummy_states, dummy_actions, dummy_rtgs, dummy_timesteps)
    
    print("\n--- Forward Pass Successful ---")
    print(f"Output Logits Shape: {logits.shape}") 
    print("Expected Shape:      torch.Size([32, 20, 2])")