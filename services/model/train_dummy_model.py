"""
Script to create a dummy model for development/testing
This can be replaced with actual training code later
"""
import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple autoencoder for anomaly detection"""
    def __init__(self, input_dim: int, hidden_dims=None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [16, 8]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
            ])
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_dummy_model(input_dim=3, hidden_dims=None):
    """Create and save a dummy model with random weights"""
    
    if hidden_dims is None:
        hidden_dims = [16, 8]
    
    # Create model
    model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
    
    # Generate dummy training statistics
    feature_mean = np.random.randn(input_dim).astype(np.float32) * 0.1
    feature_std = np.ones(input_dim, dtype=np.float32) + np.random.rand(input_dim).astype(np.float32) * 0.5
    
    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    
    torch.save(checkpoint, "model.pt")
    print(f"Dummy model saved to model.pt")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Feature mean: {feature_mean}")
    print(f"  Feature std: {feature_std}")


if __name__ == "__main__":
    create_dummy_model(input_dim=3, hidden_dims=[16, 8])
