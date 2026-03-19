"""
LSTM Autoencoder Model for Anomaly Detection

A lightweight LSTM-based autoencoder for learning normal patterns
in time series data.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """LSTM-based encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bottleneck layer (compress to latent representation)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (latent_representation, hidden_state)
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Compress to latent space
        latent = self.fc(last_hidden)  # (batch, latent_dim)
        
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """LSTM-based decoder."""
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Expand from latent space
        self.fc = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            latent: Latent representation (batch, latent_dim)
            seq_len: Length of sequence to reconstruct
            
        Returns:
            Reconstructed sequence (batch, seq_len, output_dim)
        """
        batch_size = latent.size(0)
        
        # Expand latent to hidden dimension
        hidden_state = self.fc(latent)  # (batch, hidden_dim)
        
        # Repeat for sequence length
        decoder_input = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        
        # LSTM decoding
        lstm_out, _ = self.lstm(decoder_input)  # (batch, seq_len, hidden_dim)
        
        # Output projection
        output = self.output_layer(lstm_out)  # (batch, seq_len, output_dim)
        
        return output


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection.
    
    Learns to reconstruct normal time series patterns.
    Anomalies have higher reconstruction error.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout  # Store dropout value
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch, seq_len, input_dim)
        """
        seq_len = x.size(1)
        
        # Encode
        latent, _ = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent, seq_len)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        latent, _ = self.encoder(x)
        return latent
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE).
        
        Args:
            x: Input tensor
            reduction: 'mean' for average over all, 'sample' for per-sample
            
        Returns:
            Reconstruction error
        """
        reconstructed = self.forward(x)
        
        if reduction == 'mean':
            return nn.functional.mse_loss(reconstructed, x)
        elif reduction == 'sample':
            # MSE per sample (average over seq_len and features)
            return ((reconstructed - x) ** 2).mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }


def create_model(
    input_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    device: Optional[str] = None
) -> LSTMAutoencoder:
    """
    Factory function to create an LSTM Autoencoder.
    
    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        latent_dim: Latent/bottleneck dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        device: Device to use (cuda/cpu/None for auto)
        
    Returns:
        Initialized LSTMAutoencoder
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = model.to(device)
    
    print(f"Created LSTM Autoencoder:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def save_model(
    model: LSTMAutoencoder,
    filepath: str,
    feature_mean: Optional[list] = None,
    feature_std: Optional[list] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        filepath: Path to save checkpoint
        feature_mean: Normalization mean
        feature_std: Normalization std
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }
    
    if feature_mean is not None:
        checkpoint["feature_mean"] = feature_mean
    if feature_std is not None:
        checkpoint["feature_std"] = feature_std
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    filepath: str,
    device: Optional[str] = None
) -> Tuple[LSTMAutoencoder, dict]:
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to checkpoint
        device: Device to load to
        
    Returns:
        Tuple of (model, checkpoint)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"]
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_len = 50
    input_dim = 14
    
    # Create model
    model = create_model(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Forward
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Reconstruction error
    error = model.compute_reconstruction_error(x, reduction='mean')
    print(f"Mean reconstruction error: {error.item():.6f}")
    
    sample_errors = model.compute_reconstruction_error(x, reduction='sample')
    print(f"Per-sample errors shape: {sample_errors.shape}")
    print(f"Per-sample errors: {sample_errors[:5].tolist()}")
