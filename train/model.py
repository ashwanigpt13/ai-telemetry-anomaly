"""
LSTM Autoencoder Model for Anomaly Detection

A lightweight LSTM-based autoencoder for learning normal patterns
in time series data.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, NamedTuple

try:
    from quantum_layer import QuantumLatentLayer
except ImportError:  # pragma: no cover - allows direct package/module execution
    from train.quantum_layer import QuantumLatentLayer


class EncoderOutput(NamedTuple):
    """Structured encoder output for the VAE model."""
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    attention_weights: torch.Tensor


class TemporalAttention(nn.Module):
    """Differentiable temporal attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: Tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Tuple of (attention_weights, context_vector)
            where attention_weights has shape (batch, seq_len)
            and context_vector has shape (batch, hidden_dim)
        """
        scores = self.attention(lstm_outputs).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (batch, hidden_dim)
        return weights, context


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
        
        # Temporal attention over the full sequence
        self.attention = TemporalAttention(hidden_dim)

        # Bottleneck layers for probabilistic latent distribution
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Structured encoder output with latent, mean, logvar, and attention weights.
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use a context vector built from all timesteps via attention
        attention_weights, context_vector = self.attention(lstm_out)
        
        # Produce Gaussian latent parameters
        latent_mean = self.fc_mean(context_vector)  # (batch, latent_dim)
        latent_logvar = self.fc_logvar(context_vector)  # (batch, latent_dim)
        latent = latent_mean
        
        return EncoderOutput(
            latent=latent,
            mean=latent_mean,
            logvar=latent_logvar,
            attention_weights=attention_weights,
        )

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for the provided input sequence."""
        lstm_out, _ = self.lstm(x)
        attention_weights, _ = self.attention(lstm_out)
        return attention_weights


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


def compute_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute the average KL divergence for a Gaussian latent distribution."""
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return kl_per_sample.mean()


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
        dropout: float = 0.1,
        use_quantum: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout  # Store dropout value
        self.use_quantum = use_quantum
        
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

        self.compression_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.quantum_layer = (
            QuantumLatentLayer(
                num_qubits=4,
                num_layers=2,
                input_dim=8,
                output_dim=8,
            )
            if use_quantum
            else None
        )

        self.expansion_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        
    def reparameterize(self, latent_mean: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent Gaussian distribution using the reparameterization trick."""
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        return latent_mean + eps * std

    def _apply_hybrid_bottleneck(self, latent: torch.Tensor) -> torch.Tensor:
        """Compress the sampled latent code, optionally refine it with the quantum layer, and expand it back to decoder size."""
        compressed = self.compression_net(latent)
        if self.use_quantum and self.quantum_layer is not None:
            refined = self.quantum_layer(compressed)
        else:
            refined = compressed
        return self.expansion_net(refined)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (reconstructed, latent_mean, latent_logvar, attention_weights)
        """
        seq_len = x.size(1)
        
        # Encode to latent distribution parameters
        encoder_output = self.encoder(x)
        latent_mean = encoder_output.mean
        latent_logvar = encoder_output.logvar
        latent = self.reparameterize(latent_mean, latent_logvar)
        
        hybrid_latent = self._apply_hybrid_bottleneck(latent)

        # Decode using the refined latent vector
        reconstructed = self.decoder(hybrid_latent, seq_len)
        
        return reconstructed, latent_mean, latent_logvar, encoder_output.attention_weights
    
    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Get a deterministic latent mean or a sampled latent vector after the hybrid bottleneck."""
        encoder_output = self.encoder(x)
        if sample:
            latent = self.reparameterize(encoder_output.mean, encoder_output.logvar)
        else:
            latent = encoder_output.mean
        return self._apply_hybrid_bottleneck(latent)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible wrapper for retrieving attention weights."""
        return self.encoder.get_attention_weights(x)

    def get_latent_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log variance for the provided input."""
        encoder_output = self.encoder(x)
        return encoder_output.mean, encoder_output.logvar
    
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
        reconstructed, _, _, _ = self.forward(x)
        
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
            "dropout": self.dropout,
            "use_quantum": self.use_quantum
        }

    def count_quantum_parameters(self) -> int:
        """Return the number of trainable parameters in the quantum layer, if enabled."""
        if self.use_quantum and self.quantum_layer is not None:
            return self.quantum_layer.count_quantum_parameters()
        return 0


def create_model(
    input_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    use_quantum: bool = True,
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
        dropout=dropout,
        use_quantum=use_quantum
    )
    
    model = model.to(device)
    
    print(f"Created LSTM Autoencoder:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Quantum enabled: {use_quantum}")
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
        "use_quantum": model.use_quantum,
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
    model_config = checkpoint.get("model_config", {})
    
    model = LSTMAutoencoder(
        input_dim=checkpoint.get("input_dim", model_config.get("input_dim", 16)),
        hidden_dim=checkpoint.get("hidden_dim", model_config.get("hidden_dim", 64)),
        latent_dim=checkpoint.get("latent_dim", model_config.get("latent_dim", 32)),
        num_layers=checkpoint.get("num_layers", model_config.get("num_layers", 1)),
        dropout=checkpoint.get("dropout", model_config.get("dropout", 0.1)),
        use_quantum=checkpoint.get("use_quantum", model_config.get("use_quantum", True))
    )
    
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with compatible state-dict updates: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    
    return model, checkpoint


if __name__ == "__main__":
    print("Hybrid Quantum VAE Created")

    batch_size = 8
    seq_len = 50
    input_dim = 16

    model = create_model(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        use_quantum=True
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    if torch.cuda.is_available():
        x = x.cuda()

    reconstructed, _, _, _ = model(x)

    print("Input Shape:")
    print(tuple(x.shape))

    print("Output Shape:")
    print(tuple(reconstructed.shape))

    print("Quantum Enabled:")
    print(model.use_quantum)

    print("Quantum Parameters")
    print(model.count_quantum_parameters())

    print("Forward Successful")

    loss = reconstructed.sum()
    loss.backward()
    print("Gradient Successful")
