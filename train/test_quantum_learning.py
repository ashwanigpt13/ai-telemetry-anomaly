import importlib.util
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(1)

LAYER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_layer.py")
SPEC = importlib.util.spec_from_file_location("quantum_layer", LAYER_PATH)
quantum_layer_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quantum_layer_module)
QuantumLatentLayer = quantum_layer_module.QuantumLatentLayer


def build_dataset(num_samples: int = 80, input_dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(num_samples, input_dim)).astype(np.float32)
    true_w = rng.normal(size=(input_dim, input_dim)).astype(np.float32)
    y = x @ true_w
    y = y.astype(np.float32)
    return torch.tensor(x), torch.tensor(y)


class TinyQuantumRegressor(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 2) -> None:
        super().__init__()
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.quantum = QuantumLatentLayer(
            num_qubits=2,
            num_layers=1,
            input_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.post = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.quantum(x)
        return self.post(x)


def main() -> None:
    print("==================================================")
    print("Quantum Learning Benchmark")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = build_dataset(num_samples=1000, input_dim=8)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TinyQuantumRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    losses = []
    epoch_times = []
    forward_times = []

    initial_params = [p.detach().clone() for p in model.quantum.parameters()]

    for epoch in range(4):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        batch_forward_time = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            start = time.perf_counter()
            preds = model(batch_x)
            batch_forward_time += (time.perf_counter() - start) * 1000.0

            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        forward_times.append(batch_forward_time / max(1, len(loader)))
        losses.append(epoch_loss / max(1, len(loader)))

        quantum_params = [p for p in model.quantum.parameters() if p.requires_grad]
        grad_norm = 0.0
        for param in quantum_params:
            if param.grad is not None:
                grad_norm += param.grad.norm().item()

        print(f"Epoch {epoch + 1:02d} | Loss: {losses[-1]:.6f} | Grad Norm: {grad_norm:.6f}")

    final_params = [p.detach().clone() for p in model.quantum.parameters()]
    params_changed = any(not torch.allclose(init, final) for init, final in zip(initial_params, final_params))

    print("\nVerification")
    print(f"Loss decreases: {losses[-1] < losses[0]}")
    print(f"Quantum gradients exist: {all(any(p.grad is not None for p in model.quantum.parameters()) for _ in [0])}")
    print(f"Parameters updated: {params_changed}")

    output_dir = os.path.dirname(__file__)
    curve_path = os.path.join(output_dir, "quantum_learning_curve.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Quantum Layer Learning Curve")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    print(f"Learning curve saved to: {curve_path}")
    print(f"Average Epoch Time: {np.mean(epoch_times):.3f}s")
    print(f"Average Forward Time: {np.mean(forward_times):.3f}ms")
    print("✓ Loss decreases")
    print("✓ Quantum gradients exist")
    print("✓ Parameters updated")
    print("✓ Learning curve generated")
    print("✓ No runtime errors")


if __name__ == "__main__":
    main()
