import time

import torch
import torch.nn as nn
import pennylane as qml


class QuantumLatentLayer(nn.Module):
    """A reusable hybrid quantum-classical latent layer for PyTorch."""

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        input_dim: int = 8,
        output_dim: int = 8,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device_name = device_name

        self.projection = nn.Linear(input_dim, num_qubits)

        dev = qml.device(device_name, wires=num_qubits, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.quantum_net = qml.qnn.TorchLayer(
            circuit,
            {"weights": (num_layers, num_qubits, 3)},
        )
        self.output_layer = nn.Linear(num_qubits, output_dim)

    def get_config(self):
        return {
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device_name,
        }

    def count_quantum_parameters(self) -> int:
        return sum(p.numel() for p in self.quantum_net.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"Expected input of shape (batch_size, input_dim), got {tuple(x.shape)}"
            )
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(1)}"
            )

        projected = self.projection(x)
        batch_size = projected.size(0)
        quantum_outputs = []

        for index in range(batch_size):
            sample = projected[index]
            quantum_outputs.append(self.quantum_net(sample))

        quantum_outputs = torch.stack(quantum_outputs, dim=0)
        return self.output_layer(quantum_outputs)


if __name__ == "__main__":
    print("Created QuantumLatentLayer")

    batch_size = 16
    input_dim = 8
    output_dim = 8
    layer = QuantumLatentLayer(num_qubits=4, num_layers=2, input_dim=input_dim, output_dim=output_dim)

    x = torch.randn(batch_size, input_dim)
    print("Input Shape:")
    print(tuple(x.shape))

    start = time.perf_counter()
    with torch.no_grad():
        y = layer(x)
    forward_time = (time.perf_counter() - start) * 1000.0

    print("Output Shape:")
    print(tuple(y.shape))

    print("Quantum Parameters:")
    print(layer.count_quantum_parameters())

    print(f"Forward successful in {forward_time:.2f} ms")

    x_req = x.clone().requires_grad_(True)
    y_req = layer(x_req)
    loss = y_req.sum()
    loss.backward()
    print("Gradient successful")
