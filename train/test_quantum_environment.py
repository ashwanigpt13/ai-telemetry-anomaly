import sys
import time

try:
    import pennylane as qml
    import torch
except ImportError as exc:
    print("Quantum Environment Test")
    print("PennyLane or Torch is not available.")
    print(f"Reason: {exc}")
    print("Install the required dependencies with: python -m pip install \"pennylane>=0.40.0\"")
    sys.exit(0)


def main() -> int:
    print("==================================================")
    print("Quantum Environment Test")
    print("==================================================")
    print(f"PennyLane Version : {qml.__version__}")
    print(f"PyTorch Version   : {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA available; using GPU for torch operations")
    else:
        print("CUDA unavailable; continuing on CPU")

    try:
        dev = qml.device("default.qubit", wires=4, shots=None)
        print("✓ Quantum device initialized")
    except Exception as exc:
        print(f"Failed to initialize quantum device: {exc}")
        return 1

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(4))
        qml.StronglyEntanglingLayers(weights, wires=range(4))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    try:
        qnn = qml.qnn.TorchLayer(circuit, {"weights": (1, 4, 3)})
        inputs = torch.randn(4, dtype=torch.float32, requires_grad=True)

        start = time.perf_counter()
        outputs = qnn(inputs)
        forward_time_ms = (time.perf_counter() - start) * 1000.0

        print(f"Input Shape  : {tuple(inputs.shape)}")
        print(f"Output Shape : {tuple(outputs.shape)}")
        print("✓ Forward pass successful")

        loss = outputs.sum()
        print(f"Loss : {loss.item():.6f}")

        start = time.perf_counter()
        loss.backward()
        backward_time_ms = (time.perf_counter() - start) * 1000.0

        grads = [param.grad for param in qnn.parameters() if param.requires_grad]
        grads_exist = all(grad is not None for grad in grads)
        grads_finite = all(torch.isfinite(grad).all().item() for grad in grads if grad is not None)

        if grads_exist and grads_finite:
            print("✓ Gradient computation successful")
        else:
            print("Gradient computation did not produce valid values")
            return 1

        print(f"Forward Time : {forward_time_ms:.2f} ms")
        print(f"Backward Time: {backward_time_ms:.2f} ms")
        print("==================================================")
        print("Quantum Environment Ready")
        print("==================================================")
        return 0
    except Exception as exc:
        print(f"Quantum environment check failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
