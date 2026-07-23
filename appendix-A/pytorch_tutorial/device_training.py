"""사용 가능한 CPU 또는 GPU 장치에서 학습하는 예제."""

import torch

from .data import create_data_loaders
from .model import NeuralNetwork
from .training import compute_accuracy, train_model


def select_device() -> torch.device:
    """CUDA 사용 가능 여부에 따라 학습 장치를 선택한다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_device_training_demo() -> NeuralNetwork:
    """사용 가능한 장치에서 모델을 학습한다."""
    torch.manual_seed(123)

    device = select_device()
    print(f"사용 장치: {device}")

    train_loader, test_loader = create_data_loaders()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    model = train_model(
        model,
        train_loader,
        num_epochs=3,
        learning_rate=0.5,
        device=device,
    )

    test_accuracy = compute_accuracy(
        model,
        test_loader,
        device=device,
    )
    print(f"테스트 정확도: {test_accuracy:.4f}")
    return model


if __name__ == "__main__":
    run_device_training_demo()
