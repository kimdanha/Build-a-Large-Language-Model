"""일반적인 PyTorch 학습 루프와 정확도 계산."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import create_data_loaders
from .model import NeuralNetwork


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    num_epochs: int = 3,
    learning_rate: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """주어진 장치에서 모델을 학습한다."""
    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(num_epochs):
        model.train()

        for batch_index, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"에포크: {epoch + 1:03d}/{num_epochs:03d}"
                f" | 배치 {batch_index:03d}/{len(train_loader):03d}"
                f" | 훈련 손실: {loss.item():.2f}"
            )

    return model


def compute_accuracy(
    model: torch.nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str = "cpu",
) -> float:
    """데이터 로더에 대한 분류 정확도를 계산한다."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            correct += int(torch.sum(labels == predictions).item())
            total_examples += labels.shape[0]

    if total_examples == 0:
        raise ValueError("정확도를 계산할 데이터가 없습니다.")

    return correct / total_examples


def run_cpu_training_demo() -> NeuralNetwork:
    """CPU에서 장난감 모델을 학습하고 정확도를 출력한다."""
    torch.manual_seed(123)

    train_loader, test_loader = create_data_loaders()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model = train_model(
        model,
        train_loader,
        num_epochs=3,
        learning_rate=0.5,
        device="cpu",
    )

    train_accuracy = compute_accuracy(model, train_loader, device="cpu")
    test_accuracy = compute_accuracy(model, test_loader, device="cpu")

    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    return model


if __name__ == "__main__":
    run_cpu_training_demo()
