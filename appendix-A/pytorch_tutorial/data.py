"""사용자 정의 Dataset과 DataLoader 예제."""

import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """특성과 레이블 텐서를 감싸는 간단한 데이터셋."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.shape[0]


def create_toy_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """훈련 및 테스트용 장난감 데이터를 만든다."""
    x_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    x_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    return x_train, y_train, x_test, y_test


def create_data_loaders(
    batch_size: int = 2,
    seed: int = 123,
) -> tuple[DataLoader, DataLoader]:
    """훈련 및 테스트 DataLoader를 만든다."""
    x_train, y_train, x_test, y_test = create_toy_tensors()

    train_dataset = ToyDataset(x_train, y_train)
    test_dataset = ToyDataset(x_test, y_test)

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def run_data_loader_demo() -> tuple[DataLoader, DataLoader]:
    """훈련 데이터 로더의 배치를 출력한다."""
    train_loader, test_loader = create_data_loaders()

    for batch_index, (features, labels) in enumerate(train_loader, start=1):
        print(f"배치 {batch_index}:")
        print("features =", features)
        print("labels =", labels)

    return train_loader, test_loader


if __name__ == "__main__":
    run_data_loader_demo()
