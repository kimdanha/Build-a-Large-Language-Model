"""모델의 state_dict 저장과 로드 예제."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .data import create_data_loaders
from .model import NeuralNetwork
from .training import compute_accuracy, train_model


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    """모델 파라미터를 파일에 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    *,
    num_inputs: int,
    num_outputs: int,
    device: torch.device | str = "cpu",
) -> NeuralNetwork:
    """저장된 파라미터로 모델을 복원한다."""
    device = torch.device(device)
    model = NeuralNetwork(num_inputs=num_inputs, num_outputs=num_outputs)

    state_dict = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.to(device)


def run_persistence_demo() -> NeuralNetwork:
    """모델을 임시 파일에 저장한 뒤 다시 불러온다."""
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

    with TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model.pth"
        save_model(model, model_path)

        loaded_model = load_model(
            model_path,
            num_inputs=2,
            num_outputs=2,
            device="cpu",
        )

        accuracy = compute_accuracy(
            loaded_model,
            test_loader,
            device="cpu",
        )

    print(f"복원한 모델의 테스트 정확도: {accuracy:.4f}")
    return loaded_model


if __name__ == "__main__":
    run_persistence_demo()
