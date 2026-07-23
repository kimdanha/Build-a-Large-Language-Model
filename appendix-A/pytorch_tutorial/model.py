"""다층 퍼셉트론 모델 정의."""

import torch


class NeuralNetwork(torch.nn.Module):
    """두 개의 은닉층을 가진 완전 연결 신경망."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 텐서로부터 로짓을 계산한다."""
        return self.layers(x)


def run_model_demo() -> NeuralNetwork:
    """입력 특성 50개, 출력 클래스 3개인 모델 구조를 출력한다."""
    model = NeuralNetwork(num_inputs=50, num_outputs=3)
    print(model)
    return model


if __name__ == "__main__":
    run_model_demo()
