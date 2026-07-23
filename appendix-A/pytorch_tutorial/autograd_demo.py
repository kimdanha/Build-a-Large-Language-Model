"""계산 그래프와 autograd 예제."""

import torch
import torch.nn.functional as F
from torch.autograd import grad


def run_computation_graph_demo() -> torch.Tensor:
    """간단한 로지스틱 회귀의 정방향 계산을 실행한다."""
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2])
    b = torch.tensor([0.0])

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    print("순입력 z:", z)
    print("활성화 출력 a:", a)
    print("손실:", loss)
    return loss


def run_autograd_demo() -> tuple[torch.Tensor, torch.Tensor]:
    """가중치와 편향에 대한 손실 그레이디언트를 계산한다."""
    y = torch.tensor([1.0])
    x1 = torch.tensor(1.1)
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    grad_loss_w1 = grad(loss, w1, retain_graph=True)[0]
    grad_loss_b = grad(loss, b)[0]

    print("grad_L_w1 =", grad_loss_w1)
    print("grad_L_b =", grad_loss_b)
    return grad_loss_w1, grad_loss_b


if __name__ == "__main__":
    run_computation_graph_demo()
    run_autograd_demo()
