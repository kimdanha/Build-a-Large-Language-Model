"""스칼라, 벡터, 행렬 텐서 예제."""

import torch


def run_tensor_demo() -> None:
    """기본 텐서 생성과 변형 연산을 실행한다."""
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    tensor3d = torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])

    print("0차원 텐서:", tensor0d)
    print("1차원 텐서:", tensor1d)
    print("2차원 텐서:")
    print(tensor2d)
    print("3차원 텐서:")
    print(tensor3d)

    print("tensor2d.shape:", tensor2d.shape)
    print("reshape(3, 2):")
    print(tensor2d.reshape(3, 2))
    print("view(3, 2):")
    print(tensor2d.view(3, 2))
    print("전치 행렬:")
    print(tensor2d.T)
    print("matmul 결과:")
    print(tensor2d.matmul(tensor2d.T))
    print("@ 연산 결과:")
    print(tensor2d @ tensor2d.T)


if __name__ == "__main__":
    run_tensor_demo()
