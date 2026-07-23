"""명령행에서 PyTorch 튜토리얼 예제를 실행한다."""

import argparse
from collections.abc import Callable

from .autograd_demo import run_autograd_demo, run_computation_graph_demo
from .data import run_data_loader_demo
from .device_training import run_device_training_demo
from .model import run_model_demo
from .persistence import run_persistence_demo
from .tensors import run_tensor_demo
from .training import run_cpu_training_demo


def print_section(title: str) -> None:
    """출력 구역을 구분한다."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_autograd_section() -> None:
    print_section("2. 계산 그래프")
    run_computation_graph_demo()

    print_section("3. Autograd")
    run_autograd_demo()


def run_all() -> None:
    """전체 튜토리얼을 원래 학습 순서대로 실행한다."""
    sections: list[tuple[str, Callable[[], object]]] = [
        ("1. 텐서 기초", run_tensor_demo),
        ("2~3. 계산 그래프와 Autograd", run_autograd_section),
        ("4. 다층 신경망", run_model_demo),
        ("5. Dataset과 DataLoader", run_data_loader_demo),
        ("6. CPU 학습 루프", run_cpu_training_demo),
        ("7. 모델 저장과 로드", run_persistence_demo),
        ("8. CPU/GPU 장치 학습", run_device_training_demo),
    ]

    for title, function in sections:
        print_section(title)
        function()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Appendix A PyTorch 튜토리얼 실행기",
    )
    parser.add_argument(
        "--section",
        choices=[
            "all",
            "tensors",
            "autograd",
            "model",
            "data",
            "training",
            "persistence",
            "device",
        ],
        default="all",
        help="실행할 튜토리얼 구역",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    section_functions: dict[str, Callable[[], object]] = {
        "all": run_all,
        "tensors": run_tensor_demo,
        "autograd": run_autograd_section,
        "model": run_model_demo,
        "data": run_data_loader_demo,
        "training": run_cpu_training_demo,
        "persistence": run_persistence_demo,
        "device": run_device_training_demo,
    }

    section_functions[args.section]()


if __name__ == "__main__":
    main()
