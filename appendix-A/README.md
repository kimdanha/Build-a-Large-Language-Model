# Appendix A: PyTorch Tutorial

기존 `pytorch_tutorial.py` 한 파일을 학습 주제별 Python 모듈로 분리한 구조다.

## 폴더 구조

```text
appendix-A/
├── README.md
├── requirements.txt
├── run_tutorial.py
└── pytorch_tutorial/
    ├── __init__.py
    ├── __main__.py
    ├── tensors.py
    ├── autograd_demo.py
    ├── model.py
    ├── data.py
    ├── training.py
    ├── persistence.py
    └── device_training.py
```

## 설치

저장소 루트에서 다음 명령어를 실행한다.

```bash
python -m pip install -r appendix-A/requirements.txt
```

## 전체 실행

`appendix-A` 폴더로 이동한 뒤 실행한다.

```bash
cd appendix-A
python -m pytorch_tutorial
```

다음 명령어도 같은 방식으로 동작한다.

```bash
python run_tutorial.py
```

## 특정 구역만 실행

```bash
python -m pytorch_tutorial --section tensors
python -m pytorch_tutorial --section autograd
python -m pytorch_tutorial --section model
python -m pytorch_tutorial --section data
python -m pytorch_tutorial --section training
python -m pytorch_tutorial --section persistence
python -m pytorch_tutorial --section device
```

각 모듈을 직접 모듈 방식으로 실행할 수도 있다.

```bash
python -m pytorch_tutorial.tensors
python -m pytorch_tutorial.autograd_demo
python -m pytorch_tutorial.model
python -m pytorch_tutorial.data
python -m pytorch_tutorial.training
python -m pytorch_tutorial.persistence
python -m pytorch_tutorial.device_training
```
