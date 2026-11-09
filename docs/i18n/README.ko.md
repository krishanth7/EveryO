# EveryO — 개방형 신경망 컴퓨팅

**언어:** 한국어 · [English](../../README.md) · [모든 언어](../../README.md#documentation-in-10-languages)

EveryO는 신경망 연구, 교육 및 재현 가능한 실험을 위한 오픈소스 Python 프레임워크입니다. NumPy 기반 코어는 텐서, 자동 미분, 신경망 계층, 최적화 및 학습 기능을 제공합니다. CUDA, TensorFlow, ONNX는 선택적 통합 기능입니다.

## 빠른 시작

Python 3.9–3.12가 필요합니다. 저장소 루트에서 실행하세요.

```bash
python -m pip install -e .
everyo doctor
```

## 실행 예제

```python
import everyo as eo
x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**예상 출력:**

```text
[4.]
```

이는 `x = 2`에서 `x²`의 도함수 값입니다. 모델과 학습 예제는 [examples](../../examples/)를 참고하세요.

## 기술 범위

- **코어:** NumPy, 자동 미분, 계층, 최적화 및 직렬화.
- **검증:** 회귀 테스트, 수치 검증, CPU 벤치마크.
- **통합:** 선택적 CUDA 가속, TensorFlow 비교, ONNX 내보내기.
- **제한:** 독립적인 프로덕션 인증은 아직 없습니다. [검증 보고서](../production-qualification-2026-11-09.md)를 확인하세요.

[영어 문서](../../README.md)가 공식 기술 기준입니다.
