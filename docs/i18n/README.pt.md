# EveryO — Computação neural aberta

**Idioma:** português · [English](../../README.md) · [Todos os idiomas](../../README.md#documentation-in-10-languages)

EveryO é um framework Python de código aberto para investigação, ensino e experiências reproduzíveis com redes neurais. O núcleo NumPy implementa tensores, diferenciação automática, camadas neurais, otimizadores e treino. CUDA, TensorFlow e ONNX são integrações opcionais.

## Início rápido

Requer Python 3.9–3.12. Na raiz do repositório:

```bash
python -m pip install -e .
everyo doctor
```

## Exemplo executável

```python
import everyo as eo
x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**Resultado esperado:**

```text
[4.]
```

O resultado é a derivada de `x²` em `x = 2`. Consulte os [exemplos](../../examples/) de treino e modelos.

## Âmbito técnico

- **Núcleo:** NumPy, autograd, camadas, otimização e serialização.
- **Validação:** testes de regressão, verificações numéricas e benchmarks de CPU.
- **Integrações:** CUDA, verificações com TensorFlow e exportação ONNX opcionais.
- **Limitação:** sem certificação independente para produção. Ver o [relatório de qualificação](../production-qualification-2026-11-09.md).

A [documentação inglesa](../../README.md) é a referência técnica oficial.
