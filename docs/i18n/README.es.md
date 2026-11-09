# EveryO — Computación neuronal abierta

**Idioma:** español · [English](../../README.md) · [Todos los idiomas](../../README.md#documentation-in-10-languages)

EveryO es un framework de Python de código abierto para investigación, educación y experimentos reproducibles con redes neuronales. Su núcleo basado en NumPy incluye tensores, diferenciación automática, capas, optimizadores y entrenamiento. CUDA, TensorFlow y ONNX son integraciones opcionales.

## Inicio rápido

Requiere Python 3.9–3.12. Desde el repositorio:

```bash
python -m pip install -e .
everyo doctor
```

## Ejemplo funcional

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**Salida esperada:**

```text
[4.]
```

El resultado corresponde a la derivada de `x²` cuando `x = 2`. Consulta los [ejemplos](../../examples/) de entrenamiento y modelos.

## Alcance técnico

- **Núcleo:** NumPy, autograd, capas, optimización y serialización.
- **Verificación:** pruebas de regresión, verificaciones numéricas y benchmarks de CPU.
- **Integraciones:** CUDA, comprobaciones con TensorFlow y exportación ONNX opcionales.
- **Límite:** todavía no dispone de certificación independiente para producción. Consulta el [informe de cualificación](../production-qualification-2026-11-09.md).

La [documentación en inglés](../../README.md) es la referencia técnica principal.
