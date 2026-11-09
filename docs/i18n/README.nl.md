# EveryO — Open neurale berekeningen

**Taal:** Nederlands · [English](../../README.md) · [Alle talen](../../README.md#documentation-in-10-languages)

EveryO is een open-source Python-framework voor onderzoek, onderwijs en reproduceerbare experimenten met neurale netwerken. De NumPy-kern bevat tensors, automatische differentiatie, neurale lagen, optimalisatie en training. CUDA, TensorFlow en ONNX zijn optionele uitbreidingen.

## Snel starten

Vereist Python 3.9–3.12. Installeer vanuit de repository:

```bash
python -m pip install -e .
everyo doctor
```

## Werkend voorbeeld

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**Verwachte uitvoer:**

```text
[4.]
```

Dit is de afgeleide van `x²` bij `x = 2`. Zie [voorbeelden](../../examples/) voor modellen en training.

## Technische reikwijdte

- **Kern:** NumPy, autograd, lagen, optimizers en serialisatie.
- **Validatie:** regressietests, numerieke controles en CPU-benchmarks.
- **Integraties:** optionele CUDA-versnelling, TensorFlow-controles en ONNX-export.
- **Beperking:** geen onafhankelijk gecertificeerde productieomgeving. Raadpleeg het [kwalificatierapport](../production-qualification-2026-11-09.md).

De [Engelse documentatie](../../README.md) is de gezaghebbende technische referentie.
