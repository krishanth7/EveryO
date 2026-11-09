# EveryO — Offenes neuronales Rechnen

**Sprache:** Deutsch · [English](../../README.md) · [Alle Sprachen](../../README.md#documentation-in-10-languages)

EveryO ist ein quelloffenes Python-Framework für Forschung, Lehre und reproduzierbare Experimente mit neuronalen Netzen. Der NumPy-Kern umfasst Tensoren, automatische Differenzierung, Netzwerkschichten, Optimierer und Training. CUDA, TensorFlow und ONNX sind optionale Erweiterungen.

## Schnellstart

Voraussetzung: Python 3.9–3.12. Im Repository ausführen:

```bash
python -m pip install -e .
everyo doctor
```

## Ausführbares Beispiel

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**Erwartete Ausgabe:**

```text
[4.]
```

Das Ergebnis ist die Ableitung von `x²` bei `x = 2`. Weitere [Beispiele](../../examples/) zeigen Training und Modelle.

## Technischer Umfang

- **Kern:** NumPy, Autograd, Schichten, Optimierung und Serialisierung.
- **Prüfung:** Regressionstests, numerische Verifikation und CPU-Benchmarks.
- **Integrationen:** optionales CUDA, TensorFlow-Vergleiche und ONNX-Export.
- **Grenze:** keine unabhängige Produktionszertifizierung. Siehe den [Qualifikationsbericht](../production-qualification-2026-11-09.md).

Die [englische Dokumentation](../../README.md) ist die maßgebliche technische Referenz.
