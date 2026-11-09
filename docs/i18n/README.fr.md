# EveryO — Calcul neuronal ouvert

**Langue :** français · [English](../../README.md) · [Toutes les langues](../../README.md#documentation-in-10-languages)

EveryO est un framework Python open source dédié à la recherche, à l'enseignement et aux expériences reproductibles sur les réseaux de neurones. Son cœur NumPy comprend les tenseurs, la différentiation automatique, les couches neuronales, les optimiseurs et l'entraînement. CUDA, TensorFlow et ONNX sont facultatifs.

## Démarrage rapide

Python 3.9–3.12 est requis. Depuis le dépôt :

```bash
python -m pip install -e .
everyo doctor
```

## Exemple exécutable

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**Résultat attendu :**

```text
[4.]
```

Il s'agit de la dérivée de `x²` en `x = 2`. Consultez les [exemples](../../examples/) pour l'entraînement et les modèles.

## Portée technique

- **Noyau :** NumPy, autograd, couches, optimisation et sérialisation.
- **Validation :** tests de régression, contrôles numériques et benchmarks CPU.
- **Intégrations :** CUDA, vérification TensorFlow et export ONNX facultatifs.
- **Limite :** aucune certification de production indépendante. Voir le [rapport de qualification](../production-qualification-2026-11-09.md).

La [documentation anglaise](../../README.md) constitue la référence technique officielle.
