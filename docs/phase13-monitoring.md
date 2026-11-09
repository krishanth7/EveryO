# Phase 13 — CPU monitoring and drift detection

Phase 13 supplies **aggregate, offline monitoring** for numeric features and
probabilistic outputs. It does not collect, retain, or transmit raw samples.

| Milestone | Implementation | Acceptance test |
| --- | --- | --- |
| 13.1 | `build_reference_profile` | Feature means, standard deviations, histograms |
| 13.2 | `population_stability_index` | Finite PSI with fixed reference bin edges |
| 13.3 | `mean_shift_scores` | Standardized feature shifts |
| 13.4 | `prediction_entropy` | Valid probability rows and mean entropy |
| 13.5 | `RollingAccuracy` | Bounded feedback window and aligned labels |
| 13.6 | `drift_report` | Configurable thresholds and alerted feature indices |
| 13.7 | `monitoring_summary` | JSON-compatible aggregate-only results |
| 13.8 | CPU integration | `tests/test_phase13_monitoring.py` and example |

Run `python -m pytest -q tests/test_phase13_monitoring.py` and
`python examples/cpu_monitoring_robustness.py`.

PSI depends on sample size and the chosen reference histogram. Clipping values
to reference edges makes the two populations comparable but loses some tail
information. Mean shifts and PSI are heuristic alarms, **not** proof of harmful
concept drift; output-quality monitoring needs trustworthy, representative
ground-truth feedback. A release gate should not auto-rollback a model from a
single noisy signal without independent validation.
