"""Offline Phase 13–14 monitoring and robustness demonstration."""

from __future__ import annotations

import numpy as np

from everyo import Linear
from everyo.monitoring import build_reference_profile, drift_report
from everyo.robustness import batch_consistency_check, repeatability_check, robustness_report


def main() -> None:
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(100, 2))
    shifted = rng.normal(loc=[3, 0], size=(100, 2))
    profile = build_reference_profile(reference)
    observed = drift_report(profile, shifted)
    model = Linear(2, 2, seed=7)
    x = rng.normal(size=(20, 2))
    labels = model(x).data.argmax(axis=1)
    repeat = repeatability_check(model, x)
    batch = batch_consistency_check(model, x)
    robustness = robustness_report(model, x, labels, sigma=0.05, dropout=0.05)
    print("Drift detected:", observed["drift_detected"])
    print("Repeatability:", repeat["passed"])
    print("Batch consistency:", batch["passed"])
    print("Toy robustness report:", robustness)
    assert observed["drift_detected"] and repeat["passed"] and batch["passed"]


if __name__ == "__main__":
    main()
