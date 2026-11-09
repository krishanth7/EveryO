"""Acceptance tests for scoped CPU-first Phase 15–20 foundations."""

import numpy as np
import pytest

from everyo.agent_tools import SafeToolRegistry, ToolSpec, cpu_math_registry
from everyo.cpu_optimization import batched_cpu_predict, cpu_parity_report, plan_microbatches
from everyo.deployment_policy import DeploymentPolicy, validate_deployment_artifact
from everyo.distributed_foundations import aggregate_weighted_metrics, evaluate_shards, plan_shards
from everyo.governance import sha256_file
from everyo.multimodal_retrieval import CPUEmbeddingIndex, l2_normalize_embeddings
from everyo.release_qualification import (
    qualify_metric_gates,
    qualify_release,
    release_decision_json,
)


def test_phase15_microbatch_planning_and_prediction():
    values = np.arange(60).reshape(30, 2)
    assert len(plan_microbatches(30, batch_size=7)) == 5

    def fn(x):
        return x * 2

    np.testing.assert_array_equal(batched_cpu_predict(fn, values, batch_size=7), values * 2)
    assert cpu_parity_report(fn, values, batch_size=7)["passed"]
    with pytest.raises(ValueError):
        plan_microbatches(30, batch_size=100, max_batch=64)


def test_phase15_catches_batch_dependent_models():
    values = np.arange(12).reshape(6, 2)
    report = cpu_parity_report(lambda x: x - x.mean(axis=0), values, batch_size=2)
    assert not report["passed"]


def test_phase16_shard_partition_and_weighted_metrics():
    shards = plan_shards(11, workers=3)
    assert sorted(int(i) for shard in shards for i in shard) == list(range(11))
    result = aggregate_weighted_metrics(
        [{"samples": 3.0, "loss": 2.0}, {"samples": 1.0, "loss": 6.0}]
    )
    assert result == {"samples": 4.0, "loss": 3.0}
    with pytest.raises(ValueError):
        aggregate_weighted_metrics([])


def test_phase16_sharded_reference_evaluation():
    values = np.arange(24).reshape(12, 2)
    metrics = evaluate_shards(values, lambda rows: {"mean": float(rows.mean())}, workers=3)
    assert metrics["samples"] == 12 and metrics["mean"] == pytest.approx(values.mean())


def test_phase17_exact_cosine_search():
    embeddings = np.eye(3)
    index = CPUEmbeddingIndex(embeddings, ["image", "audio", "text"])
    assert index.search([[1, 0, 0]], k=2)[0][0][0] == "image"
    np.testing.assert_allclose(np.linalg.norm(l2_normalize_embeddings(embeddings), axis=1), 1)
    with pytest.raises(ValueError):
        CPUEmbeddingIndex(embeddings, ["dup", "dup", "unique"])
    with pytest.raises(ValueError):
        index.search([[0, 0, 0]])


def test_phase18_registry_rejects_unknown_tools_and_budget():
    registry = cpu_math_registry()
    assert registry.call("add", {"a": 1, "b": 2}) == 3
    with pytest.raises(ValueError):
        registry.call("unknown", {})
    with pytest.raises(ValueError):
        registry.call("add", {"a": 1, "unexpected": 2})
    small = SafeToolRegistry(max_calls=1)
    small.register(ToolSpec("identity", "Identity", ("x",)), lambda x: x)
    assert small.call("identity", {"x": 1}) == 1
    with pytest.raises(RuntimeError):
        small.call("identity", {"x": 1})


def test_phase19_fail_closed_deployment_policy(tmp_path):
    DeploymentPolicy().validate()
    with pytest.raises(ValueError, match="TLS"):
        DeploymentPolicy(host="0.0.0.0").validate()
    with pytest.raises(ValueError, match="production"):
        DeploymentPolicy().validate(environment="production")
    DeploymentPolicy(tls_terminated=True, authentication_enabled=True, audit_enabled=True).validate(
        environment="production"
    )
    artifact = tmp_path / "model.evo"
    artifact.write_bytes(b"test-archive-digest")
    assert validate_deployment_artifact(artifact, expected_digest=sha256_file(artifact))
    with pytest.raises(ValueError):
        validate_deployment_artifact(artifact, expected_digest="0" * 64)


def test_phase20_evidence_is_required_and_serializable():
    missing = qualify_release("1.0.0", {"unit_tests": True})
    assert not missing.approved and "security_review" in missing.missing
    passed = qualify_release("1.0.0", dict.fromkeys(missing.checks, True))
    assert passed.approved and '"approved": true' in release_decision_json(passed)
    assert qualify_metric_gates({"accuracy": 0.9}, {"accuracy": (">=", 0.8)})["passed"]
    with pytest.raises(ValueError):
        qualify_release("1.0.0", {"unit_tests": 1})
