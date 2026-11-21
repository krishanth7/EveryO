"""Comprehensive unit tests for SBOM qualification, transitive discovery, and determinism."""

from __future__ import annotations

import json

from everyo.security.sbom import generate_cyclonedx_sbom, generate_spdx_sbom


def test_sbom_transitive_discovery_cyclonedx():
    sbom = generate_cyclonedx_sbom(include_optional=True, include_transitive=True)
    comp_names = {c["name"].lower() for c in sbom["components"]}

    # Direct dependencies
    assert "numpy" in comp_names
    assert "matplotlib" in comp_names

    # Check that transitive dependencies are captured in components
    direct_flags = {
        c["name"].lower(): any(
            p["name"] == "is_direct" and p["value"] == "true" for p in c.get("properties", [])
        )
        for c in sbom["components"]
    }
    assert direct_flags.get("numpy") is True
    assert direct_flags.get("matplotlib") is True

    # Check CycloneDX dependencies graph
    assert "dependencies" in sbom
    assert len(sbom["dependencies"]) > 0

    root_dep = next((d for d in sbom["dependencies"] if d["ref"].startswith("everyo@")), None)
    assert root_dep is not None
    assert any("numpy" in ref for ref in root_dep["dependsOn"])
    assert any("matplotlib" in ref for ref in root_dep["dependsOn"])


def test_sbom_transitive_discovery_spdx():
    sbom = generate_spdx_sbom(include_optional=True, include_transitive=True)
    packages = {p["name"].lower(): p for p in sbom["packages"]}

    # Direct packages present
    assert "everyo" in packages
    assert "numpy" in packages
    assert "matplotlib" in packages

    # SPDX relationships
    relationships = sbom["relationships"]
    rel_types = {r["relationshipType"] for r in relationships}
    assert "DESCRIBES" in rel_types
    assert "DEPENDS_ON" in rel_types
    assert "CONTAINS" in rel_types or "DEPENDS_ON" in rel_types


def test_sbom_deterministic_mode():
    cdx_1 = generate_cyclonedx_sbom(deterministic=True)
    cdx_2 = generate_cyclonedx_sbom(deterministic=True)
    assert json.dumps(cdx_1, sort_keys=True) == json.dumps(cdx_2, sort_keys=True)
    assert cdx_1["metadata"]["timestamp"] == "2026-01-01T00:00:00Z"
    assert cdx_1["serialNumber"] == cdx_2["serialNumber"]

    spdx_1 = generate_spdx_sbom(deterministic=True)
    spdx_2 = generate_spdx_sbom(deterministic=True)
    assert json.dumps(spdx_1, sort_keys=True) == json.dumps(spdx_2, sort_keys=True)
    assert spdx_1["creationInfo"]["created"] == "2026-01-01T00:00:00Z"


def test_sbom_direct_only_flag():
    cdx_all = generate_cyclonedx_sbom(include_transitive=True)
    cdx_direct = generate_cyclonedx_sbom(include_transitive=False)
    assert len(cdx_all["components"]) >= len(cdx_direct["components"])

    spdx_all = generate_spdx_sbom(include_transitive=True)
    spdx_direct = generate_spdx_sbom(include_transitive=False)
    assert len(spdx_all["packages"]) >= len(spdx_direct["packages"])
