"""Software Bill of Materials (SBOM) generation for EveryO.

Supports CycloneDX (v1.5 JSON) and SPDX (v2.3 JSON) standard specifications.
Performs dynamic environment discovery capturing direct, optional, development,
and transitive dependencies, complete package URLs (purl), licenses, and dependency graphs.
Supports deterministic output mode for reproducible builds.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from everyo.version import __version__

# Canonical direct dependency declarations for EveryO
CORE_DEPENDENCIES: list[dict[str, str]] = [
    {
        "name": "numpy",
        "requirement": ">=1.22",
        "license": "BSD-3-Clause",
        "description": "Fundamental package for scientific computing with Python",
        "scope": "required",
    },
    {
        "name": "matplotlib",
        "requirement": ">=3.5",
        "license": "PSF",
        "description": "Plotting library for visualization utilities",
        "scope": "required",
    },
]

OPTIONAL_DEPENDENCIES: list[dict[str, str]] = [
    {
        "name": "tensorflow",
        "requirement": ">=2.12",
        "license": "Apache-2.0",
        "description": "Optional accelerator backend and reference framework",
        "scope": "optional",
    },
    {
        "name": "PyYAML",
        "requirement": ">=6.0",
        "license": "MIT",
        "description": "YAML configuration file parsing support",
        "scope": "optional",
    },
    {
        "name": "onnx",
        "requirement": ">=1.15",
        "license": "Apache-2.0",
        "description": "Open Neural Network Exchange export specification",
        "scope": "optional",
    },
    {
        "name": "onnxruntime",
        "requirement": ">=1.17",
        "license": "MIT",
        "description": "Cross-platform inference engine for ONNX models",
        "scope": "optional",
    },
    {
        "name": "cryptography",
        "requirement": ">=42",
        "license": "Apache-2.0 OR BSD-3-Clause",
        "description": "Cryptographic recipes and primitives for artifact signing",
        "scope": "optional",
    },
    {
        "name": "pybind11",
        "requirement": ">=2.11",
        "license": "BSD-3-Clause",
        "description": "C++/Python bindings for native CUDA acceleration",
        "scope": "optional",
    },
    {
        "name": "pip-audit",
        "requirement": ">=2.7.0",
        "license": "Apache-2.0",
        "description": "Scans Python environments for packages with known vulnerabilities",
        "scope": "optional",
    },
    {
        "name": "pytest",
        "requirement": ">=7.0",
        "license": "MIT",
        "description": "Testing framework for test qualification",
        "scope": "optional",
    },
    {
        "name": "ruff",
        "requirement": "==0.17.5",
        "license": "MIT OR Apache-2.0",
        "description": "Fast Python linter and code formatter",
        "scope": "optional",
    },
]


def _canonical_name(name: str) -> str:
    """Normalize package name according to PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _get_installed_version(pkg_name: str) -> str | None:
    """Attempt to resolve installed package version via importlib.metadata."""
    try:
        import importlib.metadata

        return importlib.metadata.version(pkg_name)
    except Exception:
        return None


def _calculate_file_sha256(path: Path) -> str:
    """Calculate the hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_requirement_name(req_str: str) -> str:
    """Extract bare package name from requirement specification."""
    # Match leading package name before version specifiers, extras or environment markers
    m = re.match(r"^([a-zA-Z0-9_.\-]+)", req_str.strip())
    if m:
        return m.group(1)
    return req_str.split(";")[0].split("[")[0].strip()


def _discover_installed_environment(
    include_optional: bool = True,
    include_transitive: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Dynamically discover all direct, optional, and transitive dependencies in environment.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, list[str]]]
        A list of component metadata dicts and a mapping of canonical package names to
        their dependencies (child packages).
    """
    direct_core_names = {_canonical_name(d["name"]): d for d in CORE_DEPENDENCIES}
    direct_opt_names = {_canonical_name(d["name"]): d for d in OPTIONAL_DEPENDENCIES}

    installed_dists: dict[str, Any] = {}
    try:
        import importlib.metadata

        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                installed_dists[_canonical_name(name)] = dist
    except Exception:
        pass

    components_by_name: dict[str, dict[str, Any]] = {}
    dep_graph: dict[str, list[str]] = {}

    # 1. Add direct core dependencies
    for canon_name, decl in direct_core_names.items():
        installed_ver = _get_installed_version(decl["name"]) or decl["requirement"].lstrip("=>~")
        comp = {
            "name": decl["name"],
            "canonical_name": canon_name,
            "version": installed_ver,
            "description": decl["description"],
            "license": decl["license"],
            "scope": "required",
            "is_direct": True,
            "requirement": decl["requirement"],
        }
        components_by_name[canon_name] = comp

    # 2. Add direct optional dependencies if requested
    if include_optional:
        for canon_name, decl in direct_opt_names.items():
            installed_ver = _get_installed_version(decl["name"]) or decl["requirement"].lstrip(
                "=>~"
            )
            comp = {
                "name": decl["name"],
                "canonical_name": canon_name,
                "version": installed_ver,
                "description": decl["description"],
                "license": decl["license"],
                "scope": "optional",
                "is_direct": True,
                "requirement": decl["requirement"],
            }
            components_by_name[canon_name] = comp

    # 3. Add transitive dependencies if requested
    if include_transitive and installed_dists:
        for canon_name, dist in installed_dists.items():
            if canon_name == "everyo":
                continue
            if canon_name not in components_by_name:
                raw_name = dist.metadata.get("Name", canon_name)
                license_val = dist.metadata.get("License") or "NOASSERTION"
                # Strip multi-line license text if full file is in header
                if len(license_val) > 100 or "\n" in license_val:
                    license_val = "NOASSERTION"
                desc = dist.metadata.get("Summary") or f"Transitive dependency {raw_name}"

                components_by_name[canon_name] = {
                    "name": raw_name,
                    "canonical_name": canon_name,
                    "version": dist.version,
                    "description": desc,
                    "license": license_val,
                    "scope": "transitive",
                    "is_direct": False,
                    "requirement": "",
                }

            # Map dependencies
            child_deps = []
            if dist.requires:
                for req in dist.requires:
                    # Ignore optional extra markers like 'extra == "test"'
                    if "extra ==" in req:
                        continue
                    child_name = _canonical_name(_parse_requirement_name(req))
                    if child_name != canon_name and child_name != "everyo":
                        child_deps.append(child_name)
            dep_graph[canon_name] = sorted(set(child_deps))

    # Sort components deterministically by canonical name
    sorted_components = [components_by_name[k] for k in sorted(components_by_name)]
    return sorted_components, dep_graph


def generate_cyclonedx_sbom(
    *,
    include_optional: bool = True,
    include_transitive: bool = True,
    deterministic: bool = False,
    custom_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a CycloneDX v1.5 JSON Software Bill of Materials (SBOM)."""
    if deterministic:
        now = "2026-01-01T00:00:00Z"
        serial_urn = "urn:uuid:" + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"https://github.com/krishanth7/EveryO/cyclonedx/{__version__}"
            )
        )
    else:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        serial_urn = f"urn:uuid:{uuid.uuid4()}"

    components_data, dep_graph = _discover_installed_environment(
        include_optional=include_optional,
        include_transitive=include_transitive,
    )

    components: list[dict[str, Any]] = []
    direct_purls: list[str] = []

    for comp in components_data:
        name = comp["name"]
        version = comp["version"]
        purl = f"pkg:pypi/{comp['canonical_name']}@{version}"
        bom_ref = purl

        if comp["is_direct"]:
            direct_purls.append(bom_ref)

        c_dict: dict[str, Any] = {
            "type": "library",
            "bom-ref": bom_ref,
            "name": name,
            "version": version,
            "description": comp["description"],
            "scope": "required" if comp["scope"] == "required" else "optional",
            "licenses": [{"license": {"id": comp["license"]}}],
            "purl": purl,
            "properties": [
                {"name": "is_direct", "value": str(comp["is_direct"]).lower()},
            ],
        }
        if comp["requirement"]:
            c_dict["properties"].append({"name": "requirement", "value": comp["requirement"]})
        components.append(c_dict)

    # CycloneDX dependency relationships
    root_ref = f"everyo@{__version__}"
    dependencies_tree: list[dict[str, Any]] = [
        {
            "ref": root_ref,
            "dependsOn": sorted(direct_purls),
        }
    ]

    for comp in components_data:
        c_canon = comp["canonical_name"]
        c_purl = f"pkg:pypi/{c_canon}@{comp['version']}"
        child_purls = []
        for child_canon in dep_graph.get(c_canon, []):
            matching_comp = next(
                (c for c in components_data if c["canonical_name"] == child_canon), None
            )
            if matching_comp:
                child_purls.append(f"pkg:pypi/{child_canon}@{matching_comp['version']}")
        dependencies_tree.append(
            {
                "ref": c_purl,
                "dependsOn": sorted(child_purls),
            }
        )

    metadata: dict[str, Any] = {
        "timestamp": now,
        "tools": [
            {
                "vendor": "EveryO Project",
                "name": "everyo-sbom-generator",
                "version": __version__,
            }
        ],
        "component": {
            "type": "framework",
            "bom-ref": root_ref,
            "name": "everyo",
            "version": __version__,
            "description": (
                "Open-source neural computing and experimentation framework with "
                "NumPy, TensorFlow and optional CUDA acceleration."
            ),
            "licenses": [{"license": {"id": "MIT"}}],
            "purl": f"pkg:pypi/everyo@{__version__}",
            "externalReferences": [
                {
                    "type": "vcs",
                    "url": "https://github.com/krishanth7/EveryO",
                },
                {
                    "type": "issue-tracker",
                    "url": "https://github.com/krishanth7/EveryO/issues",
                },
            ],
        },
    }

    if custom_properties:
        metadata["properties"] = [
            {"name": str(k), "value": str(v)} for k, v in custom_properties.items()
        ]

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial_urn,
        "version": 1,
        "metadata": metadata,
        "components": components,
        "dependencies": dependencies_tree,
    }


def generate_spdx_sbom(
    *,
    include_optional: bool = True,
    include_transitive: bool = True,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Generate an SPDX v2.3 JSON Software Bill of Materials (SBOM)."""
    if deterministic:
        now = "2026-01-01T00:00:00Z"
        doc_namespace = f"https://github.com/krishanth7/EveryO/spdx/everyo-{__version__}"
    else:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        doc_namespace = (
            f"https://github.com/krishanth7/EveryO/spdx/everyo-{__version__}-{uuid.uuid4()}"
        )

    components_data, dep_graph = _discover_installed_environment(
        include_optional=include_optional,
        include_transitive=include_transitive,
    )

    root_spdx_id = "SPDXRef-Package-EveryO"
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": root_spdx_id,
            "name": "everyo",
            "versionInfo": __version__,
            "downloadLocation": "https://github.com/krishanth7/EveryO/releases",
            "filesAnalyzed": False,
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) 2024 EveryO Contributors",
            "summary": (
                "Open-source neural computing and experimentation framework with "
                "NumPy, TensorFlow and optional CUDA acceleration."
            ),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/everyo@{__version__}",
                }
            ],
        }
    ]

    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_spdx_id,
        }
    ]

    for comp in components_data:
        name = comp["name"]
        clean_name = comp["canonical_name"].replace("-", "_")
        pkg_id = f"SPDXRef-Package-{clean_name}"
        installed_ver = comp["version"]

        packages.append(
            {
                "SPDXID": pkg_id,
                "name": name,
                "versionInfo": installed_ver,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": comp["license"],
                "licenseDeclared": comp["license"],
                "copyrightText": "NOASSERTION",
                "summary": comp["description"],
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{comp['canonical_name']}@{installed_ver}",
                    }
                ],
            }
        )

        if comp["is_direct"]:
            rel_type = "DEPENDS_ON" if comp["scope"] == "required" else "OPTIONAL_DEPENDENCY_OF"
            if rel_type == "DEPENDS_ON":
                relationships.append(
                    {
                        "spdxElementId": root_spdx_id,
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": pkg_id,
                    }
                )
            else:
                relationships.append(
                    {
                        "spdxElementId": pkg_id,
                        "relationshipType": "OPTIONAL_DEPENDENCY_OF",
                        "relatedSpdxElement": root_spdx_id,
                    }
                )
        else:
            # Transitive dependency relationship
            relationships.append(
                {
                    "spdxElementId": root_spdx_id,
                    "relationshipType": "CONTAINS",
                    "relatedSpdxElement": pkg_id,
                }
            )

        # Transitive package-to-package relationships
        for child_canon in dep_graph.get(comp["canonical_name"], []):
            child_id = f"SPDXRef-Package-{child_canon.replace('-', '_')}"
            relationships.append(
                {
                    "spdxElementId": pkg_id,
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": child_id,
                }
            )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"EveryO-{__version__}",
        "documentNamespace": doc_namespace,
        "creationInfo": {
            "creators": [
                f"Tool: everyo-sbom-generator-{__version__}",
                "Organization: EveryO Project",
            ],
            "created": now,
        },
        "packages": packages,
        "relationships": relationships,
    }


def generate_sbom(
    format: str = "cyclonedx",
    *,
    output_path: str | Path | None = None,
    include_optional: bool = True,
    include_transitive: bool = True,
    deterministic: bool = False,
    custom_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an SBOM in the specified format ('cyclonedx' or 'spdx').

    Parameters
    ----------
    format : str
        Format to output: 'cyclonedx' or 'spdx'.
    output_path : str | Path | None
        Optional file path to write the JSON SBOM to.
    include_optional : bool
        Whether to include optional extras.
    include_transitive : bool
        Whether to include installed transitive dependencies (default: True).
    deterministic : bool
        Whether to generate deterministic timestamps and UUIDs for reproducible builds.
    custom_properties : Mapping[str, Any] | None
        Additional metadata properties.

    Returns
    -------
    dict[str, Any]
        The structured SBOM dictionary.
    """
    fmt = format.lower().strip()
    if fmt == "cyclonedx":
        data = generate_cyclonedx_sbom(
            include_optional=include_optional,
            include_transitive=include_transitive,
            deterministic=deterministic,
            custom_properties=custom_properties,
        )
    elif fmt == "spdx":
        data = generate_spdx_sbom(
            include_optional=include_optional,
            include_transitive=include_transitive,
            deterministic=deterministic,
        )
    else:
        raise ValueError(
            f"Unsupported SBOM format '{format}'. Supported formats: 'cyclonedx', 'spdx'."
        )

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    return data
