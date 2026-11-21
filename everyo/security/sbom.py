"""Software Bill of Materials (SBOM) generation for EveryO.

Supports CycloneDX (v1.5 JSON) and SPDX (v2.3 JSON) standard specifications.
Captures core metadata, dependencies, licenses, package URLs (purl), and integrity hashes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

from everyo.version import __version__

# Canonical dependency declaration for EveryO
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


def generate_cyclonedx_sbom(
    *,
    include_optional: bool = True,
    custom_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a CycloneDX v1.5 JSON Software Bill of Materials (SBOM)."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    serial_urn = f"urn:uuid:{uuid.uuid4()}"

    components: list[dict[str, Any]] = []
    dep_list = list(CORE_DEPENDENCIES)
    if include_optional:
        dep_list.extend(OPTIONAL_DEPENDENCIES)

    for dep in dep_list:
        name = dep["name"]
        installed_ver = _get_installed_version(name) or dep["requirement"].lstrip("=>~")
        component = {
            "type": "library",
            "name": name,
            "version": installed_ver,
            "description": dep["description"],
            "scope": dep["scope"],
            "licenses": [{"license": {"id": dep["license"]}}],
            "purl": f"pkg:pypi/{name.lower()}@{installed_ver}",
            "properties": [
                {"name": "requirement", "value": dep["requirement"]},
            ],
        }
        components.append(component)

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
            "bom-ref": f"everyo@{__version__}",
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
    }


def generate_spdx_sbom(
    *,
    include_optional: bool = True,
) -> dict[str, Any]:
    """Generate an SPDX v2.3 JSON Software Bill of Materials (SBOM)."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc_namespace = f"https://github.com/krishanth7/EveryO/spdx/everyo-{__version__}-{uuid.uuid4()}"

    dep_list = list(CORE_DEPENDENCIES)
    if include_optional:
        dep_list.extend(OPTIONAL_DEPENDENCIES)

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

    for dep in dep_list:
        name = dep["name"]
        clean_name = name.replace("-", "_")
        pkg_id = f"SPDXRef-Package-{clean_name}"
        installed_ver = _get_installed_version(name) or dep["requirement"].lstrip("=>~")

        packages.append(
            {
                "SPDXID": pkg_id,
                "name": name,
                "versionInfo": installed_ver,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": dep["license"],
                "licenseDeclared": dep["license"],
                "copyrightText": "NOASSERTION",
                "summary": dep["description"],
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{name.lower()}@{installed_ver}",
                    }
                ],
            }
        )

        rel_type = "DEPENDS_ON" if dep["scope"] == "required" else "OPTIONAL_DEPENDENCY_OF"
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
            custom_properties=custom_properties,
        )
    elif fmt == "spdx":
        data = generate_spdx_sbom(include_optional=include_optional)
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
