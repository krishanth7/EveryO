"""Tests for automated dependency vulnerability auditing and severity policy enforcement."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from everyo.security.vulnerability import (
    VulnerabilityException,
    parse_pip_audit_json,
    run_pip_audit,
)


def test_parse_pip_audit_clean():
    sample_json = {
        "dependencies": [
            {"name": "numpy", "version": "2.5.5", "vulns": []},
            {"name": "matplotlib", "version": "3.11.2", "vulns": []},
        ],
        "fixes": [],
    }
    report = parse_pip_audit_json(sample_json, severity_threshold="high")
    assert report.passed is True
    assert report.total_scanned == 2
    assert len(report.findings) == 0
    assert len(report.violations) == 0
    assert report.critical_count == 0
    assert report.high_count == 0


def test_parse_pip_audit_high_severity_violation():
    sample_json = {
        "dependencies": [
            {
                "name": "vulnerable-lib",
                "version": "1.0.0",
                "vulns": [
                    {
                        "id": "GHSA-xxxx-yyyy-zzzz",
                        "severity": "high",
                        "description": "Remote code execution",
                        "fix_versions": ["1.0.1"],
                    }
                ],
            }
        ],
        "fixes": [],
    }
    report = parse_pip_audit_json(sample_json, severity_threshold="high")
    assert report.passed is False
    assert len(report.findings) == 1
    assert len(report.violations) == 1
    assert report.high_count == 1
    assert report.violations[0].package == "vulnerable-lib"
    assert report.violations[0].fix_versions == ["1.0.1"]


def test_severity_threshold_filtering():
    sample_json = {
        "dependencies": [
            {
                "name": "medium-lib",
                "version": "2.0.0",
                "vulns": [
                    {
                        "id": "CVE-2026-99999",
                        "severity": "medium",
                        "description": "Information disclosure",
                        "fix_versions": ["2.0.1"],
                    }
                ],
            }
        ],
        "fixes": [],
    }
    # When threshold is 'high', a medium vulnerability should not fail policy
    report_high = parse_pip_audit_json(sample_json, severity_threshold="high")
    assert report_high.passed is True
    assert len(report_high.findings) == 1
    assert len(report_high.violations) == 0
    assert report_high.medium_count == 1

    # When threshold is 'medium', it must trigger a violation
    report_med = parse_pip_audit_json(sample_json, severity_threshold="medium")
    assert report_med.passed is False
    assert len(report_med.violations) == 1


def test_exception_suppression_active():
    sample_json = {
        "dependencies": [
            {
                "name": "flagged-pkg",
                "version": "1.2.3",
                "vulns": [
                    {
                        "id": "CVE-2026-12345",
                        "severity": "high",
                        "description": "ReDoS in parsing",
                        "fix_versions": [],
                    }
                ],
            }
        ],
        "fixes": [],
    }
    future_date = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
    exc = VulnerabilityException(
        vuln_id="CVE-2026-12345",
        package="flagged-pkg",
        justification="ReDoS vector not reachable in our pipeline",
        expires=future_date,
        reviewer="Security Audit Team",
    )
    report = parse_pip_audit_json(sample_json, severity_threshold="high", exceptions=[exc])
    assert report.passed is True
    assert len(report.violations) == 0
    assert len(report.suppressed) == 1
    assert report.suppressed[0]["exception"]["vuln_id"] == "CVE-2026-12345"


def test_expired_exception_rejected():
    sample_json = {
        "dependencies": [
            {
                "name": "flagged-pkg",
                "version": "1.2.3",
                "vulns": [
                    {
                        "id": "CVE-2026-12345",
                        "severity": "critical",
                        "description": "Critical vulnerability",
                        "fix_versions": [],
                    }
                ],
            }
        ],
        "fixes": [],
    }
    past_date = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    exc = VulnerabilityException(
        vuln_id="CVE-2026-12345",
        package="flagged-pkg",
        justification="Temporary deferral",
        expires=past_date,
        reviewer="SecOps",
    )
    report = parse_pip_audit_json(sample_json, severity_threshold="high", exceptions=[exc])
    assert report.passed is False
    assert len(report.violations) == 1
    assert any("expired" in err for err in report.errors)


def test_save_json_report(tmp_path: Path):
    sample_json = {
        "dependencies": [{"name": "numpy", "version": "2.5.5", "vulns": []}],
        "fixes": [],
    }
    report = parse_pip_audit_json(sample_json, severity_threshold="high")
    out_file = tmp_path / "report.json"
    report.save_json(out_file)

    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["passed"] is True
    assert data["total_scanned"] == 1
    assert "counts" in data
    assert data["counts"]["total_findings"] == 0


def test_run_pip_audit_mock_mode(tmp_path: Path):
    mock_data = {
        "dependencies": [
            {"name": "numpy", "version": "2.5.5", "vulns": []},
            {"name": "matplotlib", "version": "3.11.2", "vulns": []},
        ],
        "fixes": [],
    }
    out_file = tmp_path / "audit_output.json"
    report = run_pip_audit(
        severity_threshold="high",
        output_path=out_file,
        mock_json=mock_data,
    )
    assert report.passed is True
    assert out_file.exists()
