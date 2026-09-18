import json

import pytest

import everyo as eo
from everyo.profiler import profile, record_function


def test_profile_records_nested_modules_and_shapes():
    model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 2, seed=1))
    with profile() as result:
        model(eo.ones(3, 4))
    names = [event.name for event in result.events]
    assert names.count("Linear") == 2
    assert "Sequential" in names and "ReLU" in names
    linear = next(event for event in result.events if event.name == "Linear")
    assert linear.metadata["input_shapes"] == [[3, 4]]


def test_summary_aggregates_calls():
    layer = eo.ReLU()
    with profile() as result:
        layer(eo.ones(2))
        layer(eo.ones(2))
    row = next(row for row in result.summary() if row["name"] == "ReLU")
    assert row["calls"] == 2
    assert row["total_ms"] >= row["max_ms"] >= 0


def test_custom_region_and_exports(tmp_path):
    with profile() as result, record_function("data-loading", rows=10):
        pass
    raw = result.export_json(tmp_path / "profile.json")
    trace = result.export_chrome_trace(tmp_path / "trace.json")
    assert json.loads(raw.read_text())[0]["name"] == "data-loading"
    assert json.loads(trace.read_text())["traceEvents"][0]["ph"] == "X"


def test_profiler_cannot_be_nested():
    with profile(), pytest.raises(RuntimeError, match="cannot be nested"), profile():
        pass


def test_no_active_profiler_is_no_op():
    with record_function("nothing"):
        pass
