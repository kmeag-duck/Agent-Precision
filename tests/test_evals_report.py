"""Unit tests for the Layer 2 aggregator (evals/layer2/report.py).

These tests build synthetic run.py-shaped payloads (matching
schema_version=2 as documented in evals/layer2/run.py:_serialize_results)
and assert that report.py:

  - validates schema_version and refuses unknown/missing versions,
  - aggregates outcome counts, per-category counts, and per-variable
    totals correctly,
  - detects regressions (best > last) and surfaces them in both the
    text report and the aggregated JSON,
  - renders a text report with sections that adapt to content
    (failure / regression sections collapse when empty),
  - returns exit codes 0 / 1 / 2 matching the contract documented in
    the report.py module docstring.

No real run.py invocation occurs; the tests construct dict payloads
directly to keep the surface area pinned to the JSON contract
between the two modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.layer2.report import (
    Aggregate,
    _exit_code,
    _SUPPORTED_SCHEMA_VERSIONS,
    aggregate,
    load_results,
    main,
    render_text,
    serialize_aggregate,
)


# ----------------------------------------------------------------------
# Payload builders
# ----------------------------------------------------------------------


def _make_run(
    path: str = "test-kernels/kokkos/lowerable/vector_add.cpp",
    category: str = "lowerable",
    outcome: str = "pass",
    pv_total: int = 3,
    pv_matched: int = 3,
    pv_best_matched: int | None = None,
    mismatches: list[dict] | None = None,
    analyst_absent: bool = False,
    timed_out: bool = False,
    returncode: int | None = 0,
    trace_exists: bool = True,
    cycle_count: int = 1,
    comparator_status: str | None = "ok",
    comparator_inapplicable: bool = False,
) -> dict:
    """Build a single run dict matching run.py's schema_version=2.

    `pv_best_matched` defaults to `pv_matched` (no regression). Pass a
    larger value to simulate a regression (best > last).
    """
    if pv_best_matched is None:
        pv_best_matched = pv_matched
    return {
        "path": path,
        "category": category,
        "tolerance_kind": "sig_figs",
        "tolerance_value": 6,
        "argv": ["python", "-m", "workflow.run", path, "--auto",
                 "--sig-figs", "6"],
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_sec": 12.3,
        "trace_path": f"/tmp/baselines/{Path(path).stem}/orchestrator_trace.jsonl",
        "trace_exists": trace_exists,
        "outcome": outcome,
        "reached_finish": outcome == "pass",
        "comparator_status": comparator_status,
        "cycle_count": cycle_count,
        "comparator_inapplicable": comparator_inapplicable,
        "per_variable": {
            "total": pv_total,
            "matched": pv_matched,
            "missing_count": 0,
            "analyst_absent": analyst_absent,
            "mismatches": mismatches or [],
        },
        "per_variable_best": {
            "total": pv_total,
            "matched": pv_best_matched,
            "missing_count": 0,
            "analyst_absent": analyst_absent,
            "mismatches": [],
        },
        "analyst_verdict": {},
        "stderr_tail": "",
    }


def _make_payload(runs: list[dict], schema_version: int = 2) -> dict:
    """Wrap a list of run dicts in the top-level schema run.py emits."""
    summary: dict[str, int] = {}
    for r in runs:
        summary[r["outcome"]] = summary.get(r["outcome"], 0) + 1
    return {
        "schema_version": schema_version,
        "total": len(runs),
        "summary": summary,
        "runs": runs,
    }


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    """Write a payload to a JSON file and return its path."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))
    return path


# ----------------------------------------------------------------------
# load_results
# ----------------------------------------------------------------------


def test_load_results_accepts_supported_schema(tmp_path: Path):
    """A well-formed v2 payload loads without complaint."""
    path = _write_payload(tmp_path, _make_payload([_make_run()]))
    payload = load_results(path)
    assert payload["schema_version"] == 2
    assert len(payload["runs"]) == 1


def test_load_results_missing_file(tmp_path: Path):
    """A missing input file exits with SystemExit and a helpful message."""
    with pytest.raises(SystemExit, match="not found"):
        load_results(tmp_path / "nope.json")


def test_load_results_malformed_json(tmp_path: Path):
    """Invalid JSON yields SystemExit naming the parse failure."""
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_results(path)


def test_load_results_non_object_toplevel(tmp_path: Path):
    """A JSON list at the top level is rejected; we require an object."""
    path = tmp_path / "list.json"
    path.write_text("[]")
    with pytest.raises(SystemExit, match="top-level must be a JSON object"):
        load_results(path)


def test_load_results_missing_schema_version(tmp_path: Path):
    """A payload without schema_version is rejected with a hint about run.py."""
    path = tmp_path / "noversion.json"
    path.write_text(json.dumps({"runs": [], "total": 0, "summary": {}}))
    with pytest.raises(SystemExit, match="no schema_version"):
        load_results(path)


def test_load_results_unknown_schema_version(tmp_path: Path):
    """An unsupported schema_version exits cleanly, listing what IS supported."""
    payload = _make_payload([], schema_version=99)
    path = _write_payload(tmp_path, payload)
    with pytest.raises(SystemExit, match="schema_version=99"):
        load_results(path)


def test_load_results_missing_required_keys(tmp_path: Path):
    """Missing 'runs' / 'total' / 'summary' is a config error."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"schema_version": 2, "total": 0,
                                "summary": {}}))  # no 'runs'
    with pytest.raises(SystemExit, match="missing required top-level key 'runs'"):
        load_results(path)


def test_load_results_runs_not_a_list(tmp_path: Path):
    """'runs' must be a list, not a dict or scalar."""
    path = tmp_path / "wrongtype.json"
    path.write_text(json.dumps({
        "schema_version": 2, "total": 0, "summary": {}, "runs": {}
    }))
    with pytest.raises(SystemExit, match="'runs' must be a list"):
        load_results(path)


def test_supported_schema_versions_contains_v2():
    """Sanity: v2 is the contract documented in run.py:_serialize_results."""
    assert 2 in _SUPPORTED_SCHEMA_VERSIONS


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------


def test_aggregate_empty_runs():
    """Zero runs produces all-zero counts; no division-by-zero."""
    agg = aggregate(_make_payload([]))
    assert agg.total == 0
    assert agg.outcome_counts == {}
    assert agg.category_counts == {}
    assert agg.per_variable_total == 0
    assert agg.per_variable_matched == 0
    assert agg.per_variable_matched_best == 0
    assert agg.analyst_absent_count == 0
    assert agg.regressions == []


def test_aggregate_counts_outcomes():
    """outcome_counts tallies each outcome string."""
    payload = _make_payload([
        _make_run(path="a.cpp", outcome="pass"),
        _make_run(path="b.cpp", outcome="fail_comparator"),
        _make_run(path="c.cpp", outcome="fail_comparator"),
        _make_run(path="d.cpp", outcome="error"),
    ])
    agg = aggregate(payload)
    assert agg.outcome_counts == {
        "pass": 1, "fail_comparator": 2, "error": 1,
    }


def test_aggregate_category_counts():
    """category_counts is keyed by (category, outcome)."""
    payload = _make_payload([
        _make_run(path="a.cpp", category="lowerable", outcome="pass"),
        _make_run(path="b.cpp", category="lowerable", outcome="pass"),
        _make_run(path="c.cpp", category="needs_precision",
                  outcome="fail_comparator"),
        _make_run(path="d.cpp", category="mixed", outcome="pass"),
    ])
    agg = aggregate(payload)
    assert agg.category_counts[("lowerable", "pass")] == 2
    assert agg.category_counts[("needs_precision", "fail_comparator")] == 1
    assert agg.category_counts[("mixed", "pass")] == 1


def test_aggregate_per_variable_totals():
    """per_variable totals sum across all non-analyst-absent runs."""
    payload = _make_payload([
        _make_run(path="a.cpp", pv_total=3, pv_matched=3),
        _make_run(path="b.cpp", pv_total=2, pv_matched=1),
        _make_run(path="c.cpp", pv_total=5, pv_matched=4),
    ])
    agg = aggregate(payload)
    assert agg.per_variable_total == 10
    assert agg.per_variable_matched == 8
    assert agg.per_variable_matched_best == 8  # no regression
    assert agg.analyst_absent_count == 0


def test_aggregate_excludes_analyst_absent_from_pv_totals():
    """Runs with analyst_absent=True are counted separately, not in pv totals."""
    payload = _make_payload([
        _make_run(path="a.cpp", pv_total=3, pv_matched=3),
        _make_run(path="b.cpp", pv_total=0, pv_matched=0,
                  analyst_absent=True, outcome="error"),
    ])
    agg = aggregate(payload)
    assert agg.per_variable_total == 3
    assert agg.per_variable_matched == 3
    assert agg.analyst_absent_count == 1


def test_aggregate_detects_regressions():
    """When per_variable_best.matched > per_variable.matched, that run
    is appended to .regressions; it still contributes to pv_matched_best."""
    regressed = _make_run(path="r.cpp", pv_total=4,
                          pv_matched=2, pv_best_matched=4)
    payload = _make_payload([
        _make_run(path="ok.cpp", pv_total=3, pv_matched=3),
        regressed,
    ])
    agg = aggregate(payload)
    assert len(agg.regressions) == 1
    assert agg.regressions[0]["path"] == "r.cpp"
    assert agg.per_variable_matched == 5  # 3 + 2
    assert agg.per_variable_matched_best == 7  # 3 + 4


def test_aggregate_no_regression_when_best_equals_last():
    """best == last is NOT a regression; the regressions list stays empty."""
    payload = _make_payload([
        _make_run(path="a.cpp", pv_total=3, pv_matched=2, pv_best_matched=2),
    ])
    agg = aggregate(payload)
    assert agg.regressions == []


# ----------------------------------------------------------------------
# render_text
# ----------------------------------------------------------------------


def test_render_text_header_and_outcome_summary():
    """Text report includes a header with the total count and outcome lines."""
    payload = _make_payload([
        _make_run(path="a.cpp", outcome="pass"),
        _make_run(path="b.cpp", outcome="fail_comparator"),
    ])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "Total kernels: 2" in text
    assert "PASS" in text
    assert "FCMP" in text
    # Percent formatting present somewhere.
    assert "(50.0%)" in text


def test_render_text_omits_regression_section_when_none():
    """A clean run has no Regressions block."""
    payload = _make_payload([_make_run(outcome="pass")])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "Regressions" not in text


def test_render_text_includes_regression_section_when_present():
    """A regressed run produces a Regressions block listing the kernel."""
    payload = _make_payload([
        _make_run(path="r.cpp", pv_total=4, pv_matched=2, pv_best_matched=4,
                  cycle_count=3),
    ])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "Regressions (1)" in text
    assert "r.cpp" in text
    assert "last=2/4" in text
    assert "best=4/4" in text
    assert "cycles=3" in text


def test_render_text_omits_failures_section_when_all_pass():
    """All-pass runs do not produce a Failures block."""
    payload = _make_payload([_make_run(outcome="pass")])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "Failures" not in text


def test_render_text_failure_block_includes_mismatches():
    """A failed run lists each variable mismatch under its block."""
    payload = _make_payload([_make_run(
        path="b.cpp", outcome="fail_comparator",
        pv_total=2, pv_matched=1,
        mismatches=[
            {"name": "x", "expected_action": "keep",
             "observed_action": "downcast"},
        ],
        comparator_status="mismatch",
    )])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "Failures (1)" in text
    assert "[FCMP] b.cpp" in text
    assert "variable x: expected=keep, observed=downcast" in text
    assert "comparator status='mismatch'" in text


def test_render_text_failure_block_surfaces_timeout():
    """A timed-out subprocess is annotated in the failure block."""
    payload = _make_payload([_make_run(
        path="t.cpp", outcome="error",
        pv_total=0, pv_matched=0, analyst_absent=True,
        timed_out=True, returncode=None, trace_exists=False,
        comparator_status=None,
    )])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "subprocess timed out" in text
    assert "no trace file written" in text


def test_render_text_failure_block_surfaces_nonzero_rc():
    """Non-zero subprocess returncode is surfaced when not a timeout."""
    payload = _make_payload([_make_run(
        path="x.cpp", outcome="error",
        timed_out=False, returncode=2,
    )])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "returncode=2" in text


def test_render_text_per_variable_only_shows_best_when_different():
    """The 'best' per-variable line is suppressed when best == last."""
    payload = _make_payload([_make_run(pv_total=3, pv_matched=2,
                                       pv_best_matched=2,
                                       outcome="fail_comparator")])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "matched: 2/3" in text
    assert "best:" not in text


def test_render_text_per_variable_shows_best_when_regressed():
    """When a regression exists, the 'best' line is shown."""
    payload = _make_payload([_make_run(pv_total=3, pv_matched=1,
                                       pv_best_matched=3,
                                       outcome="fail_comparator")])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "matched: 1/3" in text
    assert "best:    3/3" in text


def test_render_text_unknown_outcome_surfaces():
    """An unrecognized outcome string is shown with a ???? label rather than
    silently dropped, so future enum additions are visible."""
    payload = _make_payload([_make_run(outcome="weird_new_thing")])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "????" in text
    assert "weird_new_thing" in text


def test_render_text_category_breakdown_present_when_categories():
    """By-category table appears when at least one category is present."""
    payload = _make_payload([
        _make_run(category="lowerable", outcome="pass"),
        _make_run(category="needs_precision", outcome="fail_comparator"),
    ])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "By category:" in text
    assert "lowerable" in text
    assert "needs_precision" in text


def test_render_text_per_variable_handles_all_absent():
    """All-analyst-absent runs render '(n/a)' percent without dividing by zero."""
    payload = _make_payload([_make_run(
        pv_total=0, pv_matched=0, analyst_absent=True, outcome="error",
    )])
    agg = aggregate(payload)
    text = render_text(payload, agg)
    assert "matched: 0/0 (n/a)" in text
    assert "excluded 1 run(s) with no analyst call" in text


# ----------------------------------------------------------------------
# serialize_aggregate
# ----------------------------------------------------------------------


def test_serialize_aggregate_schema_v1_shape():
    """The aggregated JSON has schema_version=1 and the documented keys."""
    payload = _make_payload([_make_run()])
    agg = aggregate(payload)
    out = serialize_aggregate(payload, agg)
    assert out["schema_version"] == 1
    assert out["source_schema_version"] == 2
    assert set(out.keys()) >= {
        "schema_version", "source_schema_version", "total",
        "outcome_counts", "category_counts", "per_variable",
        "regressions", "failures",
    }


def test_serialize_aggregate_failures_list():
    """Each non-pass run appears in the failures list with path/outcome/category."""
    payload = _make_payload([
        _make_run(path="a.cpp", outcome="pass"),
        _make_run(path="b.cpp", outcome="fail_comparator",
                  category="needs_precision"),
        _make_run(path="c.cpp", outcome="error", category="mixed"),
    ])
    agg = aggregate(payload)
    out = serialize_aggregate(payload, agg)
    assert {f["path"] for f in out["failures"]} == {"b.cpp", "c.cpp"}
    by_path = {f["path"]: f for f in out["failures"]}
    assert by_path["b.cpp"]["outcome"] == "fail_comparator"
    assert by_path["b.cpp"]["category"] == "needs_precision"


def test_serialize_aggregate_regressions_list():
    """Regressed runs appear with last/best/cycle metadata."""
    payload = _make_payload([_make_run(
        path="r.cpp", pv_total=4, pv_matched=2, pv_best_matched=4,
        cycle_count=2,
    )])
    agg = aggregate(payload)
    out = serialize_aggregate(payload, agg)
    assert len(out["regressions"]) == 1
    r = out["regressions"][0]
    assert r["path"] == "r.cpp"
    assert r["last_matched"] == 2
    assert r["best_matched"] == 4
    assert r["cycle_count"] == 2


def test_serialize_aggregate_category_counts_sorted():
    """category_counts list is sorted by (category, outcome) for reproducibility."""
    payload = _make_payload([
        _make_run(path="a.cpp", category="mixed", outcome="pass"),
        _make_run(path="b.cpp", category="lowerable", outcome="pass"),
        _make_run(path="c.cpp", category="lowerable",
                  outcome="fail_comparator"),
    ])
    agg = aggregate(payload)
    out = serialize_aggregate(payload, agg)
    keys = [(c["category"], c["outcome"]) for c in out["category_counts"]]
    assert keys == sorted(keys)


def test_serialize_aggregate_round_trips_through_json():
    """The aggregated dict is JSON-serializable (no Path/dataclass leaks)."""
    payload = _make_payload([_make_run()])
    agg = aggregate(payload)
    out = serialize_aggregate(payload, agg)
    encoded = json.dumps(out)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == 1


# ----------------------------------------------------------------------
# _exit_code
# ----------------------------------------------------------------------


def test_exit_code_zero_when_all_pass():
    """All-pass + no regressions returns 0."""
    payload = _make_payload([_make_run(outcome="pass"),
                             _make_run(path="b.cpp", outcome="pass")])
    agg = aggregate(payload)
    assert _exit_code(agg) == 0


def test_exit_code_one_on_failure():
    """Any non-pass outcome returns 1."""
    payload = _make_payload([_make_run(outcome="fail_comparator")])
    agg = aggregate(payload)
    assert _exit_code(agg) == 1


def test_exit_code_one_on_regression_even_when_outcome_pass():
    """A regression flips the exit code to 1 even if every outcome is pass.

    Rationale: a regression means the analyst had a better answer
    earlier and self-corrected the wrong way — exactly the kind of
    behavior the eval harness exists to catch."""
    payload = _make_payload([_make_run(
        outcome="pass", pv_total=3, pv_matched=2, pv_best_matched=3,
    )])
    agg = aggregate(payload)
    assert _exit_code(agg) == 1


def test_exit_code_zero_on_empty_payload():
    """An empty payload trivially 'passes' (vacuously)."""
    agg = aggregate(_make_payload([]))
    assert _exit_code(agg) == 0


# ----------------------------------------------------------------------
# main (CLI)
# ----------------------------------------------------------------------


def test_main_all_pass_exits_zero(tmp_path: Path, capsys):
    """End-to-end: all-pass payload yields exit 0 and prints a report."""
    path = _write_payload(tmp_path, _make_payload([_make_run()]))
    rc = main([str(path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Total kernels: 1" in captured.out


def test_main_failure_exits_one(tmp_path: Path, capsys):
    """A failure payload yields exit 1."""
    path = _write_payload(tmp_path, _make_payload([
        _make_run(outcome="fail_comparator"),
    ]))
    rc = main([str(path)])
    assert rc == 1


def test_main_missing_file_exits_two(tmp_path: Path, capsys):
    """A missing input file yields exit 2 (config error, not run failure)."""
    rc = main([str(tmp_path / "missing.json")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not found" in captured.err


def test_main_unknown_schema_exits_two(tmp_path: Path, capsys):
    """An unknown schema_version yields exit 2."""
    path = _write_payload(tmp_path, _make_payload([], schema_version=99))
    rc = main([str(path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "schema_version=99" in captured.err


def test_main_writes_output_json_when_requested(tmp_path: Path, capsys):
    """--output PATH writes the aggregated JSON (schema_version=1)."""
    in_path = _write_payload(tmp_path, _make_payload([_make_run()]))
    out_path = tmp_path / "agg.json"
    rc = main([str(in_path), "--output", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["schema_version"] == 1
    assert data["total"] == 1


def test_main_does_not_write_output_when_not_requested(tmp_path: Path):
    """Without --output, no aggregated JSON file is created."""
    in_path = _write_payload(tmp_path, _make_payload([_make_run()]))
    sentinel = tmp_path / "agg.json"
    rc = main([str(in_path)])
    assert rc == 0
    assert not sentinel.exists()


def test_main_creates_output_parent_dirs(tmp_path: Path):
    """--output's parent directory is created if it doesn't exist
    (mirrors run.py's behavior)."""
    in_path = _write_payload(tmp_path, _make_payload([_make_run()]))
    out_path = tmp_path / "nested" / "deeper" / "agg.json"
    rc = main([str(in_path), "--output", str(out_path)])
    assert rc == 0
    assert out_path.exists()


def test_main_exit_one_on_regression_pass_outcomes(tmp_path: Path):
    """End-to-end: regression with all-pass outcomes still exits 1."""
    path = _write_payload(tmp_path, _make_payload([_make_run(
        outcome="pass", pv_total=3, pv_matched=1, pv_best_matched=3,
    )]))
    rc = main([str(path)])
    assert rc == 1
