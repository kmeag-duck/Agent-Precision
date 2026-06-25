"""Unit tests for the Layer 2 batch driver (evals/layer2/run.py).

The driver shells out to `python -m workflow.run --auto` per kernel
and reads back the trace it leaves at `baselines/<stem>/
orchestrator_trace.jsonl`. These tests never invoke that real
subprocess — they pass a fake `subprocess_runner` into `run_one` /
`_run_batch` that:

  - asserts the argv shape matches what `build_argv` would produce,
  - writes a synthetic trace JSONL to the expected path under a
    tmp_path "fake repo root",
  - returns whatever (returncode, timed_out, stderr_tail) the test
    case wants to simulate.

That keeps the tests fast, hermetic, and pinned to the subprocess
CONTRACT (argv shape + trace path) rather than to a live workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.layer2.expected import EXPECTED, ExpectedKernel
from evals.layer2.run import (
    RunRecord,
    _format_summary_line,
    _resolve_timeout_sec,
    _run_batch,
    _serialize_results,
    build_argv,
    main,
    run_one,
    select_kernels,
    trace_path_for,
)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


def _kernel(
    path: str = "test-kernels/kokkos/lowerable/vector_add.cpp",
    category: str = "lowerable",
    tol_kind: str = "sig_figs",
    tol_value: int = 6,
    per_variable: dict | None = None,
) -> ExpectedKernel:
    return ExpectedKernel(
        path=path,
        category=category,
        tolerance_kind=tol_kind,
        tolerance_value=tol_value,
        per_variable=per_variable
        if per_variable is not None
        else {"x": "downcast", "y": "downcast", "z": "downcast"},
    )


def _happy_trace_records(actions: dict[str, str] | None = None) -> list[dict]:
    """A minimal happy-path trace: analyst with the given per-variable
    actions, then a single rewrite cycle ending in finish-honored."""
    actions = actions or {"x": "downcast", "y": "downcast", "z": "downcast"}
    variables = [
        {"name": name, "action": action} for name, action in actions.items()
    ]
    return [
        {"turn": 1, "tool_name": "spawn_baseline_harness",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 2, "tool_name": "compile_baseline_driver",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 3, "tool_name": "run_baseline_driver",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 4, "tool_name": "spawn_analyst",
         "tool_input": {},
         "exec_result": {"status": "ok", "result": {
             "variables": variables,
             "rework": {"suggested": False, "transformation": "",
                        "rationale": "", "affected_variables": []},
             "precision_budget": {"target_kind": "sig_figs",
                                  "target_value": 6, "source": "user_cli",
                                  "claimed_output_precision": "ok",
                                  "headroom_argument": ""},
         }}},
        {"turn": 5, "tool_name": "spawn_rewriter",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 6, "tool_name": "spawn_verifier",
         "tool_input": {},
         "exec_result": {"status": "ok", "result": {"verdict": "accept"}}},
        {"turn": 7, "tool_name": "splice_rewritten_kernel",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 8, "tool_name": "compile_rewritten_driver",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 9, "tool_name": "run_rewritten_driver",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 10, "tool_name": "compare_outputs",
         "tool_input": {}, "exec_result": {"status": "ok"}},
        {"turn": 11, "tool_name": "finish",
         "tool_input": {}, "exec_result": {"status": "ok", "honored": True}},
    ]


def _write_trace(repo_root: Path, kernel: ExpectedKernel,
                 records: list[dict]) -> Path:
    """Write a trace JSONL to the location run.py will read it from."""
    path = trace_path_for(kernel, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


# ----------------------------------------------------------------------
# build_argv
# ----------------------------------------------------------------------


def test_build_argv_sig_figs():
    """sig_figs tolerance produces --sig-figs N in the argv."""
    k = _kernel(tol_kind="sig_figs", tol_value=6)
    argv = build_argv(Path("/usr/bin/python"), k)
    assert argv == [
        "/usr/bin/python",
        "-m",
        "workflow.run",
        k.path,
        "--auto",
        "--sig-figs",
        "6",
    ]


def test_build_argv_decimal_digits():
    """decimal_digits tolerance produces --decimal-digits N in the argv."""
    k = _kernel(tol_kind="decimal_digits", tol_value=2)
    argv = build_argv(Path("/usr/bin/python"), k)
    assert argv[-2:] == ["--decimal-digits", "2"]
    assert "--auto" in argv


def test_build_argv_rejects_unknown_kind():
    """An unknown tolerance_kind raises ValueError, not SystemExit."""
    k = _kernel(tol_kind="bogus", tol_value=6)
    with pytest.raises(ValueError, match="bogus"):
        build_argv(Path("/usr/bin/python"), k)


# ----------------------------------------------------------------------
# trace_path_for
# ----------------------------------------------------------------------


def test_trace_path_for_uses_kernel_stem(tmp_path: Path):
    """Trace path is `<root>/baselines/<stem>/orchestrator_trace.jsonl`."""
    k = _kernel(path="test-kernels/kokkos/lowerable/vector_add.cpp")
    p = trace_path_for(k, tmp_path)
    assert p == tmp_path / "baselines" / "vector_add" / "orchestrator_trace.jsonl"


def test_trace_path_for_cuda_extension_stem(tmp_path: Path):
    """`.cu` files have the suffix stripped for the stem too."""
    k = _kernel(path="test-kernels/cuda/lowerable/saxpy.cu")
    p = trace_path_for(k, tmp_path)
    assert p == tmp_path / "baselines" / "saxpy" / "orchestrator_trace.jsonl"


# ----------------------------------------------------------------------
# select_kernels
# ----------------------------------------------------------------------


def test_select_kernels_no_filters_returns_all():
    """With no filters, every EXPECTED entry is selected, in order."""
    selected = select_kernels(EXPECTED, None, None)
    assert [s.path for s in selected] == list(EXPECTED.keys())


def test_select_kernels_category_filter():
    """--category filter restricts to that category."""
    selected = select_kernels(EXPECTED, ["lowerable"], None)
    assert selected
    assert all(s.category == "lowerable" for s in selected)


def test_select_kernels_multiple_categories():
    """Multiple --category flags union the categories."""
    selected = select_kernels(EXPECTED, ["lowerable", "mixed"], None)
    cats = {s.category for s in selected}
    assert cats == {"lowerable", "mixed"}


def test_select_kernels_path_substring_case_insensitive():
    """--path matches case-insensitively as a substring."""
    selected = select_kernels(EXPECTED, None, ["VECTOR_add"])
    assert selected
    assert all("vector_add" in s.path for s in selected)


def test_select_kernels_multiple_path_substrings_are_or():
    """Multiple --path flags match if ANY substring is present."""
    selected = select_kernels(EXPECTED, None, ["vector_add", "saxpy"])
    paths = [s.path for s in selected]
    assert any("vector_add" in p for p in paths)
    assert any("saxpy" in p for p in paths)
    assert all(("vector_add" in p) or ("saxpy" in p) for p in paths)


def test_select_kernels_category_and_path_compose():
    """--category AND --path: both must match."""
    selected = select_kernels(EXPECTED, ["lowerable"], ["vector_add"])
    assert selected
    assert all(s.category == "lowerable" and "vector_add" in s.path
               for s in selected)


def test_select_kernels_empty_when_nothing_matches():
    """A filter that excludes everything returns []."""
    assert select_kernels(EXPECTED, None, ["no_such_kernel"]) == []


# ----------------------------------------------------------------------
# _resolve_timeout_sec
# ----------------------------------------------------------------------


def test_resolve_timeout_default(monkeypatch):
    """Unset env var yields the 600s default."""
    monkeypatch.delenv("AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC",
                       raising=False)
    assert _resolve_timeout_sec() == 600


def test_resolve_timeout_explicit(monkeypatch):
    """A positive int env value is honored."""
    monkeypatch.setenv("AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC", "30")
    assert _resolve_timeout_sec() == 30


def test_resolve_timeout_rejects_non_integer(monkeypatch):
    """A non-int env value raises SystemExit (config error)."""
    monkeypatch.setenv("AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC", "abc")
    with pytest.raises(SystemExit):
        _resolve_timeout_sec()


def test_resolve_timeout_rejects_non_positive(monkeypatch):
    """A zero or negative env value raises SystemExit."""
    monkeypatch.setenv("AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC", "0")
    with pytest.raises(SystemExit):
        _resolve_timeout_sec()


# ----------------------------------------------------------------------
# run_one: subprocess contract + trace handoff
# ----------------------------------------------------------------------


def test_run_one_invokes_runner_with_expected_argv_and_cwd(tmp_path: Path):
    """run_one calls the runner with the build_argv result and cwd=repo_root."""
    k = _kernel()
    seen: dict = {}

    def runner(argv, cwd, timeout_sec):
        seen["argv"] = argv
        seen["cwd"] = cwd
        seen["timeout"] = timeout_sec
        _write_trace(tmp_path, k, _happy_trace_records())
        return 0, False, ""

    rec = run_one(k, Path("/usr/bin/python"), tmp_path,
                  timeout_sec=42, subprocess_runner=runner)

    assert seen["argv"] == build_argv(Path("/usr/bin/python"), k)
    assert seen["cwd"] == tmp_path
    assert seen["timeout"] == 42
    assert rec.returncode == 0
    assert rec.timed_out is False
    assert rec.trace_exists is True
    assert rec.score.outcome.outcome == "pass"
    assert rec.score.per_variable.matched == 3


def test_run_one_records_timeout(tmp_path: Path):
    """A runner that signals timeout produces timed_out=True and outcome=error."""
    k = _kernel()

    def runner(argv, cwd, timeout_sec):
        # Don't write a trace — timeout struck before the workflow
        # could persist one.
        return None, True, "partial stderr"

    rec = run_one(k, Path("/usr/bin/python"), tmp_path,
                  timeout_sec=1, subprocess_runner=runner)
    assert rec.timed_out is True
    assert rec.returncode is None
    assert rec.trace_exists is False
    # No trace -> score_trace_file maps to outcome='error'.
    assert rec.score.outcome.outcome == "error"
    assert rec.stderr_tail == "partial stderr"


def test_run_one_nonzero_returncode_with_trace(tmp_path: Path):
    """A nonzero exit with a partial trace still gets scored from the trace."""
    k = _kernel()
    # Trace contains an analyst call but no finish.
    partial = _happy_trace_records()[:4]

    def runner(argv, cwd, timeout_sec):
        _write_trace(tmp_path, k, partial)
        return 1, False, "boom"

    rec = run_one(k, Path("/usr/bin/python"), tmp_path,
                  timeout_sec=60, subprocess_runner=runner)
    assert rec.returncode == 1
    assert rec.trace_exists is True
    assert rec.score.outcome.outcome == "fail_no_finish"
    assert rec.score.per_variable.matched == 3  # analyst ran, all good


def test_run_one_records_wallclock_duration(tmp_path: Path):
    """rec.duration_sec is populated and non-negative."""
    k = _kernel()

    def runner(argv, cwd, timeout_sec):
        _write_trace(tmp_path, k, _happy_trace_records())
        return 0, False, ""

    rec = run_one(k, Path("/usr/bin/python"), tmp_path,
                  timeout_sec=60, subprocess_runner=runner)
    assert rec.duration_sec >= 0.0


# ----------------------------------------------------------------------
# _run_batch
# ----------------------------------------------------------------------


def test_run_batch_sequential_preserves_submission_order(tmp_path: Path):
    """--jobs=1 results are returned in submission order."""
    kernels = [
        _kernel(path="test-kernels/kokkos/lowerable/vector_add.cpp"),
        _kernel(path="test-kernels/kokkos/lowerable/saxpy_bounded.cpp",
                per_variable={"a": "downcast", "x": "downcast", "y": "downcast"}),
    ]

    def runner(argv, cwd, timeout_sec):
        # Recover the kernel from argv[3] (the kernel path) and write
        # a tailored trace.
        kernel_path = argv[3]
        kernel = next(k for k in kernels if k.path == kernel_path)
        actions = {name: action for name, action
                   in kernel.per_variable.items()}
        _write_trace(tmp_path, kernel, _happy_trace_records(actions))
        return 0, False, ""

    seen_order: list[str] = []
    records = _run_batch(
        kernels=kernels, python=Path("/usr/bin/python"),
        repo_root=tmp_path, timeout_sec=60, jobs=1,
        on_complete=lambda i, t, r: seen_order.append(r.expected.path),
        subprocess_runner=runner,
    )
    assert [r.expected.path for r in records] == [k.path for k in kernels]
    assert seen_order == [k.path for k in kernels]


def test_run_batch_parallel_preserves_submission_order(tmp_path: Path):
    """--jobs>1 results are still returned in submission order even if
    completion order is different. The on_complete callback runs in
    completion order, but the returned list is submission-ordered."""
    kernels = [
        _kernel(path=f"test-kernels/kokkos/lowerable/kernel_{i}.cpp",
                per_variable={"x": "downcast"})
        for i in range(4)
    ]

    def runner(argv, cwd, timeout_sec):
        kernel_path = argv[3]
        kernel = next(k for k in kernels if k.path == kernel_path)
        _write_trace(tmp_path, kernel,
                     _happy_trace_records({"x": "downcast"}))
        return 0, False, ""

    records = _run_batch(
        kernels=kernels, python=Path("/usr/bin/python"),
        repo_root=tmp_path, timeout_sec=60, jobs=4,
        on_complete=lambda i, t, r: None,
        subprocess_runner=runner,
    )
    # Submission order = the order we passed kernels in.
    assert [r.expected.path for r in records] == [k.path for k in kernels]


def test_run_batch_on_complete_called_once_per_kernel(tmp_path: Path):
    """on_complete fires exactly len(kernels) times with idx in [1..total]."""
    kernels = [
        _kernel(path=f"test-kernels/kokkos/lowerable/kernel_{i}.cpp",
                per_variable={"x": "downcast"})
        for i in range(3)
    ]

    def runner(argv, cwd, timeout_sec):
        kernel_path = argv[3]
        kernel = next(k for k in kernels if k.path == kernel_path)
        _write_trace(tmp_path, kernel,
                     _happy_trace_records({"x": "downcast"}))
        return 0, False, ""

    seen_idx: list[int] = []
    seen_total: list[int] = []
    _run_batch(
        kernels=kernels, python=Path("/usr/bin/python"),
        repo_root=tmp_path, timeout_sec=60, jobs=1,
        on_complete=lambda i, t, r: (seen_idx.append(i), seen_total.append(t)),
        subprocess_runner=runner,
    )
    assert seen_idx == [1, 2, 3]
    assert seen_total == [3, 3, 3]


# ----------------------------------------------------------------------
# Summary line formatting
# ----------------------------------------------------------------------


def _make_record(outcome: str = "pass", **score_overrides) -> RunRecord:
    """Build a RunRecord with a synthetic score for formatting tests.

    Per-variable best defaults to a copy of per-variable last (the
    common case: 0 or 1 analyst cycles, so best == last). Pass
    `pv_best_matched=N` to force a divergence — used by the
    regression-annotation tests.
    """
    from evals.layer2.score import (
        KernelResult, OutcomeScore, PerVariableScore,
    )
    k = _kernel()
    pv_total = score_overrides.get("pv_total", 3)
    pv_matched = score_overrides.get("pv_matched", 3)
    pv = PerVariableScore(
        total=pv_total,
        matched=pv_matched,
        mismatches=[],
        missing_count=score_overrides.get("pv_missing", 0),
        analyst_absent=score_overrides.get("analyst_absent", False),
    )
    pv_best = PerVariableScore(
        total=pv_total,
        matched=score_overrides.get("pv_best_matched", pv_matched),
        mismatches=[],
        missing_count=score_overrides.get(
            "pv_best_missing", score_overrides.get("pv_missing", 0)),
        analyst_absent=score_overrides.get("analyst_absent", False),
    )
    out = OutcomeScore(
        outcome=outcome,
        reached_finish=score_overrides.get("reached_finish", outcome == "pass"),
        comparator_status=score_overrides.get("comparator_status", "ok"
                                              if outcome == "pass" else None),
        cycle_count=score_overrides.get("cycle_count", 1),
        comparator_inapplicable=False,
    )
    return RunRecord(
        expected=k,
        argv=build_argv(Path("/usr/bin/python"), k),
        returncode=score_overrides.get("returncode", 0),
        timed_out=score_overrides.get("timed_out", False),
        duration_sec=score_overrides.get("duration_sec", 12.3),
        trace_path=Path("/tmp/fake"),
        trace_exists=score_overrides.get("trace_exists", True),
        score=KernelResult(path=k.path, category=k.category,
                           per_variable=pv, per_variable_best=pv_best,
                           outcome=out, analyst_verdict={}),
        stderr_tail="",
    )


def test_format_summary_line_pass():
    """PASS lines include vars matched, cycle count, duration."""
    rec = _make_record(outcome="pass")
    line = _format_summary_line(1, 17, rec)
    assert line.startswith("[1/17] PASS  ")
    assert "vars=3/3" in line
    assert "cycles=1" in line
    assert "(12.3s)" in line
    assert "[" not in line.split("(12.3s)")[1]  # no extras


def test_format_summary_line_timeout_extra():
    """Timeout surfaces as an extra annotation."""
    rec = _make_record(outcome="error", timed_out=True, returncode=None,
                       trace_exists=False, pv_matched=0, pv_missing=3,
                       analyst_absent=True, cycle_count=0,
                       reached_finish=False, comparator_status=None)
    line = _format_summary_line(2, 17, rec)
    assert "ERR" in line
    assert "[timeout, no trace]" in line


def test_format_summary_line_nonzero_returncode():
    """A nonzero return code without timeout surfaces as `rc=N`."""
    rec = _make_record(outcome="fail_no_finish", returncode=1,
                       reached_finish=False)
    line = _format_summary_line(3, 17, rec)
    assert "FNOF" in line
    assert "[rc=1]" in line


def test_format_summary_line_omits_best_annotation_when_equal():
    """When best == last (the common case: 0 or 1 analyst cycles, or
    every cycle scored the same), the line shows only `vars=M/N` —
    no parenthetical — to keep things scannable."""
    rec = _make_record(outcome="pass", pv_matched=3, pv_best_matched=3)
    line = _format_summary_line(1, 17, rec)
    assert "vars=3/3" in line
    assert "(best" not in line


def test_format_summary_line_shows_best_annotation_when_regression():
    """The vector_add real-world case: last is worse than best.
    The one-liner surfaces `(best B/N)` so the regression is visible
    at a glance during a batch run."""
    rec = _make_record(outcome="pass", pv_matched=0, pv_best_matched=3)
    line = _format_summary_line(1, 17, rec)
    assert "vars=0/3 (best 3/3)" in line


def test_format_summary_line_shows_best_annotation_when_last_better():
    """Symmetric: if for some reason last > best (shouldn't happen by
    construction, but the formatter is purely comparison-based and
    should still annotate any divergence rather than silently lying)."""
    rec = _make_record(outcome="pass", pv_matched=3, pv_best_matched=2)
    line = _format_summary_line(1, 17, rec)
    assert "vars=3/3 (best 2/3)" in line


# ----------------------------------------------------------------------
# _serialize_results
# ----------------------------------------------------------------------


def test_serialize_results_round_trips_through_json():
    """The serialized dict is JSON-safe and re-parseable."""
    rec = _make_record(outcome="pass")
    blob = _serialize_results([rec])
    # Should not raise.
    text = json.dumps(blob)
    parsed = json.loads(text)
    assert parsed["schema_version"] == 2
    assert parsed["total"] == 1
    assert parsed["summary"] == {"pass": 1}
    assert len(parsed["runs"]) == 1
    run = parsed["runs"][0]
    assert run["path"] == rec.expected.path
    assert run["outcome"] == "pass"
    assert run["per_variable"]["matched"] == 3
    # v2 adds per_variable_best with the same shape as per_variable.
    assert run["per_variable_best"]["matched"] == 3
    assert run["per_variable_best"]["total"] == 3
    assert run["argv"][:3] == ["/usr/bin/python", "-m", "workflow.run"]
    assert run["trace_path"]  # non-empty


def test_serialize_results_aggregates_summary_counts():
    """Multiple kernels with mixed outcomes aggregate correctly."""
    recs = [
        _make_record(outcome="pass"),
        _make_record(outcome="pass"),
        _make_record(outcome="fail_no_finish", reached_finish=False,
                     comparator_status=None),
        _make_record(outcome="error", timed_out=True, returncode=None,
                     trace_exists=False, analyst_absent=True),
    ]
    blob = _serialize_results(recs)
    assert blob["summary"] == {"pass": 2, "fail_no_finish": 1, "error": 1}
    assert blob["total"] == 4


# ----------------------------------------------------------------------
# main: end-to-end CLI exit codes
# ----------------------------------------------------------------------


def test_main_dry_run_exits_zero_without_running(capsys, monkeypatch):
    """--dry-run lists kernels and exits 0 without invoking subprocess."""
    # Belt-and-suspenders: replace the default runner with one that
    # would fail loudly if accidentally called.
    monkeypatch.setattr(
        "evals.layer2.run._default_subprocess_runner",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("subprocess should not run under --dry-run")
        ),
    )
    rc = main(["--dry-run", "--category", "lowerable"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "test-kernels/kokkos/lowerable/vector_add.cpp" in out
    assert "--auto" in out


def test_main_empty_selection_returns_two(capsys):
    """A filter that matches nothing returns exit code 2."""
    rc = main(["--path", "no_such_kernel"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No kernels matched" in err


def test_main_invalid_jobs_returns_two(capsys):
    """--jobs=0 is a config error and returns 2."""
    rc = main(["--jobs", "0", "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--jobs" in err
