"""Unit tests for the Layer 2 trace scorer.

These tests build synthetic JSONL traces in-memory (matching the schema
emitted by workflow/orchestrator.py:_append_trace) and feed them
through score_trace. The point is to exercise every branch of the
scorer — finish-gate state machine, per-variable matching, missing
analyst, malformed trace — without standing up the real workflow.

If you change the trace schema or the finish-gate semantics in
workflow/orchestrator.py:_FinishGateState, update the
_make_* helpers below in the SAME change. The scorer mirrors that
state machine; this test file pins the mirror.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.layer2.expected import ExpectedKernel
from evals.layer2.score import (
    load_trace,
    score_outcome,
    score_per_variable,
    score_trace,
    score_trace_file,
)


# ----------------------------------------------------------------------
# Helpers: build trace records.
# ----------------------------------------------------------------------


def _rec(turn: int, tool_name: str, tool_input: dict, exec_result: dict) -> dict:
    """Build one trace record matching _append_trace's schema."""
    return {
        "turn": turn,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "exec_result": exec_result,
    }


def _analyst_ok(variables: list[dict]) -> dict:
    """exec_result for a successful spawn_analyst, with the variables
    list the scorer reads. Other schema fields are omitted because the
    scorer doesn't look at them."""
    return {
        "status": "ok",
        "result": {
            "variables": variables,
            "rework": {
                "suggested": False,
                "transformation": "",
                "rationale": "",
                "affected_variables": [],
            },
            "precision_budget": {
                "target_kind": "sig_figs",
                "target_value": 6,
                "source": "user_cli",
                "claimed_output_precision": "ok",
                "headroom_argument": "test stub",
            },
        },
    }


def _finalizer_ok(variables: list[dict]) -> dict:
    """exec_result for a successful spawn_analyst_finalizer. Payload
    shape is identical to _analyst_ok (the two agents share
    ANALYST_OUTPUT_SCHEMA verbatim by AGENTS.md contract); this helper
    exists so scoring tests can spell out which agent produced a given
    record at the call site."""
    return _analyst_ok(variables)


def _verifier_ok(verdict: str) -> dict:
    return {"status": "ok", "result": {"verdict": verdict}}


def _compare_ok() -> dict:
    return {"status": "ok"}


def _compare_fail() -> dict:
    return {"status": "error", "stderr": "mismatch"}


def _finish_honored() -> dict:
    return {"status": "ok", "honored": True}


def _finish_gate_violation() -> dict:
    return {"status": "error", "is_error": True, "stderr": "gate"}


_KERNEL = ExpectedKernel(
    path="test-kernels/kokkos/lowerable/vector_add.cpp",
    category="lowerable",
    tolerance_kind="sig_figs",
    tolerance_value=6,
    per_variable={"x": "downcast", "y": "downcast", "z": "downcast"},
)


# ----------------------------------------------------------------------
# Per-variable scoring.
# ----------------------------------------------------------------------


def test_per_variable_all_match():
    """All three expected variables match the analyst's verdict."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.total == 3
    assert score.matched == 3
    assert score.mismatches == []
    assert score.missing_count == 0
    assert score.analyst_absent is False


def test_per_variable_one_mismatch_two_match():
    """Mismatched action surfaces in the mismatches list, missing_count stays 0."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "keep"},  # wrong
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.matched == 2
    assert score.missing_count == 0
    assert len(score.mismatches) == 1
    m = score.mismatches[0]
    assert (m.name, m.expected_action, m.observed_action) == (
        "y", "downcast", "keep",
    )


def test_per_variable_missing_variables_counted():
    """A variable the analyst omits counts as missing AND a mismatch."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            # y and z omitted
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.matched == 1
    assert score.missing_count == 2
    assert len(score.mismatches) == 2
    assert all(m.observed_action == "<missing>" for m in score.mismatches)


def test_per_variable_name_normalization():
    """Variable names match case-insensitively and ignore surrounding whitespace."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "X", "action": "downcast"},
            {"name": " y ", "action": "downcast"},
            {"name": "Z", "action": "downcast"},
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.matched == 3


def test_per_variable_extra_observed_variables_ignored():
    """Variables in the verdict but not in EXPECTED don't affect the score."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
            {"name": "scratch", "action": "keep"},  # not in EXPECTED
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.matched == 3
    assert score.mismatches == []


def test_per_variable_last_analyst_wins():
    """When analyst is re-spawned, only the LAST verdict is scored."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "keep"},  # wrong, but superseded
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
        _rec(5, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.matched == 3


def test_per_variable_failed_analyst_skipped():
    """A spawn_analyst whose exec_result is an error is ignored."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
        _rec(3, "spawn_analyst", {}, {"status": "error", "stderr": "boom"}),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    # The successful earlier verdict still counts.
    assert score.matched == 3


def test_per_variable_no_analyst_call():
    """Trace with no analyst call surfaces as analyst_absent + zero matches."""
    records = [
        _rec(1, "spawn_baseline_harness", {}, {"status": "ok"}),
    ]
    score, _best, _verdict = score_per_variable(records, _KERNEL)
    assert score.analyst_absent is True
    assert score.matched == 0
    assert score.missing_count == 3
    assert score.mismatches == []  # not enumerated when analyst never ran


def test_per_variable_empty_expected_map():
    """A kernel with empty per_variable is trivially passing."""
    kernel = ExpectedKernel(
        path="test-kernels/foo.cpp",
        category="mixed",
        tolerance_kind="sig_figs",
        tolerance_value=6,
        per_variable={},
    )
    records = [_rec(1, "spawn_analyst", {}, _analyst_ok([]))]
    score, _best, _verdict = score_per_variable(records, kernel)
    assert score.total == 0
    assert score.matched == 0
    assert score.analyst_absent is False


# ----------------------------------------------------------------------
# Finalizer / legacy-analyst dispatch (Step 5c: spawn_analyst_finalizer
# is the production analyst-verdict agent; spawn_analyst is a
# legacy-trace fallback).
# ----------------------------------------------------------------------


def test_per_variable_reads_finalizer_verdict():
    """A trace whose only analyst-verdict record is spawn_analyst_finalizer
    scores exactly like the legacy spawn_analyst equivalent — the two
    agents share ANALYST_OUTPUT_SCHEMA so scoring is agnostic. Pins
    the primary Step-5c contract: the scorer recognizes the finalizer.
    """
    records = [
        _rec(1, "spawn_analyst_finalizer", {}, _finalizer_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, _best, verdict = score_per_variable(records, _KERNEL)
    assert score.analyst_absent is False
    assert score.total == 3
    assert score.matched == 3
    assert score.missing_count == 0
    assert score.mismatches == []
    # Verdict dict is returned for diagnostic dumping.
    assert verdict.get("variables"), "finalizer verdict should be returned"


def test_per_variable_prefers_finalizer_over_legacy_analyst():
    """When both spawn_analyst_finalizer AND legacy spawn_analyst records
    exist in the trace (impossible on the production path but possible
    in a hand-crafted or archived trace), the finalizer wins
    unconditionally. Legacy is silently ignored — NOT merged into
    best-of-K, NOT used as a tiebreak. Pins the precedence rule."""
    records = [
        # Legacy first with a wrong verdict.
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "keep"},
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
        # Finalizer later with the correct verdict.
        _rec(5, "spawn_analyst_finalizer", {}, _finalizer_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, best, _ = score_per_variable(records, _KERNEL)
    assert score.matched == 3, "finalizer's verdict should win"
    assert best.matched == 3, "legacy should not appear in best-of-K"


def test_per_variable_falls_back_to_spawn_analyst_when_no_finalizer():
    """A trace that contains only legacy spawn_analyst records (e.g. a
    pre-Step-5c archived trace) is scored via the legacy fallback.
    Pins backward compatibility so archived traces stay scorable."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    score, _best, verdict = score_per_variable(records, _KERNEL)
    assert score.analyst_absent is False
    assert score.matched == 3
    assert verdict.get("variables"), "legacy verdict should be returned"


def test_per_variable_errored_finalizer_does_not_fall_back():
    """When the finalizer errors AND a successful legacy spawn_analyst
    record ALSO exists in the trace, the scorer must NOT fall back to
    the legacy verdict — an errored finalizer means the run genuinely
    failed at its authoritative final step, and grading a stale
    earlier verdict would be wrong signal. Result: analyst_absent=True,
    same as if no verdict existed at all. Pins the errored-finalizer
    no-fallback rule."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
        _rec(5, "spawn_analyst_finalizer", {}, {"status": "error", "stderr": "boom"}),
    ]
    score, best, verdict = score_per_variable(records, _KERNEL)
    assert score.analyst_absent is True
    assert score.matched == 0
    assert score.missing_count == 3
    assert best.analyst_absent is True
    assert verdict == {}


def test_per_variable_multiple_finalizer_cycles_last_wins():
    """The orchestrator may re-spawn the finalizer after a comparator
    failure (per the system prompt's retry guidance). last_score uses
    the most recent successful finalizer; best_score iterates across
    all cycles for its highest-match pick. Pins that retry semantics
    carry over from the legacy spawn_analyst behavior."""
    records = [
        # First cycle: matches 1/3.
        _rec(1, "spawn_analyst_finalizer", {}, _finalizer_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
        # Second cycle after retry: matches 3/3.
        _rec(5, "spawn_analyst_finalizer", {}, _finalizer_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.matched == 3, "LAST cycle wins for last_score"
    assert best.matched == 3, "best_score picks the highest-match cycle"


# ----------------------------------------------------------------------
# Best-cycle scoring (per_variable_best).
# ----------------------------------------------------------------------


def test_per_variable_best_equals_last_with_single_analyst_call():
    """With one analyst cycle, best == last by construction."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.matched == 2
    assert best.matched == 2
    assert best == last


def test_per_variable_best_equals_last_with_no_analyst_call():
    """With zero analyst cycles, best == last (both analyst_absent)."""
    records = [_rec(1, "spawn_baseline_harness", {}, {"status": "ok"})]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.analyst_absent is True
    assert best.analyst_absent is True
    assert best == last


def test_per_variable_best_picks_better_earlier_cycle():
    """The vector_add real-world case: cycle 1 matches all 3, cycle 2
    regresses to 0. Last shows the regression; best preserves the win."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
        _rec(5, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "keep"},
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.matched == 0
    assert best.matched == 3
    # And best != last in this case.
    assert best != last


def test_per_variable_best_picks_better_later_cycle():
    """When the LATER cycle is the strongest, best matches last."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "keep"},  # wrong
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
        _rec(5, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.matched == 3
    assert best.matched == 3


def test_per_variable_best_tiebreak_earliest_cycle():
    """Ties on matched count are broken by EARLIEST cycle. Both cycles
    here score 2/3; best should be the one from cycle 1 (mismatch on y),
    not cycle 5 (mismatch on z)."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "keep"},  # cycle 1 mismatches y
            {"name": "z", "action": "downcast"},
        ])),
        _rec(5, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "keep"},  # cycle 5 mismatches z
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    # Last is cycle 5 -> mismatches z.
    assert last.matched == 2
    assert any(m.name == "z" for m in last.mismatches)
    # Best is cycle 1 (earliest of the tied pair) -> mismatches y.
    assert best.matched == 2
    assert any(m.name == "y" for m in best.mismatches)


def test_per_variable_best_ignores_failed_analyst_cycle():
    """A failed-status spawn_analyst doesn't count in best-of-K either."""
    records = [
        _rec(1, "spawn_analyst", {}, {"status": "error", "stderr": "boom"}),
        _rec(3, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
    ]
    last, best, _ = score_per_variable(records, _KERNEL)
    assert last.matched == 3
    assert best.matched == 3
    # Same object since only one successful verdict exists.
    assert best == last


def test_score_trace_exposes_per_variable_best():
    """score_trace surfaces per_variable_best on the KernelResult."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
        _rec(5, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "keep"},
            {"name": "y", "action": "keep"},
            {"name": "z", "action": "keep"},
        ])),
    ]
    result = score_trace(records, _KERNEL)
    assert result.per_variable.matched == 0       # last
    assert result.per_variable_best.matched == 3  # best


# ----------------------------------------------------------------------
# Outcome scoring: finish-gate state machine.
# ----------------------------------------------------------------------


def _happy_path_records() -> list[dict]:
    """The canonical Kokkos pipeline: harness -> compile -> run -> analyst
    -> rewriter -> verifier(accept) -> splice -> compile_rewritten ->
    run_rewritten -> compare(ok) -> finish(honored)."""
    return [
        _rec(1, "spawn_baseline_harness", {}, {"status": "ok"}),
        _rec(2, "compile_baseline_driver", {}, {"status": "ok"}),
        _rec(3, "run_baseline_driver", {}, {"status": "ok"}),
        _rec(4, "spawn_analyst", {}, _analyst_ok([
            {"name": "x", "action": "downcast"},
            {"name": "y", "action": "downcast"},
            {"name": "z", "action": "downcast"},
        ])),
        _rec(5, "spawn_rewriter", {}, {"status": "ok"}),
        _rec(6, "spawn_verifier", {}, _verifier_ok("accept")),
        _rec(7, "splice_rewritten_kernel", {}, {"status": "ok"}),
        _rec(8, "compile_rewritten_driver", {}, {"status": "ok"}),
        _rec(9, "run_rewritten_driver", {}, {"status": "ok"}),
        _rec(10, "compare_outputs", {}, _compare_ok()),
        _rec(11, "finish", {}, _finish_honored()),
    ]


def test_outcome_happy_path_passes():
    """Canonical successful pipeline scores as pass."""
    out = score_outcome(_happy_path_records())
    assert out.outcome == "pass"
    assert out.reached_finish is True
    assert out.comparator_status == "ok"
    assert out.cycle_count == 1
    assert out.comparator_inapplicable is False


def test_outcome_no_finish_is_fail_no_finish():
    """A run that never calls finish is fail_no_finish."""
    records = _happy_path_records()[:-1]  # drop finish
    out = score_outcome(records)
    assert out.outcome == "fail_no_finish"
    assert out.reached_finish is False
    assert out.comparator_status == "ok"  # state at end-of-trace


def test_outcome_only_gate_violation_finish_is_fail_no_finish():
    """A finish call that the gate rejected counts as not-honored."""
    records = [
        _rec(1, "spawn_baseline_harness", {}, {"status": "ok"}),
        _rec(2, "spawn_analyst", {}, _analyst_ok([])),
        _rec(3, "finish", {}, _finish_gate_violation()),
    ]
    out = score_outcome(records)
    assert out.outcome == "fail_no_finish"
    assert out.reached_finish is False


def test_outcome_finish_honored_after_gate_violation_passes():
    """The orchestrator may finish-then-recover; only the honored
    finish matters for the outcome."""
    records = _happy_path_records()
    # Insert a rejected finish before the honored one.
    records.insert(10, _rec(10, "finish", {}, _finish_gate_violation()))
    # Renumber turns (cosmetic; scorer ignores turn).
    for i, r in enumerate(records, start=1):
        r["turn"] = i
    out = score_outcome(records)
    assert out.outcome == "pass"
    assert out.reached_finish is True


def test_outcome_compare_failed_at_finish_is_fail_comparator():
    """If finish was honored but compare_outputs failed (gate bug or
    workflow version drift), surface as fail_comparator."""
    records = _happy_path_records()
    # Flip the compare to a fail. The honored finish in this synthetic
    # trace then represents a gate bug we want surfaced.
    records[9] = _rec(10, "compare_outputs", {}, _compare_fail())
    out = score_outcome(records)
    assert out.outcome == "fail_comparator"
    assert out.reached_finish is True
    assert out.comparator_status == "error"


def test_outcome_spawn_rewriter_resets_compare_status():
    """A second rewrite cycle invalidates the prior compare status; if
    the new cycle's compare never runs, finish should not be honored —
    but if the trace SHOWS an honored finish anyway (gate bug), we
    surface fail_comparator because last_compare is None at finish."""
    records = _happy_path_records()
    # Inject a second rewriter call AFTER the first compare, with no
    # subsequent compare before the (still-honored) finish.
    records.insert(10, _rec(11, "spawn_rewriter", {}, {"status": "ok"}))
    for i, r in enumerate(records, start=1):
        r["turn"] = i
    out = score_outcome(records)
    assert out.outcome == "fail_comparator"
    assert out.comparator_status is None
    assert out.cycle_count == 2


def test_outcome_splice_resets_compare_status_only():
    """splice_rewritten_kernel invalidates last_compare but not the cycle count."""
    records = _happy_path_records()
    # Inject a splice AFTER the compare, before finish. With no
    # subsequent compare, finish appearing as honored is again a gate
    # bug we want surfaced.
    records.insert(10, _rec(11, "splice_rewritten_kernel", {}, {"status": "ok"}))
    for i, r in enumerate(records, start=1):
        r["turn"] = i
    out = score_outcome(records)
    assert out.outcome == "fail_comparator"
    assert out.comparator_status is None
    assert out.cycle_count == 1  # splice does NOT bump cycle


@pytest.mark.parametrize(
    "invalidator",
    ["splice_rewritten_kernel", "compile_rewritten_driver", "run_rewritten_driver"],
)
def test_outcome_each_rewritten_step_invalidates_compare(invalidator: str):
    """splice/compile_rewritten/run_rewritten each reset last_compare."""
    records = _happy_path_records()
    records.insert(10, _rec(11, invalidator, {}, {"status": "ok"}))
    for i, r in enumerate(records, start=1):
        r["turn"] = i
    out = score_outcome(records)
    assert out.comparator_status is None


def test_outcome_no_dynamic_verification_chain_is_comparator_inapplicable():
    """A trace without harness/compare calls reports comparator_inapplicable."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([])),
        _rec(2, "spawn_rewriter", {}, {"status": "ok"}),
        _rec(3, "spawn_verifier", {}, _verifier_ok("accept")),
        _rec(4, "finish", {}, _finish_honored()),
    ]
    out = score_outcome(records)
    assert out.outcome == "pass"
    assert out.comparator_inapplicable is True
    assert out.comparator_status is None


def test_outcome_cycle_count_increments_on_each_rewriter_call():
    """cycle_count tracks the number of spawn_rewriter invocations."""
    records = [
        _rec(1, "spawn_analyst", {}, _analyst_ok([])),
        _rec(2, "spawn_rewriter", {}, {"status": "ok"}),
        _rec(3, "spawn_verifier", {}, _verifier_ok("reject")),
        _rec(4, "spawn_rewriter", {}, {"status": "ok"}),
        _rec(5, "spawn_verifier", {}, _verifier_ok("reject")),
        _rec(6, "spawn_rewriter", {}, {"status": "ok"}),
    ]
    out = score_outcome(records)
    assert out.cycle_count == 3
    assert out.reached_finish is False


# ----------------------------------------------------------------------
# Top-level entry points.
# ----------------------------------------------------------------------


def test_score_trace_combines_per_variable_and_outcome():
    """score_trace bundles both metrics and the analyst verdict."""
    records = _happy_path_records()
    result = score_trace(records, _KERNEL)
    assert result.path == _KERNEL.path
    assert result.category == _KERNEL.category
    assert result.per_variable.matched == 3
    assert result.outcome.outcome == "pass"
    # Analyst verdict surfaced for diagnostic dumping.
    assert "variables" in result.analyst_verdict


def test_load_trace_parses_jsonl(tmp_path: Path):
    """load_trace reads JSONL line-by-line, skipping blanks."""
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"turn": 1, "tool_name": "spawn_analyst", "tool_input": {}, '
        '"exec_result": {"status": "ok"}}\n'
        '\n'  # blank line tolerated
        '{"turn": 2, "tool_name": "finish", "tool_input": {}, '
        '"exec_result": {"status": "ok"}}\n'
    )
    records = load_trace(p)
    assert len(records) == 2
    assert records[0]["tool_name"] == "spawn_analyst"


def test_load_trace_raises_on_malformed_line(tmp_path: Path):
    """load_trace raises ValueError with line number on malformed JSONL."""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"turn": 1}\nnot json at all\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_trace(p)


def test_score_trace_file_missing_returns_error_outcome(tmp_path: Path):
    """A missing trace file scores as outcome='error' with all vars missing."""
    result = score_trace_file(tmp_path / "absent.jsonl", _KERNEL)
    assert result.outcome.outcome == "error"
    assert result.outcome.reached_finish is False
    assert result.per_variable.analyst_absent is True
    assert result.per_variable.missing_count == 3
    assert result.analyst_verdict == {}


def test_score_trace_file_malformed_returns_error_outcome(tmp_path: Path):
    """A malformed trace file scores as outcome='error' (not raised)."""
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n")
    result = score_trace_file(p, _KERNEL)
    assert result.outcome.outcome == "error"
    assert result.per_variable.analyst_absent is True


def test_score_trace_file_happy_path(tmp_path: Path):
    """End-to-end: write a JSONL trace and score it via the file entry point."""
    p = tmp_path / "ok.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _happy_path_records()) + "\n")
    result = score_trace_file(p, _KERNEL)
    assert result.outcome.outcome == "pass"
    assert result.per_variable.matched == 3
