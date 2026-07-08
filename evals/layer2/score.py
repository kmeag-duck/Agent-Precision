"""Trace parsing and scoring for Layer 2 evaluation.

Reads an orchestrator JSONL trace (produced by `python -m workflow.run
--auto`) and scores it against an ExpectedKernel from expected.py.
Produces a KernelResult that the report module aggregates and prints.

Trace record schema (one JSON object per line, written by
workflow/orchestrator.py:_append_trace):
  {turn: int, tool_name: str, tool_input: dict, exec_result: dict}

This module is strictly read-only: it parses traces and returns
structured results. It does not run the workflow (run.py does that),
does not write anywhere, and does not import from workflow.* (so it
can be vendored or run against archived traces from a different
workflow revision).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from .expected import ExpectedKernel

# Tools that the orchestrator may emit. Kept here as a closed set so a
# typo in a trace (or a workflow version drift) surfaces as an
# unknown-tool warning rather than silently being ignored.
_KNOWN_TOOLS = frozenset({
    "spawn_analyst",
    "spawn_candidate_finder",
    "spawn_variable_analyst",
    "spawn_analyst_finalizer",
    "spawn_rewriter",
    "spawn_verifier",
    "spawn_baseline_harness",
    "compile_baseline_driver",
    "run_baseline_driver",
    "splice_rewritten_kernel",
    "compile_rewritten_driver",
    "run_rewritten_driver",
    "compare_outputs",
    "measure_speedup",
    "probe_step",
    "probe_compare",
    "test_variable_downcast",
    "test_variable_union_downcast",
    "bisect_variable_downcast",
    "finish",
})

# Outcome enum values, exported as a Literal for type-checking. Mirrors
# the sketch in the conversation that preceded this file:
#   pass             — reached finish AND comparator ok (or comparator inapplicable)
#   fail_comparator  — reached finish but comparator failed (gate bug)
#   fail_no_finish   — never reached finish (turn limit / crash / unrecovered error)
#   error            — trace malformed, missing, or unparseable
Outcome = Literal["pass", "fail_comparator", "fail_no_finish", "error"]


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VariableMismatch:
    """One per-variable disagreement between expected and observed action."""

    name: str
    expected_action: str
    observed_action: str  # "<missing>" if the analyst didn't classify it at all


@dataclass(frozen=True)
class PerVariableScore:
    """Per-variable agreement metric for one kernel.

    matched + mismatched + missing == total. They are tracked
    separately because "the analyst classified this variable wrong" is
    a different failure mode than "the analyst never mentioned this
    variable" — the second is worse (incomplete coverage), and
    distinguishing them is useful for diagnosing prompt issues.
    """

    total: int
    matched: int
    mismatches: list[VariableMismatch] = field(default_factory=list)
    # Variables expected.per_variable named but the analyst's verdict
    # did not include. Counted in `mismatches` too (with
    # observed_action='<missing>'), but also surfaced as a count so
    # callers can distinguish "wrong action" from "no action".
    missing_count: int = 0
    # True iff no spawn_analyst call appears in the trace at all (the
    # workflow died before reaching the analyst). When True, total
    # equals len(expected.per_variable) and matched is 0; mismatches
    # is left empty since we don't want to attribute K mismatches to
    # an analyst that never ran.
    analyst_absent: bool = False


@dataclass(frozen=True)
class OutcomeScore:
    """Final-outcome metric for one kernel."""

    outcome: Outcome
    reached_finish: bool
    # Status of the most recent compare_outputs call in the SAME rewrite
    # cycle as `finish` (or as the end of the trace if finish never
    # ran). None if no compare_outputs call was ever made, or if the
    # comparator was invalidated by a later splice/compile/run before
    # any finish or end-of-trace. Mirrors _FinishGateState semantics.
    comparator_status: str | None
    # Number of spawn_rewriter calls; rough proxy for how much retry
    # the run needed. 0 means the workflow gave up before any rewrite.
    cycle_count: int
    # True if the kernel's language profile carries
    # dynamic_verification=False (e.g. a future profile registered
    # before a baseline harness exists). When True, comparator_status
    # is expected to be None and reached_finish alone determines
    # pass/fail. Derived from the absence of any baseline_harness/
    # compare_outputs call in the trace AND the inability to infer
    # from category alone — see _classify_outcome for the precise
    # rule.
    comparator_inapplicable: bool = False


@dataclass(frozen=True)
class KernelResult:
    """Per-kernel scoring bundle returned to the runner.

    Carries TWO per-variable scores against the same ground truth:

      - `per_variable` is the LAST successful analyst cycle, i.e. the
        verdict the workflow ultimately committed to. This is the
        primary score and the one that should be used for headline
        aggregates.

      - `per_variable_best` is the analyst cycle (across all
        successful spawn_analyst calls in the trace) with the highest
        match count, with ties broken by earliest cycle. This is a
        diagnostic that surfaces "the analyst got the right answer at
        some point but the workflow regressed on retry" — a real
        failure mode observed on `vector_add` where the first
        analyst's correct downcast verdict was followed by a numerical
        regression and a conservative `keep`-everything retry.

    The two are identical when there is exactly one (or zero)
    successful analyst call, which is the common case.
    """

    path: str
    category: str
    per_variable: PerVariableScore
    per_variable_best: PerVariableScore
    outcome: OutcomeScore
    # Raw analyst verdict (the .result dict from the last spawn_analyst
    # exec_result) for diagnostic dumping. Empty dict if no analyst
    # call was found.
    analyst_verdict: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Trace I/O
# ----------------------------------------------------------------------


def load_trace(path: Path) -> list[dict]:
    """Load a JSONL trace file into a list of records.

    Raises FileNotFoundError if the file is missing and ValueError if
    any line fails to parse as JSON (caller decides whether to wrap
    this into an `error` outcome).
    """
    records: list[dict] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: malformed JSONL record: {exc}"
                ) from exc
    return records


# ----------------------------------------------------------------------
# Per-variable scoring
# ----------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Match aggregator.py's normalization so omitted/duplicate-cased
    variables are treated consistently."""
    return name.strip().lower()


def _collect_analyst_results(records: Iterable[dict]) -> list[dict]:
    """Return the .result payloads of ALL successful spawn_analyst
    calls in trace order.

    The orchestrator may re-spawn the analyst after a comparator
    failure (per the system prompt's retry guidance). Callers that
    want only the last verdict take `results[-1] if results else None`;
    callers that want best-of-K iterate.
    """
    out: list[dict] = []
    for rec in records:
        if rec.get("tool_name") != "spawn_analyst":
            continue
        exec_result = rec.get("exec_result", {})
        if exec_result.get("status") != "ok":
            continue
        result = exec_result.get("result")
        if isinstance(result, dict):
            out.append(result)
    return out


def _score_single_verdict(
    verdict: dict | None, expected_map: dict[str, str]
) -> PerVariableScore:
    """Score one analyst verdict against the expected-variable map.

    `verdict=None` means "no analyst call available" — return
    analyst_absent=True with total matched as 0 and missing_count
    equal to total. Mismatches is left empty in that case so the
    output does not pretend the analyst made K wrong calls when in
    fact it made zero.

    Variables present in the verdict but NOT named in expected_map
    are silently ignored (the deliberate "borderline variables aren't
    scored" policy from expected.py — extra coverage is neither
    rewarded nor punished).
    """
    total = len(expected_map)

    if total == 0:
        return PerVariableScore(
            total=0,
            matched=0,
            mismatches=[],
            missing_count=0,
            analyst_absent=verdict is None,
        )

    if verdict is None:
        return PerVariableScore(
            total=total,
            matched=0,
            mismatches=[],
            missing_count=total,
            analyst_absent=True,
        )

    observed_map: dict[str, str] = {}
    for var in verdict.get("variables", []):
        name = var.get("name")
        action = var.get("action")
        if not isinstance(name, str) or not isinstance(action, str):
            continue
        observed_map[_normalize_name(name)] = action

    matched = 0
    mismatches: list[VariableMismatch] = []
    missing_count = 0
    # Iterate in expected-insertion order so the mismatch list is
    # stable and diff-friendly across runs.
    for norm_name, exp_action in expected_map.items():
        if norm_name not in observed_map:
            mismatches.append(VariableMismatch(
                name=norm_name,
                expected_action=exp_action,
                observed_action="<missing>",
            ))
            missing_count += 1
            continue
        obs_action = observed_map[norm_name]
        if obs_action == exp_action:
            matched += 1
        else:
            mismatches.append(VariableMismatch(
                name=norm_name,
                expected_action=exp_action,
                observed_action=obs_action,
            ))

    return PerVariableScore(
        total=total,
        matched=matched,
        mismatches=mismatches,
        missing_count=missing_count,
        analyst_absent=False,
    )


def score_per_variable(
    records: list[dict], expected: ExpectedKernel
) -> tuple[PerVariableScore, PerVariableScore, dict]:
    """Score the analyst's per-variable verdict against expected.

    Returns (last_cycle_score, best_cycle_score, last_analyst_verdict_dict).

    `last_cycle_score` scores the LAST successful spawn_analyst —
    the verdict the workflow ultimately committed to. This is the
    primary score.

    `best_cycle_score` scores the analyst call with the highest match
    count, with ties broken by earliest cycle (smallest trace index).
    This surfaces "the analyst got it right once but the workflow
    regressed on retry" as a diagnostic distinct from "the analyst
    never got it right". When there are 0 or 1 successful analyst
    calls, best == last by construction.

    The verdict dict (from the LAST analyst) is returned alongside so
    the caller can stash it on KernelResult for diagnostic dumping
    without re-walking the trace.
    """
    expected_map = {
        _normalize_name(k): v for k, v in expected.per_variable.items()
    }

    verdicts = _collect_analyst_results(records)
    last_verdict = verdicts[-1] if verdicts else None

    last_score = _score_single_verdict(last_verdict, expected_map)

    # Best-of-K. When 0 or 1 verdicts exist, best == last by
    # construction — short-circuit to avoid an unnecessary list pass
    # and to make the (best is last) invariant obvious in the trivial
    # cases.
    if len(verdicts) <= 1:
        best_score = last_score
    else:
        # Score every cycle, then pick highest matched, ties broken by
        # earliest cycle (smallest index). enumerate keeps the index
        # paired with the score for the tie-break.
        scored = [
            (i, _score_single_verdict(v, expected_map))
            for i, v in enumerate(verdicts)
        ]
        # max() with key= picks the FIRST element on ties (since
        # iteration is in submission order), which is exactly the
        # "earliest cycle" tie-break we want. Negate index in the key
        # so that "higher matched, smaller index" both push the same
        # direction; equivalently, sort by (-matched, +index) and pick
        # element 0.
        scored.sort(key=lambda pair: (-pair[1].matched, pair[0]))
        best_score = scored[0][1]

    return (last_score, best_score, last_verdict or {})


# ----------------------------------------------------------------------
# Outcome scoring
# ----------------------------------------------------------------------


# Tools that invalidate `last_compare_status` when they run (mirrors
# _FinishGateState.observe in workflow/orchestrator.py — keep these two
# in sync).
_COMPARATOR_INVALIDATING_TOOLS = frozenset({
    "splice_rewritten_kernel",
    "compile_rewritten_driver",
    "run_rewritten_driver",
})


def _walk_gate_state(records: list[dict]) -> tuple[
    str | None, str | None, bool, int, bool
]:
    """Re-derive the finish-gate state by walking the trace.

    Returns (last_verifier_verdict, last_compare_status, finish_seen,
    cycle_count, finish_honored).

    last_verifier_verdict and last_compare_status are the values at the
    moment `finish` was honored (if it was) OR at end-of-trace (if it
    wasn't). This mirrors _FinishGateState semantics: spawn_rewriter
    resets both; splice/compile/run-rewritten reset only
    compare_status; spawn_verifier sets verifier_verdict;
    compare_outputs sets compare_status.

    finish_honored is True iff a `finish` record appears whose
    exec_result indicates success (not a synthesized gate-violation
    error). The orchestrator writes a gate-violation as
    {status:'error', is_error:True, ...} via _append_trace, so we
    distinguish the two by exec_result.status.
    """
    last_verdict: str | None = None
    last_compare: str | None = None
    finish_seen = False
    finish_honored = False
    cycle_count = 0

    for rec in records:
        tool = rec.get("tool_name")
        exec_result = rec.get("exec_result", {}) or {}

        if tool == "finish":
            finish_seen = True
            # Honored finish writes {status:'ok', honored:True};
            # gate-violation writes {status:'error', ...}.
            if exec_result.get("status") == "ok":
                finish_honored = True
                # Stop walking: state at this moment is what mattered.
                break
            # Otherwise it was a synthesized gate-violation; keep
            # walking, the orchestrator may self-correct and try
            # again.
            continue

        if tool == "spawn_rewriter":
            cycle_count += 1
            last_verdict = None
            last_compare = None
            continue
        if tool in _COMPARATOR_INVALIDATING_TOOLS:
            last_compare = None
            continue
        if tool == "spawn_verifier":
            if exec_result.get("status") == "ok":
                result = exec_result.get("result", {}) or {}
                last_verdict = result.get("verdict")
            continue
        if tool == "compare_outputs":
            last_compare = exec_result.get("status")
            continue
        # Other tools (advisor, analyst, harness, compile/run baseline)
        # do not affect the gate. Unknown tool names are silently
        # ignored here; the caller can audit via _KNOWN_TOOLS if
        # desired.

    return last_verdict, last_compare, finish_seen, cycle_count, finish_honored


def _comparator_was_applicable(records: list[dict]) -> bool:
    """True iff the trace contains any baseline_harness or
    compare_outputs call — i.e. the workflow at least attempted the
    dynamic-verification chain. We infer applicability from trace
    contents rather than from the language profile so this module
    stays independent of workflow.*.

    Note: a kernel whose profile has dynamic_verification=True but
    whose harness call was REJECTED before the chain started will
    appear as "inapplicable" here. That's an acceptable approximation
    for v0 — all current profiles with dynamic_verification=True
    (Kokkos, CUDA) emit a harness call, and a rejected harness call
    still leaves a record in the trace under --auto (auto-mode cannot
    reject).
    """
    for rec in records:
        tool = rec.get("tool_name")
        if tool in ("spawn_baseline_harness", "compare_outputs"):
            return True
    return False


def score_outcome(records: list[dict]) -> OutcomeScore:
    """Classify the run's final outcome from the trace."""
    (
        _last_verdict,
        last_compare,
        finish_seen,
        cycle_count,
        finish_honored,
    ) = _walk_gate_state(records)

    comparator_applicable = _comparator_was_applicable(records)

    if not finish_honored:
        # Includes finish_seen-but-rejected (only gate violations) and
        # finish-never-seen (turn limit, crash, etc.).
        return OutcomeScore(
            outcome="fail_no_finish",
            reached_finish=False,
            comparator_status=last_compare,
            cycle_count=cycle_count,
            comparator_inapplicable=not comparator_applicable,
        )

    # finish was honored. Two sub-cases:
    if not comparator_applicable:
        # Profile without dynamic verification (none today, reserved
        # for future). Honored finish == pass.
        return OutcomeScore(
            outcome="pass",
            reached_finish=True,
            comparator_status=None,
            cycle_count=cycle_count,
            comparator_inapplicable=True,
        )

    # Dynamic-verification profile. The gate should have required
    # compare_status='ok' for finish to be honored. If we see anything
    # else here, the gate has a bug (or the trace is from a workflow
    # version that didn't have the gate yet) — surface it as
    # fail_comparator so it doesn't quietly count as pass.
    if last_compare == "ok":
        return OutcomeScore(
            outcome="pass",
            reached_finish=True,
            comparator_status="ok",
            cycle_count=cycle_count,
            comparator_inapplicable=False,
        )
    return OutcomeScore(
        outcome="fail_comparator",
        reached_finish=True,
        comparator_status=last_compare,
        cycle_count=cycle_count,
        comparator_inapplicable=False,
    )


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------


def score_trace(
    records: list[dict], expected: ExpectedKernel
) -> KernelResult:
    """Score a parsed trace against the expected verdicts for one kernel."""
    per_var_last, per_var_best, verdict = score_per_variable(records, expected)
    outcome = score_outcome(records)
    return KernelResult(
        path=expected.path,
        category=expected.category,
        per_variable=per_var_last,
        per_variable_best=per_var_best,
        outcome=outcome,
        analyst_verdict=verdict,
    )


def score_trace_file(
    trace_path: Path, expected: ExpectedKernel
) -> KernelResult:
    """Convenience: load and score in one call, mapping I/O errors to
    an `error` outcome rather than raising. Per-variable score is
    zeroed out (matched=0, missing_count=total) when the trace is
    missing or malformed, since the analyst could not have run.
    """
    try:
        records = load_trace(trace_path)
    except (FileNotFoundError, ValueError):
        total = len(expected.per_variable)
        absent_score = PerVariableScore(
            total=total,
            matched=0,
            mismatches=[],
            missing_count=total,
            analyst_absent=True,
        )
        # On a missing/malformed trace there is no analyst at all, so
        # best and last collapse to the same all-missing score.
        return KernelResult(
            path=expected.path,
            category=expected.category,
            per_variable=absent_score,
            per_variable_best=absent_score,
            outcome=OutcomeScore(
                outcome="error",
                reached_finish=False,
                comparator_status=None,
                cycle_count=0,
                comparator_inapplicable=False,
            ),
            analyst_verdict={},
        )
    return score_trace(records, expected)
