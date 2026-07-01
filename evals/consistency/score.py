"""Run-to-run consistency scoring across N orchestrator traces.

Answers the question: "I ran the same kernel with the same flags N
times. Did I get the same verdict each time?"

Reads N JSONL traces (one per run, produced by `python -m workflow.run
--auto`), extracts three signals per run, and emits an agreement
report. The three signals were chosen to map directly onto the three
variance sources documented in AGENTS.md (analyst per-variable
verdicts, verifier accept/reject, orchestrator retry-cycle count):

  - per_variable_actions: dict[var_name -> action]. Source is the LAST
    successful spawn_verifier's `per_variable[*].observed_action` field
    (the verifier reliably emits a structured list per
    VERIFIER_OUTPUT_SCHEMA; the analyst's top-level `variables` field
    has been observed malformed in real traces, so reading the verifier
    is the more robust signal). Falls back to the LAST successful
    spawn_analyst's `variables[*].action` when no verifier ran, and to
    {} when neither ran.

  - finish_outcome: 'pass' | 'fail_comparator' | 'fail_no_finish' |
    'error'. Same enum and the same derivation rules as
    evals/layer2/score.py:score_outcome (re-implemented here, not
    imported, because Layer 2 takes an ExpectedKernel which is not
    meaningful for this question \u2014 we want raw outcome only).

  - cycle_count: int. Number of spawn_rewriter calls in the trace.
    Proxy for how many rewrite-retry trips the orchestrator needed
    before reaching finish (or giving up).

The script is invoked with explicit trace paths:

    python -m evals.consistency.score \\
        runs/1/trace.jsonl runs/2/trace.jsonl runs/3/trace.jsonl

No conventions about where traces live; the caller stashes them. This
module makes zero network calls, does not import from workflow.*, and
does not write anywhere. Output goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Outcome enum mirrors evals/layer2/score.py:Outcome. Re-declared
# rather than imported to keep this module independent of layer2 \u2014
# the two address different questions and we don't want a cross-import
# coupling them.
Outcome = Literal["pass", "fail_comparator", "fail_no_finish", "error"]

# Tools that invalidate the tracked compare_status when they run.
# Mirrors _FinishGateState.observe in workflow/orchestrator.py and
# evals/layer2/score.py:_COMPARATOR_INVALIDATING_TOOLS. If you change
# the gate, change all three.
_COMPARATOR_INVALIDATING_TOOLS = frozenset({
    "splice_rewritten_kernel",
    "compile_rewritten_driver",
    "run_rewritten_driver",
})


# ----------------------------------------------------------------------
# Per-run extraction
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunSignals:
    """The three signals extracted from one trace."""

    trace_path: str
    # Empty dict iff no analyst AND no verifier produced a usable
    # verdict. Variable names are normalized lowercase+stripped so
    # cross-run comparison doesn't trip on cosmetic differences (the
    # analyst aggregator uses the same normalization).
    per_variable_actions: dict[str, str]
    # Provenance of per_variable_actions: 'verifier' | 'analyst' |
    # 'none'. Surfaced so a divergent run that fell back to the
    # analyst is visible in the report (mixing sources would silently
    # bias the agreement number).
    actions_source: str
    finish_outcome: Outcome
    # Mirrors evals/layer2 OutcomeScore.comparator_status: status of
    # the most recent compare_outputs call in the same rewrite cycle
    # as `finish` (or at end-of-trace if finish never ran). None if
    # never called or invalidated.
    comparator_status: str | None
    cycle_count: int
    # True if the run reached an honored `finish` (exec_result.status
    # == 'ok'). Distinct from finish_outcome=='pass' because a profile
    # without dynamic verification can pass without a comparator.
    reached_finish: bool


def _normalize_name(name: str) -> str:
    """Match aggregator.py / evals/layer2/score.py normalization."""
    return name.strip().lower()


def _load_trace(path: Path) -> list[dict]:
    """Load a JSONL trace into a list of records.

    Returns [] for a missing/empty/malformed file rather than raising,
    so callers can map "no trace" onto a clean 'error' RunSignals
    without try/except scaffolding. The empty-list case is
    indistinguishable from a trace with zero records; both correctly
    score as outcome='fail_no_finish' (no finish was ever honored).
    """
    if not path.is_file():
        return []
    records: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _extract_verifier_actions(records: list[dict]) -> dict[str, str]:
    """Per-variable actions from the LAST successful spawn_verifier.

    Reads `per_variable[*].observed_action` (per VERIFIER_OUTPUT_SCHEMA
    in workflow/registry.py). Skips entries missing `name` or
    `observed_action`. Returns {} when no successful verifier call
    appears.
    """
    last_verifier: dict | None = None
    for rec in records:
        if rec.get("tool_name") != "spawn_verifier":
            continue
        exec_result = rec.get("exec_result", {}) or {}
        if exec_result.get("status") != "ok":
            continue
        result = exec_result.get("result")
        if isinstance(result, dict):
            last_verifier = result

    if last_verifier is None:
        return {}

    actions: dict[str, str] = {}
    for var in last_verifier.get("per_variable", []) or []:
        if not isinstance(var, dict):
            continue
        name = var.get("name")
        action = var.get("observed_action")
        if isinstance(name, str) and isinstance(action, str):
            actions[_normalize_name(name)] = action
    return actions


def _extract_analyst_actions(records: list[dict]) -> dict[str, str]:
    """Per-variable actions from the LAST successful spawn_analyst.

    Reads `variables[*].action` (per ANALYST_OUTPUT_SCHEMA). Fallback
    source used only when the verifier produced nothing. Resilient to
    malformed payloads where `variables` is absent or not a list (a
    real failure mode observed in baselines/nbody_force/).
    """
    last_analyst: dict | None = None
    for rec in records:
        if rec.get("tool_name") != "spawn_analyst":
            continue
        exec_result = rec.get("exec_result", {}) or {}
        if exec_result.get("status") != "ok":
            continue
        result = exec_result.get("result")
        if isinstance(result, dict):
            last_analyst = result

    if last_analyst is None:
        return {}

    variables = last_analyst.get("variables")
    if not isinstance(variables, list):
        return {}

    actions: dict[str, str] = {}
    for var in variables:
        if not isinstance(var, dict):
            continue
        name = var.get("name")
        action = var.get("action")
        if isinstance(name, str) and isinstance(action, str):
            actions[_normalize_name(name)] = action
    return actions


def _score_outcome(records: list[dict]) -> tuple[
    Outcome, str | None, bool, int
]:
    """Walk the trace and classify the final outcome.

    Returns (outcome, comparator_status, reached_finish, cycle_count).

    Mirrors evals/layer2/score.py:score_outcome semantics:
      - spawn_rewriter resets both tracked verifier_verdict and
        compare_status (it starts a new rewrite cycle).
      - splice/compile/run-rewritten invalidate compare_status only
        (they overwrite files the comparator reads).
      - 'pass' requires honored finish AND (comparator inapplicable OR
        compare_status == 'ok').
      - 'fail_comparator' is honored finish with comparator applicable
        but compare_status != 'ok'. This means the code-side gate let
        a bad rewrite through \u2014 surface it loudly rather than count
        as pass.
      - 'fail_no_finish' is everything else (turn limit, gate kept
        rejecting, etc.).

    'error' is reserved for callers handling missing trace files; this
    function does not return 'error' for any well-formed (possibly
    empty) record list.
    """
    last_compare: str | None = None
    finish_honored = False
    cycle_count = 0

    # Applicability is a pure existence check independent of the gate
    # walk; do it as a single scan up front so the loop below stays
    # focused on the gate-state machine.
    comparator_applicable = any(
        rec.get("tool_name") in ("spawn_baseline_harness", "compare_outputs")
        for rec in records
    )

    for rec in records:
        tool = rec.get("tool_name")
        exec_result = rec.get("exec_result", {}) or {}

        if tool == "finish":
            if exec_result.get("status") == "ok":
                finish_honored = True
                break
            # Synthesized gate-violation; orchestrator may try again.
            continue

        if tool == "spawn_rewriter":
            cycle_count += 1
            last_compare = None
            continue
        if tool in _COMPARATOR_INVALIDATING_TOOLS:
            last_compare = None
            continue
        if tool == "compare_outputs":
            last_compare = exec_result.get("status")
            continue
        # Other tools (advisor, analyst, verifier, harness, baseline
        # compile/run, probe tools) do not affect the finish gate.

    if not finish_honored:
        return ("fail_no_finish", last_compare, False, cycle_count)

    if not comparator_applicable:
        return ("pass", None, True, cycle_count)

    if last_compare == "ok":
        return ("pass", "ok", True, cycle_count)

    return ("fail_comparator", last_compare, True, cycle_count)


def extract_run_signals(trace_path: Path) -> RunSignals:
    """Load one trace and return the three consistency signals.

    A missing or unreadable trace becomes outcome='error' with empty
    actions; the report counts these separately so a flaky disk or a
    forgotten `--auto` doesn't get scored as a verdict regression.
    """
    records = _load_trace(trace_path)

    if not records:
        return RunSignals(
            trace_path=str(trace_path),
            per_variable_actions={},
            actions_source="none",
            finish_outcome="error",
            comparator_status=None,
            cycle_count=0,
            reached_finish=False,
        )

    actions = _extract_verifier_actions(records)
    source = "verifier"
    if not actions:
        actions = _extract_analyst_actions(records)
        source = "analyst" if actions else "none"

    outcome, comparator_status, reached_finish, cycle_count = _score_outcome(
        records
    )

    return RunSignals(
        trace_path=str(trace_path),
        per_variable_actions=actions,
        actions_source=source,
        finish_outcome=outcome,
        comparator_status=comparator_status,
        cycle_count=cycle_count,
        reached_finish=reached_finish,
    )


# ----------------------------------------------------------------------
# Cross-run aggregation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VariableAgreement:
    """Per-variable agreement across N runs.

    `actions_by_run` is parallel to the input run list (length N);
    each entry is the action that run assigned, or '<missing>' if the
    run's per_variable_actions did not include this variable.

    `unanimous` is True iff every run assigned the same non-'<missing>'
    action. A variable that one run missed and the others agreed on
    is NOT unanimous \u2014 missing coverage is itself a consistency
    failure.
    """

    name: str
    actions_by_run: list[str]
    unanimous: bool
    distinct_actions: list[str]  # sorted unique non-missing actions


@dataclass(frozen=True)
class ConsistencyReport:
    """Aggregate consistency across N runs of the same kernel."""

    n_runs: int
    runs: list[RunSignals]
    # All variable names seen across any run, sorted for stable
    # presentation. The union (not intersection) so a run that
    # silently dropped a variable shows up as a row of mostly-agreement
    # with one '<missing>' cell.
    variables: list[str]
    variable_agreement: list[VariableAgreement]
    # Counts across runs.
    outcome_distribution: dict[str, int]
    cycle_count_distribution: dict[int, int]
    # True iff every run agreed on every variable (no missing, no
    # divergence) AND every run had the same finish_outcome AND the
    # same cycle_count. The strict "everything matches" headline.
    fully_consistent: bool


def build_report(signals: list[RunSignals]) -> ConsistencyReport:
    """Aggregate per-run signals into a cross-run consistency report."""
    n = len(signals)

    # Union of variable names seen anywhere.
    variables: list[str] = sorted({
        name for s in signals for name in s.per_variable_actions
    })

    variable_agreement: list[VariableAgreement] = []
    all_vars_unanimous = True
    for var in variables:
        actions_by_run = [
            s.per_variable_actions.get(var, "<missing>") for s in signals
        ]
        non_missing = [a for a in actions_by_run if a != "<missing>"]
        # Unanimous requires every run named the variable AND assigned
        # the same action.
        unanimous = (
            len(non_missing) == n
            and len(set(non_missing)) == 1
        )
        if not unanimous:
            all_vars_unanimous = False
        distinct = sorted(set(non_missing))
        variable_agreement.append(VariableAgreement(
            name=var,
            actions_by_run=actions_by_run,
            unanimous=unanimous,
            distinct_actions=distinct,
        ))

    outcome_dist = dict(Counter(s.finish_outcome for s in signals))
    cycle_dist = dict(Counter(s.cycle_count for s in signals))

    outcomes_uniform = len(outcome_dist) == 1
    cycles_uniform = len(cycle_dist) == 1

    fully_consistent = (
        n > 0 and all_vars_unanimous and outcomes_uniform and cycles_uniform
    )

    return ConsistencyReport(
        n_runs=n,
        runs=signals,
        variables=variables,
        variable_agreement=variable_agreement,
        outcome_distribution=outcome_dist,
        cycle_count_distribution=cycle_dist,
        fully_consistent=fully_consistent,
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _render_per_variable_table(report: ConsistencyReport) -> str:
    """Markdown-ish fixed-width table: rows are variables, columns are
    runs. Last column flags non-unanimous rows with an asterisk."""
    if not report.variables:
        return "  (no variables found in any run)\n"

    n = report.n_runs
    # Column widths: variable name, then one column per run, then flag.
    name_w = max(len("variable"), max(len(v) for v in report.variables))
    run_headers = [f"run{i+1}" for i in range(n)]
    # Per-run column width: max of header and any action string seen.
    run_widths = []
    for i, header in enumerate(run_headers):
        col_max = max(
            len(header),
            max(
                (len(va.actions_by_run[i]) for va in report.variable_agreement),
                default=0,
            ),
        )
        run_widths.append(col_max)

    lines = []
    # Header.
    header_cells = [f"{'variable':<{name_w}}"] + [
        f"{h:<{w}}" for h, w in zip(run_headers, run_widths)
    ] + ["agree"]
    lines.append("  " + " | ".join(header_cells))
    lines.append("  " + "-+-".join(
        ["-" * name_w] + ["-" * w for w in run_widths] + ["-----"]
    ))
    # Rows.
    for va in report.variable_agreement:
        cells = [f"{va.name:<{name_w}}"] + [
            f"{a:<{w}}" for a, w in zip(va.actions_by_run, run_widths)
        ]
        cells.append("yes" if va.unanimous else "NO ")
        lines.append("  " + " | ".join(cells))
    return "\n".join(lines) + "\n"


def render_report(report: ConsistencyReport) -> str:
    """Human-readable text report. Stdout-friendly, no colors."""
    out: list[str] = []
    out.append(f"Consistency report across {report.n_runs} run(s)")
    out.append("=" * 60)
    out.append("")

    # Per-run summary table.
    out.append("Per-run summary:")
    out.append(
        f"  {'run':<4} {'outcome':<16} {'cycles':<7} {'cmp':<6} "
        f"{'actions':<9} trace"
    )
    for i, s in enumerate(report.runs):
        cmp_status = s.comparator_status or "-"
        out.append(
            f"  {i+1:<4} {s.finish_outcome:<16} {s.cycle_count:<7} "
            f"{cmp_status:<6} {s.actions_source:<9} {s.trace_path}"
        )
    out.append("")

    out.append("Per-variable agreement (verdict source noted per run above):")
    out.append(_render_per_variable_table(report))

    out.append("Outcome distribution:")
    for outcome, count in sorted(report.outcome_distribution.items()):
        out.append(f"  {outcome:<20} {count}")
    out.append("")

    out.append("Rewriter-cycle distribution:")
    for cycles, count in sorted(report.cycle_count_distribution.items()):
        out.append(f"  cycles={cycles:<3}  {count} run(s)")
    out.append("")

    # Headline.
    if report.fully_consistent:
        out.append("HEADLINE: fully consistent across all runs.")
    else:
        out.append("HEADLINE: divergence detected. See table above.")
        # Surface the WHY: which dimension diverged.
        non_unanimous = [
            va.name for va in report.variable_agreement if not va.unanimous
        ]
        if non_unanimous:
            out.append(
                f"  - per-variable divergence on: {', '.join(non_unanimous)}"
            )
        if len(report.outcome_distribution) > 1:
            out.append(
                f"  - outcome divergence: "
                f"{dict(report.outcome_distribution)}"
            )
        if len(report.cycle_count_distribution) > 1:
            out.append(
                f"  - cycle-count divergence: "
                f"{dict(report.cycle_count_distribution)}"
            )
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.consistency.score",
        description=(
            "Score run-to-run consistency across N orchestrator traces "
            "from the same kernel + same flags. Reads JSONL traces "
            "produced by `python -m workflow.run --auto`; writes a "
            "text report to stdout. Exit 0 if fully consistent, 1 "
            "otherwise."
        ),
    )
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help=(
            "Paths to JSONL trace files, one per run. The caller is "
            "responsible for stashing baselines/<stem>/"
            "orchestrator_trace.jsonl between runs (e.g. via mv)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON alongside the text report. The "
            "text report still goes to stdout; the JSON is appended "
            "after a '---' separator line so a downstream consumer can "
            "split on it. Kept dead-simple instead of split files."
        ),
    )
    args = parser.parse_args(argv)

    if len(args.traces) < 2:
        # N=1 is degenerate (no variance possible), but allowed so the
        # script is usable for smoke-testing. Emit a warning rather
        # than refusing.
        print(
            "WARNING: scoring with N=1 trace; consistency is trivially "
            "true. Run the workflow >=2 times to get a real signal.",
            file=sys.stderr,
        )

    signals = [extract_run_signals(p) for p in args.traces]
    report = build_report(signals)
    print(render_report(report))

    if args.json:
        print("---")
        payload = {
            "n_runs": report.n_runs,
            "fully_consistent": report.fully_consistent,
            "runs": [
                {
                    "trace_path": s.trace_path,
                    "per_variable_actions": s.per_variable_actions,
                    "actions_source": s.actions_source,
                    "finish_outcome": s.finish_outcome,
                    "comparator_status": s.comparator_status,
                    "cycle_count": s.cycle_count,
                    "reached_finish": s.reached_finish,
                }
                for s in report.runs
            ],
            "variables": report.variables,
            "variable_agreement": [
                {
                    "name": va.name,
                    "actions_by_run": va.actions_by_run,
                    "unanimous": va.unanimous,
                    "distinct_actions": va.distinct_actions,
                }
                for va in report.variable_agreement
            ],
            "outcome_distribution": report.outcome_distribution,
            "cycle_count_distribution": {
                str(k): v for k, v in report.cycle_count_distribution.items()
            },
        }
        print(json.dumps(payload, indent=2))

    return 0 if report.fully_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
