"""Layer 2 aggregator: turn a run.py --output JSON file into a
human-readable report (and optionally a smaller aggregated JSON).

This module is a pure aggregator — it consumes the JSON written by
`evals/layer2/run.py --output PATH` and produces a summary. It does
NOT re-invoke the workflow, re-score traces, or read the raw
`baselines/<stem>/orchestrator_trace.jsonl` files. All scoring lives
in `score.py`; report.py just shapes and prints what run.py already
computed.

CLI:
    python -m evals.layer2.report INPUT_JSON [--output PATH]

Exit code:
  - 0 if every kernel scored as `pass` AND no regressions
    (`per_variable_best.matched > per_variable.matched`) are present.
  - 1 if any kernel scored as a non-`pass` outcome OR any kernel
    regressed from a better earlier analyst cycle.
  - 2 on argument / configuration errors (missing file, malformed
    JSON, unknown schema_version).

Schema compatibility: report.py refuses unknown schema_versions. v2
is the only currently-recognized schema (per run.py:_serialize_results
docstring). If run.py bumps the schema, update _SUPPORTED_SCHEMA_VERSIONS
in the same change.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Bump this set when run.py adds a new schema version that report.py
# knows how to consume. Removing a version is a breaking change for
# operators with old JSON files; prefer to keep older versions in this
# set and branch on `schema_version` inside the loader if/when the
# shape diverges.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({2})

# Outcome -> short uppercase label, matching run.py's _OUTCOME_LABEL.
# Duplicated rather than imported so report.py is a clean downstream
# consumer of the JSON; the JSON itself uses the lowercase outcome
# strings ("pass", "fail_comparator", ...) which are the stable
# contract between the two modules.
_OUTCOME_LABEL = {
    "pass": "PASS",
    "fail_comparator": "FCMP",
    "fail_no_finish": "FNOF",
    "error": "ERR",
}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """Aggregated statistics across all runs in a results file.

    `outcome_counts` maps outcome string -> count. `category_counts`
    maps (category, outcome) -> count; useful for spotting whether
    lowerable kernels pass at a different rate than needs_precision
    ones. `per_variable_totals` sums matched/total across every run
    that had an analyst verdict, so the operator gets a single
    "M/N variables correct overall" number; `analyst_absent_count`
    is the number of runs with no analyst call (excluded from
    per_variable_totals).

    `regressions` lists the runs where `per_variable_best.matched`
    exceeded `per_variable.matched` — i.e. the analyst had a better
    answer in an earlier cycle and self-corrected the wrong way.
    """

    total: int
    outcome_counts: dict[str, int]
    category_counts: dict[tuple[str, str], int]
    per_variable_total: int  # sum of `total` across runs with analyst
    per_variable_matched: int  # sum of `matched` across same
    per_variable_matched_best: int  # sum of best-cycle matched
    analyst_absent_count: int
    regressions: list[dict]  # subset of `runs` list; same dict shape


def aggregate(payload: dict) -> Aggregate:
    """Compute Aggregate from a loaded run.py JSON payload.

    The payload must already have been validated by `load_results`;
    this function assumes schema_version is supported and the shape
    matches run.py:_serialize_results.
    """
    runs = payload["runs"]
    outcome_counts: dict[str, int] = {}
    category_counts: dict[tuple[str, str], int] = {}
    pv_total = 0
    pv_matched = 0
    pv_matched_best = 0
    analyst_absent = 0
    regressions: list[dict] = []

    for run in runs:
        outcome = run["outcome"]
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        key = (run["category"], outcome)
        category_counts[key] = category_counts.get(key, 0) + 1

        pv = run["per_variable"]
        pv_best = run["per_variable_best"]
        if pv["analyst_absent"]:
            analyst_absent += 1
        else:
            pv_total += pv["total"]
            pv_matched += pv["matched"]
            pv_matched_best += pv_best["matched"]
            if pv_best["matched"] > pv["matched"]:
                regressions.append(run)

    return Aggregate(
        total=len(runs),
        outcome_counts=outcome_counts,
        category_counts=category_counts,
        per_variable_total=pv_total,
        per_variable_matched=pv_matched,
        per_variable_matched_best=pv_matched_best,
        analyst_absent_count=analyst_absent,
        regressions=regressions,
    )


# ----------------------------------------------------------------------
# Loading & validation
# ----------------------------------------------------------------------


def load_results(path: Path) -> dict:
    """Load a run.py --output JSON file, validating its schema_version
    and top-level shape.

    Raises SystemExit(2) with a human-readable message on:
      - file missing or unreadable,
      - JSON parse error,
      - missing/unknown schema_version,
      - missing required top-level keys ('runs', 'total', 'summary').

    The check is intentionally shallow: per-run shape errors will
    surface as KeyError downstream and that's fine — the operator is
    closer to the JSON-producer (run.py) than to a network boundary.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SystemExit(f"report.py: input file not found: {path}")
    except OSError as exc:
        raise SystemExit(f"report.py: cannot read {path}: {exc}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"report.py: {path} is not valid JSON: {exc}")

    if not isinstance(payload, dict):
        raise SystemExit(
            f"report.py: {path} top-level must be a JSON object, "
            f"got {type(payload).__name__}."
        )

    version = payload.get("schema_version")
    if version is None:
        raise SystemExit(
            f"report.py: {path} has no schema_version; was it produced "
            f"by `python -m evals.layer2.run --output`?"
        )
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise SystemExit(
            f"report.py: {path} has schema_version={version}; "
            f"this report.py supports {{{supported}}}. Regenerate the "
            f"results JSON with a matching run.py, or update report.py."
        )

    for key in ("runs", "total", "summary"):
        if key not in payload:
            raise SystemExit(
                f"report.py: {path} is missing required top-level key "
                f"{key!r}."
            )

    if not isinstance(payload["runs"], list):
        raise SystemExit(
            f"report.py: {path} 'runs' must be a list, "
            f"got {type(payload['runs']).__name__}."
        )

    return payload


# ----------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------


def _fmt_percent(num: int, denom: int) -> str:
    """Format `num/denom` as `(NN.N%)` or `(n/a)` when denom is zero."""
    if denom == 0:
        return "(n/a)"
    return f"({100.0 * num / denom:.1f}%)"


def render_text(payload: dict, agg: Aggregate) -> str:
    """Produce the human-readable text report.

    Sections (in order):
      1. Header (total kernel count, source path metadata if present).
      2. Outcome summary (count + percent per outcome).
      3. Per-category breakdown (category x outcome matrix).
      4. Per-variable totals (matched/total, including best-cycle if
         it differs).
      5. Regression list (kernels where best > last).
      6. Failure detail (one block per non-pass run, with mismatches
         and triage hints).

    Sections 5 and 6 are omitted when empty so a clean run renders as
    a short, scannable report.
    """
    lines: list[str] = []

    # 1. Header
    lines.append(f"Layer 2 evaluation report")
    lines.append(f"  Total kernels: {agg.total}")
    lines.append("")

    # 2. Outcome summary
    lines.append("Outcomes:")
    # Stable order: PASS, FCMP, FNOF, ERR (matches _OUTCOME_LABEL
    # insertion order and is the order operators learn to read).
    for outcome, label in _OUTCOME_LABEL.items():
        count = agg.outcome_counts.get(outcome, 0)
        pct = _fmt_percent(count, agg.total)
        lines.append(f"  {label:4s}  {count:4d}  {pct}")
    # Surface any unrecognized outcome strings so a future enum
    # addition doesn't silently vanish from the report.
    unknown = sorted(set(agg.outcome_counts) - set(_OUTCOME_LABEL))
    for outcome in unknown:
        count = agg.outcome_counts[outcome]
        lines.append(f"  ????  {count:4d}  (unknown outcome {outcome!r})")
    lines.append("")

    # 3. Per-category breakdown
    categories = sorted({cat for (cat, _) in agg.category_counts})
    if categories:
        lines.append("By category:")
        # Column header
        header = "  " + " " * 18 + "  " + "  ".join(
            f"{lbl:>4s}" for lbl in _OUTCOME_LABEL.values()
        ) + "  total"
        lines.append(header)
        for cat in categories:
            row_total = sum(
                c for (c_cat, _), c in agg.category_counts.items()
                if c_cat == cat
            )
            cells = []
            for outcome in _OUTCOME_LABEL:
                cells.append(
                    f"{agg.category_counts.get((cat, outcome), 0):4d}"
                )
            lines.append(
                f"  {cat:18s}  " + "  ".join(cells) + f"  {row_total:5d}"
            )
        lines.append("")

    # 4. Per-variable totals
    lines.append("Per-variable verdicts (last analyst cycle):")
    pct = _fmt_percent(agg.per_variable_matched, agg.per_variable_total)
    lines.append(
        f"  matched: {agg.per_variable_matched}/{agg.per_variable_total} {pct}"
    )
    if agg.per_variable_matched_best != agg.per_variable_matched:
        pct_best = _fmt_percent(
            agg.per_variable_matched_best, agg.per_variable_total
        )
        lines.append(
            f"  best:    {agg.per_variable_matched_best}/"
            f"{agg.per_variable_total} {pct_best}"
        )
    if agg.analyst_absent_count:
        lines.append(
            f"  (excluded {agg.analyst_absent_count} run(s) with no "
            f"analyst call)"
        )
    lines.append("")

    # 5. Regressions (best > last)
    if agg.regressions:
        lines.append(
            f"Regressions ({len(agg.regressions)}): last analyst cycle "
            f"scored worse than an earlier cycle"
        )
        for run in agg.regressions:
            pv = run["per_variable"]
            pv_best = run["per_variable_best"]
            lines.append(
                f"  {run['path']}  last={pv['matched']}/{pv['total']}  "
                f"best={pv_best['matched']}/{pv_best['total']}  "
                f"cycles={run['cycle_count']}"
            )
        lines.append("")

    # 6. Failure detail
    failures = [r for r in payload["runs"] if r["outcome"] != "pass"]
    if failures:
        lines.append(f"Failures ({len(failures)}):")
        for run in failures:
            label = _OUTCOME_LABEL.get(run["outcome"], "????")
            lines.append(f"  [{label}] {run['path']}")
            extras: list[str] = []
            if run["timed_out"]:
                extras.append("subprocess timed out")
            elif run["returncode"] not in (0, None):
                extras.append(f"returncode={run['returncode']}")
            if not run["trace_exists"]:
                extras.append("no trace file written")
            if run["per_variable"]["analyst_absent"] and run["trace_exists"]:
                extras.append("trace had no analyst call")
            comp = run["comparator_status"]
            if comp is not None and comp != "ok":
                extras.append(f"comparator status={comp!r}")
            if extras:
                lines.append("    " + "; ".join(extras))
            for mm in run["per_variable"]["mismatches"]:
                lines.append(
                    f"    variable {mm['name']}: "
                    f"expected={mm['expected_action']}, "
                    f"observed={mm['observed_action']}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------
# Aggregated JSON output (--output)
# ----------------------------------------------------------------------


def serialize_aggregate(payload: dict, agg: Aggregate) -> dict:
    """Produce a small JSON-safe summary of the run.

    Schema (top-level):
      {
        "schema_version": 1,
        "source_schema_version": int,    # from input payload
        "total": int,
        "outcome_counts": {outcome: int, ...},
        "category_counts": [
          {"category": str, "outcome": str, "count": int}, ...
        ],
        "per_variable": {
          "total": int,
          "matched_last": int,
          "matched_best": int,
          "analyst_absent_count": int
        },
        "regressions": [
          {"path": str,
           "last_matched": int, "last_total": int,
           "best_matched": int, "best_total": int,
           "cycle_count": int}, ...
        ],
        "failures": [
          {"path": str, "outcome": str, "category": str}, ...
        ]
      }

    This schema is intentionally distinct from run.py's schema; it
    drops the per-run details (argv, stderr_tail, raw analyst_verdict)
    and keeps only the aggregated stats plus minimally-actionable
    failure/regression pointers. Bump report.py's schema_version
    independently of run.py's when this shape changes.
    """
    category_counts_list = [
        {"category": cat, "outcome": outcome, "count": count}
        for (cat, outcome), count in sorted(agg.category_counts.items())
    ]
    regressions = [
        {
            "path": run["path"],
            "last_matched": run["per_variable"]["matched"],
            "last_total": run["per_variable"]["total"],
            "best_matched": run["per_variable_best"]["matched"],
            "best_total": run["per_variable_best"]["total"],
            "cycle_count": run["cycle_count"],
        }
        for run in agg.regressions
    ]
    failures = [
        {
            "path": run["path"],
            "outcome": run["outcome"],
            "category": run["category"],
        }
        for run in payload["runs"]
        if run["outcome"] != "pass"
    ]
    return {
        "schema_version": 1,
        "source_schema_version": payload["schema_version"],
        "total": agg.total,
        "outcome_counts": dict(agg.outcome_counts),
        "category_counts": category_counts_list,
        "per_variable": {
            "total": agg.per_variable_total,
            "matched_last": agg.per_variable_matched,
            "matched_best": agg.per_variable_matched_best,
            "analyst_absent_count": agg.analyst_absent_count,
        },
        "regressions": regressions,
        "failures": failures,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.layer2.report",
        description=(
            "Aggregate the JSON written by `python -m evals.layer2.run "
            "--output PATH` into a human-readable report, and optionally "
            "into a smaller aggregated JSON."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        metavar="INPUT_JSON",
        help=(
            "Path to the run.py results JSON (schema_version=2). "
            "Required."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write the aggregated summary JSON to this path. Schema is "
            "documented in evals/layer2/report.py:serialize_aggregate. "
            "When omitted, only the text report is produced."
        ),
    )
    return parser.parse_args(argv)


def _exit_code(agg: Aggregate) -> int:
    """0 iff every kernel passed AND no regressions; 1 otherwise.

    Regressions count as a failure even when outcome=='pass' because
    they signal an analyst-self-correction bug: the workflow had a
    correct answer in an earlier cycle and replaced it with a worse
    one. That's the kind of behavior the eval harness exists to
    surface, so it should fail loudly in CI / batch contexts.
    """
    passes = agg.outcome_counts.get("pass", 0)
    if passes != agg.total:
        return 1
    if agg.regressions:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        payload = load_results(args.input)
    except SystemExit as exc:
        # Re-emit the message to stderr and return 2 so the caller
        # sees a config-error exit code distinct from "ran and
        # something failed" (1).
        msg = exc.code if isinstance(exc.code, str) else str(exc)
        print(msg, file=sys.stderr)
        return 2

    agg = aggregate(payload)
    sys.stdout.write(render_text(payload, agg))
    sys.stdout.flush()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(serialize_aggregate(payload, agg), indent=2)
        )
        print(f"Wrote {args.output}", file=sys.stderr)

    return _exit_code(agg)


if __name__ == "__main__":
    raise SystemExit(main())
