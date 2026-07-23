"""Layer 2 batch driver: run the workflow over every (or a subset of)
test kernel and score the results.

Each kernel is invoked as a subprocess:

    python -m workflow.run <kernel_path> --auto \
        (--sig-figs N | --decimal-digits N)

The subprocess is launched with cwd=repo-root so its baselines/<stem>/
output lands where the workflow normally writes it. After the
subprocess finishes (or times out), this module reads
`baselines/<stem>/orchestrator_trace.jsonl` and scores it via
`score_trace_file`, then streams a one-line summary to stdout.

This module is deliberately separate from `workflow/run.py` (the
single-kernel CLI) so the Layer 2 harness does not depend on any
workflow internals beyond the subprocess contract:

  - argv shape: `python -m workflow.run <kernel> --auto --sig-figs N`
  - trace path: `baselines/<Path(kernel).stem>/orchestrator_trace.jsonl`
  - trace schema: as documented in workflow/orchestrator.py:_append_trace

If any of those three change, update this module in the same change.

CLI:
    python -m evals.layer2.run \
        [--category {lowerable,needs_precision,mixed}]... \
        [--path PATTERN]... \
        [--jobs N] \
        [--output PATH] \
        [--python PATH] \
        [--dry-run]

Filters compose: a kernel runs if it matches ANY --category AND ANY
--path. Both default to "match everything" if not given. `--path`
matches as a case-insensitive substring against the kernel path
(simpler and more predictable than glob; the EXPECTED keys are stable
short strings).

The per-kernel timeout is governed by the
`AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC` env var (default 1800s); it
is intentionally not a CLI flag, mirroring the env-var convention used
by `AGENT_PRECISION_RUN_TIMEOUT_SEC` etc. in workflow/tools.py.

Output:
  - One line per kernel to stdout, real-time (see _format_summary_line).
  - If --output PATH given, writes the full structured results as JSON
    to that path on completion. The schema is documented on
    _serialize_results.

Exit code:
  - 0 if every kernel scored as `pass`.
  - 1 if any kernel scored as `fail_comparator`, `fail_no_finish`, or
    `error`. This matches the convention of pytest and other test
    runners: "did anything go wrong?"
  - 2 on argument / configuration errors before any kernel runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .expected import EXPECTED, ExpectedKernel
from .score import KernelResult, score_trace_file


# Repo root: evals/layer2/run.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_TIMEOUT_SEC = 1800
_TIMEOUT_ENV_VAR = "AGENT_PRECISION_ORCHESTRATOR_TIMEOUT_SEC"

_VALID_CATEGORIES = ("lowerable", "needs_precision", "mixed")


# ----------------------------------------------------------------------
# Per-run record
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """One kernel's invocation outcome.

    `score` is None if the subprocess crashed before producing a
    parseable trace (in that case `score_trace_file` would still
    return outcome='error', but we separately record the subprocess
    failure mode so the operator can distinguish "workflow ran and
    failed" from "the workflow never got to run at all").
    """

    expected: ExpectedKernel
    argv: list[str]
    returncode: int | None  # None iff timed out
    timed_out: bool
    duration_sec: float
    trace_path: Path
    trace_exists: bool
    score: KernelResult
    stderr_tail: str  # last few KB of subprocess stderr, for triage


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.layer2.run",
        description=(
            "Run the precision-rewrite workflow over the test-kernels/ "
            "corpus and score the results against evals/layer2/expected.py."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=_VALID_CATEGORIES,
        default=None,
        metavar="NAME",
        help=(
            "Restrict to kernels in this category. Repeatable; the "
            "intersection of all --category flags is taken. If omitted, "
            "all categories run."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Restrict to kernels whose path contains this substring "
            "(case-insensitive). Repeatable; a kernel must match at "
            "least one --path. If omitted, all paths match."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of kernels to run in parallel. Default 1 (sequential). "
            "Each job is a thread that blocks on `subprocess.run`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write structured JSON results to this path on completion. "
            "The schema is documented in evals/layer2/run.py:_serialize_results."
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        metavar="PATH",
        help=(
            "Python interpreter to use for the workflow subprocess. "
            f"Default: the interpreter running this script ({sys.executable})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the kernels that WOULD run (with their argv and "
            "tolerance) and exit without invoking any subprocess."
        ),
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def select_kernels(
    expected: dict[str, ExpectedKernel],
    categories: list[str] | None,
    path_substrings: list[str] | None,
) -> list[ExpectedKernel]:
    """Filter EXPECTED by category and path substring.

    Both filters default to "match everything" when None or empty. The
    result preserves EXPECTED's insertion order so runs are
    reproducible.
    """
    cat_filter = set(categories) if categories else None
    needles = (
        [s.lower() for s in path_substrings] if path_substrings else None
    )
    out: list[ExpectedKernel] = []
    for entry in expected.values():
        if cat_filter is not None and entry.category not in cat_filter:
            continue
        if needles is not None and not any(n in entry.path.lower() for n in needles):
            continue
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# Subprocess construction
# ----------------------------------------------------------------------


def build_argv(
    python: Path, kernel: ExpectedKernel
) -> list[str]:
    """Construct the argv to invoke the workflow on one kernel.

    Always passes --auto (the only sane mode for batch runs) and the
    appropriate tolerance flag from `kernel.tolerance_kind`. The kernel
    path is forwarded verbatim — the workflow CLI accepts it as a
    relative path, which is what EXPECTED stores.
    """
    if kernel.tolerance_kind == "sig_figs":
        tol_flag = "--sig-figs"
    elif kernel.tolerance_kind == "decimal_digits":
        tol_flag = "--decimal-digits"
    else:
        # Should be impossible thanks to test_expected_entry_is_well_formed,
        # but fail loudly here rather than building a malformed argv.
        raise ValueError(
            f"{kernel.path}: unknown tolerance_kind {kernel.tolerance_kind!r}; "
            "expected 'sig_figs' or 'decimal_digits'."
        )
    return [
        str(python),
        "-m",
        "workflow.run",
        kernel.path,
        "--auto",
        tol_flag,
        str(kernel.tolerance_value),
    ]


def _resolve_timeout_sec() -> int:
    """Read the per-kernel timeout from the env var, defaulting to
    _DEFAULT_TIMEOUT_SEC.

    A non-int or non-positive value is rejected with SystemExit(2) —
    same convention as `workflow/run.py` uses for its tolerance flags.
    Mirrors AGENT_PRECISION_RUN_TIMEOUT_SEC's parsing strictness in
    workflow/tools.py.
    """
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is None or raw == "":
        return _DEFAULT_TIMEOUT_SEC
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"{_TIMEOUT_ENV_VAR}={raw!r} is not an integer; "
            f"set it to a positive number of seconds or unset it "
            f"to use the default ({_DEFAULT_TIMEOUT_SEC})."
        )
    if value <= 0:
        raise SystemExit(
            f"{_TIMEOUT_ENV_VAR}={raw!r} must be > 0; "
            f"got {value}."
        )
    return value


# ----------------------------------------------------------------------
# Trace location
# ----------------------------------------------------------------------


def trace_path_for(kernel: ExpectedKernel, repo_root: Path) -> Path:
    """Where the workflow subprocess writes the trace for this kernel.

    The orchestrator builds this path itself in
    workflow/orchestrator.py:run_orchestrator as
    `baselines/<Path(kernel_path).stem>/orchestrator_trace.jsonl`, with
    the `baselines/` prefix interpreted relative to the SUBPROCESS's
    cwd. Since we launch from repo_root, the absolute path is just
    `repo_root / baselines / <stem> / orchestrator_trace.jsonl`.

    If the workflow's trace-path convention ever changes, fix it here
    and update the docstring comment in workflow/orchestrator.py that
    documents it.
    """
    stem = Path(kernel.path).stem
    return repo_root / "baselines" / stem / "orchestrator_trace.jsonl"


# ----------------------------------------------------------------------
# Per-kernel execution
# ----------------------------------------------------------------------


# Default factory for the subprocess runner. Pulled out as a parameter
# so tests can inject a mock without monkeypatching the subprocess
# module globally (and so the production call site is a single,
# trivially-mockable callable).
def _default_subprocess_runner(
    argv: list[str], cwd: Path, timeout_sec: int
) -> tuple[int | None, bool, str]:
    """Run `argv` in `cwd` with the given timeout.

    Returns (returncode, timed_out, stderr_tail). On timeout,
    returncode is None and timed_out is True; the partial stderr
    captured by Popen is included in the tail. On any other error
    (FileNotFoundError, OSError) we re-raise — those indicate a broken
    invocation environment, not a kernel-level failure, and should
    halt the batch.
    """
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # exc.stderr may be bytes or None; normalize to str.
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            try:
                stderr = stderr.decode("utf-8", errors="replace")
            except Exception:
                stderr = ""
        return None, True, _tail(stderr)
    return completed.returncode, False, _tail(completed.stderr or "")


def _tail(text: str, max_bytes: int = 4000) -> str:
    """Return the last max_bytes characters of `text`. The stderr_tail
    on RunRecord is for triage; the full stderr would bloat the JSON
    output and isn't useful when the trace itself is the source of
    truth for scoring."""
    if len(text) <= max_bytes:
        return text
    return "...[truncated]...\n" + text[-max_bytes:]


def run_one(
    kernel: ExpectedKernel,
    python: Path,
    repo_root: Path,
    timeout_sec: int,
    subprocess_runner=_default_subprocess_runner,
) -> RunRecord:
    """Run the workflow on one kernel and score the resulting trace."""
    argv = build_argv(python, kernel)
    trace_path = trace_path_for(kernel, repo_root)

    start = time.monotonic()
    returncode, timed_out, stderr_tail = subprocess_runner(
        argv, repo_root, timeout_sec
    )
    duration = time.monotonic() - start

    trace_exists = trace_path.exists()
    score = score_trace_file(trace_path, kernel)

    return RunRecord(
        expected=kernel,
        argv=argv,
        returncode=returncode,
        timed_out=timed_out,
        duration_sec=duration,
        trace_path=trace_path,
        trace_exists=trace_exists,
        score=score,
        stderr_tail=stderr_tail,
    )


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


# Outcome -> short uppercase label for one-line summaries. The padding
# matters: keeping each label 4 chars wide makes the streamed output
# column-aligned without a format() width spec.
_OUTCOME_LABEL = {
    "pass": "PASS",
    "fail_comparator": "FCMP",
    "fail_no_finish": "FNOF",
    "error": "ERR ",
}


def _format_summary_line(idx: int, total: int, rec: RunRecord) -> str:
    """One-line summary printed to stdout per kernel.

    Format:
        [i/total] LABEL  path  vars=M/N[ (best B/N)]  cycles=K  (S.Ts)  [<extra>]?

    where:
      LABEL    = PASS | FCMP | FNOF | ERR
      M/N      = matched/total per-variable verdicts on the LAST
                 analyst cycle (the verdict the workflow committed to)
      B/N      = best per-variable match across ALL analyst cycles;
                 emitted in parentheses only when B != M (i.e. the
                 workflow regressed from a better earlier verdict).
                 This is a diagnostic for analyst self-correction
                 going the wrong direction — see
                 KernelResult.per_variable_best for the rationale.
      K        = number of spawn_rewriter calls
      S.T      = wall-clock seconds
      <extra>  = optional notes ("timeout", "rc=N", "no trace", ...)
                 surfaced only when they add info beyond LABEL.
    """
    score = rec.score
    label = _OUTCOME_LABEL.get(score.outcome.outcome, "????")
    extras: list[str] = []
    if rec.timed_out:
        extras.append("timeout")
    elif rec.returncode not in (0, None):
        extras.append(f"rc={rec.returncode}")
    if not rec.trace_exists:
        extras.append("no trace")
    if score.per_variable.analyst_absent and rec.trace_exists:
        extras.append("no analyst call")

    extra_str = f"  [{', '.join(extras)}]" if extras else ""
    pv = score.per_variable
    pv_best = score.per_variable_best
    # Only annotate when best differs from last; suppress when equal
    # (common case: 0 or 1 analyst cycles, or every cycle matched
    # equally well) to keep the line scannable.
    best_str = (
        f" (best {pv_best.matched}/{pv_best.total})"
        if pv_best.matched != pv.matched
        else ""
    )
    return (
        f"[{idx}/{total}] {label}  {rec.expected.path}  "
        f"vars={pv.matched}/{pv.total}{best_str}  "
        f"cycles={score.outcome.cycle_count}  "
        f"({rec.duration_sec:.1f}s){extra_str}"
    )


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def _serialize_results(records: list[RunRecord]) -> dict:
    """Produce a JSON-safe dict for --output.

    Schema (top-level):
      {
        "schema_version": 2,
        "total": int,
        "summary": {<outcome>: int, ...},
        "runs": [<run>, ...]
      }

    Each <run> mirrors RunRecord but with Paths and dataclasses
    flattened to plain dicts/strings:
      {
        "path": str,
        "category": str,
        "tolerance_kind": str,
        "tolerance_value": int,
        "argv": [str, ...],
        "returncode": int | null,
        "timed_out": bool,
        "duration_sec": float,
        "trace_path": str,
        "trace_exists": bool,
        "outcome": str,           # "pass" | "fail_comparator" | ...
        "reached_finish": bool,
        "comparator_status": str | null,
        "cycle_count": int,
        "comparator_inapplicable": bool,
        "per_variable": {            # LAST analyst cycle
          "total": int,
          "matched": int,
          "missing_count": int,
          "analyst_absent": bool,
          "mismatches": [
            {"name": str, "expected_action": str, "observed_action": str},
            ...
          ]
        },
        "per_variable_best": {       # BEST analyst cycle (highest
          ...                        # matched, ties broken by earliest;
        },                           # same shape as per_variable)
        "analyst_verdict": <dict>,   # raw last-analyst .result payload
        "stderr_tail": str
      }

    schema_version is bumped if this dict shape changes in a
    non-additive way; report.py should refuse unknown schema_versions.
    v2 added `per_variable_best` (additive in shape but semantically
    new), bumped so downstream consumers know to expect it.
    """
    def _pv_dict(pv) -> dict:
        return {
            "total": pv.total,
            "matched": pv.matched,
            "missing_count": pv.missing_count,
            "analyst_absent": pv.analyst_absent,
            "mismatches": [dataclasses.asdict(m) for m in pv.mismatches],
        }

    summary: dict[str, int] = {}
    runs: list[dict] = []
    for rec in records:
        score = rec.score
        out = score.outcome
        outcome = out.outcome
        summary[outcome] = summary.get(outcome, 0) + 1
        runs.append({
            "path": rec.expected.path,
            "category": rec.expected.category,
            "tolerance_kind": rec.expected.tolerance_kind,
            "tolerance_value": rec.expected.tolerance_value,
            "argv": list(rec.argv),
            "returncode": rec.returncode,
            "timed_out": rec.timed_out,
            "duration_sec": rec.duration_sec,
            "trace_path": str(rec.trace_path),
            "trace_exists": rec.trace_exists,
            "outcome": outcome,
            "reached_finish": out.reached_finish,
            "comparator_status": out.comparator_status,
            "cycle_count": out.cycle_count,
            "comparator_inapplicable": out.comparator_inapplicable,
            "per_variable": _pv_dict(score.per_variable),
            "per_variable_best": _pv_dict(score.per_variable_best),
            "analyst_verdict": score.analyst_verdict,
            "stderr_tail": rec.stderr_tail,
        })
    return {
        "schema_version": 2,
        "total": len(records),
        "summary": summary,
        "runs": runs,
    }


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def _run_batch(
    kernels: list[ExpectedKernel],
    python: Path,
    repo_root: Path,
    timeout_sec: int,
    jobs: int,
    on_complete,
    subprocess_runner=_default_subprocess_runner,
) -> list[RunRecord]:
    """Run all `kernels`, dispatching `jobs` at a time.

    `on_complete(idx, total, rec)` is called once per kernel as it
    finishes, in completion order (NOT submission order). The returned
    list is in submission order, so the JSON output is reproducible
    even when --jobs > 1.
    """
    total = len(kernels)
    results: list[RunRecord | None] = [None] * total

    if jobs <= 1:
        # Sequential path: no thread pool. Keeps tracebacks clean and
        # makes the --jobs=1 default trivially debuggable.
        for idx, kernel in enumerate(kernels, start=1):
            rec = run_one(
                kernel, python, repo_root, timeout_sec,
                subprocess_runner=subprocess_runner,
            )
            results[idx - 1] = rec
            on_complete(idx, total, rec)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_idx = {
                pool.submit(
                    run_one, kernel, python, repo_root, timeout_sec,
                    subprocess_runner,
                ): i
                for i, kernel in enumerate(kernels)
            }
            completed = 0
            for fut in as_completed(future_to_idx):
                completed += 1
                idx = future_to_idx[fut]
                rec = fut.result()
                results[idx] = rec
                # `completed` is the completion-order index for the
                # one-liner; the submission-order slot is still
                # `idx + 1` (1-based) but the operator cares about
                # progress-as-it-happens here.
                on_complete(completed, total, rec)

    # mypy/lint: we filled every slot.
    return [r for r in results if r is not None]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.jobs < 1:
        print(f"--jobs must be >= 1, got {args.jobs}", file=sys.stderr)
        return 2

    try:
        timeout_sec = _resolve_timeout_sec()
    except SystemExit as exc:
        # _resolve_timeout_sec raises SystemExit(message); re-emit and
        # return 2 so the caller sees a config error code.
        print(exc.code if isinstance(exc.code, str) else str(exc), file=sys.stderr)
        return 2

    kernels = select_kernels(EXPECTED, args.category, args.path)
    if not kernels:
        print(
            "No kernels matched the given filters; nothing to do.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        for k in kernels:
            print(f"{k.path}  ({k.category}, "
                  f"{k.tolerance_kind}={k.tolerance_value})")
            print("  argv: " + " ".join(build_argv(args.python, k)))
        return 0

    print(
        f"Running {len(kernels)} kernel(s) with --jobs={args.jobs}, "
        f"timeout={timeout_sec}s, cwd={_REPO_ROOT}",
        file=sys.stderr,
    )

    def _on_complete(idx: int, total: int, rec: RunRecord) -> None:
        print(_format_summary_line(idx, total, rec), flush=True)

    records = _run_batch(
        kernels=kernels,
        python=args.python,
        repo_root=_REPO_ROOT,
        timeout_sec=timeout_sec,
        jobs=args.jobs,
        on_complete=_on_complete,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_serialize_results(records), indent=2))
        print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate summary to stderr so stdout stays clean for piping the
    # one-liners.
    counts: dict[str, int] = {}
    for rec in records:
        outcome = rec.score.outcome.outcome
        counts[outcome] = counts.get(outcome, 0) + 1
    summary = "  ".join(
        f"{label}={counts.get(out, 0)}"
        for out, label in _OUTCOME_LABEL.items()
    )
    print(f"\n=== Summary: {summary} ===", file=sys.stderr)

    return 0 if counts.get("pass", 0) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
