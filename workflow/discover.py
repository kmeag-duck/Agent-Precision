"""CLI: python -m workflow.discover <path> [options]

Kernel discovery over a codebase (step 1, discovery-only). Points at a
directory tree (a real codebase such as Kokkos Kernels) or a single file,
finds candidate numerical compute kernels via a hybrid pipeline, lets the
operator SELECT which ones, and prints a table (optionally a JSON
manifest). It is read-only w.r.t. the codebase and does NOT rewrite or
pipe into `python -m workflow.run` — that bridge depends on a later
extraction step and is deliberately out of scope here.

Pipeline:
  1. Deterministic scan (workflow.discovery.scan_codebase): walk the
     tree, filter to profile-claimed suffixes, grep kernel markers.
     No API calls.
  2. LLM confirm (kernel_extractor agent, one call per shortlisted
     file): identify + name the real kernels, flag floating_point /
     self_contained.
  3. Filter (--only-fp / --only-self-contained), rank, present a
     numbered table, resolve selection (interactive prompt, --select,
     or --yes), print the selection and optionally write --json.

Usage:
  python -m workflow.discover path/to/repo
  python -m workflow.discover path/to/repo --only-self-contained --only-fp
  python -m workflow.discover path/to/repo --select 1,3,5 --json out.json
  python -m workflow.discover file.cpp --yes --json out.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .discovery import CandidateFile, scan_codebase
from .run_agent import run_agent

MANIFEST_SCHEMA_VERSION = 1


@dataclass
class DiscoveredKernel:
    """A kernel confirmed by the kernel_extractor agent, tied to its file.

    Flat record combining the source file path with one kernel entry
    from the agent's output. This is the unit the selection table lists
    and the JSON manifest serializes.
    """

    file: str
    function_name: str
    language: str
    start_line: int
    end_line: int
    floating_point: bool
    self_contained: bool
    rationale: str
    # Informational only: the kernel's template parameters (empty for a
    # non-templated kernel). Each item is {name, kind, suggested}. The
    # workflow does NOT instantiate on this; it surfaces to the operator
    # what instantiation a templated kernel would need. See the design
    # note in AGENTS.md ("instantiation specializes a kernel").
    template_params: list[dict] = field(default_factory=list)


def build_extractor_task(candidate: CandidateFile, text: str) -> str:
    """Build the kernel_extractor task string for one shortlisted file.

    Prepends 1-based line numbers to the source (the agent reports line
    ranges, so accurate numbering matters) and surfaces the deterministic
    scan's coarse guess + matched markers as context. Pure function so it
    can be unit-tested without an API call.
    """
    numbered = "\n".join(
        f"{i:6d}: {line}"
        for i, line in enumerate(text.splitlines(), start=1)
    )
    markers = ", ".join(candidate.matched_markers) or "(none)"
    return (
        f"FILE: {candidate.path}\n"
        f"SCAN LANGUAGE GUESS: {candidate.language_guess}\n"
        f"SCAN MATCHED MARKERS: {markers}\n\n"
        "SOURCE (1-based line numbers prefixed; ignore the prefix when "
        "reading the code, use it only to report accurate line ranges):\n"
        "```\n"
        f"{numbered}\n"
        "```\n"
    )


def extract_kernels(
    candidates: list[CandidateFile],
    run_agent_fn=run_agent,
    on_progress=None,
) -> list[DiscoveredKernel]:
    """Run the kernel_extractor agent once per candidate file, aggregate.

    `run_agent_fn` is injectable so tests can pass a stub instead of the
    real (network) run_agent. `on_progress(i, n, candidate)` is an
    optional callback for CLI progress output. Files whose agent call
    raises are skipped with a stderr note (discovery is best-effort; one
    bad file must not abort the whole scan).
    """
    discovered: list[DiscoveredKernel] = []
    n = len(candidates)
    for i, candidate in enumerate(candidates, start=1):
        if on_progress is not None:
            on_progress(i, n, candidate)
        try:
            text = Path(candidate.path).read_text(errors="replace")
        except OSError as exc:
            print(
                f"  ! skipping {candidate.path}: {exc}", file=sys.stderr
            )
            continue
        task = build_extractor_task(candidate, text)
        try:
            result = run_agent_fn("kernel_extractor", task)
        except Exception as exc:  # noqa: BLE001 - best-effort per file
            print(
                f"  ! kernel_extractor failed on {candidate.path}: {exc}",
                file=sys.stderr,
            )
            continue
        for k in result.get("kernels", []):
            discovered.append(
                DiscoveredKernel(
                    file=str(candidate.path),
                    function_name=k["function_name"],
                    language=k["language"],
                    start_line=k["start_line"],
                    end_line=k["end_line"],
                    floating_point=bool(k["floating_point"]),
                    self_contained=bool(k["self_contained"]),
                    rationale=k["rationale"],
                    # Tolerant of stub / older agent output that omits
                    # the field: default to no template params.
                    template_params=list(k.get("template_params", [])),
                )
            )
    return discovered


def apply_filters(
    kernels: list[DiscoveredKernel],
    only_fp: bool,
    only_self_contained: bool,
) -> list[DiscoveredKernel]:
    """Narrow the kernel list by the --only-* flags (pure function)."""
    out = kernels
    if only_fp:
        out = [k for k in out if k.floating_point]
    if only_self_contained:
        out = [k for k in out if k.self_contained]
    return out


def rank_kernels(kernels: list[DiscoveredKernel]) -> list[DiscoveredKernel]:
    """Order kernels most-promising-first for the selection table.

    Ranking heuristic (all deterministic): self-contained before not,
    floating-point before not, then by file then by start line so the
    order is stable and eyeball-friendly. This is only a display order;
    the operator selects by the printed index.
    """
    return sorted(
        kernels,
        key=lambda k: (
            not k.self_contained,
            not k.floating_point,
            k.file,
            k.start_line,
        ),
    )


def parse_selection(spec: str, n: int) -> list[int]:
    """Parse a selection spec into a sorted list of 0-based indices.

    Accepts `all` (case-insensitive) for every index, or a
    comma-separated list of 1-based indices (`1,3,5`). Whitespace around
    items is tolerated. Raises ValueError on an empty spec, a
    non-integer item, a duplicate, or an out-of-range index (1..n). The
    caller (interactive loop or --select handler) decides how to surface
    the error.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty selection")
    if spec.lower() == "all":
        return list(range(n))
    indices: list[int] = []
    seen: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            one = int(item)
        except ValueError:
            raise ValueError(f"not a number: {item!r}")
        if one < 1 or one > n:
            raise ValueError(f"index out of range (1..{n}): {one}")
        if one in seen:
            raise ValueError(f"duplicate index: {one}")
        seen.add(one)
        indices.append(one - 1)
    if not indices:
        raise ValueError("empty selection")
    return sorted(indices)


def render_table(kernels: list[DiscoveredKernel]) -> str:
    """Render the numbered selection table (pure function -> str)."""
    if not kernels:
        return "(no candidate kernels)"
    rows = []
    header = (
        f"{'#':>3}  {'FUNCTION':<28} {'LANG':<8} {'LINES':<12} "
        f"{'FP':<4} {'SELF':<5} {'TMPL':<14} FILE"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for i, k in enumerate(kernels, start=1):
        lines = f"{k.start_line}-{k.end_line}"
        fp = "yes" if k.floating_point else "no"
        sc = "yes" if k.self_contained else "no"
        fn = (k.function_name[:27] + "…") if len(k.function_name) > 28 else k.function_name
        tmpl = _summarize_template_params(k.template_params)
        rows.append(
            f"{i:>3}  {fn:<28} {k.language:<8} {lines:<12} "
            f"{fp:<4} {sc:<5} {tmpl:<14} {k.file}"
        )
    return "\n".join(rows)


def _summarize_template_params(params: list[dict]) -> str:
    """Compact one-cell summary of template params for the table.

    Empty -> "-". Otherwise a comma-joined list of parameter names,
    truncated with an ellipsis so the column stays fixed-ish width.
    """
    if not params:
        return "-"
    names = [str(p.get("name", "?")) for p in params]
    joined = ",".join(names)
    if len(joined) > 13:
        joined = joined[:12] + "…"
    return joined


def build_manifest(
    root: str, selected: list[DiscoveredKernel]
) -> dict:
    """Build the JSON manifest dict for --json (forward-looking schema).

    schema_version is pinned so a future step-2 extraction consumer can
    detect shape changes. generated_at is UTC ISO-8601.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "root": root,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "selected": [
            {
                "file": k.file,
                "function_name": k.function_name,
                "language": k.language,
                "start_line": k.start_line,
                "end_line": k.end_line,
                "floating_point": k.floating_point,
                "self_contained": k.self_contained,
                "template_params": k.template_params,
                "rationale": k.rationale,
            }
            for k in selected
        ],
    }


def _resolve_selection(
    kernels: list[DiscoveredKernel],
    select: str | None,
    assume_yes: bool,
) -> list[DiscoveredKernel]:
    """Turn the ranked kernel list + flags into the chosen subset.

    Precedence: an explicit --select wins; else --yes selects all; else
    an interactive prompt loops until a valid selection or quit. On quit
    the process exits 0 (the operator chose to bow out, not an error).
    """
    n = len(kernels)
    if select is not None:
        try:
            idxs = parse_selection(select, n)
        except ValueError as exc:
            print(f"Invalid --select: {exc}", file=sys.stderr)
            raise SystemExit(2)
        return [kernels[i] for i in idxs]
    if assume_yes:
        return list(kernels)
    # Interactive loop.
    while True:
        try:
            raw = input(
                "\nSelect kernels to record (e.g. 1,3,5 / all / q): "
            )
        except EOFError:
            print("\n(no input; nothing selected)", file=sys.stderr)
            return []
        if raw.strip().lower() in {"q", "quit"}:
            print("Quit; nothing selected.")
            raise SystemExit(0)
        try:
            idxs = parse_selection(raw, n)
        except ValueError as exc:
            print(f"  invalid selection: {exc}. Try again.")
            continue
        return [kernels[i] for i in idxs]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m workflow.discover",
        description=(
            "Discover candidate numerical kernels in a codebase and "
            "select which ones to record. Read-only; does not rewrite."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Directory tree (walked) or single source file to scan.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Cap on how many marker-matching files are sent to the LLM "
            "kernel_extractor (cost guard on large repos). Default 50."
        ),
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Non-interactive selection: comma-separated 1-based indices "
            "(e.g. 1,3,5) or 'all'. Skips the interactive prompt."
        ),
    )
    parser.add_argument(
        "--only-fp",
        action="store_true",
        help="Show/record only kernels flagged floating_point.",
    )
    parser.add_argument(
        "--only-self-contained",
        action="store_true",
        help="Show/record only kernels flagged self_contained.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write the selected kernels as a JSON manifest to OUTFILE.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Non-interactive: if --select is not given, select all "
            "candidates. Intended for scripted/batch runs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 2

    print(f"Scanning {args.path} ...", file=sys.stderr)
    try:
        candidates = scan_codebase(args.path, max_files=args.max_files)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Scan error: {exc}", file=sys.stderr)
        return 2

    if not candidates:
        print(
            "No kernel-shaped files found (no profile-claimed suffix with "
            "a kernel marker).",
            file=sys.stderr,
        )
        return 0

    print(
        f"Found {len(candidates)} candidate file(s); confirming with "
        f"kernel_extractor ...",
        file=sys.stderr,
    )

    def _progress(i, n, candidate):
        print(f"  [{i}/{n}] {candidate.path}", file=sys.stderr)

    kernels = extract_kernels(candidates, on_progress=_progress)
    kernels = apply_filters(
        kernels, args.only_fp, args.only_self_contained
    )
    kernels = rank_kernels(kernels)

    if not kernels:
        print("No kernels confirmed after filtering.", file=sys.stderr)
        return 0

    print()
    print(render_table(kernels))

    selected = _resolve_selection(kernels, args.select, args.yes)

    print()
    print(f"Selected {len(selected)} kernel(s):")
    for k in selected:
        print(f"  - {k.function_name}  ({k.file}:{k.start_line}-{k.end_line})")

    if args.json is not None:
        manifest = build_manifest(str(args.path), selected)
        args.json.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nWrote manifest to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
