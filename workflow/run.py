"""CLI entrypoint: python -m workflow.run <kernel_file> [tolerance flags]

Tolerance flags (mutually exclusive, exactly one required):
  --sig-figs N         Output-precision tolerance as N significant figures
                       (relative tolerance: relative error < 10^-N).
  --decimal-digits N   Output-precision tolerance as N decimal digits
                       after the point (absolute tolerance: abs error < 10^-N).

A run without either flag is rejected by argparse at exit code 2. There
is no in-workflow fallback: the operator must supply an explicit
numerical target so batch runs stay comparable across invocations.

Test-config side-channel: if a file named `<kernel_file>.testconfig.json`
exists next to the kernel source, it is auto-loaded and its parsed JSON
is threaded into the baseline_harness agent's BASELINE STEP block so the
harness uses the operator-supplied test parameters (N, seed, eps, dt,
per-array distribution / ranges, etc.) verbatim instead of inventing
them. The schema is freeform JSON — the harness system prompt describes
the conventional keys — but a malformed JSON file is a hard CLI error
so the operator can't silently drift into "harness-invented" territory.
"""

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import run_orchestrator


def _load_test_config(kernel_path: Path) -> dict | None:
    """Auto-load the sibling `<kernel>.testconfig.json` file, if it exists.

    Returns the parsed dict, or None when no sibling file is present.
    Raises SystemExit on a JSON parse error, on a non-object top-level
    value (test-config must be a JSON object, not a list / scalar), or
    on an IOError reading the file — the operator explicitly opted into
    a config by dropping the file next to the kernel, so silent fallback
    to "harness invents inputs" would defeat the whole point.
    """
    config_path = kernel_path.with_suffix(kernel_path.suffix + ".testconfig.json")
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text()
    except OSError as exc:
        raise SystemExit(f"Failed to read {config_path}: {exc}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Failed to parse {config_path} as JSON: {exc}"
        )
    if not isinstance(parsed, dict):
        raise SystemExit(
            f"{config_path} must contain a JSON object at the top "
            f"level (got {type(parsed).__name__})"
        )
    return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m workflow.run",
        description=(
            "Rewrite a numerical kernel to reduce precision cost while "
            "keeping output within a tolerance."
        ),
    )
    parser.add_argument(
        "kernel_file",
        type=Path,
        help="Path to the kernel source file to rewrite.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Skip the interactive y/n/q pause before every tool call and "
            "approve every tool automatically. Writes a JSONL trace of "
            "all executed tools to baselines/<kernel_stem>/"
            "orchestrator_trace.jsonl for post-hoc inspection. Use this "
            "for batch runs (e.g. consistency measurements); not "
            "recommended for first-time debugging of a kernel."
        ),
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help=(
            "Skip the precision probe step. By default, on language "
            "profiles whose probe_precisions is non-empty (Kokkos in "
            "v1), the orchestrator runs a small (precision, seed) "
            "matrix of probe drivers before invoking the analyst and "
            "attaches the aggregated evidence to the analyst task. "
            "Pass this flag to reproduce the v0 (pre-probe) behavior "
            "exactly, or for kernels where the probe's wall-clock cost "
            "is not worth its evidence value. Profiles without probe "
            "templates (every profile other than Kokkos in v1) ignore "
            "this flag — they never had a probe step to skip."
        ),
    )
    tol_group = parser.add_mutually_exclusive_group(required=True)
    tol_group.add_argument(
        "--sig-figs",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Required correct significant figures of the kernel's output "
            "(relative tolerance). Mutually exclusive with --decimal-digits. "
            "Exactly one of --sig-figs / --decimal-digits is required."
        ),
    )
    tol_group.add_argument(
        "--decimal-digits",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Required correct decimal digits after the point of the "
            "kernel's output (absolute tolerance). Mutually exclusive "
            "with --sig-figs. Exactly one of --sig-figs / "
            "--decimal-digits is required."
        ),
    )
    return parser.parse_args(argv)


def _normalize_tolerance(args: argparse.Namespace) -> dict:
    """Turn parsed CLI args into the {kind, value, source} dict.

    Argparse guarantees exactly one of --sig-figs / --decimal-digits
    was passed (the tol_group is required=True), so this function
    always returns a dict; there is no None fallback.
    """
    if args.sig_figs is not None:
        if args.sig_figs <= 0:
            raise SystemExit("--sig-figs must be a positive integer")
        return {"kind": "sig_figs", "value": args.sig_figs, "source": "user_cli"}
    if args.decimal_digits <= 0:
        raise SystemExit("--decimal-digits must be a positive integer")
    return {
        "kind": "decimal_digits",
        "value": args.decimal_digits,
        "source": "user_cli",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.kernel_file.exists():
        print(f"File not found: {args.kernel_file}", file=sys.stderr)
        return 2

    kernel_source = args.kernel_file.read_text()
    tolerance = _normalize_tolerance(args)
    test_config = _load_test_config(args.kernel_file)

    result = run_orchestrator(
        str(args.kernel_file),
        kernel_source,
        tolerance=tolerance,
        auto=args.auto,
        run_probe=not args.no_probe,
        test_config=test_config,
    )

    if result is None:
        return 1

    print()
    print("=" * 72)
    print("=== FINAL REWRITTEN KERNEL ===")
    print("=" * 72)
    print(result["rewritten_code"])
    print()
    print("--- notes ---")
    print(result["notes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
