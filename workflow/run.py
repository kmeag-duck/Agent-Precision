"""CLI entrypoint: python -m workflow.run <kernel_file> [tolerance flags]

Tolerance flags (mutually exclusive, exactly one required):
  --sig-figs N         Output-precision tolerance as N significant figures
                       (relative tolerance: relative error < 10^-N).
  --decimal-digits N   Output-precision tolerance as N decimal digits
                       after the point (absolute tolerance: abs error < 10^-N).

A run without either flag is rejected by argparse at exit code 2. There
is no in-workflow fallback: the operator must supply an explicit
numerical target so batch runs stay comparable across invocations.
"""

import argparse
import sys
from pathlib import Path

from .orchestrator import run_orchestrator


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

    result = run_orchestrator(
        str(args.kernel_file),
        kernel_source,
        tolerance=tolerance,
        auto=args.auto,
        run_probe=not args.no_probe,
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
