"""CLI entrypoint: python -m workflow.run <kernel_file> [tolerance flags]

Tolerance flags (mutually exclusive, both optional):
  --sig-figs N         Output-precision tolerance as N significant figures
                       (relative tolerance: relative error < 10^-N).
  --decimal-digits N   Output-precision tolerance as N decimal digits
                       after the point (absolute tolerance: abs error < 10^-N).

If neither flag is given, the orchestrator will call the
precision_advisor agent to infer a tolerance from the kernel source.
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
    tol_group = parser.add_mutually_exclusive_group()
    tol_group.add_argument(
        "--sig-figs",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Required correct significant figures of the kernel's output "
            "(relative tolerance). Mutually exclusive with --decimal-digits."
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
            "with --sig-figs."
        ),
    )
    return parser.parse_args(argv)


def _normalize_tolerance(args: argparse.Namespace) -> dict | None:
    """Turn parsed CLI args into the {kind, value, source} dict or None."""
    if args.sig_figs is not None:
        if args.sig_figs <= 0:
            raise SystemExit("--sig-figs must be a positive integer")
        return {"kind": "sig_figs", "value": args.sig_figs, "source": "user_cli"}
    if args.decimal_digits is not None:
        if args.decimal_digits <= 0:
            raise SystemExit("--decimal-digits must be a positive integer")
        return {
            "kind": "decimal_digits",
            "value": args.decimal_digits,
            "source": "user_cli",
        }
    return None


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
