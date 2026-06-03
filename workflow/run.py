"""CLI entrypoint: python -m workflow.run <kernel_file>"""

import sys
from pathlib import Path

from .orchestrator import run_orchestrator


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m workflow.run <kernel_file>", file=sys.stderr)
        return 2

    kernel_path = Path(sys.argv[1])
    if not kernel_path.exists():
        print(f"File not found: {kernel_path}", file=sys.stderr)
        return 2

    kernel_source = kernel_path.read_text()
    result = run_orchestrator(str(kernel_path), kernel_source)

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
