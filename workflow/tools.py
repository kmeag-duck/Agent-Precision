"""Deterministic (non-LLM) tools the orchestrator can call.

This module hosts the small, mechanical helpers that the orchestrator
exposes as tools alongside the LLM-backed `spawn_*` agents. Unlike the
agents, these functions do real I/O (run a compiler, read/write files)
and have no model call inside them. They return a uniform
`{status, stdout, stderr, artifacts}` shape so the orchestrator's
tool-result handling is the same regardless of whether the tool was an
LLM agent or a local subprocess. That shape is also what the planned
remote-batch verifier tools will return (see AGENTS.md "JLSE / async
toolchain migration"), so the orchestrator loop does not need to change
when "run g++ locally" becomes "submit a compile job and poll".

Currently exposes:

  - compile_baseline_driver(kernel_stem): compile
    baselines/<kernel_stem>/driver.cpp produced by the baseline_harness
    agent into a native executable at baselines/<kernel_stem>/driver,
    linking against the Kokkos install named by the
    AGENT_PRECISION_KOKKOS_ROOT environment variable.

  - run_baseline_driver(kernel_stem): execute the compiled driver at
    baselines/<kernel_stem>/driver and verify that it produces a
    parseable baselines/<kernel_stem>/reference.json. Subject to a
    per-run wall-clock timeout configured via the
    AGENT_PRECISION_RUN_TIMEOUT_SEC environment variable (default 60s).

  - splice_rewritten_kernel(kernel_stem, rewritten_kernel_source): read
    the baseline driver at baselines/<kernel_stem>/driver.cpp, replace
    the region strictly between the KERNEL BEGIN / KERNEL END sentinels
    with the supplied rewritten kernel source, and write the result to
    baselines/<kernel_stem>/rewritten/driver.cpp. Pure text I/O; never
    invokes a subprocess and never modifies the baseline file in place.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Environment variable that names the Kokkos install prefix (i.e. the
# directory containing include/ and lib/). Intentionally namespaced so
# it does not collide with Kokkos's own CMake convention (Kokkos_ROOT)
# or with a system-wide install.
KOKKOS_ROOT_ENV = "AGENT_PRECISION_KOKKOS_ROOT"

# Environment variable that caps the wall-clock seconds run_baseline_driver
# will wait for the compiled driver to finish. Namespaced for the same
# reason as KOKKOS_ROOT_ENV; the explicit "_SEC" suffix avoids ms/s
# ambiguity. Kernels in v0 are small enough that 60s is generous; raise
# this as kernels grow (e.g. once deployment-scale inputs land).
RUN_TIMEOUT_ENV = "AGENT_PRECISION_RUN_TIMEOUT_SEC"
DEFAULT_RUN_TIMEOUT_SEC = 60

# Compile flags. C++20 is required by the baseline driver template; the
# OpenMP host backend is what the Kokkos install in this repo was built
# with, so -fopenmp is mandatory at link time. Kokkos here is shipped as
# static archives (libkokkoscore.a, libkokkoscontainers.a), so no rpath
# is needed and the resulting binary has no libkokkos* in its dynamic
# NEEDED list.
CXX = "g++"
CXX_STD = "-std=c++20"
OPT_FLAGS = ["-O2", "-fopenmp"]
KOKKOS_LIBS = ["-lkokkoscore", "-lkokkoscontainers"]
EXTRA_LIBS = ["-lpthread", "-ldl"]

# Splice sentinels. These exact byte strings are mandated by the
# baseline_harness agent's system prompt (see BASELINE_HARNESS_SYSTEM_PROMPT
# in workflow/registry.py): the inlined kernel in baselines/<stem>/driver.cpp
# is bracketed by these two lines, each on its own line with no
# surrounding indentation. splice_rewritten_kernel relies on that exact
# contract to identify the region to replace. If you ever change either
# string here, you MUST also update the literal sentinel lines spelled
# out in the harness prompt; LLM prompts cannot reference Python
# identifiers, so the prompt deliberately re-states the bytes.
KERNEL_BEGIN_SENTINEL = "// ---- KERNEL BEGIN ----"
KERNEL_END_SENTINEL = "// ---- KERNEL END ----"


def _error(stderr: str) -> dict:
    """Build a uniform error result with empty stdout and no artifacts."""
    return {
        "status": "error",
        "stdout": "",
        "stderr": stderr,
        "artifacts": [],
    }


def compile_baseline_driver(kernel_stem: str) -> dict:
    """Compile baselines/<kernel_stem>/driver.cpp against the local Kokkos.

    Reads the Kokkos install prefix from the AGENT_PRECISION_KOKKOS_ROOT
    environment variable. Returns a `{status, stdout, stderr, artifacts}`
    dict, where `status` is 'ok' on a successful compile and 'error'
    otherwise. `artifacts` is a list of created/expected output paths
    (the driver binary) — empty on error.

    The driver source is expected to already exist at
    baselines/<kernel_stem>/driver.cpp (written by the baseline_harness
    agent on HITL approval). This helper does not run the compiled
    binary; that is a separate step.
    """
    kokkos_root = os.environ.get(KOKKOS_ROOT_ENV)
    if not kokkos_root:
        return _error(
            f"{KOKKOS_ROOT_ENV} is not set. Point it at a Kokkos install "
            f"prefix (the directory containing include/ and lib/)."
        )
    root = Path(kokkos_root)
    include_dir = root / "include"
    lib_dir = root / "lib"
    if not include_dir.is_dir() or not lib_dir.is_dir():
        return _error(
            f"{KOKKOS_ROOT_ENV}={kokkos_root!r} does not look like a "
            f"Kokkos install prefix (missing include/ or lib/)."
        )

    driver_dir = Path("baselines") / kernel_stem
    driver_src = driver_dir / "driver.cpp"
    driver_bin = driver_dir / "driver"
    if not driver_src.is_file():
        return _error(
            f"Driver source not found at {driver_src}. Did "
            f"spawn_baseline_harness run and get approved for this "
            f"kernel_stem?"
        )

    cmd = [
        CXX,
        CXX_STD,
        *OPT_FLAGS,
        f"-I{include_dir}",
        f"-L{lib_dir}",
        str(driver_src),
        *KOKKOS_LIBS,
        *EXTRA_LIBS,
        "-o",
        str(driver_bin),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        # g++ not on PATH.
        return _error(f"Failed to invoke {CXX!r}: {exc}")

    if proc.returncode != 0:
        return {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": (
                f"{CXX} exited with code {proc.returncode}.\n"
                f"Command: {' '.join(cmd)}\n\n{proc.stderr}"
            ),
            "artifacts": [],
        }

    return {
        "status": "ok",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifacts": [str(driver_bin)],
    }


def _parse_timeout(raw: str) -> int | None:
    """Parse RUN_TIMEOUT_ENV into a positive int; return None on bad input."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def run_baseline_driver(kernel_stem: str) -> dict:
    """Execute baselines/<kernel_stem>/driver and validate reference.json.

    The driver is invoked with cwd=baselines/<kernel_stem>/ so the
    `./reference.json` path the baseline_harness prompt mandates lands
    next to the driver source/binary. Returns a uniform
    `{status, stdout, stderr, artifacts}` dict (the same shape as
    compile_baseline_driver and the planned remote-batch verifier
    tools), where `status` is 'ok' on a clean run + parseable JSON and
    'error' otherwise. On success, `artifacts` is the single-element
    list `["baselines/<kernel_stem>/reference.json"]`.

    Any pre-existing reference.json at the target path is deleted
    before the subprocess runs so that, on a failed run, the orchestrator
    does not see a misleadingly-stale reference. Compile/run failures
    are non-fatal to the surrounding pipeline; the orchestrator treats
    this whole side artifact as optional.
    """
    raw_timeout = os.environ.get(RUN_TIMEOUT_ENV)
    if raw_timeout is None:
        timeout = DEFAULT_RUN_TIMEOUT_SEC
    else:
        parsed = _parse_timeout(raw_timeout)
        if parsed is None:
            return _error(
                f"{RUN_TIMEOUT_ENV}={raw_timeout!r} is not a positive "
                f"integer number of seconds."
            )
        timeout = parsed

    driver_dir = Path("baselines") / kernel_stem
    driver_bin = driver_dir / "driver"
    reference_path = driver_dir / "reference.json"

    if not driver_bin.is_file():
        return _error(
            f"Driver binary not found at {driver_bin}. Did "
            f"compile_baseline_driver run and succeed for this "
            f"kernel_stem?"
        )
    if not os.access(driver_bin, os.X_OK):
        return _error(
            f"Driver binary at {driver_bin} is not executable."
        )

    # Drop any stale reference.json from a prior run so a failed
    # subprocess cannot leave the orchestrator with a misleading file.
    try:
        reference_path.unlink()
    except FileNotFoundError:
        pass

    try:
        proc = subprocess.run(
            ["./driver"],
            cwd=str(driver_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _error(
            f"Driver at {driver_bin} exceeded timeout of {timeout}s "
            f"(configure via {RUN_TIMEOUT_ENV}).\n"
            f"Partial stdout: {exc.stdout or ''}\n"
            f"Partial stderr: {exc.stderr or ''}"
        )
    except FileNotFoundError as exc:
        return _error(f"Failed to invoke driver at {driver_bin}: {exc}")

    if proc.returncode != 0:
        return {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": (
                f"Driver exited with code {proc.returncode}.\n\n"
                f"{proc.stderr}"
            ),
            "artifacts": [],
        }

    if not reference_path.is_file():
        return {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": (
                f"Driver exited 0 but did not write {reference_path}. "
                f"Check that the driver writes ./reference.json "
                f"relative to its CWD.\n\n{proc.stderr}"
            ),
            "artifacts": [],
        }

    try:
        with reference_path.open() as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": (
                f"Driver wrote {reference_path} but it is not valid "
                f"JSON: {exc}\n\n{proc.stderr}"
            ),
            "artifacts": [],
        }

    return {
        "status": "ok",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifacts": [str(reference_path)],
    }


def _find_unique_sentinel_line(lines: list[str], sentinel: str) -> int | None:
    """Return the index of the unique line equal to `sentinel`, or None.

    Returns None if the sentinel appears zero times or more than once.
    Comparison is byte-exact: a line with leading or trailing whitespace
    around the sentinel does NOT match (the sentinel contract requires
    each sentinel on its own line with no surrounding indentation).
    """
    matches = [i for i, line in enumerate(lines) if line == sentinel]
    if len(matches) != 1:
        return None
    return matches[0]


def splice_rewritten_kernel(
    kernel_stem: str, rewritten_kernel_source: str
) -> dict:
    """Splice `rewritten_kernel_source` into the baseline driver template.

    Reads baselines/<kernel_stem>/driver.cpp, locates the unique
    KERNEL BEGIN / KERNEL END sentinel lines (byte-exact, each on its
    own line with no surrounding indentation), replaces the text
    strictly BETWEEN them (sentinels themselves preserved) with
    `rewritten_kernel_source`, and writes the result to
    baselines/<kernel_stem>/rewritten/driver.cpp. The directory is
    created if needed. The baseline driver.cpp is never modified.

    The result has the uniform `{status, stdout, stderr, artifacts}`
    shape shared with compile_baseline_driver and run_baseline_driver:

      - On success: status='ok', stdout='', stderr='',
        artifacts=['baselines/<stem>/rewritten/driver.cpp'].
      - On any error: status='error', stdout='', a descriptive stderr,
        and artifacts=[].

    This is pure text I/O. It never invokes a subprocess.
    """
    if not rewritten_kernel_source:
        return _error(
            "rewritten_kernel_source is empty; nothing to splice."
        )

    baseline_path = Path("baselines") / kernel_stem / "driver.cpp"
    if not baseline_path.is_file():
        return _error(
            f"Baseline driver source not found at {baseline_path}. Did "
            f"spawn_baseline_harness run and get approved for this "
            f"kernel_stem?"
        )

    try:
        baseline_text = baseline_path.read_text()
    except OSError as exc:
        return _error(f"Failed to read {baseline_path}: {exc}")

    # splitlines(keepends=False) so we can do byte-exact line comparisons
    # against the sentinel constants without worrying about \n on the
    # right-hand side. We rejoin with "\n" below.
    lines = baseline_text.splitlines()

    begin_idx = _find_unique_sentinel_line(lines, KERNEL_BEGIN_SENTINEL)
    if begin_idx is None:
        return _error(
            f"Baseline driver at {baseline_path} does not contain "
            f"exactly one {KERNEL_BEGIN_SENTINEL!r} line on its own "
            f"(no surrounding indentation or whitespace)."
        )

    end_idx = _find_unique_sentinel_line(lines, KERNEL_END_SENTINEL)
    if end_idx is None:
        return _error(
            f"Baseline driver at {baseline_path} does not contain "
            f"exactly one {KERNEL_END_SENTINEL!r} line on its own "
            f"(no surrounding indentation or whitespace)."
        )

    if begin_idx >= end_idx:
        return _error(
            f"Baseline driver at {baseline_path} has "
            f"{KERNEL_END_SENTINEL!r} at or before "
            f"{KERNEL_BEGIN_SENTINEL!r} (line {end_idx + 1} vs line "
            f"{begin_idx + 1}); cannot splice."
        )

    # Splice: keep everything up to and including BEGIN, drop the
    # current kernel body, insert the rewritten kernel source (stripped
    # of a single trailing newline so we don't double up when rejoining
    # with "\n"), then keep END and everything after.
    rewritten_body = rewritten_kernel_source
    if rewritten_body.endswith("\n"):
        rewritten_body = rewritten_body[:-1]
    rewritten_lines = rewritten_body.split("\n")

    new_lines = (
        lines[: begin_idx + 1]
        + rewritten_lines
        + lines[end_idx:]
    )

    # Preserve a trailing newline iff the baseline had one (it normally
    # does). Use "\n" joins so we don't accidentally inherit "\r\n" from
    # a Windows-authored baseline; the baseline_harness emits Unix line
    # endings by convention.
    trailing_newline = "\n" if baseline_text.endswith("\n") else ""
    new_text = "\n".join(new_lines) + trailing_newline

    out_dir = Path("baselines") / kernel_stem / "rewritten"
    out_path = out_dir / "driver.cpp"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text)
    except OSError as exc:
        return _error(f"Failed to write {out_path}: {exc}")

    return {
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "artifacts": [str(out_path)],
    }
