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
