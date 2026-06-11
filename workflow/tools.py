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
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Environment variable that names the Kokkos install prefix (i.e. the
# directory containing include/ and lib/). Intentionally namespaced so
# it does not collide with Kokkos's own CMake convention (Kokkos_ROOT)
# or with a system-wide install.
KOKKOS_ROOT_ENV = "AGENT_PRECISION_KOKKOS_ROOT"

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
