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

  - compile_rewritten_driver(kernel_stem): compile
    baselines/<kernel_stem>/rewritten/driver.cpp (produced by
    splice_rewritten_kernel) into baselines/<kernel_stem>/rewritten/
    driver. Shares the env-var contract, compile flags, and result
    schema of compile_baseline_driver; only the directory differs.

  - run_rewritten_driver(kernel_stem): execute the compiled rewritten
    driver at baselines/<kernel_stem>/rewritten/driver and verify it
    produces a parseable baselines/<kernel_stem>/rewritten/reference.json.
    Shares the env-var contract (AGENT_PRECISION_RUN_TIMEOUT_SEC),
    subprocess shape, and result schema of run_baseline_driver; only
    the directory differs. The baseline tree is never touched.

  - compare_outputs(kernel_stem, tolerance_json): numerically compare
    baselines/<kernel_stem>/reference.json (baseline) and
    baselines/<kernel_stem>/rewritten/reference.json (rewritten) under
    the operator-agreed tolerance. Tolerance kinds are 'sig_figs' (a
    value passes when |a-b| < 10^-N * max(|a|,|b|), with both-zero
    treated as a pass) and 'decimal_digits' (|a-b| < 10^-N). NaN
    ALWAYS mismatches — including NaN vs NaN — so that any NaN in
    either output forces a regression-flag. Matching same-sign infs
    pass; mismatched inf signs and inf-vs-finite fail. Shape mismatches
    (different top-level keys, different output array names, different
    array lengths) are surfaced as status='error' with a populated
    shape_error field, distinct from numerical mismatches. A
    comparison.json artifact is written under the rewritten subtree on
    both pass and fail paths. Pure file + arithmetic I/O; no
    subprocess, no env-var contract.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Iterator

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


def _compile_driver(driver_dir: Path, missing_source_hint: str) -> dict:
    """Compile <driver_dir>/driver.cpp into <driver_dir>/driver.

    Shared implementation behind compile_baseline_driver and
    compile_rewritten_driver. The two public wrappers differ only in
    which directory they target — the env-var checks, command shape,
    subprocess invocation, error wrapping, and result schema are
    identical. Pulling the body here keeps the dynamic-verification
    chain (splice -> compile_rewritten -> run_rewritten -> compare)
    from drifting from the baseline chain it parallels.

    `missing_source_hint` is the human-readable hint appended to the
    "driver source not found" error so the operator knows which
    upstream tool was supposed to have written that file (e.g.
    spawn_baseline_harness for the baseline, splice_rewritten_kernel
    for the rewritten variant). Everything else about the error result
    is identical across the two wrappers.
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

    driver_src = driver_dir / "driver.cpp"
    driver_bin = driver_dir / "driver"
    if not driver_src.is_file():
        return _error(
            f"Driver source not found at {driver_src}. {missing_source_hint}"
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
    return _compile_driver(
        Path("baselines") / kernel_stem,
        missing_source_hint=(
            "Did spawn_baseline_harness run and get approved for this "
            "kernel_stem?"
        ),
    )


def compile_rewritten_driver(kernel_stem: str) -> dict:
    """Compile baselines/<kernel_stem>/rewritten/driver.cpp.

    Companion to compile_baseline_driver, targeting the rewritten
    driver produced by splice_rewritten_kernel. Same env-var contract
    (AGENT_PRECISION_KOKKOS_ROOT), same compile flags, same result
    schema. The compiled binary lands at
    baselines/<kernel_stem>/rewritten/driver, alongside the source.

    Like the baseline compile, this is a side artifact: a non-zero
    compile result is non-fatal to the surrounding pipeline (the
    analyst -> rewriter -> verifier loop still runs, and finish remains
    reachable on verifier accept).
    """
    return _compile_driver(
        Path("baselines") / kernel_stem / "rewritten",
        missing_source_hint=(
            "Did splice_rewritten_kernel run and get approved for this "
            "kernel_stem?"
        ),
    )


def _parse_timeout(raw: str) -> int | None:
    """Parse RUN_TIMEOUT_ENV into a positive int; return None on bad input."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _run_driver(driver_dir: Path, missing_binary_hint: str) -> dict:
    """Execute <driver_dir>/driver and validate <driver_dir>/reference.json.

    Shared implementation behind run_baseline_driver and
    run_rewritten_driver. The two public wrappers differ only in
    which directory they target — the env-var parsing, preflight
    checks, stale-reference deletion, subprocess invocation, timeout
    handling, JSON validation, and result schema are identical.
    Pulling the body here keeps the dynamic-verification chain
    (splice -> compile_rewritten -> run_rewritten -> compare) from
    drifting from the baseline chain it parallels.

    `missing_binary_hint` is the human-readable hint appended to the
    "driver binary not found" error so the operator knows which
    upstream tool was supposed to have produced that file (e.g.
    compile_baseline_driver for the baseline, compile_rewritten_driver
    for the rewritten variant). Everything else about the error
    results is identical across the two wrappers.

    Any pre-existing reference.json at the target path is deleted
    before the subprocess runs so that, on a failed run, the orchestrator
    does not see a misleadingly-stale reference. Run failures are
    non-fatal to the surrounding pipeline; the orchestrator treats this
    whole side artifact as optional.
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

    driver_bin = driver_dir / "driver"
    reference_path = driver_dir / "reference.json"

    if not driver_bin.is_file():
        return _error(
            f"Driver binary not found at {driver_bin}. {missing_binary_hint}"
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
    return _run_driver(
        Path("baselines") / kernel_stem,
        missing_binary_hint=(
            "Did compile_baseline_driver run and succeed for this "
            "kernel_stem?"
        ),
    )


def run_rewritten_driver(kernel_stem: str) -> dict:
    """Execute baselines/<kernel_stem>/rewritten/driver and validate JSON.

    Companion to run_baseline_driver, targeting the rewritten driver
    produced by compile_rewritten_driver. Same env-var contract
    (AGENT_PRECISION_RUN_TIMEOUT_SEC), same subprocess shape, same
    result schema. The rewritten driver runs with cwd set to
    baselines/<kernel_stem>/rewritten/, so its `./reference.json`
    lands inside the rewritten subtree and the baseline tree
    (baselines/<kernel_stem>/{driver.cpp, driver, reference.json}) is
    never touched by this call.

    Like the baseline run, this is a side artifact: a non-zero run
    result is non-fatal to the surrounding pipeline (the analyst ->
    rewriter -> verifier loop still runs, and finish remains reachable
    on verifier accept). On success, `artifacts` is the single-element
    list `["baselines/<kernel_stem>/rewritten/reference.json"]`.
    """
    return _run_driver(
        Path("baselines") / kernel_stem / "rewritten",
        missing_binary_hint=(
            "Did compile_rewritten_driver run and succeed for this "
            "kernel_stem?"
        ),
    )


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


# ---------- compare_outputs: baseline-vs-rewritten numerical comparator ----------

# Cap on how many concrete mismatch tuples make it into the result's
# stderr / comparison.json. Anything past the cap is summarized as a
# "+ K more mismatches suppressed" footer so an LLM caller sees a
# bounded payload but the operator still learns the true mismatch count.
# The full comparison.json has the same truncation; if a future
# regression hunt needs the full list, re-run the comparator after
# raising this constant or read it out of the driver run directly.
_MAX_REPORTED_MISMATCHES = 10


def _load_reference(path: Path) -> object:
    """Read and JSON-parse a reference.json at `path`.

    Raises FileNotFoundError if the file does not exist, and
    json.JSONDecodeError if the contents do not parse. The caller is
    expected to translate both into the uniform _error result; the
    helper deliberately does not catch them so the failure mode (which
    side was missing / unparseable) reaches the error message untouched.
    """
    with path.open("r") as fp:
        return json.load(fp)


def _value_within_tolerance(
    a: float, b: float, kind: str, value: int
) -> tuple[bool, float | None, float | None]:
    """Decide whether `a` and `b` agree under (kind, value) tolerance.

    Returns a `(passed, abs_err, threshold)` triple. `abs_err` and
    `threshold` are floats when finite arithmetic was used and None for
    the special-value paths (NaN, inf) where neither quantity is
    meaningful. The triple is what the caller serializes into a
    mismatch record so the operator can see the numbers behind the
    pass/fail decision.

    Semantics (from the dynamic-verification chain spec):
      - NaN ALWAYS mismatches — including NaN vs NaN. This is
        intentional asymmetry from IEEE 754: a NaN in either side of
        the comparison should fail the regression check, never silently
        agree with another NaN.
      - +inf vs +inf and -inf vs -inf pass; mismatched inf signs and
        inf-vs-finite fail.
      - Finite-vs-finite uses STRICT '<' against the threshold:
          sig_figs=N         -> threshold = 10^-N * max(|a|, |b|);
                                both-exactly-zero is a special pass.
          decimal_digits=N   -> threshold = 10^-N.
    """
    if math.isnan(a) or math.isnan(b):
        return (False, None, None)
    if math.isinf(a) or math.isinf(b):
        same_inf = (
            math.isinf(a)
            and math.isinf(b)
            and ((a > 0) == (b > 0))
        )
        return (same_inf, None, None)

    diff = abs(a - b)
    if kind == "sig_figs":
        max_abs = max(abs(a), abs(b))
        if max_abs == 0.0:
            return (True, 0.0, 0.0)
        threshold = (10.0 ** -value) * max_abs
        return (diff < threshold, diff, threshold)
    if kind == "decimal_digits":
        threshold = 10.0 ** -value
        return (diff < threshold, diff, threshold)
    # Unknown kinds reach this only via a pre-validated tolerance dict;
    # treat as a fail rather than raise so the caller's result schema
    # stays uniform.
    return (False, diff, None)


def _shape_error(baseline: object, rewritten: object) -> str | None:
    """Return a human-readable shape error, or None if shapes match.

    'Shape' here is the contract enforced by the baseline_harness
    prompt (see workflow/registry.py): the reference.json document has
    exactly {kernel, seed, inputs, outputs} at the top level, and
    `outputs` is a dict mapping array name -> flat array of doubles.
    The comparator only does numerical comparison under `outputs`; the
    other top-level keys are checked for presence and exact equality so
    a baseline/rewritten mismatch on kernel name or seed is surfaced
    here rather than silently masked.
    """
    if not isinstance(baseline, dict) or not isinstance(rewritten, dict):
        return (
            f"reference.json must be a JSON object at the top level; "
            f"got {type(baseline).__name__} (baseline) vs "
            f"{type(rewritten).__name__} (rewritten)."
        )
    baseline_keys = set(baseline.keys())
    rewritten_keys = set(rewritten.keys())
    if baseline_keys != rewritten_keys:
        missing_from_rewritten = sorted(baseline_keys - rewritten_keys)
        missing_from_baseline = sorted(rewritten_keys - baseline_keys)
        parts = []
        if missing_from_rewritten:
            parts.append(
                f"missing from rewritten: {missing_from_rewritten}"
            )
        if missing_from_baseline:
            parts.append(
                f"missing from baseline: {missing_from_baseline}"
            )
        return "Top-level key mismatch: " + "; ".join(parts)

    # kernel / seed are exact-equality contract checks.
    if "kernel" in baseline and baseline["kernel"] != rewritten["kernel"]:
        return (
            f"Top-level 'kernel' differs: "
            f"{baseline['kernel']!r} (baseline) vs "
            f"{rewritten['kernel']!r} (rewritten)."
        )
    if "seed" in baseline and baseline["seed"] != rewritten["seed"]:
        return (
            f"Top-level 'seed' differs: {baseline['seed']!r} "
            f"(baseline) vs {rewritten['seed']!r} (rewritten); a seed "
            f"mismatch means the two drivers exercised different inputs."
        )

    outputs_a = baseline.get("outputs")
    outputs_b = rewritten.get("outputs")
    if not isinstance(outputs_a, dict) or not isinstance(outputs_b, dict):
        return (
            f"'outputs' must be a JSON object mapping array name -> "
            f"flat array; got {type(outputs_a).__name__} (baseline) "
            f"vs {type(outputs_b).__name__} (rewritten)."
        )
    names_a = set(outputs_a.keys())
    names_b = set(outputs_b.keys())
    if names_a != names_b:
        missing_from_rewritten = sorted(names_a - names_b)
        missing_from_baseline = sorted(names_b - names_a)
        parts = []
        if missing_from_rewritten:
            parts.append(
                f"missing from rewritten: {missing_from_rewritten}"
            )
        if missing_from_baseline:
            parts.append(
                f"missing from baseline: {missing_from_baseline}"
            )
        return "Output array name mismatch: " + "; ".join(parts)
    for name in sorted(names_a):
        arr_a = outputs_a[name]
        arr_b = outputs_b[name]
        if not isinstance(arr_a, list) or not isinstance(arr_b, list):
            return (
                f"Output {name!r} must be a flat array; got "
                f"{type(arr_a).__name__} (baseline) vs "
                f"{type(arr_b).__name__} (rewritten)."
            )
        if len(arr_a) != len(arr_b):
            return (
                f"Output {name!r} length mismatch: {len(arr_a)} "
                f"(baseline) vs {len(arr_b)} (rewritten)."
            )
    return None


def _iter_leaf_pairs(
    baseline: dict, rewritten: dict
) -> Iterator[tuple[str, int, object, object]]:
    """Yield (name, index, baseline_val, rewritten_val) under outputs/.

    Assumes the shape check has already passed, so `outputs` is a dict
    with identical names and identical per-array lengths on both sides.
    Walks output arrays in sorted-name order so the mismatch list is
    deterministic across runs of the same inputs.
    """
    outputs_a = baseline["outputs"]
    outputs_b = rewritten["outputs"]
    for name in sorted(outputs_a.keys()):
        arr_a = outputs_a[name]
        arr_b = outputs_b[name]
        for i, (va, vb) in enumerate(zip(arr_a, arr_b)):
            yield (name, i, va, vb)


def compare_outputs(kernel_stem: str, tolerance_json: str) -> dict:
    """Numerically compare baseline vs rewritten reference.json.

    Reads baselines/<kernel_stem>/reference.json and
    baselines/<kernel_stem>/rewritten/reference.json, walks every
    leaf under their `outputs` dicts in parallel, and decides whether
    each pair agrees under the supplied tolerance. Writes a
    comparison.json artifact at
    baselines/<kernel_stem>/rewritten/comparison.json on BOTH the
    pass and fail paths so the operator always has a machine-readable
    record of the most recent comparator decision.

    `tolerance_json` is a JSON string with keys {kind, value, source}
    matching what the orchestrator's tolerance_block carries. `kind`
    is 'sig_figs' or 'decimal_digits', `value` is a positive integer.

    The result has the uniform {status, stdout, stderr, artifacts}
    shape shared by every deterministic tool. `status='ok'` iff every
    pair passed tolerance; any tolerance failure, any shape mismatch,
    a missing/unparseable reference.json, or a malformed
    tolerance_json all return `status='error'`. Shape errors carry a
    `shape_error` field in the written comparison.json so the
    operator can distinguish them from regular tolerance failures.

    The stderr mismatch list is truncated to the first
    _MAX_REPORTED_MISMATCHES entries with a "+ K more mismatches
    suppressed" footer when applicable; the comparison.json file uses
    the same truncation so an LLM-driven retry never sees an
    unbounded payload.
    """
    try:
        tolerance = json.loads(tolerance_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"tolerance_json is not valid JSON: {exc}. Expected an "
            f"object like {{'kind': 'sig_figs', 'value': 3, 'source': "
            f"'precision_advisor'}}."
        )
    if not isinstance(tolerance, dict):
        return _error(
            f"tolerance_json must be a JSON object; got "
            f"{type(tolerance).__name__}."
        )
    kind = tolerance.get("kind")
    value = tolerance.get("value")
    if kind not in {"sig_figs", "decimal_digits"}:
        return _error(
            f"Unsupported tolerance kind: {kind!r}. Expected "
            f"'sig_figs' or 'decimal_digits'."
        )
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return _error(
            f"Invalid tolerance value: {value!r}. Expected a positive "
            f"integer."
        )

    baseline_dir = Path("baselines") / kernel_stem
    baseline_path = baseline_dir / "reference.json"
    rewritten_dir = baseline_dir / "rewritten"
    rewritten_path = rewritten_dir / "reference.json"
    comparison_path = rewritten_dir / "comparison.json"

    if not baseline_path.is_file():
        return _error(
            f"Baseline reference.json not found at {baseline_path}. "
            f"Did run_baseline_driver run and succeed for this "
            f"kernel_stem?"
        )
    if not rewritten_path.is_file():
        return _error(
            f"Rewritten reference.json not found at {rewritten_path}. "
            f"Did run_rewritten_driver run and succeed for this "
            f"kernel_stem?"
        )

    try:
        baseline = _load_reference(baseline_path)
    except json.JSONDecodeError as exc:
        return _error(
            f"Baseline reference.json at {baseline_path} is not valid "
            f"JSON: {exc}."
        )
    try:
        rewritten = _load_reference(rewritten_path)
    except json.JSONDecodeError as exc:
        return _error(
            f"Rewritten reference.json at {rewritten_path} is not "
            f"valid JSON: {exc}."
        )

    shape_err = _shape_error(baseline, rewritten)
    if shape_err is not None:
        comparison_doc = {
            "status": "error",
            "tolerance": tolerance,
            "total_compared": 0,
            "mismatches": [],
            "shape_error": shape_err,
        }
        try:
            rewritten_dir.mkdir(parents=True, exist_ok=True)
            comparison_path.write_text(json.dumps(comparison_doc, indent=2))
        except OSError as exc:
            return _error(
                f"Shape mismatch and failed to write {comparison_path}: "
                f"{exc}. Shape error was: {shape_err}"
            )
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Shape mismatch: {shape_err}",
            "artifacts": [str(comparison_path)],
        }

    total = 0
    mismatches: list[dict] = []
    for name, idx, va, vb in _iter_leaf_pairs(baseline, rewritten):
        total += 1
        # Coerce to float for arithmetic; ints survive losslessly.
        try:
            fa = float(va)
            fb = float(vb)
        except (TypeError, ValueError):
            # Non-numeric leaf under outputs/ is a contract violation
            # the shape check missed; record it as a mismatch with
            # no abs_err / threshold so the operator can see it.
            mismatches.append({
                "name": name,
                "index": idx,
                "a": va,
                "b": vb,
                "abs_err": None,
                "threshold": None,
            })
            continue
        passed, abs_err, threshold = _value_within_tolerance(
            fa, fb, kind, value
        )
        if not passed:
            mismatches.append({
                "name": name,
                "index": idx,
                "a": fa,
                "b": fb,
                "abs_err": abs_err,
                "threshold": threshold,
            })

    truncated = mismatches[:_MAX_REPORTED_MISMATCHES]
    suppressed = max(0, len(mismatches) - _MAX_REPORTED_MISMATCHES)
    comparison_doc = {
        "status": "ok" if not mismatches else "error",
        "tolerance": tolerance,
        "total_compared": total,
        "mismatches": truncated,
    }
    try:
        rewritten_dir.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(comparison_doc, indent=2))
    except OSError as exc:
        return _error(f"Failed to write {comparison_path}: {exc}")

    if not mismatches:
        return {
            "status": "ok",
            "stdout": (
                f"All {total} compared values agree under "
                f"{kind}={value}."
            ),
            "stderr": "",
            "artifacts": [str(comparison_path)],
        }

    stderr_lines = [
        f"Tolerance mismatch under {kind}={value}: "
        f"{len(mismatches)}/{total} values disagree."
    ]
    for m in truncated:
        stderr_lines.append(
            f"  ({m['name']!r}, idx={m['index']}, a={m['a']}, "
            f"b={m['b']}, abs_err={m['abs_err']}, "
            f"threshold={m['threshold']})"
        )
    if suppressed:
        stderr_lines.append(
            f"  + {suppressed} more mismatches suppressed"
        )
    return {
        "status": "error",
        "stdout": "",
        "stderr": "\n".join(stderr_lines),
        "artifacts": [str(comparison_path)],
    }
