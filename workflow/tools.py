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

Per-language behavior (driver-file extension, compile command, splice
sentinels, env-var contract) is owned by the language-profile objects
in workflow.languages. Phase A.5 makes `language_id` a REQUIRED
argument on every tool wrapper: the orchestrator resolves a profile
once per run via workflow.languages.detect_language and threads the
resulting profile.id through every tool call. Calling a tool without
a `language_id` is a contract bug and raises TypeError at the
_resolve_profile helper; an unknown id raises KeyError. Tests that
call these tools directly must pass `language_id='kokkos'` explicitly.

Currently exposes:

  - compile_baseline_driver(kernel_stem, language_id): compile
    baselines/<kernel_stem>/<profile.driver_filename> produced by the
    baseline_harness agent into a native executable at
    baselines/<kernel_stem>/driver, using `profile.build_compile_command`.

  - run_baseline_driver(kernel_stem, language_id): execute the compiled
    driver at baselines/<kernel_stem>/driver and verify that it
    produces a parseable baselines/<kernel_stem>/reference.json.
    Subject to a per-run wall-clock timeout configured via the
    AGENT_PRECISION_RUN_TIMEOUT_SEC environment variable (default 60s).

  - splice_rewritten_kernel(kernel_stem, rewritten_kernel_source,
    language_id): read the baseline driver at
    baselines/<kernel_stem>/<profile.driver_filename>, replace the
    region strictly between the profile's KERNEL BEGIN / KERNEL END
    sentinels with the supplied rewritten kernel source, and write
    the result to baselines/<kernel_stem>/rewritten/
    <profile.driver_filename>. Pure text I/O; never invokes a
    subprocess and never modifies the baseline file in place.

  - compile_rewritten_driver(kernel_stem, language_id): compile
    baselines/<kernel_stem>/rewritten/<profile.driver_filename>
    (produced by splice_rewritten_kernel) into
    baselines/<kernel_stem>/rewritten/driver. Shares the env-var
    contract, compile flags, and result schema of
    compile_baseline_driver; only the directory differs.

  - run_rewritten_driver(kernel_stem, language_id): execute the
    compiled rewritten driver at
    baselines/<kernel_stem>/rewritten/driver and verify it produces a
    parseable baselines/<kernel_stem>/rewritten/reference.json.
    Shares the env-var contract (AGENT_PRECISION_RUN_TIMEOUT_SEC),
    subprocess shape, and result schema of run_baseline_driver; only
    the directory differs. The baseline tree is never touched.

  - compare_outputs(kernel_stem, tolerance_json, language_id): numerically compare
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

  - probe_step(kernel_stem, precision, seed, language_id): the
    workhorse of the v1 probe pipeline. Reads the per-precision
    driver template at baselines/<kernel_stem>/probe/<precision>/
    driver.cpp (written by spawn_baseline_harness via the v1
    4-driver schema), rewrites the named RNG_SEED constant to the
    requested seed, writes the result into a per-(precision, seed)
    sibling directory at baselines/<kernel_stem>/probe/
    <precision>_seed<seed>/, then compiles and runs that copy by
    reusing _compile_driver and _run_driver. The per-(precision,
    seed) directory keeps the harness-written seed=42 template
    untouched so subsequent probe_step calls always rewrite from a
    clean baseline. On success, artifacts is the single-element
    list ["baselines/<stem>/probe/<precision>_seed<seed>/
    reference.json"]; on any failure (missing template, RNG_SEED
    line not found or non-unique, compile/run error) the standard
    error result is returned. Like the baseline run, a probe_step
    failure is non-fatal to the surrounding pipeline -- the missing
    cell is recorded by probe_compare and the analyst still runs.

  - probe_compare(kernel_stem, language_id): aggregates the 8
    per-(precision, seed) reference.json files written by probe_step
    into a single evidence.json document at baselines/<kernel_stem>/
    probe/evidence.json that the analyst prompt addendum consumes.
    For each (precision, seed) cell it records a status (ok /
    missing / load_error / shape_error) so the analyst sees which
    signals are real, and for every cell that has an `ok` partner
    quad-same-seed cell it computes per-output stats vs the quad
    ground truth (max-absrel error, mean-absrel error, max-absolute
    error, count of finite mismatches). It also computes per-output
    cross-seed deltas (how much the per-precision max-absrel error
    changes between seed 42 and seed 43) so the analyst can tell
    seed-correlated precision pain from seed-independent precision
    pain. The tool itself hard-errors only when the quad ground
    truth for the canonical seed (42) is missing -- otherwise it
    reports whatever cells are present. Pure file + arithmetic
    I/O; no subprocess.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from .languages import KOKKOS_PROFILE, LanguageProfile
from .languages.base import make_error_result

# Backward-compat re-export. Tests historically imported KOKKOS_ROOT_ENV
# from workflow.tools directly; the Kokkos profile now owns the canonical
# value as workflow.languages.kokkos.ROOT_ENV, but we re-bind it here so
# the Phase A refactor does not force a sweep through the test suite. The
# compile-flag constants (CXX, CXX_STD, OPT_FLAGS, KOKKOS_LIBS, EXTRA_LIBS)
# are now Kokkos-private and live in workflow.languages.kokkos.
from .languages import kokkos as _kokkos

KOKKOS_ROOT_ENV = _kokkos.ROOT_ENV

# Environment variable that caps the wall-clock seconds run_baseline_driver
# will wait for the compiled driver to finish. Namespaced for the same
# reason as KOKKOS_ROOT_ENV; the explicit "_SEC" suffix avoids ms/s
# ambiguity. Kernels in v0 are small enough that 60s is generous; raise
# this as kernels grow (e.g. once deployment-scale inputs land).
RUN_TIMEOUT_ENV = "AGENT_PRECISION_RUN_TIMEOUT_SEC"
DEFAULT_RUN_TIMEOUT_SEC = 60

# Splice sentinels. Historically these were the only sentinels in the
# repo and lived as module-level constants; tests and the
# baseline_harness prompt both pin them. They are now owned by
# LanguageProfile (so a future Fortran profile can override), but the
# v1 profiles all default to the C-style versions, and the Kokkos
# profile's values are what we re-export here for back-compat.
KERNEL_BEGIN_SENTINEL = KOKKOS_PROFILE.sentinel_begin
KERNEL_END_SENTINEL = KOKKOS_PROFILE.sentinel_end


def _error(stderr: str) -> dict:
    """Build a uniform error result with empty stdout and no artifacts."""
    return make_error_result(stderr)


def _compile_driver(
    driver_dir: Path,
    profile: LanguageProfile,
    missing_source_hint: str,
) -> dict:
    """Compile <driver_dir>/<profile.driver_filename> into <driver_dir>/driver.

    Shared implementation behind compile_baseline_driver and
    compile_rewritten_driver. The two public wrappers differ only in
    which directory they target — the preflight, command shape,
    subprocess invocation, error wrapping, and result schema are
    identical. Pulling the body here keeps the dynamic-verification
    chain (splice -> compile_rewritten -> run_rewritten -> compare)
    from drifting from the baseline chain it parallels.

    `profile` carries the language-specific bits: the driver filename
    to look for, the preflight check (env vars set, toolchain on PATH),
    and the build_compile_command callable that produces the subprocess
    argv. The driver-binary path is always `<driver_dir>/driver`
    regardless of language; only the source file extension varies.

    `missing_source_hint` is the human-readable hint appended to the
    "driver source not found" error so the operator knows which
    upstream tool was supposed to have written that file (e.g.
    spawn_baseline_harness for the baseline, splice_rewritten_kernel
    for the rewritten variant). Everything else about the error result
    is identical across the two wrappers.
    """
    preflight_error = profile.preflight()
    if preflight_error is not None:
        return preflight_error

    driver_src = driver_dir / profile.driver_filename
    driver_bin = driver_dir / "driver"
    if not driver_src.is_file():
        return _error(
            f"Driver source not found at {driver_src}. {missing_source_hint}"
        )

    cmd = profile.build_compile_command(driver_src, driver_bin)
    compiler_name = cmd[0] if cmd else "<empty command>"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return _error(f"Failed to invoke {compiler_name!r}: {exc}")

    if proc.returncode != 0:
        return {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": (
                f"{compiler_name} exited with code {proc.returncode}.\n"
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


def syntax_check_driver_source(
    profile: LanguageProfile, source: str, label: str
) -> dict | None:
    """Syntax-check a candidate driver source in a temp dir.

    Returns None on success (or when the profile has no gate / the
    toolchain isn't available — validation is silently skipped in both
    cases, since forcing a check would tie every harness run to a
    working install the operator might not have set up yet).

    Returns a `_error()`-shaped dict on failure, with `label` folded
    into the stderr so a multi-driver payload (Kokkos v1's 4-driver
    output) can name which precision failed. The dict shape matches
    what workflow.tools._compile_driver returns on a real compile
    failure, so the orchestrator's harness branch can hand it straight
    back as an `is_error: True` tool_result and let the harness see
    exactly what g++ said.

    The driver source is written to a NamedTemporaryFile with the
    profile's driver_filename suffix (so g++ picks the right frontend
    from the extension) and the compiler is invoked with
    -fsyntax-only, which stops after parsing + typechecking and never
    writes an object file or invokes the linker. That means we can
    validate the harness's structural correctness (aliases resolve,
    every declared name has a matching declaration, function calls
    match signatures, ...) without needing any of the compile step's
    -l<lib> / -L / -o flags. The include path IS required for Kokkos
    (its <Kokkos_Core.hpp> is what defines the View/parallel_for
    surface the harness uses); the profile's callable takes care of
    that.
    """
    build = profile.build_syntax_check_command
    # Extension-only suffix (e.g. ".cpp") so the compiler's frontend
    # dispatch picks the right language.
    suffix = Path(profile.driver_filename).suffix or ""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)
    try:
        cmd = build(tmp_path)
        if cmd is None:
            return None
        compiler_name = cmd[0] if cmd else "<empty command>"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            # Missing compiler is treated as a skip, same as a missing
            # env var: we can't check, so we don't block the harness.
            print(
                f"[syntax_check] skipping {label} check: "
                f"failed to invoke {compiler_name!r}: {exc}",
                file=sys.stderr,
            )
            return None
        if proc.returncode != 0:
            return {
                "status": "error",
                "stdout": proc.stdout,
                "stderr": (
                    f"{compiler_name} -fsyntax-only rejected the "
                    f"baseline_harness output ({label}).\n"
                    f"Fix the driver source and resubmit.\n"
                    f"Command: {' '.join(cmd)}\n\n{proc.stderr}"
                ),
                "artifacts": [],
            }
        return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def compile_baseline_driver(
    kernel_stem: str, language_id: str
) -> dict:
    """Compile baselines/<kernel_stem>/<profile.driver_filename>.

    `language_id` selects which workflow.languages profile drives the
    compile (driver-file extension, compiler argv, env-var preflight).
    Required (Phase A.5): the orchestrator resolves a profile once per
    run and threads its id through every tool call. An unknown id
    surfaces as a KeyError from _resolve_profile; a missing id surfaces
    as a TypeError (both indicate a contract bug in the caller).

    Returns a `{status, stdout, stderr, artifacts}` dict, where
    `status` is 'ok' on a successful compile and 'error' otherwise.
    `artifacts` is a list of created/expected output paths (the driver
    binary) — empty on error.

    The driver source is expected to already exist at
    baselines/<kernel_stem>/<profile.driver_filename> (written by the
    baseline_harness agent on HITL approval). This helper does not run
    the compiled binary; that is a separate step.
    """
    profile = _resolve_profile(language_id)
    return _compile_driver(
        Path("baselines") / kernel_stem,
        profile,
        missing_source_hint=(
            "Did spawn_baseline_harness run and get approved for this "
            "kernel_stem?"
        ),
    )


def compile_rewritten_driver(
    kernel_stem: str, language_id: str
) -> dict:
    """Compile baselines/<kernel_stem>/rewritten/<profile.driver_filename>.

    Companion to compile_baseline_driver, targeting the rewritten
    driver produced by splice_rewritten_kernel. Same env-var contract
    (per the profile's `env_required`), same compile flags, same result
    schema. The compiled binary lands at
    baselines/<kernel_stem>/rewritten/driver, alongside the source.

    Like the baseline compile, this is a side artifact: a non-zero
    compile result is non-fatal to the surrounding pipeline (the
    analyst -> rewriter -> verifier loop still runs, and finish remains
    reachable on verifier accept).
    """
    profile = _resolve_profile(language_id)
    return _compile_driver(
        Path("baselines") / kernel_stem / "rewritten",
        profile,
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


def run_baseline_driver(
    kernel_stem: str, language_id: str
) -> dict:
    """Execute baselines/<kernel_stem>/driver and validate reference.json.

    `language_id` is accepted for call-shape symmetry with the rest of
    the dynamic-verification chain; the run subprocess is the same
    `./driver` invocation regardless of language because the compiled
    binary always lives at the same path. Required (Phase A.5): see
    compile_baseline_driver for the rationale.

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
    # language_id retained for call-shape symmetry; resolved for early
    # validation (an unknown id should error here, not silently no-op).
    _resolve_profile(language_id)
    return _run_driver(
        Path("baselines") / kernel_stem,
        missing_binary_hint=(
            "Did compile_baseline_driver run and succeed for this "
            "kernel_stem?"
        ),
    )


def run_rewritten_driver(
    kernel_stem: str, language_id: str
) -> dict:
    """Execute baselines/<kernel_stem>/rewritten/driver and validate JSON.

    Companion to run_baseline_driver, targeting the rewritten driver
    produced by compile_rewritten_driver. Same env-var contract
    (AGENT_PRECISION_RUN_TIMEOUT_SEC), same subprocess shape, same
    result schema. The rewritten driver runs with cwd set to
    baselines/<kernel_stem>/rewritten/, so its `./reference.json`
    lands inside the rewritten subtree and the baseline tree
    (baselines/<kernel_stem>/{<driver_filename>, driver, reference.json})
    is never touched by this call.

    Like the baseline run, this is a side artifact: a non-zero run
    result is non-fatal to the surrounding pipeline (the analyst ->
    rewriter -> verifier loop still runs, and finish remains reachable
    on verifier accept). On success, `artifacts` is the single-element
    list `["baselines/<kernel_stem>/rewritten/reference.json"]`.
    """
    _resolve_profile(language_id)
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
    kernel_stem: str,
    rewritten_kernel_source: str,
    language_id: str,
) -> dict:
    """Splice `rewritten_kernel_source` into the baseline driver template.

    Reads baselines/<kernel_stem>/<profile.driver_filename>, locates
    the unique profile.sentinel_begin / sentinel_end lines (byte-exact,
    each on its own line with no surrounding indentation), replaces
    the text strictly BETWEEN them (sentinels themselves preserved)
    with `rewritten_kernel_source`, and writes the result to
    baselines/<kernel_stem>/rewritten/<profile.driver_filename>. The
    directory is created if needed. The baseline driver is never
    modified.

    The result has the uniform `{status, stdout, stderr, artifacts}`
    shape shared with compile_baseline_driver and run_baseline_driver:

      - On success: status='ok', stdout='', stderr='',
        artifacts=['baselines/<stem>/rewritten/<driver_filename>'].
      - On any error: status='error', stdout='', a descriptive stderr,
        and artifacts=[].

    This is pure text I/O. It never invokes a subprocess.
    """
    profile = _resolve_profile(language_id)

    if not rewritten_kernel_source:
        return _error(
            "rewritten_kernel_source is empty; nothing to splice."
        )

    baseline_path = (
        Path("baselines") / kernel_stem / profile.driver_filename
    )
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

    begin_idx = _find_unique_sentinel_line(lines, profile.sentinel_begin)
    if begin_idx is None:
        return _error(
            f"Baseline driver at {baseline_path} does not contain "
            f"exactly one {profile.sentinel_begin!r} line on its own "
            f"(no surrounding indentation or whitespace)."
        )

    end_idx = _find_unique_sentinel_line(lines, profile.sentinel_end)
    if end_idx is None:
        return _error(
            f"Baseline driver at {baseline_path} does not contain "
            f"exactly one {profile.sentinel_end!r} line on its own "
            f"(no surrounding indentation or whitespace)."
        )

    if begin_idx >= end_idx:
        return _error(
            f"Baseline driver at {baseline_path} has "
            f"{profile.sentinel_end!r} at or before "
            f"{profile.sentinel_begin!r} (line {end_idx + 1} vs line "
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
    out_path = out_dir / profile.driver_filename
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


def _resolve_profile(language_id: str) -> LanguageProfile:
    """Look up a LanguageProfile by required `language_id`.

    Phase A.5: `language_id` is mandatory. The orchestrator resolves
    the profile once per run via workflow.languages.detect_language and
    threads the resulting profile.id through every tool call, so a tool
    being called without one indicates a contract bug somewhere in the
    dispatch path. We raise TypeError with a clear message rather than
    silently defaulting to Kokkos, because a silent default in the
    multi-language world masks exactly the kind of mistake (forgotten
    plumbing for a new language) that this required-arg contract
    exists to catch.

    Unknown ids surface through workflow.languages.get_profile_by_id,
    which raises KeyError with the list of known ids.

    Kept as a thin private helper so the public tool wrappers do not
    import workflow.languages.get_profile_by_id directly; this keeps
    the resolution policy (required, no fallback) in one place.
    """
    if language_id is None:
        raise TypeError(
            "language_id is required (Phase A.5). The orchestrator "
            "must resolve a profile via workflow.languages."
            "detect_language and pass its id to every tool call. If "
            "you are calling a tool directly (tests, REPL), pass "
            "language_id='kokkos' explicitly."
        )
    if language_id == KOKKOS_PROFILE.id:
        return KOKKOS_PROFILE
    # Deferred import to avoid a circular dependency at module load
    # (workflow.languages imports workflow.languages.kokkos, which we
    # already have via the top-level import; this guards against a
    # future profile module that imports tools).
    from .languages import get_profile_by_id

    return get_profile_by_id(language_id)


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


def compare_outputs(
    kernel_stem: str,
    tolerance_json: str,
    language_id: str,
) -> dict:
    """Numerically compare baseline vs rewritten reference.json.

    `language_id` is accepted for call-shape symmetry with the rest of
    the dynamic-verification chain. The comparator itself is fully
    language-agnostic — it walks two reference.json files that obey
    the harness contract regardless of which language produced them —
    but accepting the arg keeps every tool's schema uniform. Required
    (Phase A.5): see compile_baseline_driver for the rationale.

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
    _resolve_profile(language_id)
    try:
        tolerance = json.loads(tolerance_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"tolerance_json is not valid JSON: {exc}. Expected an "
            f"object like {{'kind': 'sig_figs', 'value': 3, 'source': "
            f"'user_cli'}}."
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


# ---------- probe_step / probe_compare: v1 probe pipeline ----------

# RNG_SEED contract regex. The baseline_harness prompt mandates the
# seed appear EXACTLY as `static constexpr int RNG_SEED = <int>;` on
# its own line above the KERNEL BEGIN sentinel (so the splice tool
# never touches it). probe_step rewrites the integer to the requested
# seed; we keep the regex tight (multiline-anchored ^...;$) so a stray
# match inside a comment or string can't accidentally be rewritten,
# and we require exactly one match per driver so a malformed harness
# output (zero or multiple RNG_SEED lines) fails loudly rather than
# silently driving the probe with the wrong seed.
_RNG_SEED_LINE_RE = re.compile(
    r"^static constexpr int RNG_SEED = \d+;$",
    re.MULTILINE,
)


def _rewrite_rng_seed(source: str, seed: int) -> tuple[str | None, str | None]:
    """Rewrite the unique RNG_SEED line in `source` to `seed`.

    Returns `(new_source, None)` on success or `(None, error_msg)` on
    any contract violation. Used by probe_step to retarget a
    per-precision driver template (harness-written with seed=42) at a
    different seed without re-asking the LLM. The new_source preserves
    the rest of the file byte-for-byte: only the integer literal
    inside the matched RNG_SEED line is changed, so the splice
    sentinels, alias block, kernel body, and main() are all preserved
    exactly.
    """
    matches = _RNG_SEED_LINE_RE.findall(source)
    if len(matches) == 0:
        return (None, (
            "Driver source has no `static constexpr int RNG_SEED = "
            "<int>;` line on its own (required by the baseline_harness "
            "RNG_SEED contract)."
        ))
    if len(matches) > 1:
        return (None, (
            f"Driver source has {len(matches)} `static constexpr int "
            f"RNG_SEED = <int>;` lines; the RNG_SEED contract requires "
            f"exactly one so the seed-rewrite is unambiguous."
        ))
    new_source = _RNG_SEED_LINE_RE.sub(
        f"static constexpr int RNG_SEED = {seed};",
        source,
        count=1,
    )
    return (new_source, None)


def probe_step(
    kernel_stem: str,
    precision: str,
    seed: int,
    language_id: str,
) -> dict:
    """Seed-rewrite, compile, and run one per-(precision, seed) probe driver.

    The probe pipeline runs the baseline_harness-emitted per-precision
    drivers across a small set of seeds to give the analyst evidence
    about how the kernel responds to precision changes -- the analyst
    is otherwise reasoning from source only. probe_step is the
    workhorse: one call per (precision, seed) cell, fused
    seed-rewrite + compile + run so we do not multiply the HITL
    approval count by 3.

    Reads `baselines/<kernel_stem>/probe/<precision>/driver.cpp` (the
    template the v1 baseline_harness wrote alongside the canonical
    baseline; see workflow.orchestrator._execute_tool's
    spawn_baseline_harness branch and the test that pins the layout).
    Rewrites the unique `static constexpr int RNG_SEED = <int>;` line
    to the requested `seed` and writes the result to
    `baselines/<kernel_stem>/probe/<precision>_seed<seed>/driver.cpp`.
    Then reuses _compile_driver and _run_driver against that sibling
    directory so the compile artifact, the run subprocess, and the
    `reference.json` all land inside it -- the template directory
    stays untouched so re-invocations always start from a clean
    seed=42 source.

    `precision` is one of the keys the harness emitted (currently
    quad / double / float / mixed_io for the Kokkos profile). `seed`
    is an int; the v1 orchestrator drives {42, 43}. The result has
    the uniform {status, stdout, stderr, artifacts} shape; on
    success, `artifacts` is the single-element list
    ["baselines/<kernel_stem>/probe/<precision>_seed<seed>/reference.json"].
    A failure at any step (missing template, malformed RNG_SEED,
    compile error, run timeout, missing or invalid reference.json)
    returns status='error' with a descriptive stderr; probe_compare
    records the failed cell as `missing` or `error` so the analyst
    still gets whatever signal is available.

    `language_id` is accepted for call-shape symmetry with the rest
    of the dynamic-verification chain and is validated (an unknown
    id errors here, not silently). v1 only emits probe templates for
    profiles with non-empty `probe_precisions` (Kokkos in v1); the
    deferred Commit 6 extends this to CUDA / HIP / SYCL / OMP-offload.
    """
    profile = _resolve_profile(language_id)

    if not isinstance(seed, int) or isinstance(seed, bool):
        return _error(
            f"seed must be an int; got {type(seed).__name__} ({seed!r})."
        )
    if not precision or not isinstance(precision, str):
        return _error(
            f"precision must be a non-empty string; got {precision!r}."
        )

    template_dir = (
        Path("baselines") / kernel_stem / "probe" / precision
    )
    template_src = template_dir / profile.driver_filename
    if not template_src.is_file():
        return _error(
            f"Probe driver template not found at {template_src}. Did "
            f"spawn_baseline_harness run and emit a `drivers.{precision}` "
            f"key for this kernel_stem? (The v1 harness writes one "
            f"template per precision under baselines/<stem>/probe/"
            f"<precision>/.)"
        )

    try:
        template_text = template_src.read_text()
    except OSError as exc:
        return _error(f"Failed to read {template_src}: {exc}")

    rewritten, err = _rewrite_rng_seed(template_text, seed)
    if err is not None:
        return _error(f"In {template_src}: {err}")

    target_dir = (
        Path("baselines") / kernel_stem / "probe" / f"{precision}_seed{seed}"
    )
    target_src = target_dir / profile.driver_filename
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_src.write_text(rewritten)
    except OSError as exc:
        return _error(f"Failed to write {target_src}: {exc}")

    compile_result = _compile_driver(
        target_dir,
        profile,
        missing_source_hint=(
            "probe_step just wrote the driver source; this should not "
            "happen. Check filesystem permissions on "
            f"{target_dir}."
        ),
    )
    if compile_result["status"] != "ok":
        return compile_result

    run_result = _run_driver(
        target_dir,
        missing_binary_hint=(
            "probe_step just compiled the driver; this should not "
            "happen. Check filesystem permissions on "
            f"{target_dir}."
        ),
    )
    return run_result


def _probe_cell_dir(kernel_stem: str, precision: str, seed: int) -> Path:
    """Return the on-disk directory probe_step writes for one cell.

    Centralized so probe_compare and probe_step share the layout
    convention; changing the directory naming scheme means changing
    this helper.
    """
    return (
        Path("baselines") / kernel_stem / "probe"
        / f"{precision}_seed{seed}"
    )


# Probe matrix. Kept private to this module: the orchestrator drives
# probe_step once per cell and probe_compare reads back whatever cells
# happen to exist, so neither caller needs the matrix at hand. If a
# future commit changes the set of precisions or seeds, change it here
# and the probe loop in orchestrator.py together.
_PROBE_SEEDS: tuple[int, ...] = (42, 43)


def _load_probe_cell(
    kernel_stem: str, precision: str, seed: int
) -> tuple[dict | None, str, str | None]:
    """Load one (precision, seed) reference.json and classify its status.

    Returns `(payload, status, error)` where `status` is one of
    'ok' / 'missing' / 'load_error' and `error` is a human-readable
    message for the non-ok cases (None for 'ok'). Used by probe_compare
    to fill in the per-cell `status` field in evidence.json; the
    analyst then sees which cells have real data.

    A 'missing' cell means probe_step never ran or its run failed
    before reference.json was written (probe_step deletes any stale
    reference.json before invoking the subprocess, so a missing file
    here is unambiguous evidence of a failed cell, not a stale one).
    """
    cell_dir = _probe_cell_dir(kernel_stem, precision, seed)
    ref_path = cell_dir / "reference.json"
    if not ref_path.is_file():
        return (None, "missing", f"reference.json not found at {ref_path}.")
    try:
        with ref_path.open("r") as fp:
            payload = json.load(fp)
    except json.JSONDecodeError as exc:
        return (None, "load_error", f"{ref_path} is not valid JSON: {exc}.")
    except OSError as exc:
        return (None, "load_error", f"Failed to read {ref_path}: {exc}.")
    return (payload, "ok", None)


def _per_output_stats_vs_quad(
    quad_payload: dict, other_payload: dict
) -> tuple[dict | None, str | None]:
    """Compute per-output error stats of `other_payload` vs `quad_payload`.

    Returns `(stats, None)` on success or `(None, error_msg)` on a
    shape mismatch (reuses `_shape_error` so the contract stays in
    one place). `stats` maps each output array name to:

      {n: int,              total elements compared
       n_finite: int,       elements where both sides are finite
       n_nonfinite: int,    elements where either side is NaN or inf
       max_absrel: float,   max |a-q| / max(|a|, |q|, eps) over finite pairs
       mean_absrel: float,  mean of the same over finite pairs
       max_abserror: float} max |a-q| over finite pairs

    The relative-error denominator uses a tiny epsilon floor (1e-300)
    so quad-zero / other-zero pairs do not blow the stat up to inf;
    in that exact-zero case both numerator and denominator are zero,
    yielding 0.0 -- which is the analyst-friendly answer (the cell
    agreed perfectly). NaN and inf pairs are excluded from the
    finite-arithmetic stats but counted in `n_nonfinite` so the
    analyst sees that something special happened.
    """
    shape_err = _shape_error(quad_payload, other_payload)
    if shape_err is not None:
        return (None, shape_err)

    stats: dict[str, dict] = {}
    eps = 1e-300
    outputs_q = quad_payload["outputs"]
    outputs_o = other_payload["outputs"]
    for name in sorted(outputs_q.keys()):
        arr_q = outputs_q[name]
        arr_o = outputs_o[name]
        n = len(arr_q)
        n_finite = 0
        n_nonfinite = 0
        max_absrel = 0.0
        sum_absrel = 0.0
        max_abserror = 0.0
        for vq, vo in zip(arr_q, arr_o):
            try:
                fq = float(vq)
                fo = float(vo)
            except (TypeError, ValueError):
                n_nonfinite += 1
                continue
            if (
                math.isnan(fq) or math.isnan(fo)
                or math.isinf(fq) or math.isinf(fo)
            ):
                n_nonfinite += 1
                continue
            n_finite += 1
            abs_err = abs(fo - fq)
            denom = max(abs(fq), abs(fo), eps)
            rel = abs_err / denom
            if rel > max_absrel:
                max_absrel = rel
            sum_absrel += rel
            if abs_err > max_abserror:
                max_abserror = abs_err
        mean_absrel = sum_absrel / n_finite if n_finite > 0 else 0.0
        stats[name] = {
            "n": n,
            "n_finite": n_finite,
            "n_nonfinite": n_nonfinite,
            "max_absrel": max_absrel,
            "mean_absrel": mean_absrel,
            "max_abserror": max_abserror,
        }
    return (stats, None)


def probe_compare(kernel_stem: str, language_id: str) -> dict:
    """Aggregate the per-cell probe runs into evidence.json for the analyst.

    Walks the probe matrix (every (precision, seed) cell the v1 probe
    pipeline emits, currently 4 precisions x 2 seeds = 8 cells for
    Kokkos), classifies each cell's reference.json (ok / missing /
    load_error / shape_error), and for every non-quad cell that has
    an `ok` partner `quad_seed<N>` cell computes per-output stats
    against the quad reference at the same seed. Adds per-output
    cross-seed deltas so the analyst can distinguish seed-correlated
    precision pain from seed-independent precision pain.

    Writes the aggregated document to
    `baselines/<kernel_stem>/probe/evidence.json`. That path is also
    what the orchestrator's spawn_analyst branch will read to attach
    the PROBE EVIDENCE block to the analyst task prompt (Commit 4).

    Hard-errors only when the quad ground truth for the canonical
    seed (42) is missing -- without that cell there is no reference
    to compare against and the per-cell stats would all be empty.
    Any other missing or failed cell is recorded in `cells[<name>]`
    with a non-ok status and skipped during the stats walk; the
    analyst sees exactly which signals are real.

    The probe quad reference values are written as `%.34Qg` tokens by
    the harness's quadmath_snprintf call but parsed back through
    Python's `json.load`, which truncates them to IEEE 754 double
    (~15-17 decimal digits). For the purpose of comparing a
    float-precision or mixed_io driver against the quad ground
    truth, that double truncation is harmless -- the float driver's
    error floor (~2^-23 ~= 1e-7) is many orders of magnitude above
    the double round-off (~2^-52 ~= 2e-16) introduced by the parse.
    A future commit that wants to compare double-precision drivers
    against quad with quad-level resolution would need to either
    parse the JSON as decimal strings or have the harness write a
    parallel quad-as-string output array; v1 punts on this because
    no current analyst question requires the extra resolution.

    `language_id` is accepted for symmetry and validated; this tool
    itself is language-agnostic (it reads JSON files that obey the
    harness contract regardless of source language).
    """
    _resolve_profile(language_id)
    probe_dir = Path("baselines") / kernel_stem / "probe"
    evidence_path = probe_dir / "evidence.json"

    # Probe the canonical-seed quad cell first so we can fail fast
    # if the ground truth is missing.
    canonical_seed = _PROBE_SEEDS[0]
    quad_canonical_payload, quad_canonical_status, quad_canonical_err = (
        _load_probe_cell(kernel_stem, "quad", canonical_seed)
    )
    if quad_canonical_status != "ok":
        return _error(
            f"Probe ground truth missing: the canonical quad cell at "
            f"seed={canonical_seed} could not be loaded "
            f"({quad_canonical_status}: {quad_canonical_err}). "
            f"probe_compare cannot run without quad_seed{canonical_seed} "
            f"as a comparison baseline; check that probe_step("
            f"precision='quad', seed={canonical_seed}) ran and succeeded."
        )

    # Discover which precisions to walk by reading the harness-written
    # template directories rather than hardcoding the precision list;
    # this keeps probe_compare in sync with the profile's
    # probe_precisions tuple without importing it directly.
    template_root = Path("baselines") / kernel_stem / "probe"
    precisions: list[str] = []
    if template_root.is_dir():
        for entry in sorted(template_root.iterdir()):
            if not entry.is_dir():
                continue
            # Skip the per-(precision, seed) cell dirs; the template
            # dirs have plain precision names like "quad", "float".
            if "_seed" in entry.name:
                continue
            precisions.append(entry.name)
    if not precisions:
        return _error(
            f"No probe driver templates found under {template_root}. "
            f"Did spawn_baseline_harness run with the v1 4-driver "
            f"schema for this kernel_stem?"
        )

    # cells[<precision>_seed<seed>] -> {status, error?, stats?, shape_error?}
    cells: dict[str, dict] = {}
    # per_seed_quad_payload[seed] caches the quad reference for that
    # seed so the stats loop doesn't reload it once per non-quad cell.
    per_seed_quad_payload: dict[int, dict] = {
        canonical_seed: quad_canonical_payload
    }

    for seed in _PROBE_SEEDS:
        for precision in precisions:
            cell_name = f"{precision}_seed{seed}"
            payload, status, err = _load_probe_cell(
                kernel_stem, precision, seed
            )
            cell: dict = {"status": status}
            if err is not None:
                cell["error"] = err
            cells[cell_name] = cell
            if precision == "quad" and status == "ok":
                per_seed_quad_payload[seed] = payload

    # Stats walk: for each non-quad ok cell with an ok quad partner
    # at the same seed, compute per-output stats vs quad.
    for seed in _PROBE_SEEDS:
        quad_partner = per_seed_quad_payload.get(seed)
        for precision in precisions:
            if precision == "quad":
                continue
            cell_name = f"{precision}_seed{seed}"
            cell = cells[cell_name]
            if cell["status"] != "ok":
                continue
            if quad_partner is None:
                # No quad partner at this seed -> record but don't
                # promote to a hard error. The analyst sees a
                # "no_quad_partner" status and treats this cell as
                # missing comparison data.
                cell["status"] = "no_quad_partner"
                cell["error"] = (
                    f"quad_seed{seed} did not load; cannot compute "
                    f"stats for {cell_name} without a ground truth."
                )
                continue
            payload, _, _ = _load_probe_cell(
                kernel_stem, precision, seed
            )
            stats, shape_err = _per_output_stats_vs_quad(
                quad_partner, payload
            )
            if shape_err is not None:
                cell["status"] = "shape_error"
                cell["shape_error"] = shape_err
                continue
            cell["stats"] = stats

    # Cross-seed deltas: for each non-quad precision, if both
    # precision_seed42 and precision_seed43 have stats, compute the
    # per-output delta of max_absrel. Stored at the top level under
    # `cross_seed_deltas[<precision>][<output_name>]`.
    cross_seed_deltas: dict[str, dict] = {}
    if len(_PROBE_SEEDS) >= 2:
        s_a, s_b = _PROBE_SEEDS[0], _PROBE_SEEDS[1]
        for precision in precisions:
            if precision == "quad":
                continue
            cell_a = cells.get(f"{precision}_seed{s_a}", {})
            cell_b = cells.get(f"{precision}_seed{s_b}", {})
            stats_a = cell_a.get("stats")
            stats_b = cell_b.get("stats")
            if not stats_a or not stats_b:
                continue
            per_output: dict[str, float] = {}
            for name in sorted(set(stats_a.keys()) & set(stats_b.keys())):
                per_output[name] = abs(
                    stats_a[name]["max_absrel"]
                    - stats_b[name]["max_absrel"]
                )
            if per_output:
                cross_seed_deltas[precision] = per_output

    evidence = {
        "kernel_stem": kernel_stem,
        "precisions": precisions,
        "seeds": list(_PROBE_SEEDS),
        "cells": cells,
        "cross_seed_deltas": cross_seed_deltas,
    }
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2))
    except OSError as exc:
        return _error(f"Failed to write {evidence_path}: {exc}")

    # Summarize for the orchestrator's tool-result stdout: cell counts
    # by status, plus the worst-case max_absrel across all ok cells.
    status_counts: dict[str, int] = {}
    worst_absrel = 0.0
    worst_cell = ""
    worst_output = ""
    for cell_name, cell in cells.items():
        st = cell["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        stats = cell.get("stats")
        if not stats:
            continue
        for output_name, output_stats in stats.items():
            if output_stats["max_absrel"] > worst_absrel:
                worst_absrel = output_stats["max_absrel"]
                worst_cell = cell_name
                worst_output = output_name
    summary_parts = [
        f"{count} {status}" for status, count in sorted(status_counts.items())
    ]
    summary = (
        f"Probe evidence written to {evidence_path}: "
        f"{', '.join(summary_parts)}."
    )
    if worst_cell:
        summary += (
            f" Worst max_absrel vs quad: {worst_absrel:.3e} at "
            f"{worst_cell}/{worst_output}."
        )
    return {
        "status": "ok",
        "stdout": summary,
        "stderr": "",
        "artifacts": [str(evidence_path)],
    }


# ---------------------------------------------------------------------------
# Probe-vs-verdict consistency check (post-analyst safety net)
# ---------------------------------------------------------------------------
#
# See AGENTS.md "Probe pipeline" for the design rationale. Motivating case:
# the nbody_force N=5 consistency sweep had one run where the analyst
# returned action='downcast' target_precision='float' for every storage
# View despite the float probe cell showing max_absrel ~= 0.34 on vy
# against a sig_figs=6 (~1e-6) tolerance -- five orders of magnitude over.
# The verifier accepted it, the comparator rejected it, and the
# orchestrator burned four full rewrite cycles before MAX_TURNS killed
# the run. This helper catches that class of failure BEFORE spawn_rewriter
# is even called.
#
# Scope in v0 is intentionally narrow (see the emulate/target_precision
# skip conditions below): the check only flags what the probe pipeline
# has actual evidence for.
def check_analyst_verdict_against_probe(
    verdict: dict,
    evidence: dict,
    tolerance: dict,
) -> list[str]:
    """Compare an analyst verdict against probe evidence; return violation
    strings.

    An empty list means "no inconsistency detected" -- either everything
    checks out, or there was no basis to check (missing evidence,
    action='keep', action='emulate', target_precision with no matching
    probe cell, probe cell status != 'ok'). All of those are silent
    skips by design: absence of evidence is not evidence of a
    violation, and this check exists to catch analyst verdicts the
    probe positively contradicts, not to demand universal probe
    coverage.

    Arithmetic (per the design decisions in the item-#5 clarification
    round):
      - For action='downcast', look up the probe cell whose precision
        matches target_precision at the canonical seed (42 -- the
        first entry in _PROBE_SEEDS).
      - Convert the tolerance to a numerical threshold:
          * kind='sig_figs':      threshold = 10 ** -value  (relative)
          * kind='decimal_digits': threshold = 10 ** -value  (absolute)
      - Compare per-output stats:
          * sig_figs -> max_absrel
          * decimal_digits -> max_abserror
        If ANY output in the cell exceeds the threshold, flag the
        variable. Per-variable precision is coarser than per-output;
        the worst output is the honest conservative signal.

    Returns a list of one violation string per flagged variable, each
    naming the variable, the chosen action/target_precision, the
    probe cell consulted, and the observed vs allowed numbers. This
    string is what the orchestrator surfaces to the analyst on retry,
    so it needs to be specific enough that the analyst can actually
    change its mind about that variable rather than re-emitting the
    same verdict.
    """
    tol_kind = tolerance.get("kind")
    tol_value = tolerance.get("value")
    if tol_kind not in ("sig_figs", "decimal_digits"):
        # Unknown tolerance shape -- silently skip; a malformed
        # tolerance is not this checker's problem to diagnose.
        return []
    if not isinstance(tol_value, int) or tol_value <= 0:
        return []
    threshold = 10.0 ** (-tol_value)
    stat_key = "max_absrel" if tol_kind == "sig_figs" else "max_abserror"

    cells = evidence.get("cells", {})
    if not cells:
        return []
    canonical_seed = _PROBE_SEEDS[0]

    violations: list[str] = []
    for entry in verdict.get("variables", []):
        action = entry.get("action")
        if action != "downcast":
            # 'keep' has no risk; 'emulate' has no matching probe
            # cell in v0 (see AGENTS.md). Silent skip.
            continue
        target = (entry.get("target_precision") or "").strip()
        if not target:
            continue
        cell_name = f"{target}_seed{canonical_seed}"
        cell = cells.get(cell_name)
        if cell is None or cell.get("status") != "ok":
            # No usable probe evidence for this precision at the
            # canonical seed. Silent skip -- e.g. downcast to 'half'
            # when the probe matrix only covered float / mixed_io.
            continue
        stats = cell.get("stats") or {}
        # Find the worst-offending output for this cell. If nothing
        # in the cell exceeds threshold, no violation.
        worst_output = ""
        worst_value = 0.0
        for output_name, output_stats in stats.items():
            value = output_stats.get(stat_key)
            if not isinstance(value, (int, float)):
                continue
            if value > threshold and value > worst_value:
                worst_value = value
                worst_output = output_name
        if not worst_output:
            continue
        name = entry.get("name", "<unnamed>")
        violations.append(
            f"Variable '{name}': analyst said action='downcast' "
            f"target_precision='{target}', but probe cell "
            f"'{cell_name}' shows {stat_key}={worst_value:.3e} on "
            f"output '{worst_output}', which exceeds the "
            f"{tol_kind}={tol_value} tolerance threshold of "
            f"{threshold:.1e}. Reconsider this variable: either "
            f"keep it at original precision, choose a wider target "
            f"precision, or justify why the probe evidence is "
            f"misleading for this specific case."
        )
    return violations


# ---------- test_variable_downcast: per-variable singleton empirical test ----------
#
# Step 3 of the per-variable analyst pipeline (see AGENTS.md "Planned
# next steps" and the Step 3 design confirmed via HITL Q&A). After the
# candidate_finder + variable_analyst loop produces N per-variable
# verdicts, the orchestrator empirically validates each `action='downcast'`
# verdict in isolation by:
#
#   1. mutating the canonical baseline driver at
#      baselines/<stem>/<profile.driver_filename> so ONLY the one alias
#      line `using <VarName>Type = <old_type>;` is retargeted to the
#      requested precision;
#   2. writing the mutated driver to
#      baselines/<stem>/varprobe/singleton_<varname>/<driver_filename>;
#   3. compiling and running it under the same _compile_driver / _run_driver
#      machinery every other driver uses; and
#   4. comparing the resulting reference.json against the canonical
#      quad oracle at baselines/<stem>/reference.json under the operator's
#      tolerance.
#
# Design decisions (locked in HITL Q&A):
#   - Explicit `tolerance_json` arg (Option A). Same tolerance the
#     finish-gate comparator applies, so a singleton pass directly
#     predicts finish-gate survival for that variable in isolation.
#   - `emulate` verdicts are OUT OF SCOPE for Step 3; the orchestrator
#     is instructed to skip this tool for them (pass through unchanged
#     like 'keep'). This tool errors if the caller nonetheless passes a
#     non-'float' target_precision, so a bug in the caller is loud.
#   - Numerical tolerance mismatch = status='ok' with the verdict and
#     mismatch summary in stdout. The tool answers the question "does
#     this singleton downcast survive the tolerance yardstick?" and
#     that question having answer 'no' is a normal outcome, not a
#     tool-level error. status='error' is reserved for infrastructure
#     failures (missing baseline driver / oracle, malformed alias
#     block, compile failure, run failure, malformed tolerance).
#
# Artifacts land under baselines/<stem>/varprobe/singleton_<varname>/ so
# they never collide with the rewritten-tree (Step 4+ union / bisection
# artifacts will land under baselines/<stem>/varprobe/{union,bisect_...}
# alongside).

# Regex that matches a single alias line `using <VarName>Type = <RHS>;`
# on its own line (no leading/trailing whitespace). The <VarName> is
# injected per-call via re.escape so a variable named e.g. `a` cannot
# accidentally match `alphaType`. Multiline-anchored so we can search
# the full driver source at once. The captured group is the RHS
# (everything up to but not including the semicolon) which we mutate
# to swap `double` for the target precision's C++ type token.
_ALIAS_LINE_RE_TEMPLATE = r"^using {var}Type = ([^;\n]+);$"

# Set of target_precision tokens this tool knows how to splice for.
# Deliberately narrow in v0: 'float' is the only precision the probe
# pipeline empirically validates (probe_precisions = quad/double/float/
# mixed_io on Kokkos), so it's the only precision we've smoke-tested
# end to end. 'half' is advertised in ANALYST_OUTPUT_SCHEMA but has
# never been compiled in this repo; adding it here without a smoke
# test on real hardware would be a silent contract expansion. The
# orchestrator's per-variable pipeline is expected to fall back to
# `keep` for a variable it wanted to downcast to a precision this tool
# doesn't support -- the tool returns a clear status='error' so that
# fallback is explicit, not silent.
_SUPPORTED_TARGET_PRECISIONS = frozenset({"float"})

# Map from analyst target_precision token to the C++ type token that
# replaces `double` in the alias RHS. Kept alongside
# _SUPPORTED_TARGET_PRECISIONS so extending the set is a one-line change
# in each place.
_TARGET_PRECISION_TO_CXX = {
    "float": "float",
}


def _mutate_alias_rhs(old_rhs: str, target_cxx: str) -> tuple[str | None, str | None]:
    """Rewrite an alias RHS by replacing every `double` token with `target_cxx`.

    Returns `(new_rhs, None)` on success or `(None, error_msg)` when
    the RHS contains no `double` token (nothing to downcast; the alias
    is probably already float or references a non-floating type -- either
    way the caller shouldn't be asking to downcast this variable).

    Token matching uses a word-boundary regex so `double` is replaced
    but a hypothetical `MyDoubleThing` typedef is left alone. All
    occurrences are replaced (a `Kokkos::View<const double*>` alias
    stays consistent when it becomes `Kokkos::View<const float*>`).
    """
    pattern = re.compile(r"\bdouble\b")
    if not pattern.search(old_rhs):
        return (None, (
            f"Alias RHS {old_rhs!r} contains no `double` token; there "
            f"is nothing to downcast. Either the variable is already at "
            f"the target precision, or its alias references a non-"
            f"floating-point type (in which case the analyst should not "
            f"have chosen action='downcast' for it)."
        ))
    new_rhs = pattern.sub(target_cxx, old_rhs)
    return (new_rhs, None)


def _splice_singleton_alias(
    driver_text: str,
    profile: LanguageProfile,
    variable_name: str,
    target_cxx: str,
) -> tuple[str | None, str | None]:
    """Rewrite the unique `using <VarName>Type = ...;` alias line inside
    the kernel sentinels of `driver_text`, retargeting its RHS from
    `double` to `target_cxx`.

    Returns `(new_source, None)` on success or `(None, error_msg)` on
    any contract violation (missing sentinels, wrong ordering, alias
    line absent, alias line non-unique, alias RHS not downcastable).
    All error paths carry a message that names the specific violation
    so the orchestrator can retry with a different variable / target /
    fallback to `keep`.

    The mutation is scoped strictly BETWEEN the sentinel lines: this
    tool is a singleton splice, so touching main() or the kernel body
    would either duplicate the rewriter's job or (worse) desync from
    the alias contract. The alias line contract (per BASELINE_HARNESS_
    SYSTEM_PROMPT in workflow/registry.py and per language) is exactly
    what makes this splice safe: a single alias redefinition
    propagates through the kernel signature (which uses the alias
    names) and through main() (which constructs kernel arguments
    through the same aliases) without any signature-touching edit.
    """
    lines = driver_text.splitlines()

    begin_idx = _find_unique_sentinel_line(lines, profile.sentinel_begin)
    if begin_idx is None:
        return (None, (
            f"Baseline driver does not contain exactly one "
            f"{profile.sentinel_begin!r} line on its own (no "
            f"surrounding indentation or whitespace)."
        ))
    end_idx = _find_unique_sentinel_line(lines, profile.sentinel_end)
    if end_idx is None:
        return (None, (
            f"Baseline driver does not contain exactly one "
            f"{profile.sentinel_end!r} line on its own (no "
            f"surrounding indentation or whitespace)."
        ))
    if begin_idx >= end_idx:
        return (None, (
            f"Baseline driver has {profile.sentinel_end!r} at or before "
            f"{profile.sentinel_begin!r} (line {end_idx + 1} vs line "
            f"{begin_idx + 1}); cannot splice."
        ))

    # Search ONLY the region strictly between the sentinels (exclusive
    # on both ends). The alias contract puts alias lines here.
    kernel_region_start = begin_idx + 1
    kernel_region_end = end_idx  # exclusive
    kernel_region_lines = lines[kernel_region_start:kernel_region_end]

    pattern = re.compile(
        _ALIAS_LINE_RE_TEMPLATE.format(var=re.escape(variable_name))
    )
    matches: list[tuple[int, str]] = []
    for offset, line in enumerate(kernel_region_lines):
        m = pattern.match(line)
        if m is not None:
            matches.append((offset, m.group(1)))
    if not matches:
        return (None, (
            f"No `using {variable_name}Type = <type>;` alias line found "
            f"between the kernel sentinels. Either the variable name "
            f"does not match a kernel parameter, the baseline_harness "
            f"emitted a non-standard alias line (indented, split across "
            f"lines, wrong suffix), or the variable's alias is not "
            f"floating-point (integer parameters do not get aliases -- "
            f"see the precision-alias contract in the harness prompt)."
        ))
    if len(matches) > 1:
        return (None, (
            f"Found {len(matches)} `using {variable_name}Type = "
            f"<type>;` alias lines between the kernel sentinels; the "
            f"alias contract requires exactly one per kernel parameter "
            f"so the singleton splice is unambiguous."
        ))
    alias_offset, old_rhs = matches[0]

    new_rhs, err = _mutate_alias_rhs(old_rhs, target_cxx)
    if err is not None:
        return (None, err)
    absolute_line_idx = kernel_region_start + alias_offset
    new_line = f"using {variable_name}Type = {new_rhs};"
    new_lines = list(lines)
    new_lines[absolute_line_idx] = new_line

    trailing_newline = "\n" if driver_text.endswith("\n") else ""
    return ("\n".join(new_lines) + trailing_newline, None)


# Regex that matches a single-line local declaration of the form
#     [<indent>][const ]<fptype> <name> = <RHS>;
# inside the kernel body. Used as a fallback splicer for variables
# that are NOT kernel parameters (locals like `eps2`, `r2`, `inv_r`
# in nbody-shaped kernels) and therefore have no `using <Name>Type`
# alias to mutate.
#
# Deliberately narrow safe subset (per AGENTS.md's per-variable
# pipeline section):
#   - matches only `double` or `float` as the type token (word-
#     bounded so `long double` is rejected -- \b before/after)
#   - allows an optional `const ` qualifier before the type
#   - requires exactly one identifier immediately after the type;
#     multi-var-per-line (`double a, b, c;`) does not match because
#     the pattern requires `= <RHS>;` on the same line
#   - requires an initializer (`= <RHS>`) so pure declarations
#     (`double x;`) do not match -- those are almost always followed
#     by an assignment on a later line, which is a semantically
#     different pattern this splicer is not designed for
#   - rejects `auto` at the type slot (see AGENTS.md for the
#     semantic argument -- `auto` mutations are storage-only vs
#     compute-precision and are semantically incoherent as a
#     'downcast')
#   - rejects arrays (`double x[N]`) because the regex requires
#     `<name> =` immediately with no `[` between
#   - single-line only (the RHS regex forbids `;` and `\n`, so
#     multi-line initializers fall through)
#   - line-anchored with optional leading whitespace, so inline
#     comments after the semicolon do not confuse it, but a fully-
#     commented-out declaration (`// double x = ...;`) does not
#     match because `//` is not `[const ]<fptype>`
#
# The captured groups (in order) are:
#   1. leading whitespace (may be empty)
#   2. optional `const ` (may be empty)
#   3. the fp type token (`double` or `float`)
# The RHS is not captured because the mutation only rewrites the
# type token, not the initializer.
_LOCAL_DECL_RE_TEMPLATE = (
    r"^(\s*)(const\s+)?\b(double|float)\b\s+{var}\s*=\s*[^;\n]+;\s*$"
)

# Prefix of the error message _splice_singleton_alias returns when
# the alias line is absent (as opposed to non-unique or RHS-non-
# downcastable). The dispatcher below uses this prefix to decide
# whether to fall back to the local-body splicer. This is a
# stability-critical string -- if you change the error message in
# _splice_singleton_alias, keep this prefix aligned or the fallback
# stops firing (and locals silently stop getting tested).
_ALIAS_NOT_FOUND_ERR_PREFIX = "No `using "


def _mutate_local_decl(
    matched_line: str, target_cxx: str, variable_name: str
) -> tuple[str | None, str | None]:
    """Rewrite a matched local declaration line by replacing its fp
    type token (`double` or `float`) with `target_cxx`, preserving
    the leading whitespace, the optional `const ` qualifier, the
    variable name, the initializer, and any trailing whitespace.

    Returns `(new_line, None)` on success or `(None, error_msg)` when
    the mutation would be a no-op (the type token already equals
    target_cxx -- the caller should not have asked to downcast this
    variable, since it is already at the target precision).

    The whole re-parse-and-rebuild sequence is done in this helper
    rather than as an in-place regex substitution so the leading
    whitespace, const prefix, and trailing whitespace are all
    preserved verbatim -- a naive `re.sub(r'\\bdouble\\b', ...)` on
    the whole line would also touch `double` occurrences inside the
    RHS (e.g. `double eps2 = (double)eps * eps;`), which would
    silently change the initializer's computation precision on top
    of the intended storage change. Confining the mutation to the
    matched type slot is what makes this a 'storage-only' downcast.
    """
    m = re.match(
        _LOCAL_DECL_RE_TEMPLATE.format(var=re.escape(variable_name)),
        matched_line,
    )
    if m is None:
        # Defensive: caller should have matched already.
        return (None, (
            f"_mutate_local_decl called with a line that does not "
            f"match the local-decl regex for {variable_name!r}. This "
            f"is a caller bug; please report."
        ))
    indent, const_prefix, old_type = m.group(1), m.group(2) or "", m.group(3)
    if old_type == target_cxx:
        return (None, (
            f"Local declaration of {variable_name!r} already uses "
            f"type {target_cxx!r}; there is nothing to downcast."
        ))
    # Rebuild the line: preserve indent, const-prefix, and everything
    # after the type token (which lives verbatim in the original
    # line after the type slot).
    #
    # We locate the type token by span and replace only that slice,
    # so the RHS (which may itself contain `double` casts or
    # literals) is untouched.
    type_start, type_end = m.span(3)
    new_line = matched_line[:type_start] + target_cxx + matched_line[type_end:]
    del indent, const_prefix  # only needed above for regex clarity
    return (new_line, None)


def _has_top_level_comma(text: str) -> bool:
    """Return True iff `text` contains a `,` at paren/bracket/brace
    depth 0. Used by _splice_singleton_local to reject multi-var-
    per-line declarations after the initial regex match, without
    also rejecting legitimate initializers whose RHS calls a
    multi-argument function (e.g. `pow(x, 2.0)`) or references a
    braced initializer list. Character-level scan is sufficient for
    the safe subset -- we do not attempt to parse comments or string
    literals because the harness contract forbids them on a decl
    line inside the kernel body.
    """
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth > 0:
                depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


def _splice_singleton_local(
    driver_text: str,
    profile: LanguageProfile,
    variable_name: str,
    target_cxx: str,
) -> tuple[str | None, str | None]:
    """Rewrite the unique local declaration line
        `[const ]<fptype> <variable_name> = <RHS>;`
    inside the kernel sentinels of `driver_text`, retargeting its
    type token from `double`/`float` to `target_cxx`.

    Mirror of `_splice_singleton_alias` for kernel LOCAL variables
    (non-parameters). Returns `(new_source, None)` on success or
    `(None, error_msg)` on any contract violation (missing
    sentinels, wrong ordering, local decl absent, local decl non-
    unique, decl outside the safe subset). All error paths carry a
    message that names the specific violation so the caller (and,
    transitively, the orchestrator) can retry with a different
    variable / target / fallback to `keep`.

    The mutation is scoped strictly BETWEEN the sentinel lines, same
    as `_splice_singleton_alias` -- locals live in the kernel body,
    which is entirely inside the sentinels. Unlike a parameter
    downcast (which requires the alias contract because main()
    outside the sentinels constructs the parameter), a local
    downcast can be applied entirely inside the sentinels without
    any main()-side ripple.

    v0 scope: single-line `[const ] <double|float> <name> = <RHS>;`
    declarations. `auto` locals, arrays, multi-var-per-line
    declarations, and pure declarations without initializers all
    return `(None, "no declaration line found")` and get demoted to
    `keep` by the orchestrator. See AGENTS.md for the semantic
    argument -- `auto` in particular is intentionally out of scope
    because a storage-only downcast on an `auto` is semantically
    incoherent (the type is deduced from the initializer, so
    changing storage alone means the RHS is computed at one
    precision and rounded to another at exactly the assignment
    point, which is not what the analyst usually means by
    'downcast').
    """
    lines = driver_text.splitlines()

    begin_idx = _find_unique_sentinel_line(lines, profile.sentinel_begin)
    if begin_idx is None:
        return (None, (
            f"Baseline driver does not contain exactly one "
            f"{profile.sentinel_begin!r} line on its own (no "
            f"surrounding indentation or whitespace)."
        ))
    end_idx = _find_unique_sentinel_line(lines, profile.sentinel_end)
    if end_idx is None:
        return (None, (
            f"Baseline driver does not contain exactly one "
            f"{profile.sentinel_end!r} line on its own (no "
            f"surrounding indentation or whitespace)."
        ))
    if begin_idx >= end_idx:
        return (None, (
            f"Baseline driver has {profile.sentinel_end!r} at or before "
            f"{profile.sentinel_begin!r} (line {end_idx + 1} vs line "
            f"{begin_idx + 1}); cannot splice."
        ))

    # Search ONLY the region strictly between the sentinels.
    kernel_region_start = begin_idx + 1
    kernel_region_end = end_idx  # exclusive
    kernel_region_lines = lines[kernel_region_start:kernel_region_end]

    pattern = re.compile(
        _LOCAL_DECL_RE_TEMPLATE.format(var=re.escape(variable_name))
    )
    matches: list[tuple[int, str]] = []
    for offset, line in enumerate(kernel_region_lines):
        if pattern.match(line) is None:
            continue
        # Post-filter: reject multi-variable-per-line declarations like
        # `double a = 0.0, b = 0.0;` where the RHS regex greedily
        # consumes across a comma. We can't ban `,` outright in the
        # RHS because legitimate initializers may contain function
        # calls with multiple arguments (e.g. `pow(x, 2.0)`), so we
        # count top-level commas -- ones that appear at paren depth
        # 0. Any top-level comma means the "decl" is actually a
        # multi-var decl and belongs to the safe-subset reject list.
        rhs = line.split("=", 1)[1]
        if _has_top_level_comma(rhs):
            continue
        matches.append((offset, line))
    if not matches:
        return (None, (
            f"No `[const] <double|float> {variable_name} = <RHS>;` "
            f"local declaration line found between the kernel sentinels. "
            f"Either the variable is not a local in this kernel, its "
            f"declaration is outside the safe subset the singleton "
            f"splicer supports (auto-typed, multi-variable-per-line, "
            f"array-typed, or split across multiple lines), or it is "
            f"actually a kernel parameter -- parameters use the "
            f"`using <Name>Type = ...;` alias contract instead, see the "
            f"precision-alias contract in the harness prompt."
        ))
    if len(matches) > 1:
        return (None, (
            f"Found {len(matches)} `[const] <double|float> "
            f"{variable_name} = <RHS>;` local declaration lines between "
            f"the kernel sentinels; the singleton splice requires "
            f"exactly one so the mutation is unambiguous. Shadowing a "
            f"local across scopes inside the same kernel is unusual "
            f"enough that the tool refuses to guess which one to "
            f"downcast."
        ))
    line_offset, matched_line = matches[0]

    new_line, err = _mutate_local_decl(matched_line, target_cxx, variable_name)
    if err is not None:
        return (None, err)
    absolute_line_idx = kernel_region_start + line_offset
    new_lines = list(lines)
    new_lines[absolute_line_idx] = new_line

    trailing_newline = "\n" if driver_text.endswith("\n") else ""
    return ("\n".join(new_lines) + trailing_newline, None)


def _splice_singleton_variable(
    driver_text: str,
    profile: LanguageProfile,
    variable_name: str,
    target_cxx: str,
) -> tuple[str | None, str | None]:
    """Dispatch wrapper that tries the alias splicer first and, on
    a specifically alias-not-found error, falls through to the
    local-body splicer.

    Returns `(new_source, None)` on success or `(None, error_msg)`
    on failure. The error message names both attempts so the
    orchestrator (and human operator reading the trace) can tell
    the variable was tried as a parameter AND as a local before
    the tool gave up.

    Fallback trigger is specifically the "alias line absent" error
    from `_splice_singleton_alias`. Other alias errors (multi-match,
    RHS-not-downcastable, sentinel violations) are NOT fallback
    triggers -- those are real contract violations on the parameter
    path and would still be violations on the local path (same
    sentinels; a variable that has 2 alias lines but also matches
    a local decl is a harness bug, not a legitimate 'try locals').
    The rationale: only 'alias absent' is the neutral signal that
    'this variable is not a parameter, maybe it's a local'; every
    other alias error means 'this variable IS a parameter, but
    something is wrong with its alias'.

    This wrapper is the ONE call site both test_variable_downcast
    and _splice_union_aliases use; adding a third splicer in the
    future (e.g. for return types or template parameters) is a
    one-branch extension here plus a new `_splice_singleton_<kind>`
    helper.
    """
    alias_result, alias_err = _splice_singleton_alias(
        driver_text, profile, variable_name, target_cxx
    )
    if alias_err is None:
        return (alias_result, None)
    # Only fall through on the specific 'alias line absent' error.
    # See _ALIAS_NOT_FOUND_ERR_PREFIX for the coupling to the
    # error-message text produced by _splice_singleton_alias.
    if not alias_err.startswith(_ALIAS_NOT_FOUND_ERR_PREFIX):
        return (None, alias_err)

    local_result, local_err = _splice_singleton_local(
        driver_text, profile, variable_name, target_cxx
    )
    if local_err is None:
        return (local_result, None)
    # Both attempts failed. Return a combined message so the trace
    # and stderr show what was tried and why each failed.
    return (None, (
        f"variable {variable_name!r} matched neither the parameter "
        f"nor the local-declaration splicer. "
        f"Parameter attempt: {alias_err} "
        f"Local attempt: {local_err}"
    ))


def _compare_singleton_vs_oracle(
    oracle_path: Path,
    candidate_path: Path,
    tolerance_kind: str,
    tolerance_value: int,
) -> tuple[bool, int, list[dict], str | None]:
    """Compare two reference.json files under the given tolerance.

    Returns `(passed, total_compared, mismatches, shape_error)`.
    `shape_error` is non-None only when the shape check itself
    fails (top-level keys, output-array names, per-array lengths);
    that's an infrastructure error the caller surfaces as
    status='error'. Numerical mismatches populate `mismatches`
    (already truncated at _MAX_REPORTED_MISMATCHES) and set
    `passed=False`, but are NOT an infrastructure error: the caller
    reports them as status='ok' + verdict=fail in stdout.

    Reuses _load_reference / _shape_error / _iter_leaf_pairs /
    _value_within_tolerance from compare_outputs so the numerical
    yardstick is bit-for-bit identical to what the finish-gate
    comparator applies.
    """
    baseline = _load_reference(oracle_path)
    rewritten = _load_reference(candidate_path)
    shape_err = _shape_error(baseline, rewritten)
    if shape_err is not None:
        return (False, 0, [], shape_err)

    total = 0
    mismatches: list[dict] = []
    for name, idx, va, vb in _iter_leaf_pairs(baseline, rewritten):
        total += 1
        try:
            fa = float(va)
            fb = float(vb)
        except (TypeError, ValueError):
            mismatches.append({
                "name": name, "index": idx, "a": va, "b": vb,
                "abs_err": None, "threshold": None,
            })
            continue
        passed, abs_err, threshold = _value_within_tolerance(
            fa, fb, tolerance_kind, tolerance_value
        )
        if not passed:
            mismatches.append({
                "name": name, "index": idx, "a": fa, "b": fb,
                "abs_err": abs_err, "threshold": threshold,
            })

    truncated = mismatches[:_MAX_REPORTED_MISMATCHES]
    return (not mismatches, total, truncated, None)


def test_variable_downcast(
    kernel_stem: str,
    variable_name: str,
    target_precision: str,
    tolerance_json: str,
    language_id: str,
) -> dict:
    """Empirically test a single-variable downcast in isolation.

    Mutates the baseline driver at
    baselines/<kernel_stem>/<profile.driver_filename> so only the alias
    line `using <variable_name>Type = <old_type>;` inside the kernel
    sentinels is retargeted from `double` to `target_precision`,
    writes the mutated driver to baselines/<kernel_stem>/varprobe/
    singleton_<variable_name>/<driver_filename>, then compiles and
    runs it via _compile_driver + _run_driver against that directory,
    and finally compares its reference.json against the canonical
    quad oracle at baselines/<kernel_stem>/reference.json under the
    operator-supplied tolerance.

    `tolerance_json` is a JSON string with keys {kind, value, source}
    matching the tolerance dict the orchestrator threads through the
    rest of the pipeline. `kind` is 'sig_figs' or 'decimal_digits',
    `value` is a positive integer.

    Result shape: uniform {status, stdout, stderr, artifacts}. Semantic
    contract for `status`:

      - 'ok' means the tool ran end-to-end without infrastructure
        failure. The tool's VERDICT (does this singleton downcast meet
        tolerance?) is in stdout as either "VERDICT: pass" or
        "VERDICT: fail" followed by a mismatch summary; the caller
        parses stdout, not status. A tolerance-mismatch is a normal
        outcome, not an error.
      - 'error' means infrastructure failed: missing baseline driver,
        missing oracle, malformed alias block, unsupported
        target_precision, unsupported language_id, compile failure,
        run failure, or malformed tolerance_json. The caller cannot
        derive a verdict from an error result and should either retry
        with different args or fall back to `keep` for that variable.

    Artifacts on success: the mutated driver source, the compiled
    driver binary, and the produced reference.json under baselines/
    <kernel_stem>/varprobe/singleton_<variable_name>/. The singleton_
    prefix reserves the directory namespace for Step 4's union/bisect
    artifacts alongside.

    v0 scope (see the module-level comment): only Kokkos and only
    target_precision='float'. Emulate verdicts are out of scope; the
    orchestrator prompt is expected to skip this tool for them.
    """
    profile = _resolve_profile(language_id)

    if not variable_name or not isinstance(variable_name, str):
        return _error(
            f"variable_name must be a non-empty string; got "
            f"{variable_name!r}."
        )
    # Guard against accidental empty / whitespace-only inputs that
    # would make the alias regex trivially match too much.
    if variable_name.strip() != variable_name or not variable_name.strip():
        return _error(
            f"variable_name must not contain leading/trailing whitespace "
            f"or be blank; got {variable_name!r}."
        )

    if target_precision not in _SUPPORTED_TARGET_PRECISIONS:
        return _error(
            f"target_precision={target_precision!r} is not supported by "
            f"test_variable_downcast in v0. Supported precisions: "
            f"{sorted(_SUPPORTED_TARGET_PRECISIONS)}. The orchestrator "
            f"should fall back to action='keep' for this variable, or "
            f"choose a supported target_precision."
        )
    target_cxx = _TARGET_PRECISION_TO_CXX[target_precision]

    try:
        tolerance = json.loads(tolerance_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"tolerance_json is not valid JSON: {exc}. Expected an "
            f"object like {{'kind': 'sig_figs', 'value': 3, 'source': "
            f"'user_cli'}}."
        )
    if not isinstance(tolerance, dict):
        return _error(
            f"tolerance_json must be a JSON object; got "
            f"{type(tolerance).__name__}."
        )
    tol_kind = tolerance.get("kind")
    tol_value = tolerance.get("value")
    if tol_kind not in {"sig_figs", "decimal_digits"}:
        return _error(
            f"Unsupported tolerance kind: {tol_kind!r}. Expected "
            f"'sig_figs' or 'decimal_digits'."
        )
    if (
        not isinstance(tol_value, int)
        or isinstance(tol_value, bool)
        or tol_value < 1
    ):
        return _error(
            f"Invalid tolerance value: {tol_value!r}. Expected a "
            f"positive integer."
        )

    baseline_dir = Path("baselines") / kernel_stem
    baseline_driver_path = baseline_dir / profile.driver_filename
    oracle_path = baseline_dir / "reference.json"

    if not baseline_driver_path.is_file():
        return _error(
            f"Baseline driver source not found at "
            f"{baseline_driver_path}. Did spawn_baseline_harness run "
            f"and get approved for this kernel_stem?"
        )
    if not oracle_path.is_file():
        return _error(
            f"Oracle reference.json not found at {oracle_path}. Did "
            f"run_baseline_driver (and, on Kokkos, the probe pipeline's "
            f"oracle promotion in probe_compare) run and succeed for "
            f"this kernel_stem?"
        )

    try:
        baseline_text = baseline_driver_path.read_text()
    except OSError as exc:
        return _error(
            f"Failed to read {baseline_driver_path}: {exc}"
        )

    new_source, err = _splice_singleton_variable(
        baseline_text, profile, variable_name, target_cxx
    )
    if err is not None:
        return _error(f"In {baseline_driver_path}: {err}")

    singleton_dir = (
        baseline_dir / "varprobe" / f"singleton_{variable_name}"
    )
    singleton_src = singleton_dir / profile.driver_filename
    try:
        singleton_dir.mkdir(parents=True, exist_ok=True)
        singleton_src.write_text(new_source)
    except OSError as exc:
        return _error(f"Failed to write {singleton_src}: {exc}")

    compile_result = _compile_driver(
        singleton_dir,
        profile,
        missing_source_hint=(
            "test_variable_downcast just wrote the driver source; this "
            f"should not happen. Check filesystem permissions on "
            f"{singleton_dir}."
        ),
    )
    if compile_result["status"] != "ok":
        return compile_result

    run_result = _run_driver(
        singleton_dir,
        missing_binary_hint=(
            "test_variable_downcast just compiled the driver; this "
            f"should not happen. Check filesystem permissions on "
            f"{singleton_dir}."
        ),
    )
    if run_result["status"] != "ok":
        return run_result

    candidate_path = singleton_dir / "reference.json"
    try:
        passed, total, mismatches, shape_err = (
            _compare_singleton_vs_oracle(
                oracle_path, candidate_path, tol_kind, tol_value
            )
        )
    except json.JSONDecodeError as exc:
        return _error(
            f"reference.json parse failure while comparing "
            f"{candidate_path} against {oracle_path}: {exc}"
        )
    except OSError as exc:
        return _error(
            f"OS error while comparing {candidate_path} against "
            f"{oracle_path}: {exc}"
        )
    if shape_err is not None:
        return _error(
            f"Shape mismatch between singleton output and oracle: "
            f"{shape_err}"
        )

    verdict_header = (
        f"VERDICT: pass -- variable {variable_name!r} tolerates "
        f"downcast to {target_precision!r} in isolation under "
        f"{tol_kind}={tol_value} ({total} values compared)."
        if passed else
        f"VERDICT: fail -- variable {variable_name!r} does NOT "
        f"tolerate downcast to {target_precision!r} in isolation "
        f"under {tol_kind}={tol_value} "
        f"({len(mismatches)}/{total} values disagree, first "
        f"{len(mismatches)} shown)."
    )
    stdout_lines = [verdict_header]
    for m in mismatches:
        stdout_lines.append(
            f"  ({m['name']!r}, idx={m['index']}, a={m['a']}, "
            f"b={m['b']}, abs_err={m['abs_err']}, "
            f"threshold={m['threshold']})"
        )

    return {
        "status": "ok",
        "stdout": "\n".join(stdout_lines),
        "stderr": "",
        "artifacts": [
            str(singleton_src),
            str(singleton_dir / "driver"),
            str(candidate_path),
        ],
    }


# ---------- test_variable_union_downcast + bisect_variable_downcast ----------
#
# Step 4 of the per-variable analyst pipeline. After Step 3
# (test_variable_downcast) has validated each candidate variable's
# downcast in ISOLATION, we still have to prove they interact safely
# when applied TOGETHER: two singleton passes do not imply a joint
# pass, because a downstream variable that consumed the singleton-
# downcast producer at full double precision in Step 3 will see the
# reduced-precision producer in the joint case.
#
# `test_variable_union_downcast` mutates N alias lines at once and
# tests the joint downcast against the oracle -- same yardstick,
# same directory layout convention, same status contract as Step 3.
# `bisect_variable_downcast` wraps it with a greedy drop-from-end
# bisection: given the singleton-passing variables in candidate-
# finder RANK ORDER (highest rank first), it tries the full union,
# and on failure drops the last (lowest-rank) variable and retries,
# recording every attempt under baselines/<stem>/varprobe/
# bisect_iter_<n>/ and emitting a summary at
# baselines/<stem>/varprobe/bisect_result.json.
#
# The reason bisection drops from the END rather than doing a
# classic O(log n) binary search: the finder's rank encodes "how
# safe / how high-value" a downcast candidate is, and we want to
# preserve as many high-rank downcasts as possible. Linear drop-
# from-end preserves the prefix, giving us the largest passing
# prefix under the finder's ranking. O(log n) bisection over
# subsets would break that monotonicity assumption. We also
# accept the O(n) worst case here because n is small (a typical
# kernel has 3-8 candidate variables after Step 3) and each
# iteration is a full compile+run+compare cycle whose wall-clock
# cost is dominated by the compile, not by the number of
# iterations we skip.


def _splice_union_aliases(
    driver_text: str,
    profile: LanguageProfile,
    variable_names: list[str],
    target_cxxs: list[str],
) -> tuple[str | None, str | None]:
    """Apply N single-variable splices to `driver_text` in sequence.

    Each (variable_name, target_cxx) pair is passed through
    `_splice_singleton_variable` in list order (which dispatches
    between the parameter/alias splicer and the local-decl
    splicer), using the result of iteration i as the input to
    iteration i+1. This composes trivially because BOTH underlying
    splicers edit a single line in place (positions do not shift
    and the sentinels are found by position, not by pattern), and
    each per-variable regex is name-specific so mutating variable
    A never invalidates the match for variable B.

    Returns `(new_source, None)` on success or `(None, error_msg)`
    on the first splice failure. The error message names the
    offending variable and forwards the underlying splice error
    verbatim (which itself names both the parameter and local
    attempts when the dispatcher fell through), so the caller
    (and, transitively, the orchestrator) can tell exactly which
    of the N variables the union is stuck on and why.
    """
    if len(variable_names) != len(target_cxxs):
        return (None, (
            f"_splice_union_aliases requires len(variable_names) == "
            f"len(target_cxxs); got {len(variable_names)} and "
            f"{len(target_cxxs)}."
        ))
    current = driver_text
    for name, cxx in zip(variable_names, target_cxxs):
        new_source, err = _splice_singleton_variable(
            current, profile, name, cxx
        )
        if err is not None:
            return (None, f"variable {name!r}: {err}")
        current = new_source
    return (current, None)


def _validate_union_args(
    variable_names: object,
    target_precisions: object,
) -> tuple[list[str], list[str], str | None]:
    """Validate the list-shaped args for the union / bisect tools.

    Returns `(names, target_cxxs, error_msg)`. On success,
    `error_msg` is None and `names` / `target_cxxs` are validated
    parallel lists. On failure, `error_msg` names the specific
    violation and the two lists are empty.

    The validation catches: non-list inputs, length mismatch,
    empty list (a zero-variable union is a caller bug, not a
    silent no-op), each name being a non-empty stripped string,
    duplicate names (would cause the second splice to fail on
    "no double token" after the first splice already floated the
    alias), and each target_precision being in the v0 support
    set. Every rejection is a caller-contract violation, i.e.
    status='error'.
    """
    if not isinstance(variable_names, list):
        return ([], [], (
            f"variable_names must be a list; got "
            f"{type(variable_names).__name__}."
        ))
    if not isinstance(target_precisions, list):
        return ([], [], (
            f"target_precisions must be a list; got "
            f"{type(target_precisions).__name__}."
        ))
    if len(variable_names) != len(target_precisions):
        return ([], [], (
            f"variable_names and target_precisions must have the same "
            f"length; got {len(variable_names)} and "
            f"{len(target_precisions)}."
        ))
    if not variable_names:
        return ([], [], (
            "variable_names must be non-empty; a zero-variable union "
            "is a caller bug, not a silent no-op."
        ))
    names: list[str] = []
    target_cxxs: list[str] = []
    seen: set[str] = set()
    for i, (name, prec) in enumerate(zip(variable_names, target_precisions)):
        if not isinstance(name, str) or not name.strip() or name.strip() != name:
            return ([], [], (
                f"variable_names[{i}]={name!r} must be a non-empty "
                f"string with no leading/trailing whitespace."
            ))
        if name in seen:
            return ([], [], (
                f"variable_names contains duplicate entry {name!r}; "
                f"each variable may appear at most once in a union."
            ))
        seen.add(name)
        if prec not in _SUPPORTED_TARGET_PRECISIONS:
            return ([], [], (
                f"target_precisions[{i}]={prec!r} is not supported by "
                f"the per-variable pipeline in v0. Supported: "
                f"{sorted(_SUPPORTED_TARGET_PRECISIONS)}."
            ))
        names.append(name)
        target_cxxs.append(_TARGET_PRECISION_TO_CXX[prec])
    return (names, target_cxxs, None)


def _parse_tolerance_json(tolerance_json: str) -> tuple[str | None, int | None, str | None]:
    """Parse and validate `tolerance_json`.

    Returns `(kind, value, error_msg)`. On success, `error_msg` is
    None and `kind` / `value` are the validated tolerance fields.
    Same contract as the inline block in `test_variable_downcast`;
    factored out so the union and bisect tools apply the same
    yardstick without drift.
    """
    try:
        tolerance = json.loads(tolerance_json)
    except json.JSONDecodeError as exc:
        return (None, None, (
            f"tolerance_json is not valid JSON: {exc}. Expected an "
            f"object like {{'kind': 'sig_figs', 'value': 3, 'source': "
            f"'user_cli'}}."
        ))
    if not isinstance(tolerance, dict):
        return (None, None, (
            f"tolerance_json must be a JSON object; got "
            f"{type(tolerance).__name__}."
        ))
    tol_kind = tolerance.get("kind")
    tol_value = tolerance.get("value")
    if tol_kind not in {"sig_figs", "decimal_digits"}:
        return (None, None, (
            f"Unsupported tolerance kind: {tol_kind!r}. Expected "
            f"'sig_figs' or 'decimal_digits'."
        ))
    if (
        not isinstance(tol_value, int)
        or isinstance(tol_value, bool)
        or tol_value < 1
    ):
        return (None, None, (
            f"Invalid tolerance value: {tol_value!r}. Expected a "
            f"positive integer."
        ))
    return (tol_kind, tol_value, None)


def _run_union_attempt(
    kernel_stem: str,
    profile: LanguageProfile,
    variable_names: list[str],
    target_cxxs: list[str],
    tol_kind: str,
    tol_value: int,
    attempt_dir: Path,
) -> dict:
    """Perform a single union attempt: splice, write, compile, run,
    compare. Writes artifacts into `attempt_dir` (which the caller
    creates fresh for each attempt so failed attempts are preserved).

    Returns the uniform `{status, stdout, stderr, artifacts}` dict.
    The caller decides how to interpret the result: for the union
    tool the result is returned verbatim; for the bisect tool the
    caller inspects status + `VERDICT:` in stdout to decide whether
    to iterate.
    """
    baseline_dir = Path("baselines") / kernel_stem
    baseline_driver_path = baseline_dir / profile.driver_filename
    oracle_path = baseline_dir / "reference.json"

    if not baseline_driver_path.is_file():
        return _error(
            f"Baseline driver source not found at "
            f"{baseline_driver_path}. Did spawn_baseline_harness run "
            f"and get approved for this kernel_stem?"
        )
    if not oracle_path.is_file():
        return _error(
            f"Oracle reference.json not found at {oracle_path}. Did "
            f"run_baseline_driver (and, on Kokkos, the probe pipeline's "
            f"oracle promotion in probe_compare) run and succeed for "
            f"this kernel_stem?"
        )

    try:
        baseline_text = baseline_driver_path.read_text()
    except OSError as exc:
        return _error(f"Failed to read {baseline_driver_path}: {exc}")

    new_source, err = _splice_union_aliases(
        baseline_text, profile, variable_names, target_cxxs
    )
    if err is not None:
        return _error(f"In {baseline_driver_path}: {err}")

    attempt_src = attempt_dir / profile.driver_filename
    try:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_src.write_text(new_source)
    except OSError as exc:
        return _error(f"Failed to write {attempt_src}: {exc}")

    compile_result = _compile_driver(
        attempt_dir,
        profile,
        missing_source_hint=(
            "test_variable_union_downcast just wrote the driver source; "
            f"this should not happen. Check filesystem permissions on "
            f"{attempt_dir}."
        ),
    )
    if compile_result["status"] != "ok":
        return compile_result

    run_result = _run_driver(
        attempt_dir,
        missing_binary_hint=(
            "test_variable_union_downcast just compiled the driver; "
            f"this should not happen. Check filesystem permissions on "
            f"{attempt_dir}."
        ),
    )
    if run_result["status"] != "ok":
        return run_result

    candidate_path = attempt_dir / "reference.json"
    try:
        passed, total, mismatches, shape_err = (
            _compare_singleton_vs_oracle(
                oracle_path, candidate_path, tol_kind, tol_value
            )
        )
    except json.JSONDecodeError as exc:
        return _error(
            f"reference.json parse failure while comparing "
            f"{candidate_path} against {oracle_path}: {exc}"
        )
    except OSError as exc:
        return _error(
            f"OS error while comparing {candidate_path} against "
            f"{oracle_path}: {exc}"
        )
    if shape_err is not None:
        return _error(
            f"Shape mismatch between union output and oracle: "
            f"{shape_err}"
        )

    names_str = ", ".join(repr(n) for n in variable_names)
    verdict_header = (
        f"VERDICT: pass -- variables [{names_str}] jointly tolerate "
        f"downcast under {tol_kind}={tol_value} "
        f"({total} values compared)."
        if passed else
        f"VERDICT: fail -- variables [{names_str}] do NOT jointly "
        f"tolerate downcast under {tol_kind}={tol_value} "
        f"({len(mismatches)}/{total} values disagree, first "
        f"{len(mismatches)} shown)."
    )
    stdout_lines = [verdict_header]
    for m in mismatches:
        stdout_lines.append(
            f"  ({m['name']!r}, idx={m['index']}, a={m['a']}, "
            f"b={m['b']}, abs_err={m['abs_err']}, "
            f"threshold={m['threshold']})"
        )

    return {
        "status": "ok",
        "stdout": "\n".join(stdout_lines),
        "stderr": "",
        "artifacts": [
            str(attempt_src),
            str(attempt_dir / "driver"),
            str(candidate_path),
        ],
    }


def test_variable_union_downcast(
    kernel_stem: str,
    variable_names: list[str],
    target_precisions: list[str],
    tolerance_json: str,
    language_id: str,
) -> dict:
    """Empirically test a joint downcast of N variables together.

    Applies each `(variable_names[i], target_precisions[i])` alias
    mutation to the baseline driver in sequence, writes the mutated
    driver to baselines/<kernel_stem>/varprobe/union/
    <driver_filename>, compiles and runs it, and compares against
    the canonical oracle at baselines/<kernel_stem>/reference.json
    under the operator-supplied tolerance.

    The purpose is to catch INTERACTIONS between downcasts that
    individually pass the Step 3 singleton test: e.g. downcasting a
    producer view AND its consumer view may compound rounding error
    that either downcast alone would not have exposed. The
    orchestrator is expected to invoke this tool once after all
    Step 3 singleton passes, using the singleton-passing subset as
    `variable_names`.

    Same status / verdict contract as `test_variable_downcast`:
    `status='ok'` means the tool ran end-to-end; verdict is in
    stdout as `VERDICT: pass` or `VERDICT: fail`. `status='error'`
    is infrastructure failure only (missing baseline / oracle,
    compile / run failure, splice contract violation, malformed
    args).
    """
    profile = _resolve_profile(language_id)

    names, target_cxxs, err = _validate_union_args(
        variable_names, target_precisions
    )
    if err is not None:
        return _error(err)

    tol_kind, tol_value, err = _parse_tolerance_json(tolerance_json)
    if err is not None:
        return _error(err)

    attempt_dir = (
        Path("baselines") / kernel_stem / "varprobe" / "union"
    )
    return _run_union_attempt(
        kernel_stem, profile, names, target_cxxs,
        tol_kind, tol_value, attempt_dir,
    )


def bisect_variable_downcast(
    kernel_stem: str,
    variable_names: list[str],
    target_precisions: list[str],
    tolerance_json: str,
    language_id: str,
) -> dict:
    """Find the largest prefix of `variable_names` (in RANK ORDER,
    highest first) whose joint downcast passes the oracle
    comparison under the operator-supplied tolerance.

    The caller passes the singleton-passing variables in candidate-
    finder rank order (most-desirable-to-downcast first). The tool
    tries the full union in
    baselines/<kernel_stem>/varprobe/bisect_iter_1/; on VERDICT:
    fail it drops the LAST (lowest-rank) variable and retries in
    bisect_iter_2/, and so on. The first attempt whose VERDICT is
    `pass` (or the empty subset, if every non-empty subset fails)
    determines the result.

    Each iteration's driver source, binary, and reference.json are
    preserved in `bisect_iter_<n>/` (not overwritten across
    iterations) so a post-mortem can inspect what the interaction
    looked like at each subset size. A summary lands at
    baselines/<kernel_stem>/varprobe/bisect_result.json with keys:
      - passed_subset: list[str], the variable names that jointly
        passed (may be empty).
      - dropped: list of {name, reason} in the order they were
        dropped (bisect_iter_1's drop is first).
      - iterations: int, number of union attempts executed.
      - union_stdout_last: str, the VERDICT header of the FINAL
        attempt (whether pass or fail) so the caller has a summary
        at a glance.

    Returns the uniform `{status, stdout, stderr, artifacts}` dict
    like every other orchestrator tool. `status='ok'` covers both
    the empty-subset-only case and the full-prefix-passed case:
    from the orchestrator's perspective, bisection ran successfully
    and produced a definite answer either way. `status='error'` is
    reserved for infra failures (as in the union tool) surfacing
    from any iteration.

    Design note: the "keep dropping from end" strategy is O(n) in
    the number of iterations rather than O(log n), but in exchange
    it preserves the finder's ranking (the answer is the largest
    passing prefix under the given order rather than an arbitrary
    largest passing subset). n is small in practice (typically
    3-8), so the O(log n) win would rarely materialize in wall
    time, and each iteration is preserved on disk for later
    inspection.
    """
    profile = _resolve_profile(language_id)

    names, target_cxxs, err = _validate_union_args(
        variable_names, target_precisions
    )
    if err is not None:
        return _error(err)

    tol_kind, tol_value, err = _parse_tolerance_json(tolerance_json)
    if err is not None:
        return _error(err)

    bisect_root = Path("baselines") / kernel_stem / "varprobe"
    dropped: list[dict] = []
    iterations = 0
    passed_subset: list[str] = []
    last_stdout_verdict = ""
    all_artifacts: list[str] = []

    current_names = list(names)
    current_cxxs = list(target_cxxs)

    while current_names:
        iterations += 1
        attempt_dir = bisect_root / f"bisect_iter_{iterations}"
        result = _run_union_attempt(
            kernel_stem, profile, current_names, current_cxxs,
            tol_kind, tol_value, attempt_dir,
        )
        # Infrastructure failures short-circuit the whole bisection:
        # a compile failure on the full union doesn't mean the empty
        # subset would compile either (well, it would, but there's
        # nothing to test), so we surface the infra error verbatim
        # and let the orchestrator decide.
        if result["status"] != "ok":
            return result
        all_artifacts.extend(result["artifacts"])
        # First line of stdout is `VERDICT: pass|fail -- ...`. We
        # detect pass vs fail by prefix rather than parsing the full
        # line so the exact wording of the verdict header can evolve
        # without breaking bisection.
        first_line = result["stdout"].split("\n", 1)[0]
        last_stdout_verdict = first_line
        if first_line.startswith("VERDICT: pass"):
            passed_subset = list(current_names)
            break
        # Drop the last (lowest-rank) variable and try again.
        dropped_name = current_names[-1]
        dropped.append({
            "name": dropped_name,
            "reason": (
                f"joint downcast failed at iteration {iterations} with "
                f"subset of {len(current_names)}; dropped lowest-rank "
                f"variable {dropped_name!r}."
            ),
        })
        current_names = current_names[:-1]
        current_cxxs = current_cxxs[:-1]

    # If we exited the loop with an empty current_names list and
    # never hit a pass, `passed_subset` is [] and `dropped` lists
    # every variable in the original order. That's a legitimate
    # (if disappointing) result, not an error.

    summary = {
        "passed_subset": passed_subset,
        "dropped": dropped,
        "iterations": iterations,
        "union_stdout_last": last_stdout_verdict,
    }
    summary_path = bisect_root / "bisect_result.json"
    try:
        bisect_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    except OSError as exc:
        return _error(f"Failed to write {summary_path}: {exc}")
    all_artifacts.append(str(summary_path))

    if passed_subset:
        stdout = (
            f"BISECT: passed_subset=[{', '.join(repr(n) for n in passed_subset)}] "
            f"after {iterations} iteration(s), dropped "
            f"{len(dropped)} variable(s)."
        )
    else:
        stdout = (
            f"BISECT: no non-empty subset of "
            f"[{', '.join(repr(n) for n in names)}] passes joint "
            f"downcast under {tol_kind}={tol_value} after {iterations} "
            f"iteration(s). All variables must fall back to action="
            f"'keep' for this rewrite cycle."
        )

    return {
        "status": "ok",
        "stdout": stdout,
        "stderr": "",
        "artifacts": all_artifacts,
    }
