"""Tests for workflow.tools.

Covers compile_baseline_driver — env-var handling, missing source file,
g++ command shape, and the success / error result schema — and
run_baseline_driver — env-var parsing, missing/non-executable binary,
clean run + reference.json validation, non-zero exit, timeout, and
the same uniform result schema. Also covers splice_rewritten_kernel —
sentinel discovery, error cases, round-trip identity, byte-preservation
outside the sentinels, and the same uniform result schema. Also
covers compile_rewritten_driver — env-var contract shared with the
baseline compile, rewritten-subdir source/output paths, missing-source
hint blaming splice_rewritten_kernel, baseline-binary preservation,
and compile-flag parity with compile_baseline_driver. Also covers
run_rewritten_driver — env-var contract shared with the baseline run,
rewritten-subdir cwd / artifact path, missing-binary hint blaming
compile_rewritten_driver, and isolation from the baseline tree.
Also covers compare_outputs — tolerance-kind dispatch (sig_figs and
decimal_digits with the documented strict-< thresholds), NaN-always-
mismatches asymmetry, ±inf rules, shape-error vs tolerance-failure
distinction, mismatch list truncation with a "+ K more suppressed"
footer, comparison.json artifact on both pass and fail paths, and
the uniform result schema. Finally covers probe_step — RNG_SEED
rewrite contract (exactly-one-match), template / sibling per-seed
directory layout, seed-type validation (rejects bool), missing
template hint blaming spawn_baseline_harness, and the uniform result
schema; and probe_compare — quad_seed42 hard-error path, cell status
classification (ok / missing / load_error / shape_error /
no_quad_partner), per-output stats vs same-seed quad partner with
the NaN/inf and eps-floor rules, cross-seed deltas, evidence.json
schema, and the uniform result schema. All tests monkeypatch
subprocess.run so no real compiler or driver invocation happens;
comparator and probe_compare tests use pure file I/O against
tmp_path.
"""

import json
import os
import stat
import subprocess

from workflow import tools
from workflow.languages import cuda as cuda_profile
from workflow.languages import hip as hip_profile
from workflow.languages import omp_offload as omp_offload_profile
from workflow.languages import sycl as sycl_profile
from workflow.languages.cuda import (
    ARCH_ENV as CUDA_ARCH_ENV,
    DEFAULT_ARCH as CUDA_DEFAULT_ARCH,
    NVCC,
)
from workflow.languages.hip import (
    ARCH_ENV as HIP_ARCH_ENV,
    DEFAULT_ARCH as HIP_DEFAULT_ARCH,
    HIPCC,
)
from workflow.languages.omp_offload import (
    CXX_ENV as OMP_CXX_ENV,
    DEFAULT_CXX as OMP_DEFAULT_CXX,
    DEFAULT_TARGET as OMP_DEFAULT_TARGET,
    OMP_FLAG,
    TARGET_ENV as OMP_TARGET_ENV,
)
from workflow.languages.sycl import (
    CXX_ENV as SYCL_CXX_ENV,
    DEFAULT_CXX as SYCL_DEFAULT_CXX,
    SYCL_FLAG,
)
from workflow.tools import (
    DEFAULT_RUN_TIMEOUT_SEC,
    KERNEL_BEGIN_SENTINEL,
    KERNEL_END_SENTINEL,
    KOKKOS_ROOT_ENV,
    RUN_TIMEOUT_ENV,
    compare_outputs,
    compile_baseline_driver,
    compile_rewritten_driver,
    probe_compare,
    probe_step,
    run_baseline_driver,
    run_rewritten_driver,
    splice_rewritten_kernel,
)


# ---------- env-var handling ----------


def test_compile_baseline_driver_errors_when_env_unset(monkeypatch, tmp_path):
    """compile_baseline_driver returns status='error' with no subprocess call when AGENT_PRECISION_KOKKOS_ROOT is unset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KOKKOS_ROOT_ENV, raising=False)

    called = []

    def fail_run(*a, **kw):
        called.append((a, kw))
        raise AssertionError("subprocess.run must not be called when env unset")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert KOKKOS_ROOT_ENV in result["stderr"]
    assert result["artifacts"] == []
    assert called == []


def test_compile_baseline_driver_errors_when_env_points_at_non_kokkos(
    monkeypatch, tmp_path
):
    """compile_baseline_driver returns status='error' when AGENT_PRECISION_KOKKOS_ROOT points at a directory missing include/ or lib/."""
    monkeypatch.chdir(tmp_path)
    bogus = tmp_path / "bogus_root"
    bogus.mkdir()
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(bogus))

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called for an invalid Kokkos root"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "include/" in result["stderr"] or "lib/" in result["stderr"]
    assert result["artifacts"] == []


# ---------- missing driver source ----------


def _make_fake_kokkos_root(tmp_path):
    """Create a directory that looks enough like a Kokkos install."""
    root = tmp_path / "kokkos"
    (root / "include").mkdir(parents=True)
    (root / "lib").mkdir(parents=True)
    return root


def test_compile_baseline_driver_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """compile_baseline_driver returns status='error' (no subprocess call) when baselines/<stem>/driver.cpp does not exist."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "driver.cpp" in result["stderr"]
    assert result["artifacts"] == []


# ---------- successful compile: command shape + result ----------


def _stage_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/driver.cpp under tmp_path."""
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    (driver_dir / "driver.cpp").write_text(body)
    return driver_dir


def test_compile_baseline_driver_success_returns_artifacts_and_uses_env_root(
    monkeypatch, tmp_path
):
    """On a successful compile, compile_baseline_driver returns status='ok' with the driver path in artifacts and shells out to g++ with -I/-L pointing at AGENT_PRECISION_KOKKOS_ROOT."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "nbody_force")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/nbody_force/driver"]
    # subprocess.run was called with capture_output=True, text=True, check=False
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False
    # The command should be g++, c++20, include the include/lib dirs, and
    # produce a binary at baselines/<stem>/driver.
    cmd = captured["cmd"]
    assert cmd[0] == "g++"
    assert "-std=c++20" in cmd
    assert f"-I{root / 'include'}" in cmd
    assert f"-L{root / 'lib'}" in cmd
    assert "-fopenmp" in cmd
    assert "-lkokkoscore" in cmd
    assert "-lkokkoscontainers" in cmd
    # Output path
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/nbody_force/driver"
    # Driver source is the input
    assert "baselines/nbody_force/driver.cpp" in cmd


# ---------- v1 quad-precision compile-link detection ----------
#
# When the baseline_harness emits a quad-precision driver (per the
# v1 BASELINE PRECISION directive resolving to `quad`), the source
# uses GCC's `__float128` extension and `quadmath_snprintf` from
# <quadmath.h>. Both require linking against `libquadmath`. The
# Kokkos compile step detects this by substring-matching the driver
# source for `__float128`; the token cannot appear in a non-quad
# build (it has no business in float / double drivers and is not in
# any Kokkos / STL header the driver transitively pulls in).
#
# These three tests pin (1) the positive case — `__float128` in
# source -> `-lquadmath` on link line; (2) the negative case — no
# such token -> no `-lquadmath`, preserving the v0-minimal link
# line; and (3) the link-order invariant — `-lquadmath` must appear
# AFTER the source file in the argv so GNU ld's left-to-right
# symbol resolution sees `__float128`-referencing symbols in the
# .o before scanning the lib.


def test_compile_baseline_driver_quad_source_adds_lquadmath(
    monkeypatch, tmp_path
):
    """When the staged driver.cpp contains the `__float128` token (the GCC quad-precision type the v1 quad-baseline harness emits), the Kokkos compile step appends `-lquadmath` to the link line. Without this the driver fails to link with undefined references to `quadmath_snprintf` / `__addtf3` / etc., breaking the entire probe pipeline at its true-ground-truth reference run."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    # Minimal source carrying the detection token; the harness's real
    # output is much larger but only the substring match matters here.
    _stage_driver(
        tmp_path,
        "quad_kernel",
        body=(
            "#include <quadmath.h>\n"
            "int main(){ __float128 x = 0; (void)x; return 0; }\n"
        ),
    )

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("quad_kernel", "kokkos")

    assert result["status"] == "ok"
    assert "-lquadmath" in captured["cmd"]


def test_compile_baseline_driver_non_quad_source_omits_lquadmath(
    monkeypatch, tmp_path
):
    """A staged driver.cpp without the `__float128` token (the historical default — every v0 driver, plus v1 float and double baselines) compiles WITHOUT `-lquadmath`. Adding the flag unconditionally would be harmless to the link but would expose every operator to a libquadmath dependency they neither use nor understand; precise detection keeps the non-quad path identical to v0."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    # Stock v0-shape driver — no `__float128`, no `<quadmath.h>`.
    _stage_driver(tmp_path, "double_kernel")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("double_kernel", "kokkos")

    assert result["status"] == "ok"
    assert "-lquadmath" not in captured["cmd"]


def test_compile_baseline_driver_quad_link_order_lquadmath_after_source(
    monkeypatch, tmp_path
):
    """When `-lquadmath` is appended, it sits AFTER the driver source file path in the argv. GNU ld resolves symbols left-to-right, so an .o referencing `quadmath_snprintf` must be scanned before the lib that defines it; the reverse order would surface as undefined-reference errors at link time even with the flag present. This is the same ordering convention `-lkokkoscore` already follows in v0."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(
        tmp_path,
        "quad_kernel",
        body="int main(){ __float128 x = 0; (void)x; return 0; }\n",
    )

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    compile_baseline_driver("quad_kernel", "kokkos")

    cmd = captured["cmd"]
    src_idx = cmd.index("baselines/quad_kernel/driver.cpp")
    lib_idx = cmd.index("-lquadmath")
    assert src_idx < lib_idx, (
        f"Link order broken: source at {src_idx}, -lquadmath at {lib_idx}; "
        f"GNU ld needs the .o before the lib"
    )


def test_compile_baseline_driver_compile_failure_propagates_stderr(
    monkeypatch, tmp_path
):
    """A non-zero g++ exit produces status='error' with the captured stderr, no artifacts, and a stderr that includes the exit code and the failing command."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "nbody_force")

    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "driver.cpp:5: error: 'foo' was not declared in this scope\n"

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: FakeProc()
    )

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "exited with code 1" in result["stderr"]
    assert "'foo' was not declared" in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_handles_missing_gxx(monkeypatch, tmp_path):
    """If g++ is not on PATH, compile_baseline_driver returns status='error' (it does not crash the orchestrator)."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "nbody_force")

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("g++")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    result = compile_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "g++" in result["stderr"]
    assert result["artifacts"] == []


# ---------- result schema invariants ----------


def test_compile_baseline_driver_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — the shape future remote-batch verifier tools will share."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}

    # 1) env unset
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KOKKOS_ROOT_ENV, raising=False)
    assert set(compile_baseline_driver("x", "kokkos").keys()) == expected_keys

    # 2) success
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x", "kokkos").keys()) == expected_keys

    # 3) failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x", "kokkos").keys()) == expected_keys


# ---------- run_baseline_driver: env-var parsing ----------


def _stage_driver_binary(tmp_path, stem, reference_payload=None):
    """Create a fake executable at baselines/<stem>/driver under tmp_path.

    The file does NOT need to actually do anything; tests monkeypatch
    subprocess.run. It just needs to exist and be executable so
    run_baseline_driver's preflight checks pass. If reference_payload is
    provided, it is written as ./reference.json next to the driver
    (matching what a real driver would do on success).
    """
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    binary = driver_dir / "driver"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    if reference_payload is not None:
        (driver_dir / "reference.json").write_text(
            json.dumps(reference_payload)
        )
    return driver_dir


def test_run_baseline_driver_uses_default_timeout_when_env_unset(
    monkeypatch, tmp_path
):
    """When AGENT_PRECISION_RUN_TIMEOUT_SEC is unset, run_baseline_driver passes DEFAULT_RUN_TIMEOUT_SEC (60) to subprocess.run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(RUN_TIMEOUT_ENV, raising=False)
    driver_dir = _stage_driver_binary(tmp_path, "nbody_force")

    captured = {}

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kwargs"] = kw
        # Simulate the real driver writing reference.json on success.
        (driver_dir / "reference.json").write_text(json.dumps({"ok": True}))
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "ok"
    assert captured["kwargs"]["timeout"] == DEFAULT_RUN_TIMEOUT_SEC
    assert DEFAULT_RUN_TIMEOUT_SEC == 60  # invariant the docs promise


def test_run_baseline_driver_honors_env_timeout_override(monkeypatch, tmp_path):
    """A valid AGENT_PRECISION_RUN_TIMEOUT_SEC value is parsed as an int and passed as the subprocess timeout."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(RUN_TIMEOUT_ENV, "120")
    driver_dir = _stage_driver_binary(tmp_path, "nbody_force")

    captured = {}

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["timeout"] = kw["timeout"]
        (driver_dir / "reference.json").write_text(json.dumps({}))
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_baseline_driver("nbody_force", "kokkos")

    assert captured["timeout"] == 120


def test_run_baseline_driver_rejects_invalid_env_timeout(monkeypatch, tmp_path):
    """A non-integer or non-positive AGENT_PRECISION_RUN_TIMEOUT_SEC makes run_baseline_driver return status='error' WITHOUT invoking subprocess.run."""
    monkeypatch.chdir(tmp_path)
    _stage_driver_binary(tmp_path, "nbody_force")

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called for invalid timeout"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    for bad in ["not_an_int", "0", "-5", "3.14"]:
        monkeypatch.setenv(RUN_TIMEOUT_ENV, bad)
        result = run_baseline_driver("nbody_force", "kokkos")
        assert result["status"] == "error"
        assert RUN_TIMEOUT_ENV in result["stderr"]
        assert result["artifacts"] == []


# ---------- run_baseline_driver: missing / non-executable binary ----------


def test_run_baseline_driver_errors_when_driver_binary_missing(
    monkeypatch, tmp_path
):
    """If baselines/<stem>/driver does not exist, run_baseline_driver returns status='error' without invoking subprocess.run."""
    monkeypatch.chdir(tmp_path)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver binary is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "driver" in result["stderr"].lower()
    assert result["artifacts"] == []


def test_run_baseline_driver_errors_when_driver_binary_not_executable(
    monkeypatch, tmp_path
):
    """If baselines/<stem>/driver exists but lacks the exec bit, run_baseline_driver returns status='error' without invoking subprocess.run."""
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "baselines" / "nbody_force"
    driver_dir.mkdir(parents=True)
    binary = driver_dir / "driver"
    binary.write_text("noop")
    # Strip exec bits explicitly.
    binary.chmod(binary.stat().st_mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver binary is non-executable"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "executable" in result["stderr"].lower()
    assert result["artifacts"] == []


# ---------- run_baseline_driver: subprocess shape + cwd ----------


def test_run_baseline_driver_invokes_driver_with_per_stem_cwd(
    monkeypatch, tmp_path
):
    """run_baseline_driver shells out to ./driver with cwd=baselines/<stem>/ so the driver writes reference.json next to itself."""
    monkeypatch.chdir(tmp_path)
    _stage_driver_binary(tmp_path, "nbody_force", reference_payload={"x": 1})

    captured = {}

    class OkProc:
        returncode = 0
        stdout = "stdout-here"
        stderr = "stderr-here"

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = kw["cwd"]
        captured["capture_output"] = kw["capture_output"]
        captured["text"] = kw["text"]
        captured["check"] = kw["check"]
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_baseline_driver("nbody_force", "kokkos")

    assert captured["cmd"] == ["./driver"]
    # cwd is the per-stem dir, relative or absolute is fine but it must
    # be that directory.
    assert os.path.basename(captured["cwd"]) == "nbody_force"
    assert os.path.basename(os.path.dirname(captured["cwd"])) == "baselines"
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


# ---------- run_baseline_driver: success / failure / timeout ----------


def test_run_baseline_driver_success_returns_reference_json_artifact(
    monkeypatch, tmp_path
):
    """On exit 0 with a parseable reference.json, run_baseline_driver returns status='ok' with the single-element artifacts list ['baselines/<stem>/reference.json']."""
    monkeypatch.chdir(tmp_path)
    driver_dir = _stage_driver_binary(tmp_path, "nbody_force")

    class OkProc:
        returncode = 0
        stdout = "all good"
        stderr = ""

    def fake_run(cmd, **kw):
        # Simulate the real driver writing reference.json on success.
        (driver_dir / "reference.json").write_text(
            json.dumps({"outputs": [1.0, 2.0]})
        )
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "ok"
    assert result["stdout"] == "all good"
    assert result["artifacts"] == ["baselines/nbody_force/reference.json"]


def test_run_baseline_driver_errors_on_nonzero_exit(monkeypatch, tmp_path):
    """A non-zero driver exit produces status='error', no artifacts, and stderr that includes the exit code and the captured stderr."""
    monkeypatch.chdir(tmp_path)
    _stage_driver_binary(tmp_path, "nbody_force")

    class FailProc:
        returncode = 7
        stdout = ""
        stderr = "segfault while integrating\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "code 7" in result["stderr"]
    assert "segfault while integrating" in result["stderr"]
    assert result["artifacts"] == []


def test_run_baseline_driver_errors_when_reference_json_missing(
    monkeypatch, tmp_path
):
    """If the driver exits 0 but does NOT write reference.json, run_baseline_driver returns status='error' (the side artifact contract is unmet)."""
    monkeypatch.chdir(tmp_path)
    # Stage the driver binary but do NOT stage a reference.json.
    _stage_driver_binary(tmp_path, "nbody_force", reference_payload=None)

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "reference.json" in result["stderr"]
    assert result["artifacts"] == []


def test_run_baseline_driver_errors_on_invalid_reference_json(
    monkeypatch, tmp_path
):
    """If reference.json exists but does not parse as JSON, run_baseline_driver returns status='error'."""
    monkeypatch.chdir(tmp_path)
    driver_dir = _stage_driver_binary(tmp_path, "nbody_force")
    (driver_dir / "reference.json").write_text("this is not json {{{")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "reference.json" in result["stderr"]
    assert "JSON" in result["stderr"] or "json" in result["stderr"]
    assert result["artifacts"] == []


def test_run_baseline_driver_deletes_stale_reference_before_run(
    monkeypatch, tmp_path
):
    """Any pre-existing reference.json is deleted before the subprocess runs, so a failed driver cannot leave the orchestrator with a misleadingly-stale file in its artifacts on a later success check."""
    monkeypatch.chdir(tmp_path)
    driver_dir = _stage_driver_binary(tmp_path, "nbody_force")
    stale = driver_dir / "reference.json"
    stale.write_text(json.dumps({"stale": True}))

    observed_at_run = {}

    class FailProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(cmd, **kw):
        # By the time the subprocess "runs", the stale file must be gone.
        observed_at_run["stale_exists"] = stale.exists()
        return FailProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert observed_at_run["stale_exists"] is False
    # And the failure is still reported (we don't accidentally pass).
    assert result["status"] == "error"


def test_run_baseline_driver_errors_on_timeout(monkeypatch, tmp_path):
    """If the driver exceeds the configured timeout, run_baseline_driver catches TimeoutExpired and returns status='error' naming the timeout and the env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(RUN_TIMEOUT_ENV, "5")
    _stage_driver_binary(tmp_path, "nbody_force")

    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=5)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "timeout" in result["stderr"].lower()
    assert "5" in result["stderr"]
    assert RUN_TIMEOUT_ENV in result["stderr"]
    assert result["artifacts"] == []


def test_run_baseline_driver_handles_file_not_found_at_exec(
    monkeypatch, tmp_path
):
    """If subprocess.run raises FileNotFoundError at exec time (race / fs glitch), run_baseline_driver returns status='error' without crashing the orchestrator."""
    monkeypatch.chdir(tmp_path)
    _stage_driver_binary(tmp_path, "nbody_force")

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("./driver")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    result = run_baseline_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "driver" in result["stderr"].lower()
    assert result["artifacts"] == []


# ---------- run_baseline_driver: result schema invariant ----------


def test_run_baseline_driver_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — the same shape compile_baseline_driver and the planned remote-batch verifier tools share."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) bad env timeout
    monkeypatch.setenv(RUN_TIMEOUT_ENV, "not_an_int")
    assert set(run_baseline_driver("x", "kokkos").keys()) == expected_keys
    monkeypatch.delenv(RUN_TIMEOUT_ENV, raising=False)

    # 2) missing binary
    assert set(run_baseline_driver("x", "kokkos").keys()) == expected_keys

    # 3) success
    _stage_driver_binary(tmp_path, "x", reference_payload={"ok": 1})

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(run_baseline_driver("x", "kokkos").keys()) == expected_keys

    # 4) non-zero exit
    class FailProc:
        returncode = 9
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(run_baseline_driver("x", "kokkos").keys()) == expected_keys

    # 5) timeout
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert set(run_baseline_driver("x", "kokkos").keys()) == expected_keys


# ---------- splice_rewritten_kernel ----------


# A minimal-but-realistic baseline driver template used by the splice
# tests. The content between the sentinels stands in for the
# baseline_harness agent's inlined kernel. The exact bytes outside the
# sentinels are what the tool must preserve byte-for-byte.
_BASELINE_DRIVER_TEMPLATE = (
    "// driver.cpp -- baseline harness driver\n"
    "#include <Kokkos_Core.hpp>\n"
    "#include <cstdio>\n"
    "\n"
    "// ---- KERNEL BEGIN ----\n"
    "void kernel(double* x, int n) {\n"
    "  for (int i = 0; i < n; ++i) x[i] = x[i] * 2.0;\n"
    "}\n"
    "// ---- KERNEL END ----\n"
    "\n"
    "int main(int argc, char** argv) {\n"
    "  Kokkos::initialize(argc, argv);\n"
    "  Kokkos::finalize();\n"
    "  return 0;\n"
    "}\n"
)
_ORIGINAL_KERNEL_BODY = (
    "void kernel(double* x, int n) {\n"
    "  for (int i = 0; i < n; ++i) x[i] = x[i] * 2.0;\n"
    "}\n"
)


def _stage_baseline_driver(tmp_path, stem, body=_BASELINE_DRIVER_TEMPLATE):
    """Write a baseline driver.cpp at baselines/<stem>/ under tmp_path."""
    d = tmp_path / "baselines" / stem
    d.mkdir(parents=True)
    (d / "driver.cpp").write_text(body)
    return d


def _ban_subprocess(monkeypatch):
    """Make any subprocess.run call from splice_rewritten_kernel fail loudly."""

    def fail_run(*a, **kw):
        raise AssertionError(
            "splice_rewritten_kernel must not invoke subprocess.run "
            "(it is pure text I/O)"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)


def test_splice_rewritten_kernel_success_writes_rewritten_driver(
    monkeypatch, tmp_path
):
    """On success splice_rewritten_kernel returns status='ok' with the rewritten driver path in artifacts and writes a file at baselines/<stem>/rewritten/driver.cpp."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    new_kernel = (
        "void kernel(float* x, int n) {\n"
        "  for (int i = 0; i < n; ++i) x[i] = x[i] * 2.0f;\n"
        "}\n"
    )

    result = splice_rewritten_kernel("k", new_kernel, "kokkos")

    assert result["status"] == "ok"
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["artifacts"] == ["baselines/k/rewritten/driver.cpp"]

    out_path = tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp"
    assert out_path.is_file()
    text = out_path.read_text()
    # Both sentinels still present, exactly once each, each on its own
    # line.
    out_lines = text.split("\n")
    assert out_lines.count(KERNEL_BEGIN_SENTINEL) == 1
    assert out_lines.count(KERNEL_END_SENTINEL) == 1
    # The new kernel body landed strictly between them.
    begin = out_lines.index(KERNEL_BEGIN_SENTINEL)
    end = out_lines.index(KERNEL_END_SENTINEL)
    spliced = "\n".join(out_lines[begin + 1 : end])
    assert spliced == new_kernel.rstrip("\n")


def test_splice_rewritten_kernel_does_not_touch_baseline(monkeypatch, tmp_path):
    """splice_rewritten_kernel must never modify baselines/<stem>/driver.cpp; the rewritten copy goes under rewritten/."""
    monkeypatch.chdir(tmp_path)
    driver_dir = _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)
    before = (driver_dir / "driver.cpp").read_bytes()

    splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    after = (driver_dir / "driver.cpp").read_bytes()
    assert before == after


def test_splice_rewritten_kernel_preserves_bytes_outside_sentinels(
    monkeypatch, tmp_path
):
    """Lines outside the sentinel region in the spliced driver must be byte-identical to the baseline (only the kernel region changes)."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    splice_rewritten_kernel("k", "void kernel() { /* new */ }\n", "kokkos")

    baseline = (tmp_path / "baselines" / "k" / "driver.cpp").read_text()
    rewritten = (
        tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp"
    ).read_text()

    baseline_lines = baseline.split("\n")
    rewritten_lines = rewritten.split("\n")
    b_begin = baseline_lines.index(KERNEL_BEGIN_SENTINEL)
    b_end = baseline_lines.index(KERNEL_END_SENTINEL)
    r_begin = rewritten_lines.index(KERNEL_BEGIN_SENTINEL)
    r_end = rewritten_lines.index(KERNEL_END_SENTINEL)

    # Prefix up to and including BEGIN: identical.
    assert baseline_lines[: b_begin + 1] == rewritten_lines[: r_begin + 1]
    # Suffix from END onward: identical.
    assert baseline_lines[b_end:] == rewritten_lines[r_end:]


def test_splice_rewritten_kernel_round_trip_is_byte_identical(
    monkeypatch, tmp_path
):
    """Splicing the ORIGINAL kernel body back in must yield a file byte-identical to the baseline driver — strong proof that splice touches only the kernel region."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", _ORIGINAL_KERNEL_BODY, "kokkos")

    assert result["status"] == "ok"
    baseline = (tmp_path / "baselines" / "k" / "driver.cpp").read_bytes()
    rewritten = (
        tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp"
    ).read_bytes()
    assert baseline == rewritten


def test_splice_rewritten_kernel_overwrites_prior_rewritten(
    monkeypatch, tmp_path
):
    """A second call to splice_rewritten_kernel must overwrite an earlier rewritten/driver.cpp without complaint (chain can re-fire on each new verifier accept)."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    r1 = splice_rewritten_kernel("k", "void kernel() { /* v1 */ }\n", "kokkos")
    assert r1["status"] == "ok"
    first = (tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp").read_text()
    assert "v1" in first

    r2 = splice_rewritten_kernel("k", "void kernel() { /* v2 */ }\n", "kokkos")
    assert r2["status"] == "ok"
    second = (tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp").read_text()
    assert "v2" in second
    assert "v1" not in second


def test_splice_rewritten_kernel_errors_when_baseline_missing(
    monkeypatch, tmp_path
):
    """If baselines/<stem>/driver.cpp does not exist, splice_rewritten_kernel returns status='error' (and the rewritten/ directory is NOT created)."""
    monkeypatch.chdir(tmp_path)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert "driver.cpp" in result["stderr"]
    assert result["artifacts"] == []
    assert not (tmp_path / "baselines" / "k" / "rewritten").exists()


def test_splice_rewritten_kernel_errors_when_rewritten_source_empty(
    monkeypatch, tmp_path
):
    """An empty rewritten_kernel_source is a programming error and is rejected without touching the filesystem."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "", "kokkos")

    assert result["status"] == "error"
    assert result["artifacts"] == []
    assert not (tmp_path / "baselines" / "k" / "rewritten").exists()


def test_splice_rewritten_kernel_errors_when_begin_sentinel_missing(
    monkeypatch, tmp_path
):
    """Zero KERNEL BEGIN sentinels in the baseline is an error (sentinel uniqueness is part of the contract)."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE.replace(KERNEL_BEGIN_SENTINEL, "// nope")
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_BEGIN_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_errors_when_end_sentinel_missing(
    monkeypatch, tmp_path
):
    """Zero KERNEL END sentinels in the baseline is an error."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE.replace(KERNEL_END_SENTINEL, "// nope")
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_END_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_errors_when_begin_sentinel_duplicated(
    monkeypatch, tmp_path
):
    """More than one KERNEL BEGIN sentinel line is an error (sentinel uniqueness contract)."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE + KERNEL_BEGIN_SENTINEL + "\n"
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_BEGIN_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_errors_when_end_sentinel_duplicated(
    monkeypatch, tmp_path
):
    """More than one KERNEL END sentinel line is an error."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE + KERNEL_END_SENTINEL + "\n"
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_END_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_errors_when_sentinels_out_of_order(
    monkeypatch, tmp_path
):
    """If KERNEL END appears before KERNEL BEGIN, splice_rewritten_kernel returns status='error' rather than producing a garbled spliced file."""
    monkeypatch.chdir(tmp_path)
    # Build a driver where the END sentinel sits above the BEGIN sentinel.
    bad = (
        "// driver.cpp\n"
        "#include <Kokkos_Core.hpp>\n"
        "\n"
        f"{KERNEL_END_SENTINEL}\n"
        "void kernel() {}\n"
        f"{KERNEL_BEGIN_SENTINEL}\n"
        "\n"
        "int main() { return 0; }\n"
    )
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    # Both sentinel strings should be named in the diagnostic.
    assert KERNEL_BEGIN_SENTINEL in result["stderr"]
    assert KERNEL_END_SENTINEL in result["stderr"]
    assert result["artifacts"] == []
    assert not (tmp_path / "baselines" / "k" / "rewritten").exists()


def test_splice_rewritten_kernel_rejects_indented_sentinel(monkeypatch, tmp_path):
    """A sentinel line that is indented (or has trailing whitespace) is NOT a valid sentinel; splice_rewritten_kernel must reject the baseline rather than silently splicing into the wrong line."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE.replace(
        KERNEL_BEGIN_SENTINEL, "  " + KERNEL_BEGIN_SENTINEL  # leading spaces
    )
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_BEGIN_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_rejects_trailing_whitespace_sentinel(
    monkeypatch, tmp_path
):
    """Trailing whitespace after a sentinel string also disqualifies the line (byte-exact contract)."""
    monkeypatch.chdir(tmp_path)
    bad = _BASELINE_DRIVER_TEMPLATE.replace(
        KERNEL_END_SENTINEL + "\n", KERNEL_END_SENTINEL + "   \n"
    )
    _stage_baseline_driver(tmp_path, "k", body=bad)
    _ban_subprocess(monkeypatch)

    result = splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos")

    assert result["status"] == "error"
    assert KERNEL_END_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — the same shape compile_baseline_driver / run_baseline_driver / planned remote-batch verifier tools share."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)
    _ban_subprocess(monkeypatch)

    # 1) empty rewritten source
    assert set(splice_rewritten_kernel("k", "", "kokkos").keys()) == expected_keys

    # 2) missing baseline
    assert (
        set(splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos").keys())
        == expected_keys
    )

    # 3) success
    _stage_baseline_driver(tmp_path, "k")
    assert (
        set(splice_rewritten_kernel("k", "void kernel() {}\n", "kokkos").keys())
        == expected_keys
    )

    # 4) malformed baseline (sentinel missing)
    bad_stem = "bad"
    _stage_baseline_driver(
        tmp_path,
        bad_stem,
        body=_BASELINE_DRIVER_TEMPLATE.replace(KERNEL_BEGIN_SENTINEL, "// x"),
    )
    assert (
        set(splice_rewritten_kernel(bad_stem, "void kernel() {}\n", "kokkos").keys())
        == expected_keys
    )


# ---------- compile_rewritten_driver: env-var + missing source + success ----------


def _stage_rewritten_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/rewritten/driver.cpp under tmp_path.

    Mirrors _stage_driver but targets the rewritten subdirectory that
    splice_rewritten_kernel populates.
    """
    rewritten_dir = tmp_path / "baselines" / stem / "rewritten"
    rewritten_dir.mkdir(parents=True)
    (rewritten_dir / "driver.cpp").write_text(body)
    return rewritten_dir


def test_compile_rewritten_driver_errors_when_env_unset(monkeypatch, tmp_path):
    """compile_rewritten_driver returns status='error' with no subprocess call when AGENT_PRECISION_KOKKOS_ROOT is unset (same env contract as compile_baseline_driver)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KOKKOS_ROOT_ENV, raising=False)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when env unset"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert KOKKOS_ROOT_ENV in result["stderr"]
    assert result["artifacts"] == []


def test_compile_rewritten_driver_errors_when_env_points_at_non_kokkos(
    monkeypatch, tmp_path
):
    """compile_rewritten_driver returns status='error' when AGENT_PRECISION_KOKKOS_ROOT points at a directory missing include/ or lib/ (same env contract as compile_baseline_driver)."""
    monkeypatch.chdir(tmp_path)
    bogus = tmp_path / "bogus_root"
    bogus.mkdir()
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(bogus))

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called for an invalid Kokkos root"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "include/" in result["stderr"] or "lib/" in result["stderr"]
    assert result["artifacts"] == []


def test_compile_rewritten_driver_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """compile_rewritten_driver returns status='error' (no subprocess call) when baselines/<stem>/rewritten/driver.cpp does not exist, and the error names splice_rewritten_kernel as the upstream tool that was supposed to have written it (NOT spawn_baseline_harness)."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    # Path must point at the rewritten subdirectory, not the baseline.
    assert "rewritten/driver.cpp" in result["stderr"]
    # Hint must blame splice, not the baseline harness.
    assert "splice_rewritten_kernel" in result["stderr"]
    assert "spawn_baseline_harness" not in result["stderr"]
    assert result["artifacts"] == []


def test_compile_rewritten_driver_success_targets_rewritten_subdir(
    monkeypatch, tmp_path
):
    """On a successful compile, compile_rewritten_driver shells out to g++ with the rewritten driver source as input and writes the binary to baselines/<stem>/rewritten/driver — NOT the baseline path."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_rewritten_driver(tmp_path, "nbody_force")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "ok"
    assert result["artifacts"] == [
        "baselines/nbody_force/rewritten/driver"
    ]
    cmd = captured["cmd"]
    # Input source path is the rewritten one, NOT the baseline.
    assert "baselines/nbody_force/rewritten/driver.cpp" in cmd
    assert "baselines/nbody_force/driver.cpp" not in cmd
    # Output binary lives in the rewritten subdir.
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/nbody_force/rewritten/driver"


def test_compile_rewritten_driver_shares_compile_flags_with_baseline(
    monkeypatch, tmp_path
):
    """compile_rewritten_driver uses the same compiler, C++ standard, Kokkos include/lib flags, and link libraries as compile_baseline_driver — only the source/output directory differs. Protects against accidental drift between the parallel chains."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "k")
    _stage_rewritten_driver(tmp_path, "k")

    cmds = []

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: (cmds.append(cmd), FakeProc())[1],
    )

    compile_baseline_driver("k", "kokkos")
    compile_rewritten_driver("k", "kokkos")

    base_cmd, rewritten_cmd = cmds
    # Strip the source path and the -o target path; the rest of the
    # command must match byte-for-byte.
    def _normalize(c):
        return [
            x for x in c
            if x != "baselines/k/driver.cpp"
            and x != "baselines/k/driver"
            and x != "baselines/k/rewritten/driver.cpp"
            and x != "baselines/k/rewritten/driver"
        ]

    assert _normalize(base_cmd) == _normalize(rewritten_cmd)


def test_compile_rewritten_driver_compile_failure_propagates_stderr(
    monkeypatch, tmp_path
):
    """A non-zero g++ exit produces status='error' with the captured stderr, no artifacts, and an exit-code line — same error shape as compile_baseline_driver."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_rewritten_driver(tmp_path, "nbody_force")

    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "rewritten/driver.cpp:9: error: invalid conversion\n"

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: FakeProc()
    )

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "exited with code 1" in result["stderr"]
    assert "invalid conversion" in result["stderr"]
    assert result["artifacts"] == []


def test_compile_rewritten_driver_handles_missing_gxx(monkeypatch, tmp_path):
    """If g++ is not on PATH, compile_rewritten_driver returns status='error' (it does not crash the orchestrator) — same as compile_baseline_driver."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_rewritten_driver(tmp_path, "nbody_force")

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("g++")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    result = compile_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "g++" in result["stderr"]
    assert result["artifacts"] == []


def test_compile_rewritten_driver_does_not_touch_baseline_binary(
    monkeypatch, tmp_path
):
    """compile_rewritten_driver must not overwrite a pre-existing baselines/<stem>/driver binary — the rewritten and baseline binaries are separate artifacts and the comparator needs both intact."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_rewritten_driver(tmp_path, "k")

    # Pre-existing baseline binary that must survive untouched.
    baseline_dir = tmp_path / "baselines" / "k"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_bin = baseline_dir / "driver"
    baseline_bin.write_bytes(b"ORIGINAL BASELINE BINARY")

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: FakeProc()
    )

    compile_rewritten_driver("k", "kokkos")

    assert baseline_bin.read_bytes() == b"ORIGINAL BASELINE BINARY"


def test_compile_rewritten_driver_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — same shape as compile_baseline_driver / run_baseline_driver / splice_rewritten_kernel / planned remote-batch verifier tools."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}

    # 1) env unset
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KOKKOS_ROOT_ENV, raising=False)
    assert set(compile_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 2) env set but no rewritten source
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    assert set(compile_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 3) success
    _stage_rewritten_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 4) failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_rewritten_driver("x", "kokkos").keys()) == expected_keys


# ---------- run_rewritten_driver: env-var + preflight + cwd + success ----------


def _stage_rewritten_driver_binary(tmp_path, stem, reference_payload=None):
    """Create a fake executable at baselines/<stem>/rewritten/driver under tmp_path.

    Mirrors _stage_driver_binary but targets the rewritten subdirectory
    that compile_rewritten_driver populates. The file does NOT need to
    actually do anything; tests monkeypatch subprocess.run. It just
    needs to exist and be executable so run_rewritten_driver's
    preflight checks pass. If reference_payload is provided, it is
    written as ./reference.json next to the driver (matching what a
    real driver would do on success).
    """
    rewritten_dir = tmp_path / "baselines" / stem / "rewritten"
    rewritten_dir.mkdir(parents=True)
    binary = rewritten_dir / "driver"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    if reference_payload is not None:
        (rewritten_dir / "reference.json").write_text(
            json.dumps(reference_payload)
        )
    return rewritten_dir


def test_run_rewritten_driver_rejects_invalid_env_timeout(monkeypatch, tmp_path):
    """A non-integer or non-positive AGENT_PRECISION_RUN_TIMEOUT_SEC makes run_rewritten_driver return status='error' WITHOUT invoking subprocess.run (same env contract as run_baseline_driver)."""
    monkeypatch.chdir(tmp_path)
    _stage_rewritten_driver_binary(tmp_path, "nbody_force")

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called for invalid timeout"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    for bad in ["not_an_int", "0", "-5", "3.14"]:
        monkeypatch.setenv(RUN_TIMEOUT_ENV, bad)
        result = run_rewritten_driver("nbody_force", "kokkos")
        assert result["status"] == "error"
        assert RUN_TIMEOUT_ENV in result["stderr"]
        assert result["artifacts"] == []


def test_run_rewritten_driver_errors_when_driver_binary_missing(
    monkeypatch, tmp_path
):
    """If baselines/<stem>/rewritten/driver does not exist, run_rewritten_driver returns status='error' (no subprocess call) and the error names compile_rewritten_driver as the upstream tool that was supposed to have produced it (NOT compile_baseline_driver)."""
    monkeypatch.chdir(tmp_path)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver binary is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    # Path must point at the rewritten subdirectory, not the baseline.
    assert "rewritten/driver" in result["stderr"]
    # Hint must blame the rewritten-compile step, not the baseline compile.
    assert "compile_rewritten_driver" in result["stderr"]
    assert "compile_baseline_driver" not in result["stderr"]
    assert result["artifacts"] == []


def test_run_rewritten_driver_errors_when_driver_binary_not_executable(
    monkeypatch, tmp_path
):
    """If baselines/<stem>/rewritten/driver exists but lacks the exec bit, run_rewritten_driver returns status='error' without invoking subprocess.run."""
    monkeypatch.chdir(tmp_path)
    rewritten_dir = tmp_path / "baselines" / "nbody_force" / "rewritten"
    rewritten_dir.mkdir(parents=True)
    binary = rewritten_dir / "driver"
    binary.write_text("noop")
    binary.chmod(binary.stat().st_mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver binary is non-executable"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "executable" in result["stderr"].lower()
    assert result["artifacts"] == []


def test_run_rewritten_driver_invokes_driver_with_rewritten_subdir_cwd(
    monkeypatch, tmp_path
):
    """run_rewritten_driver shells out to ./driver with cwd=baselines/<stem>/rewritten/ — so the driver writes reference.json inside the rewritten subtree, not next to the baseline."""
    monkeypatch.chdir(tmp_path)
    _stage_rewritten_driver_binary(
        tmp_path, "nbody_force", reference_payload={"x": 1}
    )

    captured = {}

    class OkProc:
        returncode = 0
        stdout = "stdout-here"
        stderr = "stderr-here"

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = kw["cwd"]
        captured["capture_output"] = kw["capture_output"]
        captured["text"] = kw["text"]
        captured["check"] = kw["check"]
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_rewritten_driver("nbody_force", "kokkos")

    assert captured["cmd"] == ["./driver"]
    # cwd is the per-stem rewritten dir.
    assert os.path.basename(captured["cwd"]) == "rewritten"
    assert os.path.basename(os.path.dirname(captured["cwd"])) == "nbody_force"
    assert (
        os.path.basename(os.path.dirname(os.path.dirname(captured["cwd"])))
        == "baselines"
    )
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_run_rewritten_driver_success_returns_rewritten_reference_artifact(
    monkeypatch, tmp_path
):
    """On exit 0 with a parseable reference.json, run_rewritten_driver returns status='ok' with the single-element artifacts list ['baselines/<stem>/rewritten/reference.json']."""
    monkeypatch.chdir(tmp_path)
    rewritten_dir = _stage_rewritten_driver_binary(tmp_path, "nbody_force")

    class OkProc:
        returncode = 0
        stdout = "all good"
        stderr = ""

    def fake_run(cmd, **kw):
        (rewritten_dir / "reference.json").write_text(
            json.dumps({"outputs": [1.0, 2.0]})
        )
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "ok"
    assert result["stdout"] == "all good"
    assert result["artifacts"] == [
        "baselines/nbody_force/rewritten/reference.json"
    ]


def test_run_rewritten_driver_does_not_touch_baseline_reference(
    monkeypatch, tmp_path
):
    """run_rewritten_driver must not overwrite, delete, or even observe a pre-existing baselines/<stem>/reference.json from the baseline run. The two trees are independent artifacts the future comparator needs both of."""
    monkeypatch.chdir(tmp_path)
    rewritten_dir = _stage_rewritten_driver_binary(tmp_path, "nbody_force")

    # Pre-existing baseline reference that must survive byte-identically.
    baseline_dir = tmp_path / "baselines" / "nbody_force"
    sentinel_bytes = b'{"baseline_sentinel": "do not touch"}'
    (baseline_dir / "reference.json").write_bytes(sentinel_bytes)

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        (rewritten_dir / "reference.json").write_text(json.dumps({"ok": 1}))
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_rewritten_driver("nbody_force", "kokkos")

    # Baseline file must be byte-identical to what we wrote.
    assert (baseline_dir / "reference.json").read_bytes() == sentinel_bytes


def test_run_rewritten_driver_deletes_stale_rewritten_reference_before_run(
    monkeypatch, tmp_path
):
    """Any pre-existing baselines/<stem>/rewritten/reference.json is deleted before the subprocess runs, so a failed rewritten driver cannot leave the orchestrator with a misleadingly-stale file (same guarantee run_baseline_driver makes, scoped to the rewritten subtree)."""
    monkeypatch.chdir(tmp_path)
    rewritten_dir = _stage_rewritten_driver_binary(tmp_path, "nbody_force")
    stale = rewritten_dir / "reference.json"
    stale.write_text(json.dumps({"stale": True}))

    observed_at_run = {}

    class FailProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(cmd, **kw):
        observed_at_run["stale_exists"] = stale.exists()
        return FailProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_rewritten_driver("nbody_force", "kokkos")

    assert observed_at_run["stale_exists"] is False
    assert result["status"] == "error"


def test_run_rewritten_driver_errors_on_timeout(monkeypatch, tmp_path):
    """If the rewritten driver exceeds the configured timeout, run_rewritten_driver catches TimeoutExpired and returns status='error' naming the timeout and the env var (same shape as run_baseline_driver)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(RUN_TIMEOUT_ENV, "5")
    _stage_rewritten_driver_binary(tmp_path, "nbody_force")

    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=5)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = run_rewritten_driver("nbody_force", "kokkos")

    assert result["status"] == "error"
    assert "timeout" in result["stderr"].lower()
    assert "5" in result["stderr"]
    assert RUN_TIMEOUT_ENV in result["stderr"]
    assert result["artifacts"] == []


def test_run_rewritten_driver_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — same shape as run_baseline_driver and the other deterministic tools."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) bad env timeout
    monkeypatch.setenv(RUN_TIMEOUT_ENV, "not_an_int")
    assert set(run_rewritten_driver("x", "kokkos").keys()) == expected_keys
    monkeypatch.delenv(RUN_TIMEOUT_ENV, raising=False)

    # 2) missing binary
    assert set(run_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 3) success
    _stage_rewritten_driver_binary(tmp_path, "x", reference_payload={"ok": 1})

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(run_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 4) non-zero exit
    class FailProc:
        returncode = 9
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(run_rewritten_driver("x", "kokkos").keys()) == expected_keys

    # 5) timeout
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert set(run_rewritten_driver("x", "kokkos").keys()) == expected_keys


# ---------- compare_outputs: shape + tolerance + special values + artifact ----------


def _write_reference_pair(
    tmp_path, stem, baseline_payload, rewritten_payload
):
    """Stage baseline + rewritten reference.json files for a kernel_stem.

    Mirrors the layout the live run_baseline_driver / run_rewritten_driver
    chain would have produced: baselines/<stem>/reference.json and
    baselines/<stem>/rewritten/reference.json. Either payload may be a
    Python value (will be json.dumps'd) or a raw string (written
    verbatim, so tests can stage deliberately-invalid JSON).
    """
    baseline_dir = tmp_path / "baselines" / stem
    rewritten_dir = baseline_dir / "rewritten"
    rewritten_dir.mkdir(parents=True)

    def write_one(target, payload):
        if payload is None:
            return
        if isinstance(payload, str):
            target.write_text(payload)
        else:
            target.write_text(json.dumps(payload))

    write_one(baseline_dir / "reference.json", baseline_payload)
    write_one(rewritten_dir / "reference.json", rewritten_payload)
    return baseline_dir, rewritten_dir


def _well_shaped(outputs):
    """Build a minimally-valid reference.json shell around an outputs dict."""
    return {
        "kernel": "foo",
        "seed": 1,
        "inputs": {"N": 8},
        "outputs": outputs,
    }


def _tolerance(kind, value):
    return json.dumps({"kind": kind, "value": value, "source": "user_cli"})


def test_compare_outputs_errors_when_both_references_missing(
    monkeypatch, tmp_path
):
    """If neither baseline/<stem>/reference.json nor baselines/<stem>/rewritten/reference.json exists, compare_outputs returns status='error' (no comparison.json is written because there is no rewritten dir to put it in)."""
    monkeypatch.chdir(tmp_path)

    result = compare_outputs("nbody_force", _tolerance("sig_figs", 3), "kokkos")

    assert result["status"] == "error"
    assert "baseline" in result["stderr"].lower()
    assert result["artifacts"] == []


def test_compare_outputs_errors_when_baseline_missing(monkeypatch, tmp_path):
    """If only the rewritten reference.json exists, compare_outputs blames the baseline side and names run_baseline_driver as the upstream that should have produced it."""
    monkeypatch.chdir(tmp_path)
    rewritten_dir = tmp_path / "baselines" / "x" / "rewritten"
    rewritten_dir.mkdir(parents=True)
    (rewritten_dir / "reference.json").write_text(
        json.dumps(_well_shaped({"out": [1.0]}))
    )

    result = compare_outputs("x", _tolerance("sig_figs", 3), "kokkos")

    assert result["status"] == "error"
    assert "baseline" in result["stderr"].lower()
    assert "run_baseline_driver" in result["stderr"]
    assert result["artifacts"] == []


def test_compare_outputs_errors_when_rewritten_missing(monkeypatch, tmp_path):
    """If only the baseline reference.json exists, compare_outputs blames the rewritten side and names run_rewritten_driver as the upstream that should have produced it."""
    monkeypatch.chdir(tmp_path)
    baseline_dir = tmp_path / "baselines" / "x"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "reference.json").write_text(
        json.dumps(_well_shaped({"out": [1.0]}))
    )

    result = compare_outputs("x", _tolerance("sig_figs", 3), "kokkos")

    assert result["status"] == "error"
    assert "rewritten" in result["stderr"].lower()
    assert "run_rewritten_driver" in result["stderr"]
    assert result["artifacts"] == []


def test_compare_outputs_errors_on_invalid_json_either_side(
    monkeypatch, tmp_path
):
    """A reference.json that exists but does not parse as JSON returns status='error' on whichever side is bad, citing the file path."""
    monkeypatch.chdir(tmp_path)

    # Bad baseline.
    _write_reference_pair(
        tmp_path,
        "a",
        baseline_payload="{not json",
        rewritten_payload=_well_shaped({"out": [1.0]}),
    )
    r1 = compare_outputs("a", _tolerance("sig_figs", 3), "kokkos")
    assert r1["status"] == "error"
    assert "baseline" in r1["stderr"].lower()

    # Bad rewritten.
    _write_reference_pair(
        tmp_path,
        "b",
        baseline_payload=_well_shaped({"out": [1.0]}),
        rewritten_payload="{also not json",
    )
    r2 = compare_outputs("b", _tolerance("sig_figs", 3), "kokkos")
    assert r2["status"] == "error"
    assert "rewritten" in r2["stderr"].lower()


def test_compare_outputs_errors_on_malformed_tolerance_json(
    monkeypatch, tmp_path
):
    """An unparseable or wrong-kind tolerance_json returns status='error' WITHOUT touching the reference.json files (so a bad caller cannot wedge the comparator)."""
    monkeypatch.chdir(tmp_path)
    _write_reference_pair(
        tmp_path,
        "x",
        baseline_payload=_well_shaped({"out": [1.0]}),
        rewritten_payload=_well_shaped({"out": [1.0]}),
    )

    # Unparseable.
    bad1 = compare_outputs("x", "{not json", "kokkos")
    assert bad1["status"] == "error"
    assert "tolerance_json" in bad1["stderr"]
    assert bad1["artifacts"] == []

    # Wrong kind.
    bad2 = compare_outputs(
        "x", json.dumps({"kind": "ulps", "value": 3, "source": "user_cli"}), "kokkos"
    )
    assert bad2["status"] == "error"
    assert "kind" in bad2["stderr"]

    # Bad value (non-positive or non-int).
    bad3 = compare_outputs(
        "x",
        json.dumps({"kind": "sig_figs", "value": 0, "source": "user_cli"}), "kokkos"
    )
    assert bad3["status"] == "error"
    assert "value" in bad3["stderr"]

    bad4 = compare_outputs(
        "x",
        json.dumps(
            {"kind": "sig_figs", "value": "three", "source": "user_cli"}
        ), "kokkos"
    )
    assert bad4["status"] == "error"


def test_compare_outputs_shape_mismatch_writes_shape_error_artifact(
    monkeypatch, tmp_path
):
    """Different top-level keys / output array names / array lengths each surface as status='error' with shape_error populated in the written comparison.json (so the operator can distinguish from a regular tolerance failure)."""
    monkeypatch.chdir(tmp_path)

    # Different top-level keys.
    _write_reference_pair(
        tmp_path,
        "topk",
        baseline_payload={
            "kernel": "foo",
            "seed": 1,
            "inputs": {"N": 8},
            "outputs": {"a": [1.0]},
        },
        rewritten_payload={
            "kernel": "foo",
            "seed": 1,
            "outputs": {"a": [1.0]},
        },
    )
    r_topk = compare_outputs("topk", _tolerance("sig_figs", 3), "kokkos")
    assert r_topk["status"] == "error"
    doc_topk = json.loads(
        (
            tmp_path
            / "baselines"
            / "topk"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert "shape_error" in doc_topk
    assert "top-level" in doc_topk["shape_error"].lower()

    # Different output array names.
    _write_reference_pair(
        tmp_path,
        "names",
        baseline_payload=_well_shaped({"a": [1.0], "b": [2.0]}),
        rewritten_payload=_well_shaped({"a": [1.0], "c": [2.0]}),
    )
    r_names = compare_outputs("names", _tolerance("sig_figs", 3), "kokkos")
    assert r_names["status"] == "error"
    doc_names = json.loads(
        (
            tmp_path
            / "baselines"
            / "names"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert "shape_error" in doc_names
    assert (
        "output array" in doc_names["shape_error"].lower()
        or "name mismatch" in doc_names["shape_error"].lower()
    )

    # Different array lengths.
    _write_reference_pair(
        tmp_path,
        "lens",
        baseline_payload=_well_shaped({"a": [1.0, 2.0, 3.0]}),
        rewritten_payload=_well_shaped({"a": [1.0, 2.0]}),
    )
    r_lens = compare_outputs("lens", _tolerance("sig_figs", 3), "kokkos")
    assert r_lens["status"] == "error"
    doc_lens = json.loads(
        (
            tmp_path
            / "baselines"
            / "lens"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert "shape_error" in doc_lens
    assert "length" in doc_lens["shape_error"].lower()


def test_compare_outputs_sig_figs_passes_inside_threshold(
    monkeypatch, tmp_path
):
    """sig_figs=N passes when |a-b| < 10^-N * max(|a|,|b|) (strict <), and the both-zero special case passes."""
    monkeypatch.chdir(tmp_path)
    # 1.0 vs 1.0009 with sig_figs=3 -> threshold = 0.001 * 1.0009 ~= 0.001;
    # |diff| = 0.0009 < threshold, so passes.
    _write_reference_pair(
        tmp_path,
        "inside",
        baseline_payload=_well_shaped({"a": [1.0, 0.0]}),
        rewritten_payload=_well_shaped({"a": [1.0009, 0.0]}),
    )
    result = compare_outputs("inside", _tolerance("sig_figs", 3), "kokkos")
    assert result["status"] == "ok"
    assert result["artifacts"] == [
        "baselines/inside/rewritten/comparison.json"
    ]
    doc = json.loads(
        (
            tmp_path
            / "baselines"
            / "inside"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert doc["status"] == "ok"
    assert doc["total_compared"] == 2
    assert doc["mismatches"] == []


def test_compare_outputs_sig_figs_fails_just_outside_threshold(
    monkeypatch, tmp_path
):
    """sig_figs=N is strict-<, so a value at or just beyond 10^-N * max(|a|,|b|) fails and the failure is recorded with abs_err and threshold in comparison.json."""
    monkeypatch.chdir(tmp_path)
    # 1.0 vs 1.01 with sig_figs=3 -> threshold = 0.001 * 1.01 = 0.00101;
    # |diff| = 0.01 > threshold, so fails.
    _write_reference_pair(
        tmp_path,
        "outside",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [1.01]}),
    )
    result = compare_outputs("outside", _tolerance("sig_figs", 3), "kokkos")
    assert result["status"] == "error"
    assert "1/1" in result["stderr"]
    doc = json.loads(
        (
            tmp_path
            / "baselines"
            / "outside"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert doc["status"] == "error"
    assert len(doc["mismatches"]) == 1
    m = doc["mismatches"][0]
    assert m["name"] == "a" and m["index"] == 0
    assert m["abs_err"] is not None
    assert m["threshold"] is not None


def test_compare_outputs_decimal_digits_pass_and_fail(
    monkeypatch, tmp_path
):
    """decimal_digits=N passes when |a-b| < 10^-N and fails otherwise, independent of the magnitudes (unlike sig_figs)."""
    monkeypatch.chdir(tmp_path)
    # decimal_digits=4 -> threshold = 0.0001.
    # Pair (1000.0, 1000.00005) passes; pair (1.0, 1.001) fails.
    _write_reference_pair(
        tmp_path,
        "dd",
        baseline_payload=_well_shaped({"a": [1000.0], "b": [1.0]}),
        rewritten_payload=_well_shaped({"a": [1000.00005], "b": [1.001]}),
    )
    result = compare_outputs("dd", _tolerance("decimal_digits", 4), "kokkos")
    assert result["status"] == "error"
    doc = json.loads(
        (
            tmp_path
            / "baselines"
            / "dd"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    failed_names = {m["name"] for m in doc["mismatches"]}
    assert failed_names == {"b"}  # 'a' passed, 'b' failed
    assert doc["total_compared"] == 2


def test_compare_outputs_nan_always_mismatches(monkeypatch, tmp_path):
    """NaN vs NaN, NaN vs finite, and finite vs NaN all mismatch — the comparator deliberately rejects the IEEE 754 NaN-equality asymmetry so any NaN in either output flags a regression."""
    monkeypatch.chdir(tmp_path)
    # Use Python's json non-strict parsing of NaN by writing the file raw.
    raw_base = '{"kernel":"k","seed":1,"inputs":{},"outputs":{"a":[NaN,1.0,NaN]}}'
    raw_rewr = '{"kernel":"k","seed":1,"inputs":{},"outputs":{"a":[NaN,NaN,2.0]}}'
    baseline_dir = tmp_path / "baselines" / "nans"
    (baseline_dir / "rewritten").mkdir(parents=True)
    (baseline_dir / "reference.json").write_text(raw_base)
    (baseline_dir / "rewritten" / "reference.json").write_text(raw_rewr)

    result = compare_outputs("nans", _tolerance("sig_figs", 3), "kokkos")
    assert result["status"] == "error"
    doc = json.loads(
        (baseline_dir / "rewritten" / "comparison.json").read_text()
    )
    # All three pairs are NaN-tainted: (NaN, NaN), (1.0, NaN), (NaN, 2.0).
    assert doc["total_compared"] == 3
    assert len(doc["mismatches"]) == 3


def test_compare_outputs_inf_rules(monkeypatch, tmp_path):
    """+inf vs +inf passes; -inf vs -inf passes; +inf vs -inf fails; inf vs finite fails."""
    monkeypatch.chdir(tmp_path)
    raw_base = (
        '{"kernel":"k","seed":1,"inputs":{},'
        '"outputs":{"a":[Infinity, -Infinity, Infinity, Infinity]}}'
    )
    raw_rewr = (
        '{"kernel":"k","seed":1,"inputs":{},'
        '"outputs":{"a":[Infinity, -Infinity, -Infinity, 1.0]}}'
    )
    baseline_dir = tmp_path / "baselines" / "infs"
    (baseline_dir / "rewritten").mkdir(parents=True)
    (baseline_dir / "reference.json").write_text(raw_base)
    (baseline_dir / "rewritten" / "reference.json").write_text(raw_rewr)

    result = compare_outputs("infs", _tolerance("sig_figs", 3), "kokkos")
    assert result["status"] == "error"
    doc = json.loads(
        (baseline_dir / "rewritten" / "comparison.json").read_text()
    )
    # Pairs: (+inf,+inf) OK, (-inf,-inf) OK, (+inf,-inf) FAIL, (+inf,1.0) FAIL.
    assert doc["total_compared"] == 4
    failed_indices = sorted(m["index"] for m in doc["mismatches"])
    assert failed_indices == [2, 3]


def test_compare_outputs_truncates_mismatch_list_with_footer(
    monkeypatch, tmp_path
):
    """When more than 10 leaves disagree, the stderr (and the written comparison.json) lists only the first 10 entries followed by a "+ K more mismatches suppressed" footer."""
    monkeypatch.chdir(tmp_path)
    n = 25
    _write_reference_pair(
        tmp_path,
        "many",
        baseline_payload=_well_shaped({"a": [0.0] * n}),
        rewritten_payload=_well_shaped({"a": [1.0] * n}),
    )
    result = compare_outputs("many", _tolerance("decimal_digits", 6), "kokkos")
    assert result["status"] == "error"
    # Stderr footer mentions the suppressed count.
    assert f"{n - 10} more mismatches suppressed" in result["stderr"]
    # The written comparison.json is also capped at 10 entries.
    doc = json.loads(
        (
            tmp_path
            / "baselines"
            / "many"
            / "rewritten"
            / "comparison.json"
        ).read_text()
    )
    assert len(doc["mismatches"]) == 10
    assert doc["total_compared"] == n


def test_compare_outputs_writes_comparison_json_on_both_paths(
    monkeypatch, tmp_path
):
    """comparison.json is written on the success path AND the tolerance-failure path, always under the rewritten subtree, always listed as the artifact."""
    monkeypatch.chdir(tmp_path)

    # Success path.
    _write_reference_pair(
        tmp_path,
        "okpath",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [1.0]}),
    )
    ok = compare_outputs("okpath", _tolerance("sig_figs", 3), "kokkos")
    assert ok["status"] == "ok"
    ok_path = tmp_path / "baselines" / "okpath" / "rewritten" / "comparison.json"
    assert ok_path.is_file()
    assert ok["artifacts"] == [str(ok_path.relative_to(tmp_path))]

    # Failure path.
    _write_reference_pair(
        tmp_path,
        "failpath",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [10.0]}),
    )
    bad = compare_outputs("failpath", _tolerance("sig_figs", 3), "kokkos")
    assert bad["status"] == "error"
    bad_path = (
        tmp_path / "baselines" / "failpath" / "rewritten" / "comparison.json"
    )
    assert bad_path.is_file()
    assert bad["artifacts"] == [str(bad_path.relative_to(tmp_path))]


def test_compare_outputs_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — same shape as the other deterministic tools so the orchestrator's tool-result handling is uniform."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) malformed tolerance
    r1 = compare_outputs("nope", "not json", "kokkos")
    assert set(r1.keys()) == expected_keys

    # 2) both files missing
    r2 = compare_outputs("nope", _tolerance("sig_figs", 3), "kokkos")
    assert set(r2.keys()) == expected_keys

    # 3) shape mismatch
    _write_reference_pair(
        tmp_path,
        "shp",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"b": [1.0]}),
    )
    r3 = compare_outputs("shp", _tolerance("sig_figs", 3), "kokkos")
    assert set(r3.keys()) == expected_keys

    # 4) ok
    _write_reference_pair(
        tmp_path,
        "go",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [1.0]}),
    )
    r4 = compare_outputs("go", _tolerance("sig_figs", 3), "kokkos")
    assert set(r4.keys()) == expected_keys

    # 5) tolerance fail
    _write_reference_pair(
        tmp_path,
        "no",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [10.0]}),
    )
    r5 = compare_outputs("no", _tolerance("sig_figs", 3), "kokkos")
    assert set(r5.keys()) == expected_keys


# ---------- CUDA profile: compile + splice ----------
#
# These tests exercise the CUDA language profile (workflow.languages.cuda)
# through the same six public tool wrappers as the Kokkos tests above.
# Phase B added CUDA as the second profile, so the wrappers must dispatch
# on `language_id="cuda"` to the nvcc command shape, the nvcc-on-PATH
# preflight, and the `driver.cu` filename. The Kokkos tests above
# remain the back-compat coverage for the g++ + Kokkos path; these
# tests guarantee the new branch.


def _stage_cuda_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/driver.cu under tmp_path."""
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    (driver_dir / "driver.cu").write_text(body)
    return driver_dir


def _stage_cuda_baseline_driver(tmp_path, stem, body=_BASELINE_DRIVER_TEMPLATE):
    """Write a baseline driver.cu at baselines/<stem>/ under tmp_path.

    The body content is the same Kokkos-flavored template the splice
    tests use; splice_rewritten_kernel only cares about the sentinel
    lines, not the surrounding C++. What matters here is that the file
    lives at driver.cu (not driver.cpp), so the CUDA profile finds it.
    """
    d = tmp_path / "baselines" / stem
    d.mkdir(parents=True)
    (d / "driver.cu").write_text(body)
    return d


def test_compile_baseline_driver_cuda_invokes_nvcc_with_default_arch(
    monkeypatch, tmp_path
):
    """For language_id='cuda', compile_baseline_driver shells out to nvcc with -std=c++17, -O2, -arch=sm_89 by default, and the driver.cu source path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CUDA_ARCH_ENV, raising=False)
    _stage_cuda_driver(tmp_path, "vector_add")
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "cuda")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/vector_add/driver"]
    cmd = captured["cmd"]
    assert cmd[0] == NVCC
    assert "-std=c++17" in cmd
    assert "-O2" in cmd
    assert f"-arch={CUDA_DEFAULT_ARCH}" in cmd
    assert CUDA_DEFAULT_ARCH == "sm_89"  # invariant the docs promise
    # Driver source is the .cu input, output is baselines/<stem>/driver.
    assert "baselines/vector_add/driver.cu" in cmd
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/vector_add/driver"
    # The CUDA profile must NOT inherit any Kokkos-specific flags.
    assert "-fopenmp" not in cmd
    assert "-lkokkoscore" not in cmd
    assert "-lkokkoscontainers" not in cmd


def test_compile_baseline_driver_cuda_honors_arch_env_override(
    monkeypatch, tmp_path
):
    """AGENT_PRECISION_CUDA_ARCH=sm_70 replaces the default in the nvcc -arch= flag."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CUDA_ARCH_ENV, "sm_70")
    _stage_cuda_driver(tmp_path, "vector_add")
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "cuda")

    assert result["status"] == "ok"
    cmd = captured["cmd"]
    assert "-arch=sm_70" in cmd
    assert f"-arch={CUDA_DEFAULT_ARCH}" not in cmd


def test_compile_baseline_driver_cuda_errors_when_nvcc_missing(
    monkeypatch, tmp_path
):
    """When shutil.which('nvcc') returns None, compile_baseline_driver returns status='error' WITHOUT invoking subprocess.run and the stderr names nvcc."""
    monkeypatch.chdir(tmp_path)
    # No need to stage a source file — preflight runs before the
    # source-exists check, so the error must surface from the missing
    # toolchain alone.
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: None)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when nvcc is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "cuda")

    assert result["status"] == "error"
    assert NVCC in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_cuda_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """When the CUDA driver source is absent, compile_baseline_driver returns status='error' (no subprocess) and the error names driver.cu (not driver.cpp)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "cuda")

    assert result["status"] == "error"
    assert "driver.cu" in result["stderr"]
    assert "driver.cpp" not in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_cuda_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every CUDA code path returns a dict with exactly {status, stdout, stderr, artifacts} — the uniform schema shared with the Kokkos path."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) nvcc missing
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: None)
    assert set(compile_baseline_driver("x", "cuda").keys()) == expected_keys

    # 2) success
    monkeypatch.setattr(cuda_profile.shutil, "which", lambda name: f"/usr/bin/{name}")
    _stage_cuda_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x", "cuda").keys()) == expected_keys

    # 3) compile failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x", "cuda").keys()) == expected_keys


def test_splice_rewritten_kernel_cuda_reads_and_writes_dot_cu(
    monkeypatch, tmp_path
):
    """For language_id='cuda', splice_rewritten_kernel reads baselines/<stem>/driver.cu and writes baselines/<stem>/rewritten/driver.cu — NOT driver.cpp at either path."""
    monkeypatch.chdir(tmp_path)
    _stage_cuda_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    new_kernel = (
        "__global__ void kernel(float* x, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) x[i] = x[i] * 2.0f;\n"
        "}\n"
    )

    result = splice_rewritten_kernel("k", new_kernel, "cuda")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/k/rewritten/driver.cu"]

    out_path = tmp_path / "baselines" / "k" / "rewritten" / "driver.cu"
    assert out_path.is_file()
    # The .cpp filename must not appear in either tree.
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.cpp").exists()

    text = out_path.read_text()
    out_lines = text.split("\n")
    assert out_lines.count(KERNEL_BEGIN_SENTINEL) == 1
    assert out_lines.count(KERNEL_END_SENTINEL) == 1
    begin = out_lines.index(KERNEL_BEGIN_SENTINEL)
    end = out_lines.index(KERNEL_END_SENTINEL)
    spliced = "\n".join(out_lines[begin + 1 : end])
    assert spliced == new_kernel.rstrip("\n")


# ---------- HIP profile: compile + splice ----------
#
# These tests exercise the HIP language profile (workflow.languages.hip)
# through the same public tool wrappers as the Kokkos and CUDA tests
# above. Phase C-1 added HIP as the third profile, so the wrappers must
# dispatch on `language_id="hip"` to the hipcc command shape, the
# hipcc-on-PATH preflight, and the `driver.hip` filename.
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. There is no HIP toolchain on the
# development host, so the subprocess.run calls are monkeypatched the
# same way the CUDA tests monkeypatch nvcc. End-to-end smoke against a
# real `hipcc` is deferred until a ROCm host is available.


def _stage_hip_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/driver.hip under tmp_path."""
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    (driver_dir / "driver.hip").write_text(body)
    return driver_dir


def _stage_hip_baseline_driver(tmp_path, stem, body=_BASELINE_DRIVER_TEMPLATE):
    """Write a baseline driver.hip at baselines/<stem>/ under tmp_path.

    The body content is the same Kokkos-flavored template the splice
    tests use; splice_rewritten_kernel only cares about the sentinel
    lines, not the surrounding C++. What matters here is that the file
    lives at driver.hip (not driver.cpp or driver.cu), so the HIP
    profile finds it.
    """
    d = tmp_path / "baselines" / stem
    d.mkdir(parents=True)
    (d / "driver.hip").write_text(body)
    return d


def test_compile_baseline_driver_hip_invokes_hipcc_with_default_arch(
    monkeypatch, tmp_path
):
    """For language_id='hip', compile_baseline_driver shells out to hipcc with -std=c++17, -O2, --offload-arch=gfx90a by default, and the driver.hip source path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(HIP_ARCH_ENV, raising=False)
    _stage_hip_driver(tmp_path, "vector_add")
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "hip")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/vector_add/driver"]
    cmd = captured["cmd"]
    assert cmd[0] == HIPCC
    assert "-std=c++17" in cmd
    assert "-O2" in cmd
    assert f"--offload-arch={HIP_DEFAULT_ARCH}" in cmd
    assert HIP_DEFAULT_ARCH == "gfx90a"  # invariant the docs promise
    # Driver source is the .hip input, output is baselines/<stem>/driver.
    assert "baselines/vector_add/driver.hip" in cmd
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/vector_add/driver"
    # The HIP profile must NOT inherit any Kokkos- or CUDA-specific flags.
    assert "-fopenmp" not in cmd
    assert "-lkokkoscore" not in cmd
    assert "-lkokkoscontainers" not in cmd
    # CUDA's nvcc flag is `-arch=`, NOT `--offload-arch=`. Make sure the
    # HIP command doesn't accidentally inherit nvcc syntax.
    assert not any(a.startswith("-arch=") for a in cmd)


def test_compile_baseline_driver_hip_honors_arch_env_override(
    monkeypatch, tmp_path
):
    """AGENT_PRECISION_HIP_ARCH=gfx942 replaces the default in the hipcc --offload-arch= flag (gfx942 is MI300; the default is gfx90a for MI200/MI250X)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(HIP_ARCH_ENV, "gfx942")
    _stage_hip_driver(tmp_path, "vector_add")
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "hip")

    assert result["status"] == "ok"
    cmd = captured["cmd"]
    assert "--offload-arch=gfx942" in cmd
    assert f"--offload-arch={HIP_DEFAULT_ARCH}" not in cmd


def test_compile_baseline_driver_hip_errors_when_hipcc_missing(
    monkeypatch, tmp_path
):
    """When shutil.which('hipcc') returns None, compile_baseline_driver returns status='error' WITHOUT invoking subprocess.run and the stderr names hipcc. Mirrors the nvcc-missing test for CUDA."""
    monkeypatch.chdir(tmp_path)
    # No need to stage a source file — preflight runs before the
    # source-exists check, so the error must surface from the missing
    # toolchain alone.
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: None)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when hipcc is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "hip")

    assert result["status"] == "error"
    assert HIPCC in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_hip_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """When the HIP driver source is absent, compile_baseline_driver returns status='error' (no subprocess) and the error names driver.hip (not driver.cpp or driver.cu)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "hip")

    assert result["status"] == "error"
    assert "driver.hip" in result["stderr"]
    assert "driver.cpp" not in result["stderr"]
    assert "driver.cu" not in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_hip_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every HIP code path returns a dict with exactly {status, stdout, stderr, artifacts} — the uniform schema shared with the Kokkos and CUDA paths."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) hipcc missing
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: None)
    assert set(compile_baseline_driver("x", "hip").keys()) == expected_keys

    # 2) success
    monkeypatch.setattr(hip_profile.shutil, "which", lambda name: f"/usr/bin/{name}")
    _stage_hip_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x", "hip").keys()) == expected_keys

    # 3) compile failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x", "hip").keys()) == expected_keys


def test_splice_rewritten_kernel_hip_reads_and_writes_dot_hip(
    monkeypatch, tmp_path
):
    """For language_id='hip', splice_rewritten_kernel reads baselines/<stem>/driver.hip and writes baselines/<stem>/rewritten/driver.hip — NOT driver.cpp or driver.cu at either path."""
    monkeypatch.chdir(tmp_path)
    _stage_hip_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    new_kernel = (
        "__global__ void kernel(float* x, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) x[i] = x[i] * 2.0f;\n"
        "}\n"
    )

    result = splice_rewritten_kernel("k", new_kernel, "hip")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/k/rewritten/driver.hip"]

    out_path = tmp_path / "baselines" / "k" / "rewritten" / "driver.hip"
    assert out_path.is_file()
    # Neither the .cpp nor the .cu filename must appear in either tree.
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp").exists()
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.cu").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.cpp").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.cu").exists()

    text = out_path.read_text()
    out_lines = text.split("\n")
    assert out_lines.count(KERNEL_BEGIN_SENTINEL) == 1
    assert out_lines.count(KERNEL_END_SENTINEL) == 1
    begin = out_lines.index(KERNEL_BEGIN_SENTINEL)
    end = out_lines.index(KERNEL_END_SENTINEL)
    spliced = "\n".join(out_lines[begin + 1 : end])
    assert spliced == new_kernel.rstrip("\n")


# ---------- SYCL profile: compile + splice ----------
#
# These tests exercise the SYCL language profile (workflow.languages.sycl)
# through the same public tool wrappers as the Kokkos / CUDA / HIP tests
# above. Phase C-2 added SYCL as the fourth profile, so the wrappers must
# dispatch on `language_id="sycl"` to the SYCL compile command shape (the
# resolved compiler from AGENT_PRECISION_SYCL_CXX, defaulting to `icpx`,
# with `-fsycl`), the SYCL-compiler-on-PATH preflight, and the
# `driver.cpp` filename (shared with Kokkos — SYCL also uses `.cpp`).
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. There is no SYCL toolchain (icpx,
# clang++ -fsycl, dpcpp) on the development host, so the subprocess.run
# calls are monkeypatched the same way the CUDA / HIP tests monkeypatch
# nvcc / hipcc. End-to-end smoke against a real SYCL compiler is
# deferred until an Intel-GPU or AdaptiveCpp-equipped host is available.


def _stage_sycl_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/driver.cpp under tmp_path.

    SYCL shares the `.cpp` driver_filename with Kokkos. The same staging
    file is acceptable for both profiles; what changes per profile is the
    compile command, not the source filename.
    """
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    (driver_dir / "driver.cpp").write_text(body)
    return driver_dir


def _stage_sycl_baseline_driver(tmp_path, stem, body=_BASELINE_DRIVER_TEMPLATE):
    """Write a baseline driver.cpp at baselines/<stem>/ under tmp_path.

    The body content is the Kokkos-flavored template the splice tests
    use; splice_rewritten_kernel only cares about the sentinel lines,
    not the surrounding C++. Identical to the Kokkos baseline-driver
    staging — only the language_id passed to the tool wrapper changes.
    """
    d = tmp_path / "baselines" / stem
    d.mkdir(parents=True)
    (d / "driver.cpp").write_text(body)
    return d


def test_compile_baseline_driver_sycl_invokes_default_compiler_with_fsycl(
    monkeypatch, tmp_path
):
    """For language_id='sycl', compile_baseline_driver shells out to the resolved compiler (default `icpx`) with -std=c++17, -O2, -fsycl, and the driver.cpp source path. No `-fsycl-targets=...` — SYCL device selection happens at runtime in v0."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(SYCL_CXX_ENV, raising=False)
    _stage_sycl_driver(tmp_path, "vector_add")
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "sycl")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/vector_add/driver"]
    cmd = captured["cmd"]
    assert cmd[0] == SYCL_DEFAULT_CXX == "icpx"
    assert "-std=c++17" in cmd
    assert "-O2" in cmd
    assert SYCL_FLAG in cmd
    assert SYCL_FLAG == "-fsycl"  # invariant the docs promise
    # Driver source is the .cpp input, output is baselines/<stem>/driver.
    assert "baselines/vector_add/driver.cpp" in cmd
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/vector_add/driver"
    # The SYCL profile must NOT inherit Kokkos- or CUDA- or HIP-specific
    # flags. -fopenmp / -lkokkoscore would tag this as a Kokkos compile;
    # -arch= is nvcc; --offload-arch= is hipcc; -fsycl-targets= is a
    # SYCL ahead-of-time flag deferred to a future phase.
    assert "-fopenmp" not in cmd
    assert "-lkokkoscore" not in cmd
    assert "-lkokkoscontainers" not in cmd
    assert not any(a.startswith("-arch=") for a in cmd)
    assert not any(a.startswith("--offload-arch=") for a in cmd)
    assert not any(a.startswith("-fsycl-targets=") for a in cmd)


def test_compile_baseline_driver_sycl_honors_cxx_env_override(
    monkeypatch, tmp_path
):
    """AGENT_PRECISION_SYCL_CXX=clang++ replaces the default compiler driver in the SYCL compile command. The env value is taken verbatim — there is no allowlist validation in `_resolve_compiler`; a typo (e.g. `gcc`) is caught downstream by `clang++ -fsycl`-style "unrecognized option" diagnostics rather than by a Python-side check."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(SYCL_CXX_ENV, "clang++")
    _stage_sycl_driver(tmp_path, "vector_add")
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "sycl")

    assert result["status"] == "ok"
    cmd = captured["cmd"]
    assert cmd[0] == "clang++"
    assert SYCL_DEFAULT_CXX not in cmd  # the default `icpx` is no longer the driver
    assert SYCL_FLAG in cmd  # -fsycl still present regardless of compiler choice


def test_compile_baseline_driver_sycl_errors_when_compiler_missing(
    monkeypatch, tmp_path
):
    """When shutil.which(<resolved compiler>) returns None, compile_baseline_driver returns status='error' WITHOUT invoking subprocess.run and the stderr names the missing compiler. Mirrors the nvcc-missing / hipcc-missing tests for CUDA / HIP. The error message must also name AGENT_PRECISION_SYCL_CXX so an operator on a host with a non-default SYCL compiler knows the escape hatch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(SYCL_CXX_ENV, raising=False)
    # No need to stage a source file — preflight runs before the
    # source-exists check, so the error must surface from the missing
    # toolchain alone.
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: None)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when SYCL compiler is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "sycl")

    assert result["status"] == "error"
    assert SYCL_DEFAULT_CXX in result["stderr"]
    assert SYCL_CXX_ENV in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_sycl_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """When the SYCL driver source is absent, compile_baseline_driver returns status='error' (no subprocess) and the error names driver.cpp (not driver.hip or driver.cu)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "sycl")

    assert result["status"] == "error"
    assert "driver.cpp" in result["stderr"]
    assert "driver.hip" not in result["stderr"]
    assert "driver.cu" not in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_sycl_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every SYCL code path returns a dict with exactly {status, stdout, stderr, artifacts} — the uniform schema shared with the Kokkos / CUDA / HIP paths."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) SYCL compiler missing
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: None)
    assert set(compile_baseline_driver("x", "sycl").keys()) == expected_keys

    # 2) success
    monkeypatch.setattr(sycl_profile.shutil, "which", lambda name: f"/usr/bin/{name}")
    _stage_sycl_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x", "sycl").keys()) == expected_keys

    # 3) compile failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x", "sycl").keys()) == expected_keys


def test_splice_rewritten_kernel_sycl_reads_and_writes_dot_cpp(
    monkeypatch, tmp_path
):
    """For language_id='sycl', splice_rewritten_kernel reads baselines/<stem>/driver.cpp and writes baselines/<stem>/rewritten/driver.cpp — same filename as the Kokkos profile (both use .cpp), NOT driver.cu or driver.hip. The splice logic itself is shared across profiles; this test just confirms the SYCL dispatch path resolves to the right driver_filename."""
    monkeypatch.chdir(tmp_path)
    _stage_sycl_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    new_kernel = (
        "using aType = float;\n"
        "using cType = float;\n"
        "// SYCL kernel body would go here, lambda inside q.submit(...)\n"
    )

    result = splice_rewritten_kernel("k", new_kernel, "sycl")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/k/rewritten/driver.cpp"]

    out_path = tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp"
    assert out_path.is_file()
    # The HIP / CUDA filenames must not appear in either tree.
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.hip").exists()
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.cu").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.hip").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.cu").exists()

    text = out_path.read_text()
    out_lines = text.split("\n")
    assert out_lines.count(KERNEL_BEGIN_SENTINEL) == 1
    assert out_lines.count(KERNEL_END_SENTINEL) == 1
    begin = out_lines.index(KERNEL_BEGIN_SENTINEL)
    end = out_lines.index(KERNEL_END_SENTINEL)
    spliced = "\n".join(out_lines[begin + 1 : end])
    assert spliced == new_kernel.rstrip("\n")


# ---------- OMP-offload profile: compile + splice ----------
#
# These tests exercise the OMP-offload language profile
# (workflow.languages.omp_offload) through the same public tool
# wrappers as the Kokkos / CUDA / HIP / SYCL tests above. Phase C-3
# added OMP-offload as the fifth profile, so the wrappers must
# dispatch on `language_id="omp_offload"` to the OMP-offload compile
# command shape (the resolved compiler from AGENT_PRECISION_OMP_CXX,
# defaulting to `clang++`, with -fopenmp and
# -fopenmp-targets=<triple> where the triple resolves from
# AGENT_PRECISION_OMP_TARGET, defaulting to `nvptx64-nvidia-cuda`),
# the compiler-on-PATH preflight, and the `driver.cpp` filename
# (shared with Kokkos and SYCL).
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. No host with an OMP-offload
# toolchain (clang++ with offload runtime, icpx, nvc++) was
# available at implementation time, so the subprocess.run calls are
# monkeypatched the same way the CUDA / HIP / SYCL tests
# monkeypatch nvcc / hipcc / icpx. End-to-end smoke is deferred
# until such a host is available.


def _stage_omp_offload_driver(tmp_path, stem, body="int main(){return 0;}\n"):
    """Write a placeholder baselines/<stem>/driver.cpp under tmp_path.

    OMP-offload shares the `.cpp` driver_filename with Kokkos and
    SYCL. The same staging file is acceptable for all three
    profiles; what changes per profile is the compile command, not
    the source filename.
    """
    driver_dir = tmp_path / "baselines" / stem
    driver_dir.mkdir(parents=True)
    (driver_dir / "driver.cpp").write_text(body)
    return driver_dir


def _stage_omp_offload_baseline_driver(tmp_path, stem, body=_BASELINE_DRIVER_TEMPLATE):
    """Write a baseline driver.cpp at baselines/<stem>/ under tmp_path.

    The body content is the Kokkos-flavored template the splice tests
    use; splice_rewritten_kernel only cares about the sentinel lines,
    not the surrounding C++. Identical to the Kokkos / SYCL baseline-
    driver staging — only the language_id passed to the tool wrapper
    changes.
    """
    d = tmp_path / "baselines" / stem
    d.mkdir(parents=True)
    (d / "driver.cpp").write_text(body)
    return d


def test_compile_baseline_driver_omp_offload_invokes_default_compiler_with_flags(
    monkeypatch, tmp_path
):
    """For language_id='omp_offload', compile_baseline_driver shells out to the resolved compiler (default `clang++`) with -std=c++17, -O2, -fopenmp, and -fopenmp-targets=<triple> (default `nvptx64-nvidia-cuda`), plus the driver.cpp source path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OMP_CXX_ENV, raising=False)
    monkeypatch.delenv(OMP_TARGET_ENV, raising=False)
    _stage_omp_offload_driver(tmp_path, "vector_add")
    monkeypatch.setattr(
        omp_offload_profile.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "omp_offload")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/vector_add/driver"]
    cmd = captured["cmd"]
    assert cmd[0] == OMP_DEFAULT_CXX == "clang++"
    assert "-std=c++17" in cmd
    assert "-O2" in cmd
    assert OMP_FLAG in cmd
    assert OMP_FLAG == "-fopenmp"  # invariant the docs promise
    # Target triple flag — the most distinctive OMP-offload-specific flag.
    assert f"-fopenmp-targets={OMP_DEFAULT_TARGET}" in cmd
    assert OMP_DEFAULT_TARGET == "nvptx64-nvidia-cuda"  # invariant the docs promise
    # Driver source is the .cpp input, output is baselines/<stem>/driver.
    assert "baselines/vector_add/driver.cpp" in cmd
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "baselines/vector_add/driver"
    # The OMP-offload profile must NOT inherit other profiles' flags.
    # -lkokkoscore is Kokkos-link; -arch= is nvcc; --offload-arch= is
    # hipcc; -fsycl is SYCL.
    assert "-lkokkoscore" not in cmd
    assert "-lkokkoscontainers" not in cmd
    assert not any(a.startswith("-arch=") for a in cmd)
    assert not any(a.startswith("--offload-arch=") for a in cmd)
    assert "-fsycl" not in cmd


def test_compile_baseline_driver_omp_offload_honors_cxx_env_override(
    monkeypatch, tmp_path
):
    """AGENT_PRECISION_OMP_CXX=icpx replaces the default compiler driver in the OMP-offload compile command. The env value is taken verbatim — there is no allowlist validation in `_resolve_compiler`; a typo (e.g. `gcc`) is caught downstream by the compiler's own "unrecognized option" / "no offload target" diagnostics rather than by a Python-side check."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(OMP_CXX_ENV, "icpx")
    monkeypatch.delenv(OMP_TARGET_ENV, raising=False)
    _stage_omp_offload_driver(tmp_path, "vector_add")
    monkeypatch.setattr(
        omp_offload_profile.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "omp_offload")

    assert result["status"] == "ok"
    cmd = captured["cmd"]
    assert cmd[0] == "icpx"
    assert OMP_DEFAULT_CXX not in cmd  # the default `clang++` is no longer the driver
    assert OMP_FLAG in cmd  # -fopenmp still present regardless of compiler choice
    assert f"-fopenmp-targets={OMP_DEFAULT_TARGET}" in cmd  # default target preserved


def test_compile_baseline_driver_omp_offload_honors_target_env_override(
    monkeypatch, tmp_path
):
    """AGENT_PRECISION_OMP_TARGET=amdgcn-amd-amdhsa replaces the default target triple in `-fopenmp-targets=<triple>`. The env value is taken verbatim — no Python-side allowlist of valid triples, because the universe of valid triples is compiler-version-dependent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OMP_CXX_ENV, raising=False)
    monkeypatch.setenv(OMP_TARGET_ENV, "amdgcn-amd-amdhsa")
    _stage_omp_offload_driver(tmp_path, "vector_add")
    monkeypatch.setattr(
        omp_offload_profile.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = compile_baseline_driver("vector_add", "omp_offload")

    assert result["status"] == "ok"
    cmd = captured["cmd"]
    assert "-fopenmp-targets=amdgcn-amd-amdhsa" in cmd
    # The default triple must NOT also appear (no double-target flag).
    assert f"-fopenmp-targets={OMP_DEFAULT_TARGET}" not in cmd


def test_compile_baseline_driver_omp_offload_errors_when_compiler_missing(
    monkeypatch, tmp_path
):
    """When shutil.which(<resolved compiler>) returns None, compile_baseline_driver returns status='error' WITHOUT invoking subprocess.run and the stderr names the missing compiler. Mirrors the nvcc-missing / hipcc-missing / icpx-missing tests for CUDA / HIP / SYCL. The error message must also name AGENT_PRECISION_OMP_CXX so an operator on a host with a non-default OMP compiler knows the escape hatch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OMP_CXX_ENV, raising=False)
    # No need to stage a source file — preflight runs before the
    # source-exists check.
    monkeypatch.setattr(omp_offload_profile.shutil, "which", lambda name: None)

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when OMP-offload compiler is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "omp_offload")

    assert result["status"] == "error"
    assert OMP_DEFAULT_CXX in result["stderr"]
    assert OMP_CXX_ENV in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_omp_offload_errors_when_driver_source_missing(
    monkeypatch, tmp_path
):
    """When the OMP-offload driver source is absent, compile_baseline_driver returns status='error' (no subprocess) and the error names driver.cpp (not driver.hip or driver.cu)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        omp_offload_profile.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when driver source missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = compile_baseline_driver("vector_add", "omp_offload")

    assert result["status"] == "error"
    assert "driver.cpp" in result["stderr"]
    assert "driver.hip" not in result["stderr"]
    assert "driver.cu" not in result["stderr"]
    assert result["artifacts"] == []


def test_compile_baseline_driver_omp_offload_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every OMP-offload code path returns a dict with exactly {status, stdout, stderr, artifacts} — the uniform schema shared with the Kokkos / CUDA / HIP / SYCL paths."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)

    # 1) OMP compiler missing
    monkeypatch.setattr(omp_offload_profile.shutil, "which", lambda name: None)
    assert set(compile_baseline_driver("x", "omp_offload").keys()) == expected_keys

    # 2) success
    monkeypatch.setattr(
        omp_offload_profile.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    _stage_omp_offload_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x", "omp_offload").keys()) == expected_keys

    # 3) compile failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x", "omp_offload").keys()) == expected_keys


def test_splice_rewritten_kernel_omp_offload_reads_and_writes_dot_cpp(
    monkeypatch, tmp_path
):
    """For language_id='omp_offload', splice_rewritten_kernel reads baselines/<stem>/driver.cpp and writes baselines/<stem>/rewritten/driver.cpp — same filename as the Kokkos and SYCL profiles (all use .cpp), NOT driver.cu or driver.hip. The splice logic itself is shared across profiles; this test just confirms the OMP-offload dispatch path resolves to the right driver_filename."""
    monkeypatch.chdir(tmp_path)
    _stage_omp_offload_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    new_kernel = (
        "using aType = float;\n"
        "using cType = float;\n"
        "// OMP-offload kernel body would go here, called from `#pragma omp target`\n"
    )

    result = splice_rewritten_kernel("k", new_kernel, "omp_offload")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/k/rewritten/driver.cpp"]

    out_path = tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp"
    assert out_path.is_file()
    # The HIP / CUDA filenames must not appear in either tree.
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.hip").exists()
    assert not (tmp_path / "baselines" / "k" / "rewritten" / "driver.cu").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.hip").exists()
    assert not (tmp_path / "baselines" / "k" / "driver.cu").exists()

    text = out_path.read_text()
    out_lines = text.split("\n")
    assert out_lines.count(KERNEL_BEGIN_SENTINEL) == 1
    assert out_lines.count(KERNEL_END_SENTINEL) == 1
    begin = out_lines.index(KERNEL_BEGIN_SENTINEL)
    end = out_lines.index(KERNEL_END_SENTINEL)
    spliced = "\n".join(out_lines[begin + 1 : end])
    assert spliced == new_kernel.rstrip("\n")


# ---------- probe_step ----------


def _stage_probe_template(
    tmp_path, stem, precision, seed_line="static constexpr int RNG_SEED = 42;",
    extra_body="// kernel body\nint main(){return 0;}\n",
):
    """Write a fake per-precision probe driver template under tmp_path.

    Mirrors what the v1 baseline_harness writes to
    baselines/<stem>/probe/<precision>/driver.cpp. The seed_line is
    the line probe_step rewrites; tests vary it to exercise the
    exactly-one-match contract.
    """
    template_dir = tmp_path / "baselines" / stem / "probe" / precision
    template_dir.mkdir(parents=True)
    body = f"// header\n{seed_line}\n{extra_body}"
    (template_dir / "driver.cpp").write_text(body)
    return template_dir


def test_probe_step_errors_when_template_missing(monkeypatch, tmp_path):
    """probe_step returns status='error' (no subprocess call) when baselines/<stem>/probe/<precision>/driver.cpp does not exist, and the message blames spawn_baseline_harness so the operator knows which upstream tool was supposed to write it."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when probe template is missing"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = probe_step("nbody_force", "quad", 42, "kokkos")

    assert result["status"] == "error"
    assert "spawn_baseline_harness" in result["stderr"]
    assert "quad" in result["stderr"]
    assert result["artifacts"] == []


def test_probe_step_rejects_bool_as_seed(monkeypatch, tmp_path):
    """probe_step rejects a bool seed (which is technically an int subclass in Python) without invoking subprocess.run, because True/False are almost never what the orchestrator meant to pass."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_probe_template(tmp_path, "k", "quad")

    def fail_run(*a, **kw):
        raise AssertionError("subprocess.run must not be called for bool seed")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = probe_step("k", "quad", True, "kokkos")

    assert result["status"] == "error"
    assert "seed" in result["stderr"].lower()
    assert result["artifacts"] == []


def test_probe_step_rejects_empty_precision(monkeypatch, tmp_path):
    """probe_step rejects an empty / non-string precision without invoking subprocess.run; the precision is used as a directory name and an empty string would collapse onto the template root."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))

    def fail_run(*a, **kw):
        raise AssertionError("subprocess.run must not be called for bad precision")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = probe_step("k", "", 42, "kokkos")

    assert result["status"] == "error"
    assert "precision" in result["stderr"].lower()


def test_probe_step_errors_when_rng_seed_line_missing(monkeypatch, tmp_path):
    """probe_step returns status='error' when the template has no RNG_SEED contract line (`static constexpr int RNG_SEED = <int>;` on its own line), with a message naming the RNG_SEED contract so the operator knows the harness output is malformed."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_probe_template(
        tmp_path, "k", "quad",
        seed_line="// no RNG_SEED line here",
    )

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when RNG_SEED is malformed"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = probe_step("k", "quad", 99, "kokkos")

    assert result["status"] == "error"
    assert "RNG_SEED" in result["stderr"]
    assert result["artifacts"] == []


def test_probe_step_errors_when_rng_seed_line_appears_twice(
    monkeypatch, tmp_path
):
    """probe_step returns status='error' when the template has two RNG_SEED contract lines, because the rewrite would be ambiguous (the contract requires exactly one)."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    template_dir = tmp_path / "baselines" / "k" / "probe" / "quad"
    template_dir.mkdir(parents=True)
    (template_dir / "driver.cpp").write_text(
        "static constexpr int RNG_SEED = 42;\n"
        "static constexpr int RNG_SEED = 7;\n"
        "int main(){return 0;}\n"
    )

    def fail_run(*a, **kw):
        raise AssertionError(
            "subprocess.run must not be called when RNG_SEED is ambiguous"
        )

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = probe_step("k", "quad", 99, "kokkos")

    assert result["status"] == "error"
    assert "RNG_SEED" in result["stderr"]
    assert "2" in result["stderr"]


def test_probe_step_writes_rewritten_seed_to_sibling_dir(
    monkeypatch, tmp_path
):
    """On success, probe_step writes the rewritten source to baselines/<stem>/probe/<precision>_seed<seed>/driver.cpp (a sibling of the template dir), preserves every byte outside the matched RNG_SEED line, leaves the template dir untouched, and returns reference.json as its single artifact."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    template_dir = _stage_probe_template(
        tmp_path, "k", "quad",
        seed_line="static constexpr int RNG_SEED = 42;",
        extra_body=(
            "// preserved verbatim\n"
            "static constexpr int NOT_THE_SEED = 42;  // also literally 42\n"
            "int main(){return 0;}\n"
        ),
    )
    template_path = template_dir / "driver.cpp"
    template_before = template_path.read_text()

    target_dir = tmp_path / "baselines" / "k" / "probe" / "quad_seed99"

    def fake_run(cmd, **kw):
        # Two subprocess calls land here: the compile (g++) and the run
        # (./driver). The compile must produce the driver binary; the
        # run must write reference.json. Both succeed.
        class OkProc:
            returncode = 0
            stdout = ""
            stderr = ""
        argv0 = cmd[0] if isinstance(cmd, list) else cmd
        if isinstance(argv0, str) and argv0.endswith("driver"):
            # ./driver invocation -> write reference.json
            (target_dir / "reference.json").write_text(
                json.dumps({"kernel": "k", "seed": 99, "inputs": {}, "outputs": {}})
            )
        else:
            # compile invocation -> create the binary the run check expects
            binary = target_dir / "driver"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return OkProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = probe_step("k", "quad", 99, "kokkos")

    assert result["status"] == "ok"
    assert result["artifacts"] == [
        "baselines/k/probe/quad_seed99/reference.json"
    ]

    # The sibling dir exists with the rewritten source; only the
    # RNG_SEED line differs from the template.
    rewritten_src = target_dir / "driver.cpp"
    assert rewritten_src.is_file()
    rewritten_text = rewritten_src.read_text()
    assert "RNG_SEED = 99;" in rewritten_text
    assert "RNG_SEED = 42;" not in rewritten_text
    # The other literal "42" (NOT_THE_SEED) must survive verbatim -- the
    # rewrite is on the exact contract line, not a literal-42 string
    # replace.
    assert "NOT_THE_SEED = 42" in rewritten_text

    # Template dir is untouched: re-invocations always start from
    # seed=42 source.
    assert template_path.read_text() == template_before


def test_probe_step_compile_failure_propagates(monkeypatch, tmp_path):
    """If the compile step fails, probe_step returns the compile result verbatim (status='error', g++ stderr in `stderr`) without invoking the run step."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_probe_template(tmp_path, "k", "float")

    run_calls = {"count": 0}

    def fake_run(cmd, **kw):
        run_calls["count"] += 1
        # First call is compile; fail it.
        class BadProc:
            returncode = 1
            stdout = ""
            stderr = "boom: precision_too_imprecise\n"
        return BadProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = probe_step("k", "float", 42, "kokkos")

    assert result["status"] == "error"
    assert "boom" in result["stderr"]
    assert run_calls["count"] == 1  # compile only; run never invoked


def test_probe_step_run_failure_propagates(monkeypatch, tmp_path):
    """If the compile succeeds but the run step fails (driver exits non-zero), probe_step returns the run result (status='error') and does not claim a reference.json artifact."""
    monkeypatch.chdir(tmp_path)
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_probe_template(tmp_path, "k", "float")

    target_dir = tmp_path / "baselines" / "k" / "probe" / "float_seed42"

    def fake_run(cmd, **kw):
        class Proc:
            stdout = ""
            stderr = ""
        argv0 = cmd[0] if isinstance(cmd, list) else cmd
        if isinstance(argv0, str) and argv0.endswith("driver"):
            # Run: exit non-zero, do not write reference.json.
            Proc.returncode = 7
            Proc.stderr = "kernel crashed\n"
        else:
            # Compile: succeed, create binary.
            binary = target_dir / "driver"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            Proc.returncode = 0
        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = probe_step("k", "float", 42, "kokkos")

    assert result["status"] == "error"
    assert result["artifacts"] == []


def test_probe_step_errors_on_unknown_language_id(monkeypatch, tmp_path):
    """probe_step rejects an unknown language_id loudly via _resolve_profile rather than silently treating it as a default; this keeps the call-shape symmetric with the rest of the dynamic-verification chain."""
    monkeypatch.chdir(tmp_path)

    def fail_run(*a, **kw):
        raise AssertionError("subprocess.run must not be called for unknown language")

    monkeypatch.setattr(subprocess, "run", fail_run)

    # _resolve_profile raises (it is not a normal _error result path).
    import pytest

    with pytest.raises(Exception):
        probe_step("k", "quad", 42, "not_a_real_language")


# ---------- probe_compare ----------


def _stage_probe_template_dir(tmp_path, stem, precision):
    """Create just the template directory baselines/<stem>/probe/<precision>/.

    probe_compare lists this directory to discover which precisions to
    walk, but does not actually read its contents (it only reads the
    per-(precision, seed) reference.json files). The template dir
    just needs to exist and be a directory; an empty dir is fine.
    """
    d = tmp_path / "baselines" / stem / "probe" / precision
    d.mkdir(parents=True)
    return d


def _stage_probe_cell_reference(
    tmp_path, stem, precision, seed, payload
):
    """Write baselines/<stem>/probe/<precision>_seed<seed>/reference.json."""
    cell_dir = (
        tmp_path / "baselines" / stem / "probe" / f"{precision}_seed{seed}"
    )
    cell_dir.mkdir(parents=True)
    (cell_dir / "reference.json").write_text(json.dumps(payload))
    return cell_dir


def _make_ref_payload(kernel, seed, outputs):
    """Build a reference.json payload matching the harness contract."""
    return {
        "kernel": kernel,
        "seed": seed,
        "inputs": {},
        "outputs": outputs,
    }


def test_probe_compare_errors_when_quad_seed42_missing(monkeypatch, tmp_path):
    """probe_compare hard-errors when the canonical quad cell (quad_seed42) is missing -- without ground truth there is nothing to compare against. The error message names quad_seed42 and probe_step so the operator knows which upstream call must be re-run."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")
    # Note: NO quad_seed42 cell written.

    result = probe_compare("k", "kokkos")

    assert result["status"] == "error"
    assert "quad_seed42" in result["stderr"]
    assert "probe_step" in result["stderr"]
    assert result["artifacts"] == []


def test_probe_compare_writes_evidence_with_per_output_stats(
    monkeypatch, tmp_path
):
    """Happy path with one non-quad cell: probe_compare writes evidence.json containing per-output stats vs the same-seed quad partner. The stats include n, n_finite, n_nonfinite, max_absrel, mean_absrel, max_abserror; the eps-floor handles exact-zero pairs without blowing up to inf."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    # quad ground truth, seed=42, two outputs.
    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {
            "energy": [1.0, 2.0, 0.0],
            "force":  [10.0, 20.0],
        }),
    )
    # float cell, seed=42. Inject a known ~1e-7 rel error on `energy[0]`
    # and an exact match on `energy[1]` and the exact-zero pair on
    # `energy[2]`. Force matches exactly.
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 42,
        _make_ref_payload("k", 42, {
            "energy": [1.0 + 1e-7, 2.0, 0.0],
            "force":  [10.0, 20.0],
        }),
    )
    # No seed=43 cells -> cross_seed_deltas should be empty for float.

    result = probe_compare("k", "kokkos")

    assert result["status"] == "ok"
    assert result["artifacts"] == ["baselines/k/probe/evidence.json"]

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    assert evidence["kernel_stem"] == "k"
    assert set(evidence["precisions"]) == {"quad", "float"}
    assert evidence["seeds"] == [42, 43]

    # quad_seed42 is ok; quad_seed43, float_seed43 are missing.
    cells = evidence["cells"]
    assert cells["quad_seed42"]["status"] == "ok"
    assert cells["quad_seed43"]["status"] == "missing"
    assert cells["float_seed43"]["status"] == "missing"

    # float_seed42 is ok and has stats vs quad_seed42.
    float_cell = cells["float_seed42"]
    assert float_cell["status"] == "ok"
    stats = float_cell["stats"]
    energy = stats["energy"]
    assert energy["n"] == 3
    assert energy["n_finite"] == 3
    assert energy["n_nonfinite"] == 0
    # max_absrel ~ 1e-7 / max(1.0, 1.0+1e-7, eps) ~ 1e-7
    assert 5e-8 < energy["max_absrel"] < 2e-7
    # Exact-zero pair contributed 0.0 to the sum, exact match contributed
    # 0.0, only energy[0] contributed ~1e-7; mean = sum / 3.
    assert energy["mean_absrel"] < energy["max_absrel"]
    force = stats["force"]
    assert force["max_absrel"] == 0.0
    assert force["max_abserror"] == 0.0

    # No cross-seed deltas because seed=43 cells are missing.
    assert evidence["cross_seed_deltas"] == {}


def test_probe_compare_records_nan_and_inf_as_nonfinite(
    monkeypatch, tmp_path
):
    """probe_compare counts NaN and inf entries on either side as n_nonfinite (excluded from the finite-arithmetic stats) rather than dropping them silently; this is the analyst-friendly signal that something special happened in that cell."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"out": [1.0, 2.0, 3.0, 4.0]}),
    )
    # float side has NaN, +inf, finite, finite (build via Python literals
    # in the staged payload). JSON does not have a NaN/inf literal, so
    # write them with json.dumps(..., allow_nan=True) which Python uses
    # by default; the loader json.load reads them back as float.
    cell_dir = tmp_path / "baselines" / "k" / "probe" / "float_seed42"
    cell_dir.mkdir(parents=True)
    (cell_dir / "reference.json").write_text(
        json.dumps(_make_ref_payload("k", 42, {
            "out": [float("nan"), float("inf"), 3.0, 4.0],
        }))
    )

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    out = evidence["cells"]["float_seed42"]["stats"]["out"]
    assert out["n"] == 4
    assert out["n_nonfinite"] == 2
    assert out["n_finite"] == 2
    assert out["max_absrel"] == 0.0
    assert out["max_abserror"] == 0.0


def test_probe_compare_records_shape_error_on_mismatched_outputs(
    monkeypatch, tmp_path
):
    """When a non-quad cell's reference.json has a different output-array shape than the quad partner (different keys or different lengths), probe_compare records cell.status='shape_error' with a descriptive shape_error message, rather than computing meaningless stats. Other cells in the same run are unaffected."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")
    _stage_probe_template_dir(tmp_path, "k", "double")

    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"out": [1.0, 2.0]}),
    )
    # float cell has a different output array length.
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 42,
        _make_ref_payload("k", 42, {"out": [1.0, 2.0, 3.0]}),
    )
    # double cell is well-formed.
    _stage_probe_cell_reference(
        tmp_path, "k", "double", 42,
        _make_ref_payload("k", 42, {"out": [1.0, 2.0]}),
    )

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    float_cell = evidence["cells"]["float_seed42"]
    assert float_cell["status"] == "shape_error"
    assert "shape_error" in float_cell
    # The double cell with matching shape is fine.
    double_cell = evidence["cells"]["double_seed42"]
    assert double_cell["status"] == "ok"
    assert "stats" in double_cell


def test_probe_compare_records_no_quad_partner_when_quad_seed43_missing(
    monkeypatch, tmp_path
):
    """When seed=42 has full coverage but seed=43 has a non-quad cell with no quad partner at the same seed, probe_compare records cell.status='no_quad_partner' for that cell rather than the misleading 'ok'. The seed=42 stats are unaffected."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"out": [1.0]}),
    )
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 42,
        _make_ref_payload("k", 42, {"out": [1.0]}),
    )
    # float_seed43 present but quad_seed43 missing.
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 43,
        _make_ref_payload("k", 43, {"out": [1.0]}),
    )

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    assert evidence["cells"]["float_seed42"]["status"] == "ok"
    assert evidence["cells"]["float_seed43"]["status"] == "no_quad_partner"
    assert "quad_seed43" in evidence["cells"]["float_seed43"]["error"]


def test_probe_compare_computes_cross_seed_deltas(monkeypatch, tmp_path):
    """When both precision_seed42 and precision_seed43 have ok stats, probe_compare adds cross_seed_deltas[<precision>][<output>] = |max_absrel_seed42 - max_absrel_seed43|. Quad is excluded (it is the ground truth, not a comparison target)."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    # quad at both seeds, two outputs.
    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"a": [1.0], "b": [10.0]}),
    )
    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 43,
        _make_ref_payload("k", 43, {"a": [1.0], "b": [10.0]}),
    )
    # float at seed=42: tiny error on `a`, exact on `b`.
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 42,
        _make_ref_payload("k", 42, {"a": [1.0 + 1e-7], "b": [10.0]}),
    )
    # float at seed=43: larger error on `a`, exact on `b`.
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 43,
        _make_ref_payload("k", 43, {"a": [1.0 + 5e-7], "b": [10.0]}),
    )

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    deltas = evidence["cross_seed_deltas"]
    assert "float" in deltas
    assert "quad" not in deltas  # quad excluded by construction
    assert deltas["float"]["b"] == 0.0  # exact match at both seeds
    # delta_a ~ |1e-7 - 5e-7| ~ 4e-7 (modulo tiny denominator effects)
    assert 1e-7 < deltas["float"]["a"] < 1e-6


def test_probe_compare_classifies_load_error_for_invalid_json(
    monkeypatch, tmp_path
):
    """A non-quad cell whose reference.json is not valid JSON is classified as cell.status='load_error' with a descriptive error message. Other cells continue to be processed normally."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"out": [1.0]}),
    )
    # float cell with garbage in reference.json.
    cell_dir = tmp_path / "baselines" / "k" / "probe" / "float_seed42"
    cell_dir.mkdir(parents=True)
    (cell_dir / "reference.json").write_text("{not valid json")

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"

    evidence = json.loads(
        (tmp_path / "baselines" / "k" / "probe" / "evidence.json").read_text()
    )
    float_cell = evidence["cells"]["float_seed42"]
    assert float_cell["status"] == "load_error"
    assert "error" in float_cell
    assert "JSON" in float_cell["error"] or "json" in float_cell["error"]


def test_probe_compare_summary_reports_worst_cell_and_output(
    monkeypatch, tmp_path
):
    """probe_compare's stdout summary names the worst (cell, output) pair by max_absrel across all ok cells, so the operator and the orchestrator log get a one-line view of which precision/seed/output is most pessimistic vs quad."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_template_dir(tmp_path, "k", "float")

    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"small_err": [1.0], "big_err": [1.0]}),
    )
    _stage_probe_cell_reference(
        tmp_path, "k", "float", 42,
        _make_ref_payload("k", 42, {
            "small_err": [1.0 + 1e-7],
            "big_err":   [1.0 + 1e-3],
        }),
    )

    result = probe_compare("k", "kokkos")
    assert result["status"] == "ok"
    assert "float_seed42/big_err" in result["stdout"]
    assert "max_absrel" in result["stdout"]


def test_probe_compare_result_keys_are_stable(monkeypatch, tmp_path):
    """probe_compare returns the uniform {status, stdout, stderr, artifacts} schema on both the success and the hard-error (no quad ground truth) paths, matching every other tool in workflow.tools."""
    monkeypatch.chdir(tmp_path)
    _stage_probe_template_dir(tmp_path, "k", "quad")
    _stage_probe_cell_reference(
        tmp_path, "k", "quad", 42,
        _make_ref_payload("k", 42, {"out": [1.0]}),
    )
    ok = probe_compare("k", "kokkos")
    assert set(ok.keys()) == {"status", "stdout", "stderr", "artifacts"}

    # Hard-error path: remove the ground-truth file.
    (tmp_path / "baselines" / "k" / "probe" / "quad_seed42" / "reference.json").unlink()
    err = probe_compare("k", "kokkos")
    assert set(err.keys()) == {"status", "stdout", "stderr", "artifacts"}
