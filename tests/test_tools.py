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
Finally covers compare_outputs — tolerance-kind dispatch (sig_figs and
decimal_digits with the documented strict-< thresholds), NaN-always-
mismatches asymmetry, ±inf rules, shape-error vs tolerance-failure
distinction, mismatch list truncation with a "+ K more suppressed"
footer, comparison.json artifact on both pass and fail paths, and
the uniform result schema. All tests monkeypatch subprocess.run so no
real compiler or driver invocation happens; comparator tests use pure
file I/O against tmp_path.
"""

import json
import os
import stat
import subprocess

from workflow import tools
from workflow.tools import (
    DEFAULT_RUN_TIMEOUT_SEC,
    KERNEL_BEGIN_SENTINEL,
    KERNEL_END_SENTINEL,
    KOKKOS_ROOT_ENV,
    RUN_TIMEOUT_ENV,
    compare_outputs,
    compile_baseline_driver,
    compile_rewritten_driver,
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

    result = compile_baseline_driver("nbody_force")

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

    result = compile_baseline_driver("nbody_force")

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

    result = compile_baseline_driver("nbody_force")

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

    result = compile_baseline_driver("nbody_force")

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

    result = compile_baseline_driver("nbody_force")

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

    result = compile_baseline_driver("nbody_force")

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
    assert set(compile_baseline_driver("x").keys()) == expected_keys

    # 2) success
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    _stage_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_baseline_driver("x").keys()) == expected_keys

    # 3) failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_baseline_driver("x").keys()) == expected_keys


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

    result = run_baseline_driver("nbody_force")

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

    run_baseline_driver("nbody_force")

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
        result = run_baseline_driver("nbody_force")
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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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

    result = run_baseline_driver("nbody_force")

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
    assert set(run_baseline_driver("x").keys()) == expected_keys
    monkeypatch.delenv(RUN_TIMEOUT_ENV, raising=False)

    # 2) missing binary
    assert set(run_baseline_driver("x").keys()) == expected_keys

    # 3) success
    _stage_driver_binary(tmp_path, "x", reference_payload={"ok": 1})

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(run_baseline_driver("x").keys()) == expected_keys

    # 4) non-zero exit
    class FailProc:
        returncode = 9
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(run_baseline_driver("x").keys()) == expected_keys

    # 5) timeout
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert set(run_baseline_driver("x").keys()) == expected_keys


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

    result = splice_rewritten_kernel("k", new_kernel)

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

    splice_rewritten_kernel("k", "void kernel() {}\n")

    after = (driver_dir / "driver.cpp").read_bytes()
    assert before == after


def test_splice_rewritten_kernel_preserves_bytes_outside_sentinels(
    monkeypatch, tmp_path
):
    """Lines outside the sentinel region in the spliced driver must be byte-identical to the baseline (only the kernel region changes)."""
    monkeypatch.chdir(tmp_path)
    _stage_baseline_driver(tmp_path, "k")
    _ban_subprocess(monkeypatch)

    splice_rewritten_kernel("k", "void kernel() { /* new */ }\n")

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

    result = splice_rewritten_kernel("k", _ORIGINAL_KERNEL_BODY)

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

    r1 = splice_rewritten_kernel("k", "void kernel() { /* v1 */ }\n")
    assert r1["status"] == "ok"
    first = (tmp_path / "baselines" / "k" / "rewritten" / "driver.cpp").read_text()
    assert "v1" in first

    r2 = splice_rewritten_kernel("k", "void kernel() { /* v2 */ }\n")
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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

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

    result = splice_rewritten_kernel("k", "void kernel() {}\n")

    assert result["status"] == "error"
    assert KERNEL_END_SENTINEL in result["stderr"]
    assert result["artifacts"] == []


def test_splice_rewritten_kernel_result_keys_are_stable(monkeypatch, tmp_path):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — the same shape compile_baseline_driver / run_baseline_driver / planned remote-batch verifier tools share."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}
    monkeypatch.chdir(tmp_path)
    _ban_subprocess(monkeypatch)

    # 1) empty rewritten source
    assert set(splice_rewritten_kernel("k", "").keys()) == expected_keys

    # 2) missing baseline
    assert (
        set(splice_rewritten_kernel("k", "void kernel() {}\n").keys())
        == expected_keys
    )

    # 3) success
    _stage_baseline_driver(tmp_path, "k")
    assert (
        set(splice_rewritten_kernel("k", "void kernel() {}\n").keys())
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
        set(splice_rewritten_kernel(bad_stem, "void kernel() {}\n").keys())
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

    result = compile_rewritten_driver("nbody_force")

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

    result = compile_rewritten_driver("nbody_force")

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

    result = compile_rewritten_driver("nbody_force")

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

    result = compile_rewritten_driver("nbody_force")

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

    compile_baseline_driver("k")
    compile_rewritten_driver("k")

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

    result = compile_rewritten_driver("nbody_force")

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

    result = compile_rewritten_driver("nbody_force")

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

    compile_rewritten_driver("k")

    assert baseline_bin.read_bytes() == b"ORIGINAL BASELINE BINARY"


def test_compile_rewritten_driver_result_keys_are_stable(
    monkeypatch, tmp_path
):
    """Every code path returns a dict with exactly the keys {status, stdout, stderr, artifacts} — same shape as compile_baseline_driver / run_baseline_driver / splice_rewritten_kernel / planned remote-batch verifier tools."""
    expected_keys = {"status", "stdout", "stderr", "artifacts"}

    # 1) env unset
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KOKKOS_ROOT_ENV, raising=False)
    assert set(compile_rewritten_driver("x").keys()) == expected_keys

    # 2) env set but no rewritten source
    root = _make_fake_kokkos_root(tmp_path)
    monkeypatch.setenv(KOKKOS_ROOT_ENV, str(root))
    assert set(compile_rewritten_driver("x").keys()) == expected_keys

    # 3) success
    _stage_rewritten_driver(tmp_path, "x")

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(compile_rewritten_driver("x").keys()) == expected_keys

    # 4) failure
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(compile_rewritten_driver("x").keys()) == expected_keys


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
        result = run_rewritten_driver("nbody_force")
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

    result = run_rewritten_driver("nbody_force")

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

    result = run_rewritten_driver("nbody_force")

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

    run_rewritten_driver("nbody_force")

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

    result = run_rewritten_driver("nbody_force")

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

    run_rewritten_driver("nbody_force")

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

    result = run_rewritten_driver("nbody_force")

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

    result = run_rewritten_driver("nbody_force")

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
    assert set(run_rewritten_driver("x").keys()) == expected_keys
    monkeypatch.delenv(RUN_TIMEOUT_ENV, raising=False)

    # 2) missing binary
    assert set(run_rewritten_driver("x").keys()) == expected_keys

    # 3) success
    _stage_rewritten_driver_binary(tmp_path, "x", reference_payload={"ok": 1})

    class OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: OkProc())
    assert set(run_rewritten_driver("x").keys()) == expected_keys

    # 4) non-zero exit
    class FailProc:
        returncode = 9
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FailProc())
    assert set(run_rewritten_driver("x").keys()) == expected_keys

    # 5) timeout
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["./driver"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert set(run_rewritten_driver("x").keys()) == expected_keys


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

    result = compare_outputs("nbody_force", _tolerance("sig_figs", 3))

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

    result = compare_outputs("x", _tolerance("sig_figs", 3))

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

    result = compare_outputs("x", _tolerance("sig_figs", 3))

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
    r1 = compare_outputs("a", _tolerance("sig_figs", 3))
    assert r1["status"] == "error"
    assert "baseline" in r1["stderr"].lower()

    # Bad rewritten.
    _write_reference_pair(
        tmp_path,
        "b",
        baseline_payload=_well_shaped({"out": [1.0]}),
        rewritten_payload="{also not json",
    )
    r2 = compare_outputs("b", _tolerance("sig_figs", 3))
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
    bad1 = compare_outputs("x", "{not json")
    assert bad1["status"] == "error"
    assert "tolerance_json" in bad1["stderr"]
    assert bad1["artifacts"] == []

    # Wrong kind.
    bad2 = compare_outputs(
        "x", json.dumps({"kind": "ulps", "value": 3, "source": "user_cli"})
    )
    assert bad2["status"] == "error"
    assert "kind" in bad2["stderr"]

    # Bad value (non-positive or non-int).
    bad3 = compare_outputs(
        "x",
        json.dumps({"kind": "sig_figs", "value": 0, "source": "user_cli"}),
    )
    assert bad3["status"] == "error"
    assert "value" in bad3["stderr"]

    bad4 = compare_outputs(
        "x",
        json.dumps(
            {"kind": "sig_figs", "value": "three", "source": "user_cli"}
        ),
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
    r_topk = compare_outputs("topk", _tolerance("sig_figs", 3))
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
    r_names = compare_outputs("names", _tolerance("sig_figs", 3))
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
    r_lens = compare_outputs("lens", _tolerance("sig_figs", 3))
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
    result = compare_outputs("inside", _tolerance("sig_figs", 3))
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
    result = compare_outputs("outside", _tolerance("sig_figs", 3))
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
    result = compare_outputs("dd", _tolerance("decimal_digits", 4))
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

    result = compare_outputs("nans", _tolerance("sig_figs", 3))
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

    result = compare_outputs("infs", _tolerance("sig_figs", 3))
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
    result = compare_outputs("many", _tolerance("decimal_digits", 6))
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
    ok = compare_outputs("okpath", _tolerance("sig_figs", 3))
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
    bad = compare_outputs("failpath", _tolerance("sig_figs", 3))
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
    r1 = compare_outputs("nope", "not json")
    assert set(r1.keys()) == expected_keys

    # 2) both files missing
    r2 = compare_outputs("nope", _tolerance("sig_figs", 3))
    assert set(r2.keys()) == expected_keys

    # 3) shape mismatch
    _write_reference_pair(
        tmp_path,
        "shp",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"b": [1.0]}),
    )
    r3 = compare_outputs("shp", _tolerance("sig_figs", 3))
    assert set(r3.keys()) == expected_keys

    # 4) ok
    _write_reference_pair(
        tmp_path,
        "go",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [1.0]}),
    )
    r4 = compare_outputs("go", _tolerance("sig_figs", 3))
    assert set(r4.keys()) == expected_keys

    # 5) tolerance fail
    _write_reference_pair(
        tmp_path,
        "no",
        baseline_payload=_well_shaped({"a": [1.0]}),
        rewritten_payload=_well_shaped({"a": [10.0]}),
    )
    r5 = compare_outputs("no", _tolerance("sig_figs", 3))
    assert set(r5.keys()) == expected_keys
