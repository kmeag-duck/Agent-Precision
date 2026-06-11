"""Tests for workflow.tools.

Covers compile_baseline_driver — env-var handling, missing source file,
g++ command shape, and the success / error result schema — and
run_baseline_driver — env-var parsing, missing/non-executable binary,
clean run + reference.json validation, non-zero exit, timeout, and
the same uniform result schema. All tests monkeypatch subprocess.run
so no real compiler or driver invocation happens.
"""

import json
import os
import stat
import subprocess

from workflow import tools
from workflow.tools import (
    DEFAULT_RUN_TIMEOUT_SEC,
    KOKKOS_ROOT_ENV,
    RUN_TIMEOUT_ENV,
    compile_baseline_driver,
    run_baseline_driver,
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
