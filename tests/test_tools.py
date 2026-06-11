"""Tests for workflow.tools.

Covers compile_baseline_driver — env-var handling, missing source file,
g++ command shape, and the success / error result schema. All tests
monkeypatch subprocess.run so no real compiler invocation happens.
"""

import subprocess

from workflow import tools
from workflow.tools import (
    KOKKOS_ROOT_ENV,
    compile_baseline_driver,
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
