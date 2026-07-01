"""Tests for the CLI entrypoint workflow.run.main."""

import pytest

from workflow import run as run_module


def test_main_no_args_returns_2(monkeypatch, capsys):
    """CLI exits 2 (argparse usage error) when called with no kernel path."""
    monkeypatch.setattr("sys.argv", ["workflow.run"])
    with pytest.raises(SystemExit) as excinfo:
        run_module.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "usage" in err.lower()


def test_main_missing_file_returns_2(monkeypatch, tmp_path, capsys):
    """CLI prints 'File not found' and exits 2 when the kernel path does not exist."""
    missing = tmp_path / "does_not_exist.cpp"
    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(missing), "--sig-figs", "6"]
    )
    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "File not found" in err


def test_main_orchestrator_quit_returns_1(monkeypatch, tmp_path):
    """CLI exits 1 when run_orchestrator returns None (user quit or no-tool stop)."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// fake kernel\n")
    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--sig-figs", "6"]
    )
    monkeypatch.setattr(
        run_module,
        "run_orchestrator",
        lambda p, s, tolerance, **kwargs: None,
    )

    assert run_module.main() == 1


def test_main_requires_tolerance_flag(monkeypatch, tmp_path, capsys):
    """After removing the precision_advisor agent, the CLI REQUIRES one of --sig-figs / --decimal-digits; running with neither is an argparse error (exit 2). This test locks the removal — the previous contract silently accepted no tolerance and let the orchestrator call the advisor."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// fake kernel\n")
    monkeypatch.setattr("sys.argv", ["workflow.run", str(kernel)])
    with pytest.raises(SystemExit) as excinfo:
        run_module.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    # argparse phrases mutually-exclusive-required as
    # "one of the arguments ... is required".
    assert "required" in err.lower()
    assert "--sig-figs" in err
    assert "--decimal-digits" in err


def test_main_sig_figs_flag_normalizes_to_user_cli_tolerance(
    monkeypatch, tmp_path
):
    """--sig-figs N produces tolerance={'kind':'sig_figs','value':N,'source':'user_cli'}."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// k\n")

    captured = {}

    def fake_orchestrator(path, source, tolerance=None, auto=False, **kwargs):
        captured["tolerance"] = tolerance
        return {"rewritten_code": "X", "notes": "Y"}

    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--sig-figs", "8"]
    )
    monkeypatch.setattr(run_module, "run_orchestrator", fake_orchestrator)

    assert run_module.main() == 0
    assert captured["tolerance"] == {
        "kind": "sig_figs",
        "value": 8,
        "source": "user_cli",
    }


def test_main_decimal_digits_flag_normalizes_to_user_cli_tolerance(
    monkeypatch, tmp_path
):
    """--decimal-digits N produces tolerance={'kind':'decimal_digits','value':N,'source':'user_cli'}."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// k\n")

    captured = {}

    def fake_orchestrator(path, source, tolerance=None, auto=False, **kwargs):
        captured["tolerance"] = tolerance
        return {"rewritten_code": "X", "notes": "Y"}

    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--decimal-digits", "4"]
    )
    monkeypatch.setattr(run_module, "run_orchestrator", fake_orchestrator)

    assert run_module.main() == 0
    assert captured["tolerance"] == {
        "kind": "decimal_digits",
        "value": 4,
        "source": "user_cli",
    }


def test_main_mutually_exclusive_tolerance_flags(monkeypatch, tmp_path, capsys):
    """--sig-figs and --decimal-digits are mutually exclusive (argparse rejects with exit 2)."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// k\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "workflow.run",
            str(kernel),
            "--sig-figs",
            "6",
            "--decimal-digits",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        run_module.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "not allowed" in err.lower() or "mutually exclusive" in err.lower()


def test_main_auto_flag_passes_auto_true_to_orchestrator(monkeypatch, tmp_path):
    """--auto causes the CLI to invoke run_orchestrator with auto=True; default invocation passes auto=False. Both invocations pass --sig-figs because the CLI now requires a tolerance flag."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// k\n")

    captured = {}

    def fake_orchestrator(path, source, tolerance, auto=False, **kwargs):
        captured["auto"] = auto
        return {"rewritten_code": "X", "notes": "Y"}

    monkeypatch.setattr(run_module, "run_orchestrator", fake_orchestrator)

    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--sig-figs", "6", "--auto"]
    )
    assert run_module.main() == 0
    assert captured["auto"] is True

    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--sig-figs", "6"]
    )
    assert run_module.main() == 0
    assert captured["auto"] is False


def test_main_rejects_nonpositive_sig_figs(monkeypatch, tmp_path):
    """A non-positive --sig-figs value is rejected (SystemExit with a message)."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// k\n")
    monkeypatch.setattr(
        "sys.argv", ["workflow.run", str(kernel), "--sig-figs", "0"]
    )
    with pytest.raises(SystemExit):
        run_module.main()
