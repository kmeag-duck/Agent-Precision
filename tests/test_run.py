"""Tests for the CLI entrypoint workflow.run.main."""

from workflow import run as run_module


def test_main_wrong_arg_count_returns_2(monkeypatch, capsys):
    """CLI prints usage and exits 2 when called with no kernel path."""
    monkeypatch.setattr("sys.argv", ["workflow.run"])
    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "Usage" in err


def test_main_missing_file_returns_2(monkeypatch, tmp_path, capsys):
    """CLI prints 'File not found' and exits 2 when the kernel path does not exist."""
    missing = tmp_path / "does_not_exist.cpp"
    monkeypatch.setattr("sys.argv", ["workflow.run", str(missing)])
    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "File not found" in err


def test_main_orchestrator_quit_returns_1(monkeypatch, tmp_path):
    """CLI exits 1 when run_orchestrator returns None (user quit or no-tool stop)."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// fake kernel\n")
    monkeypatch.setattr("sys.argv", ["workflow.run", str(kernel)])
    monkeypatch.setattr(run_module, "run_orchestrator", lambda p, s: None)

    assert run_module.main() == 1


def test_main_happy_path_prints_kernel_and_notes(monkeypatch, tmp_path, capsys):
    """CLI reads the kernel, calls run_orchestrator with (path, source), and prints rewritten_code + notes."""
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// fake kernel\n")

    captured = {}

    def fake_orchestrator(path, source):
        captured["path"] = path
        captured["source"] = source
        return {"rewritten_code": "REWRITTEN", "notes": "NOTES"}

    monkeypatch.setattr("sys.argv", ["workflow.run", str(kernel)])
    monkeypatch.setattr(run_module, "run_orchestrator", fake_orchestrator)

    assert run_module.main() == 0
    out = capsys.readouterr().out
    assert "REWRITTEN" in out
    assert "NOTES" in out
    assert captured["path"] == str(kernel)
    assert captured["source"] == "// fake kernel\n"
