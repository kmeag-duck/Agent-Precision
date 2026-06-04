"""Tests for the CLI entrypoint workflow.run.main."""

from workflow import run as run_module


def test_main_wrong_arg_count_returns_2(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["workflow.run"])
    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "Usage" in err


def test_main_missing_file_returns_2(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "does_not_exist.cpp"
    monkeypatch.setattr("sys.argv", ["workflow.run", str(missing)])
    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "File not found" in err


def test_main_orchestrator_quit_returns_1(monkeypatch, tmp_path):
    kernel = tmp_path / "k.cpp"
    kernel.write_text("// fake kernel\n")
    monkeypatch.setattr("sys.argv", ["workflow.run", str(kernel)])
    monkeypatch.setattr(run_module, "run_orchestrator", lambda p, s: None)

    assert run_module.main() == 1


def test_main_happy_path_prints_kernel_and_notes(monkeypatch, tmp_path, capsys):
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
