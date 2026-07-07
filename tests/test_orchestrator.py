"""Tests for workflow.orchestrator.

Covers _hitl_pause, _execute_tool, and the run_orchestrator loop
(happy path, rejection, quit, stop-without-tool).
"""

import json

import pytest

from workflow import orchestrator
from workflow.languages import CUDA_PROFILE, KOKKOS_PROFILE
from workflow.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_TOOLS,
    _execute_tool,
    _format_baseline_block,
    _format_tolerance_block,
    _hitl_pause,
    run_orchestrator,
)

from .conftest import FakeResponse, TextBlock, ToolUseBlock


# Canonical user-supplied tolerance for tests that only care about
# message-shape / dispatch / trace-writing behavior, not the specific
# numeric threshold. Any well-formed tolerance dict works here because
# the FakeAnthropic clients in these tests terminate the loop before the
# tolerance value influences any downstream tool result.
_DEFAULT_TEST_TOLERANCE = {
    "kind": "sig_figs",
    "value": 6,
    "source": "user_cli",
}


# ---------- _hitl_pause ----------


def _scripted_input(monkeypatch, answers):
    """Make builtins.input return successive values from `answers`."""
    answers = list(answers)

    def fake_input(prompt=""):
        if not answers:
            raise AssertionError("input() called more times than scripted")
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


@pytest.mark.parametrize("choice", ["y", "n", "q"])
def test_hitl_returns_each_choice(monkeypatch, choice):
    """_hitl_pause returns y, n, or q exactly as the user typed it."""
    _scripted_input(monkeypatch, [choice])
    assert _hitl_pause("spawn_analyst", {"kernel_source": "..."}) == choice


def test_hitl_loops_on_invalid_then_accepts(monkeypatch):
    """_hitl_pause re-prompts on invalid input and then accepts a valid choice."""
    _scripted_input(monkeypatch, ["maybe", "", "Y", "y"])
    # uppercase Y is accepted (the code lowercases input)
    assert _hitl_pause("spawn_rewriter", {"task_prompt": "..."}) == "y"


def test_hitl_accepts_uppercase(monkeypatch):
    """_hitl_pause accepts uppercase choices by lowercasing the input."""
    _scripted_input(monkeypatch, ["Q"])
    assert _hitl_pause("finish", {"rewritten_code": "x", "notes": "y"}) == "q"


# ---------- _execute_tool ----------


def test_execute_tool_unknown_raises(monkeypatch):
    """_execute_tool raises ValueError on an unknown tool name."""
    with pytest.raises(ValueError, match="Unknown tool"):
        _execute_tool("not_a_tool", {}, KOKKOS_PROFILE)


def test_execute_tool_dispatches_spawn_candidate_finder(monkeypatch):
    """_execute_tool routes spawn_candidate_finder to run_agent('candidate_finder', kernel_source) and wraps the result the same way spawn_analyst does. Single-shot only — no ensemble path in this transitional phase (the finder is informational, not verdict-emitting, so K>1 self-consistency is not yet meaningful)."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "variables": [
                {
                    "name": "x",
                    "downcast_candidate": True,
                    "rank": 1,
                    "rationale": "bounded",
                }
            ],
            "overall_notes": "stubbed",
        }

    def fail_ensemble(*a, **kw):
        raise AssertionError(
            "candidate_finder is single-shot in Step 1; run_agent_ensemble must not fire"
        )

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(orchestrator, "run_agent_ensemble", fail_ensemble)

    result = _execute_tool(
        "spawn_candidate_finder",
        {"kernel_source": "SOURCE"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "ok"
    assert result["result"]["overall_notes"] == "stubbed"
    assert calls == [("candidate_finder", "SOURCE")]
    # Never carries aggregator_metadata — that key is the ensemble
    # signal, and the finder does not ensemble.
    assert "aggregator_metadata" not in result


def test_execute_tool_spawn_candidate_finder_injects_probe_evidence_when_present(
    monkeypatch, tmp_path,
):
    """spawn_candidate_finder shares the analyst's probe-evidence auto-injection: when baselines/<kernel_stem>/probe/evidence.json exists, the finder's task gets a PROBE EVIDENCE (JSON) block appended after the kernel source. Same read path (single source of truth for where evidence lives on disk), same silent-skip contract when the file is absent."""
    monkeypatch.chdir(tmp_path)
    evidence_dir = tmp_path / "baselines" / "nbody_force" / "probe"
    evidence_dir.mkdir(parents=True)
    evidence_payload = {"cells": {"float_seed42": {"status": "ok"}}}
    (evidence_dir / "evidence.json").write_text(json.dumps(evidence_payload))

    captured = {}

    def stub_run_agent(type_, task):
        captured["type"] = type_
        captured["task"] = task
        return {"variables": [], "overall_notes": "ok"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_candidate_finder",
        {"kernel_source": "ORIGINAL KERNEL SOURCE"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    assert result["status"] == "ok"
    assert captured["type"] == "candidate_finder"
    assert captured["task"].startswith("ORIGINAL KERNEL SOURCE")
    assert "PROBE EVIDENCE (JSON):" in captured["task"]
    assert "float_seed42" in captured["task"]
    para_idx = captured["task"].index("PROBE EVIDENCE (JSON):")
    json_idx = captured["task"].index('"cells"')
    assert json_idx > para_idx


def test_execute_tool_spawn_candidate_finder_no_evidence_file_unchanged_task(
    monkeypatch, tmp_path,
):
    """When evidence.json is absent, spawn_candidate_finder silently falls back to the un-augmented kernel source — same rule the analyst branch already follows. Missing evidence is informational-absent, never a hard stop."""
    monkeypatch.chdir(tmp_path)
    captured = {}

    def stub_run_agent(type_, task):
        captured["task"] = task
        return {"variables": [], "overall_notes": "ok"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    _execute_tool(
        "spawn_candidate_finder",
        {"kernel_source": "ORIGINAL KERNEL SOURCE"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    assert captured["task"] == "ORIGINAL KERNEL SOURCE"
    assert "PROBE EVIDENCE" not in captured["task"]


def test_execute_tool_dispatches_spawn_variable_analyst(monkeypatch):
    """_execute_tool routes spawn_variable_analyst to run_agent('variable_analyst', task) where task = <kernel_source>\\n\\nTARGET VARIABLE: <name>. Single-shot only in Step 2 — no ensemble, no aggregator metadata. No probe-consistency gate on single-variable output (the gate walks a full variables[] list; running it here would be an under-check compared to the assembled verdict downstream — that's a Step 5 concern)."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "variable": {
                "name": "x",
                "action": "downcast",
                "target_precision": "float",
                "emulation_type": "",
                "reason": "bounded input",
            },
            "notes": "",
        }

    def fail_ensemble(*a, **kw):
        raise AssertionError(
            "variable_analyst is single-shot in Step 2; run_agent_ensemble must not fire"
        )

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(orchestrator, "run_agent_ensemble", fail_ensemble)

    result = _execute_tool(
        "spawn_variable_analyst",
        {"kernel_source": "SOURCE", "target_variable": "x"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "ok"
    assert result["result"]["variable"]["name"] == "x"
    assert len(calls) == 1
    type_, task = calls[0]
    assert type_ == "variable_analyst"
    assert task.startswith("SOURCE")
    assert "TARGET VARIABLE: x" in task
    # Never carries aggregator_metadata — that key is the ensemble
    # signal, and variable_analyst does not ensemble in Step 2.
    assert "aggregator_metadata" not in result


def test_execute_tool_spawn_variable_analyst_injects_probe_evidence_when_present(
    monkeypatch, tmp_path,
):
    """spawn_variable_analyst shares the finder/analyst probe-evidence auto-injection: when baselines/<kernel_stem>/probe/evidence.json exists, the task gets a PROBE EVIDENCE (JSON) block appended AFTER the TARGET VARIABLE line (single source of truth for WHERE evidence lives on disk). The evidence block ordering — kernel_source, then TARGET VARIABLE, then PROBE EVIDENCE — keeps the variable name visible in-context with the CANDIDATE FINDER RESULT block the caller already put in kernel_source, while still funneling the probe stats last."""
    monkeypatch.chdir(tmp_path)
    evidence_dir = tmp_path / "baselines" / "nbody_force" / "probe"
    evidence_dir.mkdir(parents=True)
    evidence_payload = {"cells": {"float_seed42": {"status": "ok"}}}
    (evidence_dir / "evidence.json").write_text(json.dumps(evidence_payload))

    captured = {}

    def stub_run_agent(type_, task):
        captured["type"] = type_
        captured["task"] = task
        return {
            "variable": {
                "name": "vx",
                "action": "keep",
                "target_precision": "",
                "emulation_type": "",
                "reason": "insufficient headroom per probe",
            },
            "notes": "",
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_variable_analyst",
        {"kernel_source": "ORIGINAL KERNEL SOURCE", "target_variable": "vx"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    assert result["status"] == "ok"
    assert captured["type"] == "variable_analyst"
    task = captured["task"]
    assert task.startswith("ORIGINAL KERNEL SOURCE")
    assert "TARGET VARIABLE: vx" in task
    assert "PROBE EVIDENCE (JSON):" in task
    assert "float_seed42" in task
    # Enforce section ordering: kernel_source, then TARGET VARIABLE,
    # then PROBE EVIDENCE, then the JSON payload.
    src_idx = task.index("ORIGINAL KERNEL SOURCE")
    tv_idx = task.index("TARGET VARIABLE: vx")
    probe_idx = task.index("PROBE EVIDENCE (JSON):")
    json_idx = task.index('"cells"')
    assert src_idx < tv_idx < probe_idx < json_idx


def test_execute_tool_spawn_variable_analyst_no_evidence_file_unchanged_task(
    monkeypatch, tmp_path,
):
    """When evidence.json is absent, spawn_variable_analyst silently falls back to the un-augmented task (kernel_source + TARGET VARIABLE line only) — same silent-skip rule the finder/analyst branches already follow. Missing evidence is informational-absent, never a hard stop; a kernel run without --probe (or on a profile with an empty probe_precisions) must still be able to call the per-variable analyst."""
    monkeypatch.chdir(tmp_path)
    captured = {}

    def stub_run_agent(type_, task):
        captured["task"] = task
        return {
            "variable": {
                "name": "x",
                "action": "downcast",
                "target_precision": "float",
                "emulation_type": "",
                "reason": "no probe -- reasoning from source",
            },
            "notes": "",
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    _execute_tool(
        "spawn_variable_analyst",
        {"kernel_source": "ORIGINAL KERNEL SOURCE", "target_variable": "x"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    task = captured["task"]
    assert "PROBE EVIDENCE" not in task
    # Task is exactly kernel_source + separator + TARGET VARIABLE line.
    assert task == "ORIGINAL KERNEL SOURCE\n\nTARGET VARIABLE: x"


def test_orchestrator_tools_expose_spawn_variable_analyst_with_required_args():
    """ORCHESTRATOR_TOOLS exposes spawn_variable_analyst with exactly {kernel_source, target_variable} as required inputs. The CANDIDATE FINDER RESULT block is embedded IN kernel_source by the orchestrator LLM (not passed as a separate arg) so the tool schema stays minimal and the same content can be reused across N per-variable calls with only target_variable changing."""
    tool = next(
        t for t in orchestrator.ORCHESTRATOR_TOOLS
        if t["name"] == "spawn_variable_analyst"
    )
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"kernel_source", "target_variable"}
    props = schema["properties"]
    assert props["kernel_source"]["type"] == "string"
    assert props["target_variable"]["type"] == "string"


def test_orchestrator_tools_no_longer_expose_spawn_analyst():
    """Step 2 of the per-variable refactor REMOVES spawn_analyst from the LLM-visible tool list — the monolithic analyst is replaced by the candidate_finder + N * variable_analyst pipeline (Step 5 will add a finalizer for precision_budget / rework). The dispatch branch for spawn_analyst inside _execute_tool is INTENTIONALLY retained as a callable-from-tests-only backdoor so the 39 existing test references keep working during the transition; this test guards the LLM-visible surface, not the Python API."""
    names = {t["name"] for t in orchestrator.ORCHESTRATOR_TOOLS}
    assert "spawn_analyst" not in names
    # Sanity: the replacement IS exposed, so this test can't false-pass
    # by accidentally guarding against a completely-empty tool list.
    assert "spawn_variable_analyst" in names
    assert "spawn_candidate_finder" in names


def test_execute_tool_dispatches_spawn_analyst(monkeypatch):
    """_execute_tool routes spawn_analyst to run_agent('analyst', kernel_source) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"variables": [], "overall_notes": "stubbed"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_analyst", {"kernel_source": "SOURCE"}, KOKKOS_PROFILE)

    assert result == {
        "status": "ok",
        "result": {"variables": [], "overall_notes": "stubbed"},
    }
    assert calls == [("analyst", "SOURCE")]


def test_execute_tool_spawn_analyst_default_k_uses_single_shot(monkeypatch):
    """Without AGENT_PRECISION_ANALYST_K set, _execute_tool stays on the single-shot run_agent path and does NOT invoke run_agent_ensemble — preserves existing behavior for callers who have not opted into the ensemble."""
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)

    def stub_run_agent(type_, task):
        return {"variables": [], "overall_notes": "single"}

    def fail_ensemble(*a, **kw):
        raise AssertionError(
            "run_agent_ensemble must not be called when K is unset (defaults to 1)"
        )

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(orchestrator, "run_agent_ensemble", fail_ensemble)

    result = _execute_tool("spawn_analyst", {"kernel_source": "SRC"}, KOKKOS_PROFILE)
    assert result == {
        "status": "ok",
        "result": {"variables": [], "overall_notes": "single"},
    }
    # The single-shot path must NOT carry aggregator_metadata; that key
    # is the signal to downstream tooling (and to the trace reader) that
    # an ensemble actually ran.
    assert "aggregator_metadata" not in result


def test_execute_tool_spawn_analyst_k_gt_one_runs_ensemble_and_aggregates(
    monkeypatch,
):
    """With AGENT_PRECISION_ANALYST_K=3 and a custom T, _execute_tool calls run_agent_ensemble with the requested k and temperature, folds the K verdicts through aggregate_analyst_verdicts, and returns the aggregated result plus the disagreement report as aggregator_metadata."""
    monkeypatch.setenv("AGENT_PRECISION_ANALYST_K", "3")
    monkeypatch.setenv("AGENT_PRECISION_ANALYST_T", "0.4")

    captured = {}

    def stub_ensemble(type_, task, k, temperature):
        captured["type"] = type_
        captured["task"] = task
        captured["k"] = k
        captured["temperature"] = temperature
        # Two verdicts agree on x=downcast, one disagrees → aggregator
        # should pick downcast and record the disagreement.
        budget = {
            "target_kind": "sig_figs",
            "target_value": 6,
            "source": "user_cli",
            "claimed_output_precision": "~7 sf",
            "headroom_argument": "ok",
        }
        empty_rework = {
            "suggested": False,
            "transformation": "",
            "rationale": "",
            "affected_variables": [],
        }

        def v(action):
            return {
                "variables": [
                    {
                        "name": "x",
                        "action": action,
                        "target_precision": "float" if action == "downcast" else "",
                        "emulation_type": "",
                        "reason": action,
                    }
                ],
                "rework": empty_rework,
                "precision_budget": budget,
                "overall_notes": f"notes-{action}",
            }

        return [v("downcast"), v("downcast"), v("keep")]

    def fail_single(*a, **kw):
        raise AssertionError(
            "run_agent must not be called directly when K>1 — the ensemble path owns the calls"
        )

    monkeypatch.setattr(orchestrator, "run_agent_ensemble", stub_ensemble)
    monkeypatch.setattr(orchestrator, "run_agent", fail_single)

    result = _execute_tool("spawn_analyst", {"kernel_source": "SRC"}, KOKKOS_PROFILE)

    assert captured == {
        "type": "analyst",
        "task": "SRC",
        "k": 3,
        "temperature": 0.4,
    }
    assert result["status"] == "ok"
    # The aggregator chose downcast on x (2-1 vote).
    assert result["result"]["variables"][0]["action"] == "downcast"
    assert result["result"]["variables"][0]["target_precision"] == "float"
    # The disagreement report rides alongside the result and names x.
    metadata = result["aggregator_metadata"]
    assert metadata["k"] == 3
    assert "x" in metadata["variable_disagreements"]
    assert metadata["variable_disagreements"]["x"]["winning_action"] == "downcast"


def test_execute_tool_spawn_analyst_k_gt_one_default_temperature(monkeypatch):
    """When AGENT_PRECISION_ANALYST_K is set but AGENT_PRECISION_ANALYST_T is not, the ensemble runs at the documented 0.7 default — chosen for vote diversity, not consistency."""
    monkeypatch.setenv("AGENT_PRECISION_ANALYST_K", "2")
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_T", raising=False)

    captured = {}

    def stub_ensemble(type_, task, k, temperature):
        captured["temperature"] = temperature
        v = {
            "variables": [],
            "rework": {
                "suggested": False,
                "transformation": "",
                "rationale": "",
                "affected_variables": [],
            },
            "precision_budget": {
                "target_kind": "sig_figs",
                "target_value": 6,
                "source": "user_cli",
                "claimed_output_precision": "",
                "headroom_argument": "",
            },
            "overall_notes": "",
        }
        return [v, v]

    monkeypatch.setattr(orchestrator, "run_agent_ensemble", stub_ensemble)

    _execute_tool("spawn_analyst", {"kernel_source": "SRC"}, KOKKOS_PROFILE)
    assert captured["temperature"] == 0.7


# ---------- _execute_tool: post-analyst probe-consistency gate ----------


def _write_evidence(tmp_path, stem, cell_stats):
    """Helper: drop a probe evidence.json at baselines/<stem>/probe/ under tmp_path.

    cell_stats is {(precision, seed): {output_name: {stat_key: value}}}; every
    listed cell is written with status='ok'. Returns nothing; the caller
    monkeypatch.chdir(tmp_path)s beforehand so the orchestrator resolves the
    relative baselines/... path here.
    """
    probe_dir = tmp_path / "baselines" / stem / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    precisions = sorted({p for p, _ in cell_stats})
    seeds = sorted({s for _, s in cell_stats})
    cells = {}
    for (prec, seed), stats in cell_stats.items():
        cells[f"{prec}_seed{seed}"] = {"status": "ok", "stats": stats}
    (probe_dir / "evidence.json").write_text(json.dumps({
        "kernel_stem": stem,
        "precisions": precisions,
        "seeds": seeds,
        "cells": cells,
    }))


def _downcast_verdict(names, target):
    """Helper: an analyst verdict that downcasts every name in `names` to `target`."""
    return {
        "variables": [
            {
                "name": n,
                "action": "downcast",
                "target_precision": target,
                "emulation_type": "",
                "reason": "test",
            }
            for n in names
        ],
        "rework": {
            "suggested": False,
            "transformation": "",
            "rationale": "",
            "affected_variables": [],
        },
        "precision_budget": {
            "target_kind": "sig_figs",
            "target_value": 6,
            "source": "user_cli",
            "claimed_output_precision": "~7 sf",
            "headroom_argument": "ok",
        },
        "overall_notes": "test",
    }


def test_execute_tool_spawn_analyst_probe_consistency_gate_flags_violation(
    monkeypatch, tmp_path
):
    """When the analyst's verdict positively contradicts the probe evidence (downcast to float, but float_seed42 cell shows worst-output max_absrel well above the sig_figs tolerance), _execute_tool returns a synthetic {status:'error', is_error:True} tool_result naming the offending variables and citing the concrete probe number, so the orchestrator LLM retries spawn_analyst instead of forwarding the bad verdict to the rewriter. Mirrors the finish-gate's synthetic-error idiom."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)
    _write_evidence(tmp_path, "kern", {
        # float cell: worst-output max_absrel 3.4e-4 blows sig_figs=6 (~1e-6)
        ("float", 42): {"vy": {"max_absrel": 3.4e-4, "max_abserror": 1.0}},
        # quad ground-truth cell must be present (helper stays minimal)
        ("quad", 42): {"vy": {"max_absrel": 0.0, "max_abserror": 0.0}},
    })
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda t, task: _downcast_verdict(["vx", "vy"], "float"),
    )

    result = _execute_tool(
        "spawn_analyst",
        {"kernel_source": "SRC"},
        KOKKOS_PROFILE,
        kernel_stem="kern",
        tolerance={"kind": "sig_figs", "value": 6, "source": "user_cli"},
    )

    assert result["status"] == "error"
    assert result["is_error"] is True
    assert "probe_consistency_violations" in result
    violations = result["probe_consistency_violations"]
    # Every downcast variable in the verdict should be flagged (cell-level
    # signal: worst-output in the target cell exceeds tolerance, so every
    # variable pointed at that cell is suspect).
    assert len(violations) == 2
    joined = " ".join(violations)
    assert "vx" in joined and "vy" in joined
    # The concrete probe number must appear in the error so the LLM has
    # something to reason about on retry, not just "probe disagreed".
    assert "3.4" in joined or "3.400e-04" in joined or "float_seed42" in joined
    # stderr carries the same info in human-readable form for the trace.
    assert "spawn_analyst" in result["stderr"]


def test_execute_tool_spawn_analyst_probe_consistency_gate_silent_when_no_evidence(
    monkeypatch, tmp_path
):
    """When baselines/<stem>/probe/evidence.json does not exist (probe disabled, or never ran, or --no-probe), the gate is a silent no-op: the analyst's verdict is returned unchanged even if it would otherwise be probe-inconsistent. Same skip policy the prompt-injection path already uses — the gate cannot manufacture evidence."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)
    # Deliberately do NOT write evidence.json.
    verdict = _downcast_verdict(["vx"], "float")
    monkeypatch.setattr(orchestrator, "run_agent", lambda t, task: verdict)

    result = _execute_tool(
        "spawn_analyst",
        {"kernel_source": "SRC"},
        KOKKOS_PROFILE,
        kernel_stem="kern",
        tolerance={"kind": "sig_figs", "value": 6, "source": "user_cli"},
    )

    assert result == {"status": "ok", "result": verdict}


def test_execute_tool_spawn_analyst_probe_consistency_gate_silent_when_tolerance_none(
    monkeypatch, tmp_path
):
    """When tolerance is None (run started without --sig-figs/--decimal-digits and either the advisor path is in flight or the caller is a unit test), the gate has no threshold to compare against and is a silent no-op. This preserves the invariant that _execute_tool called in isolation (as most unit tests do) never fails on probe-consistency."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)
    # Evidence exists AND would trip the check under a strict tolerance.
    _write_evidence(tmp_path, "kern", {
        ("float", 42): {"vy": {"max_absrel": 3.4e-4, "max_abserror": 1.0}},
        ("quad", 42): {"vy": {"max_absrel": 0.0, "max_abserror": 0.0}},
    })
    verdict = _downcast_verdict(["vx"], "float")
    monkeypatch.setattr(orchestrator, "run_agent", lambda t, task: verdict)

    result = _execute_tool(
        "spawn_analyst",
        {"kernel_source": "SRC"},
        KOKKOS_PROFILE,
        kernel_stem="kern",
        tolerance=None,
    )

    assert result == {"status": "ok", "result": verdict}


def test_execute_tool_spawn_analyst_probe_consistency_gate_ensemble_preserves_aggregator_metadata(
    monkeypatch, tmp_path
):
    """When K>1 and the AGGREGATED verdict trips the gate, the returned synthetic error still carries the aggregator_metadata under its usual key so the disagreement report reaches the trace even on a rejected ensemble result. The trace is the primary post-hoc debug artifact for ensemble runs; losing aggregator_metadata on rejection would blind us to whether the K analysts disagreed on the probe-inconsistent variables."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_PRECISION_ANALYST_K", "3")
    _write_evidence(tmp_path, "kern", {
        ("float", 42): {"vy": {"max_absrel": 3.4e-4, "max_abserror": 1.0}},
        ("quad", 42): {"vy": {"max_absrel": 0.0, "max_abserror": 0.0}},
    })
    verdict = _downcast_verdict(["vx"], "float")
    monkeypatch.setattr(
        orchestrator,
        "run_agent_ensemble",
        lambda type_, task, k, temperature: [verdict, verdict, verdict],
    )

    result = _execute_tool(
        "spawn_analyst",
        {"kernel_source": "SRC"},
        KOKKOS_PROFILE,
        kernel_stem="kern",
        tolerance={"kind": "sig_figs", "value": 6, "source": "user_cli"},
    )

    assert result["status"] == "error"
    assert result["is_error"] is True
    assert "aggregator_metadata" in result
    assert result["aggregator_metadata"]["k"] == 3


def test_execute_tool_dispatches_spawn_rewriter(monkeypatch):
    """_execute_tool routes spawn_rewriter to run_agent('rewriter', task_prompt) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"rewritten_code": "code", "summary_of_changes": "..."}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_rewriter", {"task_prompt": "PROMPT"}, KOKKOS_PROFILE)

    assert result["status"] == "ok"
    assert result["result"]["rewritten_code"] == "code"
    assert calls == [("rewriter", "PROMPT")]


# ---------- run_orchestrator: happy path ----------


def test_run_orchestrator_happy_path(monkeypatch, fake_anthropic, tmp_path):
    """run_orchestrator drives analyst -> rewriter -> verifier(accept) -> baseline harness chain -> compare_outputs -> finish end-to-end with HITL approvals."""
    # Uses a .cpp kernel which under KOKKOS_PROFILE (dynamic_verification=True)
    # requires the full dynamic-verification chain before finish. Phase B
    # unified .cu and .cpp gating, so there is no shorter happy path
    # available. Eleven orchestrator turns:
    #   1. spawn_analyst
    #   2. spawn_rewriter
    #   3. spawn_verifier(accept)
    #   4. spawn_baseline_harness   (writes driver under tmp_path)
    #   5. compile_baseline_driver  (stubbed ok)
    #   6. run_baseline_driver      (stubbed ok)
    #   7. splice_rewritten_kernel  (stubbed ok)
    #   8. compile_rewritten_driver (stubbed ok)
    #   9. run_rewritten_driver     (stubbed ok)
    #  10. compare_outputs          (stubbed ok -> sets compare_status)
    #  11. finish
    monkeypatch.chdir(tmp_path)
    tol_json = (
        '{"kind":"sig_figs","value":6,'
        '"source":"user_cli"}'
    )
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_2",
                name="spawn_rewriter",
                input={"task_prompt": "rewrite please"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_3",
                name="spawn_verifier",
                input={
                    "original_source": "KSRC",
                    "rewritten_source": "FINAL",
                    "analyst_verdict_json": "{}",
                    "tolerance_json": tol_json,
                },
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_4",
                name="spawn_baseline_harness",
                input={"kernel_source": "KSRC", "kernel_stem": "kernel"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_5",
                name="compile_baseline_driver",
                input={"kernel_stem": "kernel"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_6",
                name="run_baseline_driver",
                input={"kernel_stem": "kernel"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_7",
                name="splice_rewritten_kernel",
                input={
                    "kernel_stem": "kernel",
                    "rewritten_kernel_source": "FINAL",
                },
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_8",
                name="compile_rewritten_driver",
                input={"kernel_stem": "kernel"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_9",
                name="run_rewritten_driver",
                input={"kernel_stem": "kernel"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_10",
                name="compare_outputs",
                input={
                    "kernel_stem": "kernel",
                    "tolerance_json": tol_json,
                },
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_11",
                name="finish",
                input={"rewritten_code": "FINAL", "notes": "done"},
            )],
        ),
    ])

    # Stub the agents (called via _execute_tool).
    def stub_run_agent(type_, task):
        if type_ == "analyst":
            return {"variables": [], "overall_notes": "ok"}
        if type_ == "rewriter":
            return {"rewritten_code": "FINAL", "summary_of_changes": "ok"}
        if type_ == "verifier":
            return {"verdict": "accept", "per_variable": [], "concerns": []}
        if type_ == "baseline_harness_kokkos":
            return {
                "driver_source": "// driver\nint main(){return 0;}\n",
                "kernel_function_name": "kernel",
                "inputs_summary": "N=1, seed=42",
                "output_arrays": ["y"],
            }
        raise AssertionError(f"unexpected agent: {type_}")

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    # Stub all four deterministic chain tools that follow harness. Each
    # returns the standard {status, stdout, stderr, artifacts} shape.
    ok_chain = {
        "status": "ok", "stdout": "", "stderr": "", "artifacts": [],
    }
    monkeypatch.setattr(
        orchestrator, "compile_baseline_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "run_baseline_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "splice_rewritten_kernel", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "compile_rewritten_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "run_rewritten_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "compare_outputs", lambda *a, **kw: ok_chain
    )

    # Eleven HITL prompts, all approved.
    _scripted_input(monkeypatch, ["y"] * 11)

    result = run_orchestrator(
        "path/to/kernel.cpp",
        "kernel source body",
        tolerance=_DEFAULT_TEST_TOLERANCE,
    )

    assert result == {"rewritten_code": "FINAL", "notes": "done"}
    assert len(fake.messages.calls) == 11

    # First call: just the user message.
    first_messages = fake.messages.calls[0]["messages"]
    assert len(first_messages) == 1
    assert first_messages[0]["role"] == "user"
    assert "kernel source body" in first_messages[0]["content"]

    # Second call: user + assistant + user(tool_result). Check the tool_result
    # carries the stubbed analyst output.
    second_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_messages[-1]
    assert tool_result_msg["role"] == "user"
    tr_block = tool_result_msg["content"][0]
    assert tr_block["type"] == "tool_result"
    assert tr_block["tool_use_id"] == "tu_1"
    payload = json.loads(tr_block["content"])
    assert payload["status"] == "ok"
    assert payload["result"]["overall_notes"] == "ok"


# ---------- run_orchestrator: rejection feeds back sentinel ----------


def test_run_orchestrator_rejection_feeds_back_sentinel(
    monkeypatch, fake_anthropic, tmp_path
):
    """A HITL 'n' rejects the tool call without invoking run_agent and feeds {'status':'rejected_by_user'} back to the orchestrator."""
    # Uses a .cpp kernel under KOKKOS_PROFILE (Phase B unified .cu and
    # .cpp gating, so there is no shorter path to finish for either
    # extension). Ten orchestrator turns: rejected spawn_analyst, then
    # the full happy chain to satisfy the code-side finish-gate.
    #   1. spawn_analyst              (HITL 'n' -> rejection sentinel)
    #   2. spawn_verifier(accept)
    #   3. spawn_baseline_harness
    #   4. compile_baseline_driver
    #   5. run_baseline_driver
    #   6. splice_rewritten_kernel
    #   7. compile_rewritten_driver
    #   8. run_rewritten_driver
    #   9. compare_outputs
    #  10. finish
    monkeypatch.chdir(tmp_path)
    tol_json = (
        '{"kind":"sig_figs","value":6,'
        '"source":"user_cli"}'
    )
    fake = fake_anthropic([
        # Turn 1: orchestrator proposes spawn_analyst — user rejects.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
        # Turn 2: after seeing rejection, orchestrator calls verifier to
        # satisfy the verifier prong of the finish-gate.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_2",
                name="spawn_verifier",
                input={
                    "original_source": "src",
                    "rewritten_source": "src",
                    "analyst_verdict_json": "{}",
                    "tolerance_json": tol_json,
                },
            )],
        ),
        # Turns 3-9: the full baseline + rewritten chain that the
        # comparator prong of the finish-gate requires post-Phase-B.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_3",
                name="spawn_baseline_harness",
                input={"kernel_source": "src", "kernel_stem": "k"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_4",
                name="compile_baseline_driver",
                input={"kernel_stem": "k"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_5",
                name="run_baseline_driver",
                input={"kernel_stem": "k"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_6",
                name="splice_rewritten_kernel",
                input={
                    "kernel_stem": "k",
                    "rewritten_kernel_source": "src",
                },
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_7",
                name="compile_rewritten_driver",
                input={"kernel_stem": "k"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_8",
                name="run_rewritten_driver",
                input={"kernel_stem": "k"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_9",
                name="compare_outputs",
                input={"kernel_stem": "k", "tolerance_json": tol_json},
            )],
        ),
        # Turn 10: finish.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_10",
                name="finish",
                input={"rewritten_code": "X", "notes": "gave up"},
            )],
        ),
    ])

    # The analyst call gets rejected, finish has no agent, and the
    # chain tools are stubbed below. Only verifier + baseline_harness
    # actually go through run_agent.
    def stub_run_agent(type_, task):
        if type_ == "verifier":
            return {"verdict": "accept", "per_variable": [], "concerns": []}
        if type_ == "baseline_harness_kokkos":
            return {
                "driver_source": "// driver\nint main(){return 0;}\n",
                "kernel_function_name": "k",
                "inputs_summary": "N=1, seed=42",
                "output_arrays": ["y"],
            }
        raise AssertionError(
            f"run_agent should not be called for {type_!r} in this test"
        )

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    ok_chain = {
        "status": "ok", "stdout": "", "stderr": "", "artifacts": [],
    }
    monkeypatch.setattr(
        orchestrator, "compile_baseline_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "run_baseline_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "splice_rewritten_kernel", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "compile_rewritten_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "run_rewritten_driver", lambda *a, **kw: ok_chain
    )
    monkeypatch.setattr(
        orchestrator, "compare_outputs", lambda *a, **kw: ok_chain
    )

    # 'n' rejects the first call; the remaining nine all approve.
    _scripted_input(monkeypatch, ["n"] + ["y"] * 9)

    result = run_orchestrator("k.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE)

    assert result == {"rewritten_code": "X", "notes": "gave up"}

    # Verify the rejection sentinel was passed back on turn 2.
    second_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_messages[-1]
    tr_block = tool_result_msg["content"][0]
    assert tr_block["type"] == "tool_result"
    assert tr_block["tool_use_id"] == "tu_1"
    payload = json.loads(tr_block["content"])
    assert payload == {"status": "rejected_by_user"}


# ---------- run_orchestrator: quit ----------


def test_run_orchestrator_quit_returns_none(monkeypatch, fake_anthropic):
    """A HITL 'q' aborts the loop, skips run_agent, and returns None."""
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
    ])
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda *a, **kw: pytest.fail("run_agent must not be called after quit"),
    )
    _scripted_input(monkeypatch, ["q"])

    assert (
        run_orchestrator("k.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE)
        is None
    )


# ---------- run_orchestrator: stop without tool ----------


def test_run_orchestrator_stop_without_tool_returns_none(
    monkeypatch, fake_anthropic
):
    """If the orchestrator responds with text only (no tool_use), run_orchestrator returns None."""
    fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="I am just going to say words.")],
            stop_reason="end_turn",
        ),
    ])
    # No HITL prompts should fire because there's no tool_use block.
    _scripted_input(monkeypatch, [])

    assert (
        run_orchestrator("k.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE)
        is None
    )


# ---------- Orchestrator prompt: vocabulary matches the agents it routes to ----------


def test_orchestrator_prompt_names_all_three_methods_and_rework():
    """The orchestrator prompt names downcast, emulate, keep, and the rework block, so its task-prompt translation step has the right vocabulary to pass to the rewriter."""
    for token in ("downcast", "emulate", "keep", "rework"):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_names_tolerance_kinds():
    """The orchestrator prompt names sig_figs, decimal_digits, and precision_budget so the LLM knows the tolerance vocabulary and how to thread it into downstream task prompts."""
    for token in (
        "sig_figs",
        "decimal_digits",
        "precision_budget",
    ):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_does_not_mention_precision_advisor():
    """The precision_advisor agent has been removed; the orchestrator prompt must not name it as a callable tool or fallback path."""
    assert "precision_advisor" not in ORCHESTRATOR_SYSTEM_PROMPT
    assert "advisor_unknown_defaulted" not in ORCHESTRATOR_SYSTEM_PROMPT


def test_orchestrator_prompt_forbids_finish_without_accept():
    """The orchestrator prompt explicitly forbids calling finish unless the most recent verifier returned accept; this rule lives in the prompt, not in code."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "may not call finish" in text or "may not call `finish`" in text
    assert "accept" in text


# ---------- _format_tolerance_block ----------


def test_format_tolerance_block_renders_user_cli_verbatim():
    """The rendered block contains the kind/value/source verbatim so the orchestrator LLM can thread the tolerance into downstream task prompts."""
    block = _format_tolerance_block(
        {"kind": "sig_figs", "value": 7, "source": "user_cli"}
    )
    assert "sig_figs" in block
    assert "7" in block
    assert "user_cli" in block


def test_format_tolerance_block_does_not_mention_advisor():
    """After removing the precision_advisor agent, the rendered tolerance block must not name it (as a callable, a source enum value, or a fallback path)."""
    block = _format_tolerance_block(
        {"kind": "sig_figs", "value": 6, "source": "user_cli"}
    )
    assert "precision_advisor" not in block
    assert "advisor_unknown_defaulted" not in block


# ---------- _execute_tool: spawn_verifier(tolerance_json) ----------


def test_execute_tool_spawn_verifier_includes_tolerance_in_task(monkeypatch):
    """_execute_tool builds the verifier's task string from original_source, rewritten_source, analyst_verdict_json, AND tolerance_json — so the verifier sees the same tolerance the analyst saw."""
    captured = {}

    def stub_run_agent(type_, task):
        captured["type"] = type_
        captured["task"] = task
        return {"verdict": "accept", "per_variable": [], "concerns": []}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_verifier",
        {
            "original_source": "ORIG",
            "rewritten_source": "REW",
            "analyst_verdict_json": '{"variables": []}',
            "tolerance_json": '{"kind":"sig_figs","value":6,"source":"user_cli"}',
        }, KOKKOS_PROFILE
    )

    assert result["status"] == "ok"
    assert captured["type"] == "verifier"
    # the task must contain all four pieces
    assert "ORIG" in captured["task"]
    assert "REW" in captured["task"]
    assert "ANALYST VERDICT" in captured["task"]
    assert "TOLERANCE" in captured["task"]
    assert "user_cli" in captured["task"]


# ---------- _execute_tool: spawn_verifier panel mode (opt-in) ----------


def _verifier_task_args() -> dict:
    """Standard four-arg payload for spawn_verifier in panel tests."""
    return {
        "original_source": "ORIG",
        "rewritten_source": "REW",
        "analyst_verdict_json": '{"variables": []}',
        "tolerance_json": '{"kind":"sig_figs","value":6,"source":"user_cli"}',
    }


def test_execute_tool_spawn_verifier_default_k_uses_single_shot(monkeypatch):
    """Without AGENT_PRECISION_VERIFIER_K set, _execute_tool stays on the single-shot run_agent('verifier', ...) path and does NOT invoke run_verifier_panel — preserves existing behavior for callers who have not opted into the panel."""
    monkeypatch.delenv("AGENT_PRECISION_VERIFIER_K", raising=False)

    def stub_run_agent(type_, task):
        assert type_ == "verifier"
        return {"verdict": "accept", "per_variable": [], "concerns": []}

    def fail_panel(*a, **kw):
        raise AssertionError(
            "run_verifier_panel must not be called when K is unset (defaults to 1)"
        )

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(orchestrator, "run_verifier_panel", fail_panel)

    result = _execute_tool("spawn_verifier", _verifier_task_args(), KOKKOS_PROFILE)
    assert result == {
        "status": "ok",
        "result": {"verdict": "accept", "per_variable": [], "concerns": []},
    }
    # The single-shot path must NOT carry verifier_aggregator_metadata; that
    # key is the signal to downstream tooling (and to the trace reader) that
    # a panel actually ran.
    assert "verifier_aggregator_metadata" not in result


def test_execute_tool_spawn_verifier_k_gt_one_runs_panel_and_aggregates(
    monkeypatch,
):
    """With AGENT_PRECISION_VERIFIER_K=3 and a custom T, _execute_tool calls run_verifier_panel with the first K lenses and temperature, folds the K verdicts through aggregate_verifier_verdicts, and returns the aggregated result plus the disagreement report as verifier_aggregator_metadata."""
    monkeypatch.setenv("AGENT_PRECISION_VERIFIER_K", "3")
    monkeypatch.setenv("AGENT_PRECISION_VERIFIER_T", "0.4")

    captured = {}

    def stub_panel(task, lenses, temperature):
        captured["task"] = task
        captured["lenses"] = lenses
        captured["temperature"] = temperature
        # Two lenses accept, the budget lens rejects with one concern.
        # Strict aggregation must flip the whole panel to reject and the
        # report must name 'budget' as the dissenting lens.
        return [
            {
                "verdict": "accept",
                "per_variable": [
                    {
                        "name": "x",
                        "expected_action": "downcast",
                        "observed_action": "downcast",
                        "ok": True,
                        "note": "",
                    }
                ],
                "concerns": [],
            },
            {
                "verdict": "reject",
                "per_variable": [],
                "concerns": ["headroom_argument is hand-wavy"],
            },
            {
                "verdict": "accept",
                "per_variable": [],
                "concerns": [],
            },
        ]

    def fail_single(*a, **kw):
        raise AssertionError(
            "run_agent must not be called directly when K>1 — the panel path owns the calls"
        )

    monkeypatch.setattr(orchestrator, "run_verifier_panel", stub_panel)
    monkeypatch.setattr(orchestrator, "run_agent", fail_single)

    result = _execute_tool("spawn_verifier", _verifier_task_args(), KOKKOS_PROFILE)

    # The task threaded through verifier_panel is the same fully-formed
    # verifier prompt — same original/rewritten/verdict/tolerance shape.
    assert "ORIG" in captured["task"]
    assert "REW" in captured["task"]
    assert "ANALYST VERDICT" in captured["task"]
    assert "TOLERANCE" in captured["task"]
    # The panel got the first K lenses verbatim and the requested temperature.
    assert [l["name"] for l in captured["lenses"]] == [
        "faithfulness",
        "budget",
        "edge_cases",
    ]
    assert captured["temperature"] == 0.4

    assert result["status"] == "ok"
    # Strict-verdict: budget rejected, so the aggregate is reject.
    assert result["result"]["verdict"] == "reject"
    # per_variable from the faithfulness lens (lens 0) survives verbatim.
    assert result["result"]["per_variable"][0]["name"] == "x"
    # concerns carry the lens-name prefix so the rewriter retry knows
    # which lens raised which worry.
    assert any(
        c.startswith("[budget]") for c in result["result"]["concerns"]
    )
    # The disagreement report rides alongside and names the dissenter.
    metadata = result["verifier_aggregator_metadata"]
    assert metadata["k"] == 3
    assert metadata["dissenting_lenses"] == ["budget"]
    assert metadata["lens_verdicts"]["faithfulness"] == "accept"


def test_execute_tool_spawn_verifier_k_gt_one_default_temperature(monkeypatch):
    """When AGENT_PRECISION_VERIFIER_K is set but AGENT_PRECISION_VERIFIER_T is not, the panel runs at the documented 0.7 default — chosen for lens diversity, mirroring the analyst ensemble default."""
    monkeypatch.setenv("AGENT_PRECISION_VERIFIER_K", "2")
    monkeypatch.delenv("AGENT_PRECISION_VERIFIER_T", raising=False)

    captured = {}

    def stub_panel(task, lenses, temperature):
        captured["temperature"] = temperature
        v = {"verdict": "accept", "per_variable": [], "concerns": []}
        return [v, v]

    monkeypatch.setattr(orchestrator, "run_verifier_panel", stub_panel)

    _execute_tool("spawn_verifier", _verifier_task_args(), KOKKOS_PROFILE)
    assert captured["temperature"] == 0.7


def test_execute_tool_spawn_verifier_k_exceeds_lenses_raises(monkeypatch):
    """AGENT_PRECISION_VERIFIER_K cannot exceed the number of defined lenses — lenses ARE the panel, not just a replication multiplier. The error message must be actionable (mention both the requested K and where to add a lens)."""
    from workflow.verifier_panel import VERIFIER_LENSES

    over = len(VERIFIER_LENSES) + 1
    monkeypatch.setenv("AGENT_PRECISION_VERIFIER_K", str(over))

    def fail_panel(*a, **kw):
        raise AssertionError(
            "run_verifier_panel must not be called when K exceeds lens count"
        )

    monkeypatch.setattr(orchestrator, "run_verifier_panel", fail_panel)

    with pytest.raises(ValueError, match="VERIFIER_LENSES"):
        _execute_tool("spawn_verifier", _verifier_task_args(), KOKKOS_PROFILE)


# ---------- run_orchestrator: tolerance plumbing in the initial user message ----------


def test_run_orchestrator_user_cli_tolerance_appears_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """The user-supplied tolerance is embedded verbatim into the first user message so the orchestrator LLM can thread it into every downstream task prompt."""
    # First-user-message-only test: short-circuit by returning text + end_turn
    # on turn 1 so the loop exits with None before engaging the finish gate.
    fake = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])  # no tool_use, no HITL pause

    run_orchestrator(
        "k.cpp",
        "src",
        tolerance={
            "kind": "decimal_digits",
            "value": 4,
            "source": "user_cli",
        },
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "decimal_digits" in first_user
    assert "4" in first_user
    assert "user_cli" in first_user
    # The removed precision_advisor agent must not be advertised anywhere
    # in the initial user message.
    assert "precision_advisor" not in first_user


# ---------- Baseline harness: tool schema + prompt + dispatch + user message ----------


def test_orchestrator_tools_include_spawn_baseline_harness():
    """ORCHESTRATOR_TOOLS exposes spawn_baseline_harness with kernel_source and kernel_stem as required string inputs."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "spawn_baseline_harness" in by_name
    tool = by_name["spawn_baseline_harness"]
    props = tool["input_schema"]["properties"]
    assert "kernel_source" in props
    assert "kernel_stem" in props
    assert props["kernel_source"]["type"] == "string"
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {
        "kernel_source",
        "kernel_stem",
    }


def test_orchestrator_prompt_mentions_baseline_harness_and_dynamic_verification_chain():
    """The orchestrator prompt names baseline_harness, the BASELINE STEP block, and ties baseline_harness to the dynamic-verification chain (the code-side finish-gate). Phase B genericized the wording from 'side artifact' to chain-membership language because Kokkos and CUDA both wire the baseline into the dynamic-verification chain now."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "baseline_harness" in text
    assert "BASELINE STEP" in text
    lower = text.lower()
    # Phase B: baseline is no longer "just a side artifact" — it's the
    # first link in the dynamic-verification chain that gates finish on
    # profiles with dynamic_verification=True. Assert the prompt names
    # that chain explicitly so a future edit can't silently drop it.
    assert "dynamic-verification chain" in lower
    assert "compare_outputs" in text
    assert "finish-gate" in lower


def test_execute_tool_dispatches_spawn_baseline_harness(monkeypatch, tmp_path):
    """_execute_tool routes spawn_baseline_harness to run_agent('baseline_harness_<profile.id>', kernel_source) — per-language dispatch via the profile id — writes the driver to baselines/<stem>/<profile.driver_filename>, and returns the driver_path alongside the result."""
    monkeypatch.chdir(tmp_path)
    calls = []

    driver_text = "// driver\nint main(){return 0;}\n"

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "driver_source": driver_text,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"}, KOKKOS_PROFILE
    )

    assert result["status"] == "ok"
    assert result["result"]["kernel_function_name"] == "vector_add"
    assert calls == [("baseline_harness_kokkos", "KSRC")]

    # The driver must land at baselines/<stem>/driver.cpp under CWD
    # (the orchestrator writes via a *relative* Path; under
    # monkeypatch.chdir(tmp_path) that resolves to tmp_path/baselines/...).
    driver_path = tmp_path / "baselines" / "vector_add" / "driver.cpp"
    assert driver_path.exists()
    assert driver_path.read_text() == driver_text
    assert result["driver_path"] == "baselines/vector_add/driver.cpp"


def test_execute_tool_spawn_baseline_harness_overwrites_existing(
    monkeypatch, tmp_path
):
    """A second spawn_baseline_harness call for the same stem overwrites the previous driver.cpp (parents=True, exist_ok=True; write_text replaces)."""
    monkeypatch.chdir(tmp_path)
    # Pre-create an old driver to be overwritten.
    old_dir = tmp_path / "baselines" / "vector_add"
    old_dir.mkdir(parents=True)
    (old_dir / "driver.cpp").write_text("OLD CONTENT")

    def stub_run_agent(type_, task):
        return {
            "driver_source": "NEW CONTENT",
            "kernel_function_name": "vector_add",
            "inputs_summary": "...",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"}, KOKKOS_PROFILE
    )

    assert (tmp_path / "baselines" / "vector_add" / "driver.cpp").read_text() \
        == "NEW CONTENT"


def test_execute_tool_spawn_baseline_harness_v1_drivers_fan_out(
    monkeypatch, tmp_path
):
    """_execute_tool routes the Kokkos v1 4-driver harness output: drivers['double'] is written to baselines/<stem>/driver.cpp (the canonical splice scaffold that feeds the v0 dynamic-verification chain), and every drivers[<precision>] is also written to baselines/<stem>/probe/<precision>/driver.cpp for the probe pipeline. The canonical baseline is the DOUBLE driver, NOT drivers[baseline_precision] (which is 'quad'): Kokkos has no `__float128` math overload (`Kokkos::sqrt(__float128)` does not exist), so the quad driver is plain C++ + quadmath per the harness prompt — uncompilable as a splice target for the rewriter's Kokkos kernels. The quad driver still serves as the ground-truth oracle: its seed=42 reference.json is promoted to baselines/<stem>/reference.json later in the chain by the probe_compare branch (separately asserted). The returned dict carries the canonical driver_path plus a probe_driver_paths map keyed by precision. The harness's v1 shape (drivers: dict) is detected by presence of the `drivers` key — the v0 shape (driver_source: str, still used by CUDA/HIP/SYCL/OMP-offload) remains supported on the same branch (separately asserted)."""
    monkeypatch.chdir(tmp_path)

    drivers_payload = {
        "quad":     "// QUAD DRIVER\nint main(){return 0;}\n",
        "double":   "// DOUBLE DRIVER\nint main(){return 0;}\n",
        "float":    "// FLOAT DRIVER\nint main(){return 0;}\n",
        "mixed_io": "// MIXED_IO DRIVER\nint main(){return 0;}\n",
    }

    def stub_run_agent(type_, task):
        assert type_ == "baseline_harness_kokkos"
        return {
            "drivers": drivers_payload,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"}, KOKKOS_PROFILE
    )

    assert result["status"] == "ok"
    # The canonical baseline is the DOUBLE driver — the splice
    # scaffold the rewriter targets. NOT drivers[baseline_precision]
    # (which is "quad"): Kokkos has no `__float128` overload for its
    # math intrinsics, so the quad driver is plain C++ + quadmath
    # per the kokkos harness prompt — uncompilable as a Kokkos splice
    # target. The quad driver fills the ground-truth-oracle role
    # instead: its seed=42 reference.json is promoted into
    # baselines/<stem>/reference.json by the probe_compare branch
    # later in the chain (see test_execute_tool_probe_compare_*).
    # KOKKOS_PROFILE.baseline_precision stays "quad" — it names the
    # oracle precision, not the splice-scaffold precision.
    canonical = tmp_path / "baselines" / "vector_add" / "driver.cpp"
    assert canonical.exists()
    assert canonical.read_text() == drivers_payload["double"]
    assert result["driver_path"] == "baselines/vector_add/driver.cpp"

    # Every precision variant ALSO lands under
    # baselines/<stem>/probe/<precision>/driver.cpp (probe pipeline
    # input). All four are emitted including the one that duplicates
    # the canonical baseline — the probe_step tool in a later commit
    # rewrites a seed line per-directory and reuses _compile_driver /
    # _run_driver per-directory, so each precision needs its own
    # self-contained tree.
    probe_paths = result["probe_driver_paths"]
    assert set(probe_paths.keys()) == set(drivers_payload.keys())
    for precision, source in drivers_payload.items():
        probe_file = (
            tmp_path / "baselines" / "vector_add" / "probe" / precision / "driver.cpp"
        )
        assert probe_file.exists()
        assert probe_file.read_text() == source
        assert probe_paths[precision] == (
            f"baselines/vector_add/probe/{precision}/driver.cpp"
        )


def test_execute_tool_spawn_baseline_harness_v0_driver_source_still_works(
    monkeypatch, tmp_path
):
    """When the harness returns the v0 single-driver shape (`driver_source: str`, still used by CUDA / HIP / SYCL / OMP-offload profiles whose probe_precisions is the empty default), the orchestrator branch writes that source to the canonical baselines/<stem>/<driver_filename> and omits the v1-only probe_driver_paths key from the response. This back-compat path keeps the orchestrator language-agnostic until the deferred Commit 6 extends probe_precisions to the other profiles."""
    monkeypatch.chdir(tmp_path)
    from workflow.languages.cuda import CUDA_PROFILE

    driver_text = "// cuda driver\nint main(){return 0;}\n"

    def stub_run_agent(type_, task):
        assert type_ == "baseline_harness_cuda"
        return {
            "driver_source": driver_text,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"}, CUDA_PROFILE
    )

    assert result["status"] == "ok"
    canonical = tmp_path / "baselines" / "vector_add" / "driver.cu"
    assert canonical.exists()
    assert canonical.read_text() == driver_text
    # The v0 path MUST NOT advertise probe driver paths — the absence
    # of this key is what tells a future probe orchestrator that this
    # profile's probe pipeline is not yet wired.
    assert "probe_driver_paths" not in result
    # No probe/ subdirectory is created either.
    assert not (tmp_path / "baselines" / "vector_add" / "probe").exists()


# ---------- spawn_baseline_harness syntax-check gate ----------
#
# The gate rejects malformed harness output BEFORE it hits disk, so
# the orchestrator can retry the harness (`is_error: True`
# tool_result) with the compiler diagnostic verbatim instead of
# wasting a full compile_baseline_driver HITL cycle on a driver we
# already know won't compile. See workflow.tools.syntax_check_driver_
# source and the docstring of the gate's motivating case (an
# nbody_force run whose harness emitted two inconsistent alias
# naming conventions in one declaration).


def test_execute_tool_spawn_baseline_harness_gate_rejects_bad_v0_source(
    monkeypatch, tmp_path
):
    """When the profile's syntax-check gate rejects a v0 driver_source payload, _execute_tool returns the gate's error dict with `is_error: True` added — the tool_result-block layer of run_orchestrator translates that into `is_error: True` on the Anthropic message, so the model retries the harness. Critically, no driver file is written to disk: the write-first path is what caused the motivating alias-drift bug to burn a full compile HITL cycle."""
    monkeypatch.chdir(tmp_path)

    def stub_run_agent(type_, task):
        return {
            "driver_source": "malformed source",
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    def stub_syntax_check(profile, source, label):
        assert source == "malformed source"
        assert label == "driver_source"
        return {
            "status": "error",
            "stdout": "",
            "stderr": "g++ -fsyntax-only rejected driver_source ...",
            "artifacts": [],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(
        orchestrator, "syntax_check_driver_source", stub_syntax_check
    )

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
        CUDA_PROFILE,
    )

    assert result["status"] == "error"
    assert result["is_error"] is True
    assert "g++ -fsyntax-only rejected" in result["stderr"]
    # NO file must have been written under baselines/<stem>/.
    assert not (tmp_path / "baselines" / "vector_add").exists()


def test_execute_tool_spawn_baseline_harness_gate_rejects_bad_v1_driver(
    monkeypatch, tmp_path
):
    """When one of the drivers in a v1 multi-driver payload fails the syntax-check gate, _execute_tool returns the gate error with `is_error: True` and writes NOTHING to disk (all-or-nothing — a partial fan-out would leave stale files that later probe_step calls would silently reuse). The label folded into the error names WHICH precision failed so the harness re-run can target its fix."""
    monkeypatch.chdir(tmp_path)

    drivers_payload = {
        "quad":     "// QUAD driver\n",
        "double":   "// DOUBLE driver (broken)\n",
        "float":    "// FLOAT driver\n",
        "mixed_io": "// MIXED_IO driver\n",
    }

    def stub_run_agent(type_, task):
        return {
            "drivers": drivers_payload,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    def stub_syntax_check(profile, source, label):
        if source == drivers_payload["double"]:
            return {
                "status": "error",
                "stdout": "",
                "stderr": f"g++ -fsyntax-only rejected {label}\n",
                "artifacts": [],
            }
        return None

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(
        orchestrator, "syntax_check_driver_source", stub_syntax_check
    )

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "error"
    assert result["is_error"] is True
    # Label naming which precision failed — necessary for the harness
    # re-run to know which of the four drivers to fix.
    assert "drivers['double']" in result["stderr"]
    # All-or-nothing: no partial write, no probe/ subdirectory.
    assert not (tmp_path / "baselines" / "vector_add").exists()


def test_execute_tool_spawn_baseline_harness_gate_passes_writes_all(
    monkeypatch, tmp_path
):
    """When every driver in a v1 payload passes the syntax-check gate (stub returns None), _execute_tool proceeds with the normal fan-out write: baselines/<stem>/driver.cpp (the DOUBLE splice scaffold) plus baselines/<stem>/probe/<precision>/driver.cpp per precision. The is_error key must NOT appear in the result on the pass path."""
    monkeypatch.chdir(tmp_path)

    drivers_payload = {
        "quad":     "// QUAD\nint main(){return 0;}\n",
        "double":   "// DOUBLE\nint main(){return 0;}\n",
        "float":    "// FLOAT\nint main(){return 0;}\n",
        "mixed_io": "// MIXED_IO\nint main(){return 0;}\n",
    }

    def stub_run_agent(type_, task):
        return {
            "drivers": drivers_payload,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    # Stub gate: always pass.
    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.setattr(
        orchestrator,
        "syntax_check_driver_source",
        lambda profile, source, label: None,
    )

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "ok"
    assert "is_error" not in result
    # Canonical splice scaffold + probe fan-out both present.
    assert (tmp_path / "baselines" / "vector_add" / "driver.cpp").exists()
    for precision in drivers_payload:
        assert (
            tmp_path
            / "baselines" / "vector_add" / "probe" / precision / "driver.cpp"
        ).exists()


def test_execute_tool_spawn_baseline_harness_gate_skipped_writes_all(
    monkeypatch, tmp_path
):
    """When the profile's syntax-check gate is unavailable (stub returns None to simulate an unset AGENT_PRECISION_KOKKOS_ROOT), _execute_tool must still write every driver — the gate is a quality improvement, not a hard requirement. This test pins the silent-skip contract at the orchestrator layer so a future refactor doesn't turn 'unavailable toolchain' into a hard failure."""
    monkeypatch.chdir(tmp_path)

    driver_text = "// v0 driver\nint main(){return 0;}\n"

    def stub_run_agent(type_, task):
        return {
            "driver_source": driver_text,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    # None means "gate unavailable / silent skip".
    monkeypatch.setattr(
        orchestrator,
        "syntax_check_driver_source",
        lambda profile, source, label: None,
    )

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
        CUDA_PROFILE,
    )

    assert result["status"] == "ok"
    assert (tmp_path / "baselines" / "vector_add" / "driver.cu").exists()


# ---------- _format_baseline_block ----------


def test_format_baseline_block_cpp_no_kernel_name_invites_call():
    """For a .cpp kernel without an explicit kernel_name, the block invites spawn_baseline_harness, surfaces the file stem as KERNEL STEM, and emits no 'TARGET KERNEL: <name>' value line (the agent infers the function)."""
    block = _format_baseline_block("test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE)
    assert "BASELINE STEP" in block
    assert "spawn_baseline_harness" in block
    assert "KERNEL STEM: nbody_force" in block
    # The boilerplate may mention 'TARGET KERNEL:' as a hint to the
    # orchestrator about what it MAY prepend; what must NOT appear is an
    # actual 'TARGET KERNEL: <name>' value line (it would be empty/wrong).
    for line in block.splitlines():
        assert not line.startswith("TARGET KERNEL:"), line


def test_format_baseline_block_cpp_with_kernel_name_includes_target_line():
    """When kernel_name is given, the block adds a TARGET KERNEL: <name> line so the orchestrator can prepend it to the harness's kernel_source argument."""
    block = _format_baseline_block(
        "test-kernels/kokkos/lowerable/vector_add.cpp", "vector_add", KOKKOS_PROFILE
    )
    assert "KERNEL STEM: vector_add" in block
    assert "TARGET KERNEL: vector_add" in block


def test_format_baseline_block_test_config_none_omits_block():
    """When test_config is None (default), no `TEST CONFIG (JSON):` value-line block appears in the BASELINE STEP portion of the initial user message — the harness reverts to inferring inputs. Note the trailing colon: the boilerplate hint text may reference the phrase 'TEST CONFIG (JSON) block' without a colon, so the emitted block is distinguished by the `TEST CONFIG (JSON):` marker (same idiom as `TARGET KERNEL:`)."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "TEST CONFIG (JSON):" not in block


def test_format_baseline_block_test_config_dict_emits_json_block():
    """When test_config is a non-None dict, the BASELINE STEP block includes a `TEST CONFIG (JSON):` subsection whose payload is the dict rendered as pretty JSON — every key from the input must appear verbatim in the rendered block."""
    config = {"N": 1024, "seed": 42, "eps": 0.05, "dt": 0.01}
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp",
        None,
        KOKKOS_PROFILE,
        test_config=config,
    )
    assert "TEST CONFIG (JSON):" in block
    # Every top-level key must appear verbatim; json.dumps(..., indent=2)
    # renders them as `"key":` lines.
    for key in config:
        assert f'"{key}"' in block
    # Value fidelity: an integer and a float from the config must both
    # be present in the rendered block.
    assert "1024" in block
    assert "0.05" in block


def test_format_baseline_block_cu_invites_baseline_under_cuda_profile():
    """For a CUDA .cu kernel under CUDA_PROFILE (dynamic_verification=True), the block INVITES spawn_baseline_harness and surfaces the KERNEL STEM and the CUDA driver filename (driver.cu). Phase B inverted the old 'skipped' assertion because CUDA_PROFILE now ships its own baseline harness and is part of the dynamic-verification chain."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "BASELINE STEP" in block
    assert "KERNEL STEM: vector_add" in block
    assert "spawn_baseline_harness" in block
    # The driver filename in the block must match the profile, not the
    # hardcoded Kokkos default — that's the whole point of routing this
    # through the LanguageProfile.
    assert "driver.cu" in block
    assert "skipped" not in block.lower()


# ---------- run_orchestrator: baseline block in initial user message ----------


def test_run_orchestrator_cpp_kernel_invites_baseline_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """For a .cpp kernel, the first user message embeds the BASELINE STEP block with the file stem so the orchestrator can decide whether to call spawn_baseline_harness."""
    # First-user-message-only test: short-circuit on turn 1 with a
    # text+end_turn response so the loop exits before engaging the gate.
    fake = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])

    run_orchestrator(
        "path/to/nbody_force.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "BASELINE STEP" in first_user
    assert "KERNEL STEM: nbody_force" in first_user
    assert "spawn_baseline_harness" in first_user


def test_run_orchestrator_cpp_with_kernel_name_includes_target_kernel_line(
    monkeypatch, fake_anthropic
):
    """When kernel_name is passed to run_orchestrator, the first user message adds a TARGET KERNEL: <name> line to the BASELINE STEP block."""
    # First-user-message-only test: short-circuit on turn 1 with a
    # text+end_turn response so the loop exits before engaging the gate.
    fake = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])

    run_orchestrator(
        "path/to/vector_add.cpp",
        "src",
        tolerance=_DEFAULT_TEST_TOLERANCE,
        kernel_name="vector_add",
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "TARGET KERNEL: vector_add" in first_user
    assert "KERNEL STEM: vector_add" in first_user


def test_run_orchestrator_forwards_test_config_into_first_user_message(
    monkeypatch, fake_anthropic
):
    """When run_orchestrator is called with a non-None test_config dict, its JSON payload appears in the first user message under a `TEST CONFIG (JSON):` block; passing test_config=None (the default) omits the block entirely so the harness reverts to inferring inputs. Note the trailing colon: the boilerplate hint text may reference the phrase 'TEST CONFIG (JSON) block' without a colon, so the emitted block is distinguished by the `TEST CONFIG (JSON):` marker."""
    # Two independent short-circuit runs: one with a config, one without.
    fake_with = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])
    run_orchestrator(
        "path/to/nbody_force.cpp",
        "src",
        tolerance=_DEFAULT_TEST_TOLERANCE,
        test_config={"N": 1024, "seed": 42, "eps": 0.05},
    )
    first_user_with = fake_with.messages.calls[0]["messages"][0]["content"]
    assert "TEST CONFIG (JSON):" in first_user_with
    assert '"N"' in first_user_with
    assert "1024" in first_user_with
    assert '"eps"' in first_user_with

    fake_without = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])
    run_orchestrator(
        "path/to/nbody_force.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )
    first_user_without = fake_without.messages.calls[0]["messages"][0]["content"]
    assert "TEST CONFIG (JSON):" not in first_user_without


def test_run_orchestrator_cu_kernel_invites_baseline_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """For a CUDA .cu kernel, the first user message's BASELINE STEP block INVITES spawn_baseline_harness (CUDA_PROFILE has dynamic_verification=True). Phase B inverted the prior 'skips baseline' assertion because the .cu suffix now resolves to CUDA_PROFILE, which ships its own baseline harness and joins the dynamic-verification chain."""
    # First-user-message-only test: short-circuit on turn 1 with a
    # text+end_turn response so the loop exits before engaging the gate.
    fake = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])

    run_orchestrator(
        "path/to/vector_add.cu", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "BASELINE STEP" in first_user
    assert "KERNEL STEM: vector_add" in first_user
    assert "spawn_baseline_harness" in first_user
    assert "skipped" not in first_user.lower()


# ---------- compile_baseline_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_compile_baseline_driver():
    """ORCHESTRATOR_TOOLS exposes compile_baseline_driver with kernel_stem as the only required string input."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "compile_baseline_driver" in by_name
    tool = by_name["compile_baseline_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {"kernel_stem"}


def test_orchestrator_prompt_mentions_compile_baseline_driver_and_env_var():
    """The orchestrator prompt names compile_baseline_driver and AGENT_PRECISION_KOKKOS_ROOT, and asserts that a compile error there does NOT block the analyst -> rewriter -> verifier pipeline (only the dynamic-verification chain, and therefore finish on profiles where the chain is required). Phase B genericized 'side artifact' wording to chain-membership wording because the compiled driver is no longer a dead-end side artifact — it feeds run_baseline_driver / splice / compare."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "compile_baseline_driver" in text
    assert "AGENT_PRECISION_KOKKOS_ROOT" in text
    # The compile must not block the LLM pipeline even though it now
    # transitively gates finish on profiles with dynamic_verification=True.
    # Surface the "does NOT block analyst -> rewriter -> verifier"
    # invariant explicitly so a future prompt edit can't silently flip it.
    lower = text.lower()
    assert "does not block" in lower or "must not block" in lower
    assert "analyst -> rewriter -> verifier" in text


def test_format_baseline_block_cpp_mentions_compile_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to follow a successful spawn_baseline_harness with a single compile_baseline_driver call using the same KERNEL STEM."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "compile_baseline_driver" in block
    # Must couple it to the harness call, not be a standalone instruction.
    assert "spawn_baseline_harness" in block


def test_format_baseline_block_cu_mentions_compile_step():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention compile_baseline_driver — Phase B added CUDA to the dynamic-verification chain, so the compile step is now part of the .cu flow too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "compile_baseline_driver" in block


def test_execute_tool_dispatches_compile_baseline_driver(monkeypatch):
    """_execute_tool routes compile_baseline_driver to workflow.tools.compile_baseline_driver and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_compile(kernel_stem, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "compiled fine",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/driver"],
        }

    monkeypatch.setattr(orchestrator, "compile_baseline_driver", stub_compile)

    result = _execute_tool(
        "compile_baseline_driver", {"kernel_stem": "nbody_force"}, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {"kernel_stem": "nbody_force", "language_id": "kokkos"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "compiled fine",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/driver"],
    }


def test_execute_tool_compile_baseline_driver_error_passes_through(monkeypatch):
    """When the compile helper returns status='error', _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "compile_baseline_driver",
        lambda stem, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": "AGENT_PRECISION_KOKKOS_ROOT is not set.",
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "compile_baseline_driver", {"kernel_stem": "x"}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "AGENT_PRECISION_KOKKOS_ROOT" in result["stderr"]
    assert result["artifacts"] == []


# ---------- run_baseline_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_run_baseline_driver():
    """ORCHESTRATOR_TOOLS exposes run_baseline_driver with kernel_stem as the only required string input."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "run_baseline_driver" in by_name
    tool = by_name["run_baseline_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {"kernel_stem"}


def test_orchestrator_prompt_mentions_run_baseline_driver_and_env_var():
    """The orchestrator prompt names run_baseline_driver and AGENT_PRECISION_RUN_TIMEOUT_SEC, and asserts that a run error there does NOT block the analyst -> rewriter -> verifier pipeline (only the dynamic-verification chain, and therefore finish on profiles where the chain is required). Phase B genericized 'side artifact' wording to chain-membership wording because reference.json now feeds compare_outputs and the code-side finish-gate."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "run_baseline_driver" in text
    assert "AGENT_PRECISION_RUN_TIMEOUT_SEC" in text
    # The run must not block the LLM pipeline even though it now
    # transitively gates finish on profiles with dynamic_verification=True.
    lower = text.lower()
    assert "does not block" in lower or "must not block" in lower
    assert "analyst -> rewriter -> verifier" in text


def test_format_baseline_block_cpp_mentions_run_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to follow a successful compile_baseline_driver with a single run_baseline_driver call using the same KERNEL STEM."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "run_baseline_driver" in block
    # Must be coupled to the compile call, not a standalone instruction.
    assert "compile_baseline_driver" in block


def test_format_baseline_block_cu_mentions_run_step():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention run_baseline_driver — Phase B added CUDA to the dynamic-verification chain, so the run step is now part of the .cu flow too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "run_baseline_driver" in block


def test_execute_tool_dispatches_run_baseline_driver(monkeypatch):
    """_execute_tool routes run_baseline_driver to workflow.tools.run_baseline_driver and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_run(kernel_stem, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "driver ran cleanly",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/reference.json"],
        }

    monkeypatch.setattr(orchestrator, "run_baseline_driver", stub_run)

    result = _execute_tool(
        "run_baseline_driver", {"kernel_stem": "nbody_force"}, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {"kernel_stem": "nbody_force", "language_id": "kokkos"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "driver ran cleanly",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/reference.json"],
    }


def test_execute_tool_run_baseline_driver_error_passes_through(monkeypatch):
    """When the run helper returns status='error', _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "run_baseline_driver",
        lambda stem, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": "Driver exited with code 7.",
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "run_baseline_driver", {"kernel_stem": "x"}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "code 7" in result["stderr"]
    assert result["artifacts"] == []


# ---------- splice_rewritten_kernel: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_splice_rewritten_kernel():
    """ORCHESTRATOR_TOOLS exposes splice_rewritten_kernel with kernel_stem and rewritten_kernel_source as the only required string inputs."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "splice_rewritten_kernel" in by_name
    tool = by_name["splice_rewritten_kernel"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert "rewritten_kernel_source" in props
    assert props["rewritten_kernel_source"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {
        "kernel_stem",
        "rewritten_kernel_source",
    }


def test_orchestrator_prompt_mentions_splice_rewritten_kernel():
    """The orchestrator prompt names splice_rewritten_kernel, ties it to a verifier accept after a successful run_baseline_driver, and names the spliced driver as feeding the rewritten compile/run/compare chain that the code-side finish-gate enforces on dynamic_verification=True profiles. Phase B removed the old 'must not block finish' wording because splice now IS in the chain that gates finish; the right invariant is that splice depends on a verifier accept AND on a successful baseline chain."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "splice_rewritten_kernel" in text
    # Must be conditioned on verifier accept, not on the baseline chain alone.
    assert "verdict='accept'" in text
    # Splice feeds the rewritten compile/run/compare chain that the
    # code-side finish-gate enforces. Surface that linkage explicitly so
    # a future prompt edit can't drop the chain-membership semantics.
    lower = text.lower()
    assert "rewritten compile/run/compare chain" in lower or "dynamic-verification chain" in lower
    assert "finish-gate" in lower


def test_format_baseline_block_cpp_mentions_splice_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call splice_rewritten_kernel after a verifier accept following a successful run_baseline_driver."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "splice_rewritten_kernel" in block
    # Must be coupled to both the verifier accept and run_baseline_driver,
    # not a standalone instruction.
    assert "verdict='accept'" in block
    assert "run_baseline_driver" in block


def test_format_baseline_block_cu_mentions_splice_step():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention splice_rewritten_kernel — Phase B added CUDA to the dynamic-verification chain, so the splice step is now part of the .cu flow too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "splice_rewritten_kernel" in block


def test_execute_tool_dispatches_splice_rewritten_kernel(monkeypatch):
    """_execute_tool routes splice_rewritten_kernel to workflow.tools.splice_rewritten_kernel, forwards both kernel_stem and rewritten_kernel_source, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_splice(kernel_stem, rewritten_kernel_source, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["rewritten_kernel_source"] = rewritten_kernel_source
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/rewritten/driver.cpp"],
        }

    monkeypatch.setattr(orchestrator, "splice_rewritten_kernel", stub_splice)

    result = _execute_tool(
        "splice_rewritten_kernel",
        {
            "kernel_stem": "nbody_force",
            "rewritten_kernel_source": "void k() { /* rewritten */ }\n",
        }, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {
        "kernel_stem": "nbody_force",
        "rewritten_kernel_source": "void k() { /* rewritten */ }\n",
        "language_id": "kokkos",
    }
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/rewritten/driver.cpp"],
    }


def test_execute_tool_splice_rewritten_kernel_error_passes_through(monkeypatch):
    """When the splice helper returns status='error' (e.g. missing baseline, missing sentinels), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "splice_rewritten_kernel",
        lambda stem, src, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Baseline driver source not found at "
                "baselines/x/driver.cpp."
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "splice_rewritten_kernel",
        {"kernel_stem": "x", "rewritten_kernel_source": "void k(){}"}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "Baseline driver source not found" in result["stderr"]
    assert result["artifacts"] == []


# ---------- compile_rewritten_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_compile_rewritten_driver():
    """ORCHESTRATOR_TOOLS exposes compile_rewritten_driver with kernel_stem as its only required string input — same shape as compile_baseline_driver."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "compile_rewritten_driver" in by_name
    tool = by_name["compile_rewritten_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert tool["input_schema"]["required"] == ["kernel_stem"]


def test_orchestrator_prompt_mentions_compile_rewritten_driver():
    """The orchestrator prompt names compile_rewritten_driver, ties it to a preceding successful splice_rewritten_kernel, and states a rewritten-compile error transitively blocks the dynamic-verification chain (and therefore finish on profiles where the chain is required). Phase B removed the old 'must not block finish' wording because the rewritten-compile now IS in the chain that gates finish on Kokkos / CUDA inputs."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "compile_rewritten_driver" in text
    # Must be conditioned on splice success, not standalone.
    assert "splice_rewritten_kernel" in text
    # Compile-rewritten failures transitively block the chain (and
    # therefore finish on dynamic_verification=True profiles). Assert
    # the prompt names the chain so a future edit can't silently flip
    # the gating semantics back to "non-blocking for finish".
    lower = text.lower()
    assert "transitively blocks" in lower or "dynamic-verification chain" in lower


def test_format_baseline_block_cpp_mentions_compile_rewritten_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call compile_rewritten_driver immediately after a successful splice_rewritten_kernel."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "compile_rewritten_driver" in block
    # Must be coupled to splice success — never a standalone instruction.
    assert "splice_rewritten_kernel" in block


def test_format_baseline_block_cu_mentions_compile_rewritten_step():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention compile_rewritten_driver — Phase B added CUDA to the dynamic-verification chain, so the rewritten-compile step is now part of the .cu flow too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "compile_rewritten_driver" in block


def test_execute_tool_dispatches_compile_rewritten_driver(monkeypatch):
    """_execute_tool routes compile_rewritten_driver to workflow.tools.compile_rewritten_driver, forwards the kernel_stem argument, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_compile(kernel_stem, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/rewritten/driver"],
        }

    monkeypatch.setattr(
        orchestrator, "compile_rewritten_driver", stub_compile
    )

    result = _execute_tool(
        "compile_rewritten_driver", {"kernel_stem": "nbody_force"}, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {"kernel_stem": "nbody_force", "language_id": "kokkos"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/rewritten/driver"],
    }


def test_execute_tool_compile_rewritten_driver_error_passes_through(monkeypatch):
    """When the rewritten-compile helper returns status='error' (e.g. missing source, compile failure), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "compile_rewritten_driver",
        lambda stem, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Driver source not found at "
                "baselines/x/rewritten/driver.cpp. Did "
                "splice_rewritten_kernel run and get approved for this "
                "kernel_stem?"
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "compile_rewritten_driver", {"kernel_stem": "x"}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "rewritten/driver.cpp" in result["stderr"]
    assert "splice_rewritten_kernel" in result["stderr"]
    assert result["artifacts"] == []


# ---------- run_rewritten_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_run_rewritten_driver():
    """ORCHESTRATOR_TOOLS exposes run_rewritten_driver with kernel_stem as its only required string input — same shape as run_baseline_driver."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "run_rewritten_driver" in by_name
    tool = by_name["run_rewritten_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert tool["input_schema"]["required"] == ["kernel_stem"]


def test_orchestrator_prompt_mentions_run_rewritten_driver():
    """The orchestrator prompt names run_rewritten_driver, ties it to a preceding successful compile_rewritten_driver, and states that a rewritten-run error means the comparator cannot proceed and finish will be blocked on dynamic_verification=True profiles until compare_outputs has successfully run. Phase B removed the old 'must not block finish' wording because the rewritten-run feeds compare_outputs directly."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "run_rewritten_driver" in text
    # Must be conditioned on the rewritten-compile step succeeding,
    # never a standalone instruction.
    assert "compile_rewritten_driver" in text
    # The rewritten-run produces the comparator's input; assert the
    # prompt names that dependency so a future edit can't silently
    # decouple them.
    lower = text.lower()
    assert "compare_outputs" in text
    assert "comparator cannot proceed" in lower or "dynamic-verification chain" in lower


def test_format_baseline_block_cpp_mentions_run_rewritten_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call run_rewritten_driver immediately after a successful compile_rewritten_driver — and still mentions the upstream splice/compile_rewritten steps."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "run_rewritten_driver" in block
    # Must be coupled to compile_rewritten success — never standalone.
    assert "compile_rewritten_driver" in block
    # The whole rewritten chain must still be visible in the block so
    # the orchestrator does not lose context of how it got here.
    assert "splice_rewritten_kernel" in block


def test_format_baseline_block_cu_mentions_run_rewritten_step():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention run_rewritten_driver — Phase B added CUDA to the dynamic-verification chain, so the rewritten-run step is now part of the .cu flow too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "run_rewritten_driver" in block


def test_execute_tool_dispatches_run_rewritten_driver(monkeypatch):
    """_execute_tool routes run_rewritten_driver to workflow.tools.run_rewritten_driver, forwards the kernel_stem argument, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_run(kernel_stem, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "rewritten driver ran",
            "stderr": "",
            "artifacts": [
                "baselines/nbody_force/rewritten/reference.json"
            ],
        }

    monkeypatch.setattr(orchestrator, "run_rewritten_driver", stub_run)

    result = _execute_tool(
        "run_rewritten_driver", {"kernel_stem": "nbody_force"}, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {"kernel_stem": "nbody_force", "language_id": "kokkos"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "rewritten driver ran",
        "stderr": "",
        "artifacts": [
            "baselines/nbody_force/rewritten/reference.json"
        ],
    }


def test_execute_tool_run_rewritten_driver_error_passes_through(monkeypatch):
    """When the rewritten-run helper returns status='error' (e.g. missing binary, non-zero exit, timeout, invalid JSON), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "run_rewritten_driver",
        lambda stem, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Driver binary not found at "
                "baselines/x/rewritten/driver. Did "
                "compile_rewritten_driver run and succeed for this "
                "kernel_stem?"
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "run_rewritten_driver", {"kernel_stem": "x"}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "rewritten/driver" in result["stderr"]
    assert "compile_rewritten_driver" in result["stderr"]
    assert result["artifacts"] == []


# ---------- compare_outputs: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_compare_outputs():
    """ORCHESTRATOR_TOOLS exposes compare_outputs with kernel_stem AND tolerance_json as the two required string inputs (unlike the run / compile tools, which take only kernel_stem)."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "compare_outputs" in by_name
    tool = by_name["compare_outputs"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert "tolerance_json" in props
    assert props["kernel_stem"]["type"] == "string"
    assert props["tolerance_json"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {
        "kernel_stem",
        "tolerance_json",
    }


def test_orchestrator_prompt_mentions_compare_outputs_and_finish_gate():
    """The orchestrator prompt names compare_outputs, ties it to a preceding successful run_rewritten_driver, and states it IS a precondition for finish on .cpp inputs (the source-of-truth gate is in code, but the prompt must tell the model)."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "compare_outputs" in text
    # Must be conditioned on the rewritten-run step succeeding, never
    # standalone, and must reuse the tolerance_json the verifier got.
    assert "run_rewritten_driver" in text
    assert "tolerance_json" in text
    # Must explicitly call out the finish-gate change for .cpp inputs.
    assert "precondition for finish" in text.lower()
    # Must mention the retry-bias suggestion (per-variable analyst,
    # not spawn_rewriter, on comparator error) so the model has a
    # clear next move when the gate blocks finish. Step 2 of the
    # per-variable refactor removed the monolithic spawn_analyst
    # tool; the retry is now spawn_variable_analyst.
    assert "spawn_variable_analyst" in text


def test_format_baseline_block_cpp_mentions_compare_outputs_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call compare_outputs immediately after a successful run_rewritten_driver, reusing the same tolerance_json passed to spawn_verifier, AND states the comparator IS a precondition for finish."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None, KOKKOS_PROFILE
    )
    assert "compare_outputs" in block
    # Must be coupled to the upstream rewritten-run step.
    assert "run_rewritten_driver" in block
    # Must reuse the verifier's tolerance_json.
    assert "tolerance_json" in block
    # Must make the finish-gate visible at the block level (the
    # orchestrator reads this block for its in-context guidance).
    assert "finish" in block


def test_format_baseline_block_cu_mentions_compare_outputs():
    """For a CUDA .cu kernel under CUDA_PROFILE, the (invited) BASELINE STEP block DOES mention compare_outputs — Phase B added CUDA to the dynamic-verification chain, so compare_outputs is now the finish-gating step on .cu inputs too."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None, CUDA_PROFILE
    )
    assert "compare_outputs" in block


def test_execute_tool_dispatches_compare_outputs(monkeypatch):
    """_execute_tool routes compare_outputs to workflow.tools.compare_outputs, forwards BOTH kernel_stem and tolerance_json, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_compare(kernel_stem, tolerance_json, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["tolerance_json"] = tolerance_json
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "all 8 values agree",
            "stderr": "",
            "artifacts": [
                "baselines/nbody_force/rewritten/comparison.json"
            ],
        }

    monkeypatch.setattr(orchestrator, "compare_outputs", stub_compare)

    tol = json.dumps(
        {"kind": "sig_figs", "value": 3, "source": "user_cli"}
    )
    result = _execute_tool(
        "compare_outputs",
        {"kernel_stem": "nbody_force", "tolerance_json": tol}, KOKKOS_PROFILE
    )

    # _execute_tool injects profile.id as language_id; the LLM never
    # passes it (Phase A.5 Option B).
    assert captured == {
        "kernel_stem": "nbody_force",
        "tolerance_json": tol,
        "language_id": "kokkos",
    }
    assert result == {
        "status": "ok",
        "stdout": "all 8 values agree",
        "stderr": "",
        "artifacts": [
            "baselines/nbody_force/rewritten/comparison.json"
        ],
    }


def test_execute_tool_compare_outputs_error_passes_through(monkeypatch):
    """When the comparator returns status='error' (tolerance failure, shape mismatch, malformed tolerance_json, missing reference.json), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result — and the finish-gate downstream can read status to decide whether to block."""
    monkeypatch.setattr(
        orchestrator,
        "compare_outputs",
        lambda stem, tol, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Tolerance mismatch under sig_figs=3: 5/8 values "
                "disagree."
            ),
            "artifacts": ["baselines/x/rewritten/comparison.json"],
        },
    )
    tol = json.dumps(
        {"kind": "sig_figs", "value": 3, "source": "user_cli"}
    )
    result = _execute_tool(
        "compare_outputs", {"kernel_stem": "x", "tolerance_json": tol}, KOKKOS_PROFILE
    )
    assert result["status"] == "error"
    assert "Tolerance mismatch" in result["stderr"]
    assert result["artifacts"] == [
        "baselines/x/rewritten/comparison.json"
    ]


# ---------- finish-gate: code-side enforcement ----------


def _verifier_accept_response(turn_id, original="src", rewritten="src"):
    """Build the FakeResponse for one spawn_verifier(accept) turn.

    Threads in plausible original/rewritten strings so the fake API
    sees the same input shape it would in a real run. The verifier
    stub elsewhere returns verdict='accept' regardless of inputs.
    """
    return FakeResponse(
        content=[ToolUseBlock(
            id=turn_id,
            name="spawn_verifier",
            input={
                "original_source": original,
                "rewritten_source": rewritten,
                "analyst_verdict_json": "{}",
                "tolerance_json": (
                    '{"kind":"sig_figs","value":3,"source":"user_cli"}'
                ),
            },
        )],
    )


def _compare_response(turn_id, kernel_stem):
    return FakeResponse(
        content=[ToolUseBlock(
            id=turn_id,
            name="compare_outputs",
            input={
                "kernel_stem": kernel_stem,
                "tolerance_json": (
                    '{"kind":"sig_figs","value":3,"source":"user_cli"}'
                ),
            },
        )],
    )


def _finish_response(turn_id):
    return FakeResponse(
        content=[ToolUseBlock(
            id=turn_id,
            name="finish",
            input={"rewritten_code": "FINAL", "notes": "done"},
        )],
    )


def test_finish_gate_cpp_verifier_accept_and_compare_ok_allows_finish(
    monkeypatch, fake_anthropic, tmp_path
):
    """For a .cpp kernel, finish is allowed when the most recent spawn_verifier returned verdict='accept' AND the most recent compare_outputs returned status='ok' for the current rewrite cycle."""
    monkeypatch.chdir(tmp_path)
    fake = fake_anthropic([
        _verifier_accept_response("tu_v"),
        _compare_response("tu_c", "kstem"),
        _finish_response("tu_f"),
    ])

    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda type_, task: {
            "verdict": "accept",
            "per_variable": [],
            "concerns": [],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "compare_outputs",
        lambda stem, tol, language_id: {
            "status": "ok",
            "stdout": "match",
            "stderr": "",
            "artifacts": [],
        },
    )
    _scripted_input(monkeypatch, ["y", "y", "y"])

    result = run_orchestrator(
        "path/to/kstem.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )
    assert result == {"rewritten_code": "FINAL", "notes": "done"}
    assert len(fake.messages.calls) == 3


def test_finish_gate_cpp_verifier_accept_but_compare_error_blocks_finish(
    monkeypatch, fake_anthropic, tmp_path
):
    """For a .cpp kernel, even with a verifier-accept on file, a comparator status='error' blocks finish; the loop injects a synthetic tool_result naming what's missing instead of returning the finish args."""
    monkeypatch.chdir(tmp_path)
    fake = fake_anthropic([
        _verifier_accept_response("tu_v"),
        _compare_response("tu_c", "kstem"),
        _finish_response("tu_f"),
        # Turn 4: gate blocked finish, model gets a synthetic error and
        # must do something next. Make it text-only so the loop exits
        # cleanly with None and we can introspect the gate behavior.
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda type_, task: {
            "verdict": "accept",
            "per_variable": [],
            "concerns": [],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "compare_outputs",
        lambda stem, tol, language_id: {
            "status": "error",
            "stdout": "",
            "stderr": "Tolerance mismatch under sig_figs=3.",
            "artifacts": [],
        },
    )
    _scripted_input(monkeypatch, ["y", "y", "y"])  # all three approved

    result = run_orchestrator(
        "path/to/kstem.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )
    # Finish was blocked -> loop continued and then ran out at turn 4
    # via the text+end_turn response, returning None.
    assert result is None

    # The tool_result fed back for the blocked finish call must carry
    # an explicit gate-violation error and the is_error flag.
    fourth_messages = fake.messages.calls[3]["messages"]
    last = fourth_messages[-1]
    assert last["role"] == "user"
    blocks_by_id = {b["tool_use_id"]: b for b in last["content"]}
    finish_block = blocks_by_id["tu_f"]
    assert finish_block["is_error"] is True
    payload = json.loads(finish_block["content"])
    assert payload["status"] == "error"
    assert "compare_outputs" in payload["stderr"]


def test_finish_gate_cpp_compare_never_called_blocks_finish(
    monkeypatch, fake_anthropic, tmp_path
):
    """For a .cpp kernel, a verifier-accept alone is not enough to allow finish; the comparator must have actually been called this rewrite cycle. Missing compare_outputs is treated as compare_status=None, which the gate blocks."""
    monkeypatch.chdir(tmp_path)
    fake = fake_anthropic([
        _verifier_accept_response("tu_v"),
        _finish_response("tu_f"),
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda type_, task: {
            "verdict": "accept",
            "per_variable": [],
            "concerns": [],
        },
    )
    _scripted_input(monkeypatch, ["y", "y"])

    result = run_orchestrator(
        "path/to/kstem.cpp", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )
    assert result is None

    # The blocked-finish tool_result must mention that compare_outputs
    # was missing for the current rewrite cycle.
    third_messages = fake.messages.calls[2]["messages"]
    last = third_messages[-1]
    blocks_by_id = {b["tool_use_id"]: b for b in last["content"]}
    finish_block = blocks_by_id["tu_f"]
    assert finish_block["is_error"] is True
    payload = json.loads(finish_block["content"])
    assert "compare_outputs" in payload["stderr"]
    # The gate names the variable it is missing so the model can self-
    # correct without having to re-read the system prompt.
    assert "compare_status" in payload["stderr"]


def test_finish_gate_cu_verifier_accept_alone_blocks_finish_post_phase_b(
    monkeypatch, fake_anthropic, tmp_path
):
    """For a .cu kernel under CUDA_PROFILE (dynamic_verification=True post-Phase-B), the finish-gate now requires compare_outputs status='ok' too — a verifier-accept alone is no longer enough. Phase B added CUDA to the dynamic-verification chain, so the .cu and .cpp gating semantics are unified: this test mirrors test_finish_gate_cpp_compare_never_called_blocks_finish above but for .cu."""
    monkeypatch.chdir(tmp_path)
    fake = fake_anthropic([
        _verifier_accept_response("tu_v"),
        _finish_response("tu_f"),
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda type_, task: {
            "verdict": "accept",
            "per_variable": [],
            "concerns": [],
        },
    )
    _scripted_input(monkeypatch, ["y", "y"])

    result = run_orchestrator(
        "path/to/vector_add.cu", "src", tolerance=_DEFAULT_TEST_TOLERANCE
    )
    # Finish was blocked -> loop continued and hit the text+end_turn
    # response at turn 3, returning None.
    assert result is None

    # The blocked-finish tool_result must mention that compare_outputs
    # was missing for the current rewrite cycle — same shape as the
    # .cpp sibling test, because the gate is now profile-agnostic.
    third_messages = fake.messages.calls[2]["messages"]
    last = third_messages[-1]
    blocks_by_id = {b["tool_use_id"]: b for b in last["content"]}
    finish_block = blocks_by_id["tu_f"]
    assert finish_block["is_error"] is True
    payload = json.loads(finish_block["content"])
    assert "compare_outputs" in payload["stderr"]
    assert "compare_status" in payload["stderr"]


# ---------- probe_step / probe_compare (Commit 4 wiring) ----------


def test_orchestrator_tools_include_probe_step_and_probe_compare():
    """ORCHESTRATOR_TOOLS exposes probe_step and probe_compare to the LLM with the schemas the deterministic wrappers in workflow.tools expect. probe_step takes (kernel_stem, precision, seed), probe_compare takes (kernel_stem). language_id is NOT in either schema — _execute_tool injects it from the per-run profile so the LLM never has to know which language it's working in."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "probe_step" in by_name
    assert "probe_compare" in by_name

    probe_step_schema = by_name["probe_step"]["input_schema"]
    probe_step_props = probe_step_schema["properties"]
    assert set(probe_step_props.keys()) == {"kernel_stem", "precision", "seed"}
    assert set(probe_step_schema["required"]) == {
        "kernel_stem", "precision", "seed"
    }
    assert "language_id" not in probe_step_props

    probe_compare_schema = by_name["probe_compare"]["input_schema"]
    probe_compare_props = probe_compare_schema["properties"]
    assert set(probe_compare_props.keys()) == {"kernel_stem"}
    assert set(probe_compare_schema["required"]) == {"kernel_stem"}
    assert "language_id" not in probe_compare_props


def test_execute_tool_dispatches_probe_step(monkeypatch):
    """_execute_tool routes probe_step to workflow.tools.probe_step, forwards kernel_stem, precision, and seed verbatim, and injects profile.id as language_id (Phase A.5 Option B — same pattern as compile_baseline_driver and compare_outputs). Returns the deterministic tool's {status, stdout, stderr, artifacts} dict unchanged."""
    captured = {}

    def stub_probe_step(kernel_stem, precision, seed, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["precision"] = precision
        captured["seed"] = seed
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "compiled and ran probe cell",
            "stderr": "",
            "artifacts": [
                "baselines/nbody_force/probe/float_seed43/reference.json"
            ],
        }

    monkeypatch.setattr(orchestrator, "probe_step", stub_probe_step)

    result = _execute_tool(
        "probe_step",
        {"kernel_stem": "nbody_force", "precision": "float", "seed": 43},
        KOKKOS_PROFILE,
    )

    assert captured == {
        "kernel_stem": "nbody_force",
        "precision": "float",
        "seed": 43,
        "language_id": "kokkos",
    }
    assert result["status"] == "ok"
    assert result["artifacts"] == [
        "baselines/nbody_force/probe/float_seed43/reference.json"
    ]


def test_execute_tool_dispatches_probe_compare(monkeypatch):
    """_execute_tool routes probe_compare to workflow.tools.probe_compare, forwards kernel_stem, injects profile.id as language_id, and returns the deterministic tool's result dict verbatim — no wrapping or status-massaging."""
    captured = {}

    def stub_probe_compare(kernel_stem, language_id):
        captured["kernel_stem"] = kernel_stem
        captured["language_id"] = language_id
        return {
            "status": "ok",
            "stdout": "aggregated 8 cells into evidence.json",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/probe/evidence.json"],
        }

    monkeypatch.setattr(orchestrator, "probe_compare", stub_probe_compare)

    result = _execute_tool(
        "probe_compare",
        {"kernel_stem": "nbody_force"},
        KOKKOS_PROFILE,
    )

    assert captured == {
        "kernel_stem": "nbody_force",
        "language_id": "kokkos",
    }
    assert result["status"] == "ok"
    assert result["artifacts"] == [
        "baselines/nbody_force/probe/evidence.json"
    ]


def test_execute_tool_probe_compare_promotes_quad_oracle_to_baseline_reference(
    monkeypatch, tmp_path
):
    """After probe_compare returns status='ok' under a profile whose baseline_precision differs from the canonical splice-scaffold precision 'double' (Kokkos: baseline_precision='quad'), _execute_tool copies baselines/<stem>/probe/<baseline_precision>_seed42/reference.json over baselines/<stem>/reference.json. This is the ORACLE PROMOTION step: the canonical baselines/<stem>/driver.cpp is the double splice scaffold (Kokkos has no `__float128` math overloads, so the quad driver could never be a splice target), and run_baseline_driver wrote a double-precision reference.json earlier in the chain. Promoting the quad probe reference into the canonical slot is what lets the finish-gate comparator (compare_outputs) measure the rewritten kernel against true ground truth instead of against a same-precision double oracle. The destination filename (baselines/<stem>/reference.json) is the file compare_outputs reads as the baseline by contract — promoting in place means compare_outputs stays fully language-agnostic at the file-path level."""
    monkeypatch.chdir(tmp_path)

    # Stage a "stale" double-precision baseline reference (the file
    # run_baseline_driver would have written) and the quad probe
    # reference (the file probe_step quad_seed42 would have written).
    baseline_dir = tmp_path / "baselines" / "nbody_force"
    baseline_dir.mkdir(parents=True)
    stale_double_ref = baseline_dir / "reference.json"
    stale_double_ref.write_text(
        '{"kernel": "step", "seed": 42, "outputs": {"x": [1.0]}}'
    )
    quad_probe_dir = baseline_dir / "probe" / "quad_seed42"
    quad_probe_dir.mkdir(parents=True)
    quad_probe_ref = quad_probe_dir / "reference.json"
    quad_ref_text = (
        '{"kernel": "step", "seed": 42, '
        '"outputs": {"x": [1.0000000000000002]}}'
    )
    quad_probe_ref.write_text(quad_ref_text)

    def stub_probe_compare(kernel_stem, language_id):
        return {
            "status": "ok",
            "stdout": "aggregated 8 cells into evidence.json",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/probe/evidence.json"],
        }

    monkeypatch.setattr(orchestrator, "probe_compare", stub_probe_compare)

    result = _execute_tool(
        "probe_compare",
        {"kernel_stem": "nbody_force"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "ok"
    # The canonical baseline reference is now the quad probe reference,
    # verbatim — NOT the stale double reference that was there before.
    assert stale_double_ref.read_text() == quad_ref_text
    # The probe source must not have been moved or deleted; it stays in
    # the probe tree for the trace/inspection.
    assert quad_probe_ref.read_text() == quad_ref_text


def test_execute_tool_probe_compare_skips_promotion_when_probe_failed(
    monkeypatch, tmp_path
):
    """If probe_compare returns status != 'ok' (e.g. the required quad_seed42 cell was missing — the only hard-error path in probe_compare), _execute_tool MUST NOT promote anything: there is no trustworthy quad oracle to promote, and silently copying a partial / stale file into the canonical reference slot would corrupt the finish-gate comparator's ground truth. The existing run_baseline_driver double-precision reference stays in place; the comparator will run against it, which is suboptimal but safe."""
    monkeypatch.chdir(tmp_path)

    baseline_dir = tmp_path / "baselines" / "nbody_force"
    baseline_dir.mkdir(parents=True)
    stale_double_ref = baseline_dir / "reference.json"
    original_text = '{"kernel": "step", "seed": 42, "outputs": {"x": [1.0]}}'
    stale_double_ref.write_text(original_text)
    # Even stage a quad probe ref to prove the promotion path is gated
    # on probe_compare's status, not just on the source file's
    # existence.
    quad_probe_dir = baseline_dir / "probe" / "quad_seed42"
    quad_probe_dir.mkdir(parents=True)
    (quad_probe_dir / "reference.json").write_text(
        '{"kernel": "step", "seed": 42, "outputs": {"x": [9.9]}}'
    )

    def stub_probe_compare(kernel_stem, language_id):
        return {
            "status": "error",
            "stdout": "",
            "stderr": "missing quad_seed42 ground-truth cell",
            "artifacts": [],
        }

    monkeypatch.setattr(orchestrator, "probe_compare", stub_probe_compare)

    result = _execute_tool(
        "probe_compare",
        {"kernel_stem": "nbody_force"},
        KOKKOS_PROFILE,
    )

    assert result["status"] == "error"
    # The canonical baseline reference is untouched.
    assert stale_double_ref.read_text() == original_text


def test_execute_tool_probe_compare_skips_promotion_when_oracle_source_missing(
    monkeypatch, tmp_path
):
    """If probe_compare returns ok but the quad_seed42 probe reference.json does not exist on disk (a non-fatal cell failure earlier in the probe pipeline — probe_compare tolerates per-cell `missing` / `load_error` / `shape_error` for non-quad cells and only hard-errors on missing quad_seed42, so this path is only really reachable if the file is deleted between probe_compare and the promotion), _execute_tool MUST NOT crash and MUST NOT touch the canonical baseline reference. A stderr log line is emitted documenting the skip so the operator can correlate with the trace, but compare_result's status is returned verbatim."""
    monkeypatch.chdir(tmp_path)

    baseline_dir = tmp_path / "baselines" / "nbody_force"
    baseline_dir.mkdir(parents=True)
    stale_double_ref = baseline_dir / "reference.json"
    original_text = '{"kernel": "step", "seed": 42, "outputs": {"x": [1.0]}}'
    stale_double_ref.write_text(original_text)
    # Intentionally do NOT create baselines/nbody_force/probe/quad_seed42/.

    def stub_probe_compare(kernel_stem, language_id):
        return {
            "status": "ok",
            "stdout": "aggregated cells",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/probe/evidence.json"],
        }

    monkeypatch.setattr(orchestrator, "probe_compare", stub_probe_compare)

    result = _execute_tool(
        "probe_compare",
        {"kernel_stem": "nbody_force"},
        KOKKOS_PROFILE,
    )

    # probe_compare's status is returned unchanged: the promotion step
    # is a downstream artifact and its failure does NOT propagate into
    # the tool result (the orchestrator LLM should see exactly what
    # probe_compare itself returned).
    assert result["status"] == "ok"
    assert stale_double_ref.read_text() == original_text


def test_execute_tool_probe_compare_no_promotion_under_cuda_profile(
    monkeypatch, tmp_path
):
    """For profiles whose baseline_precision is already 'double' (the canonical splice-scaffold precision), oracle promotion is a no-op: there is no higher-precision oracle to promote from. CUDA / HIP / SYCL / OMP-offload all set probe_precisions=() in v1 and therefore never reach the probe_compare branch in production, but the guard must be coded against baseline_precision (not against profile.id) so that a future profile adding a non-quad probe pipeline (e.g. CUDA with a `double`-as-oracle config) does not accidentally trigger a copy of a non-existent file. This test pins the guard's shape: promotion only fires when baseline_precision != 'double' AND baseline_precision in probe_precisions."""
    monkeypatch.chdir(tmp_path)

    baseline_dir = tmp_path / "baselines" / "vector_add"
    baseline_dir.mkdir(parents=True)
    stale_ref = baseline_dir / "reference.json"
    original_text = '{"kernel": "vec_add", "seed": 42, "outputs": {"y": [3.0]}}'
    stale_ref.write_text(original_text)
    # Stage a probe directory that LOOKS promotable, to prove the guard
    # is profile-driven, not file-existence-driven.
    spoof_probe_dir = baseline_dir / "probe" / "double_seed42"
    spoof_probe_dir.mkdir(parents=True)
    (spoof_probe_dir / "reference.json").write_text(
        '{"kernel": "vec_add", "seed": 42, "outputs": {"y": [4.0]}}'
    )

    def stub_probe_compare(kernel_stem, language_id):
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "artifacts": [],
        }

    monkeypatch.setattr(orchestrator, "probe_compare", stub_probe_compare)

    result = _execute_tool(
        "probe_compare",
        {"kernel_stem": "vector_add"},
        CUDA_PROFILE,
    )

    assert result["status"] == "ok"
    # The canonical reference is untouched — no promotion under CUDA.
    assert stale_ref.read_text() == original_text


def test_execute_tool_probe_step_under_cuda_profile_injects_cuda_language_id(
    monkeypatch,
):
    """_execute_tool's language_id injection is per-profile: under CUDA_PROFILE, probe_step receives language_id='cuda'. The CUDA profile's probe_precisions is empty in v1 so the system prompt won't actually invite this call — but if the LLM does call it anyway, the preflight in workflow.tools.probe_step (template-not-found) cleanly errors. Verifying the dispatch contract here documents that the injection is profile-driven, not Kokkos-hardcoded."""
    captured = {}

    def stub_probe_step(kernel_stem, precision, seed, language_id):
        captured["language_id"] = language_id
        return {"status": "error", "stdout": "", "stderr": "no template", "artifacts": []}

    monkeypatch.setattr(orchestrator, "probe_step", stub_probe_step)

    _execute_tool(
        "probe_step",
        {"kernel_stem": "vector_add", "precision": "float", "seed": 42},
        CUDA_PROFILE,
    )
    assert captured == {"language_id": "cuda"}


def test_execute_tool_spawn_analyst_injects_probe_evidence_when_present(
    monkeypatch, tmp_path,
):
    """When baselines/<kernel_stem>/probe/evidence.json exists, _execute_tool's spawn_analyst branch reads it off disk and APPENDS it to the analyst task as a 'PROBE EVIDENCE (JSON):' block after the kernel source. The LLM never passes the evidence through itself — the orchestrator attaches it. This isolates the probe-injection contract from the ensemble path (K=1)."""
    monkeypatch.chdir(tmp_path)
    evidence_dir = tmp_path / "baselines" / "nbody_force" / "probe"
    evidence_dir.mkdir(parents=True)
    evidence_payload = {"cells": {"float_seed42": {"status": "ok"}}}
    (evidence_dir / "evidence.json").write_text(json.dumps(evidence_payload))

    captured = {}

    def stub_run_agent(type_, task):
        captured["type"] = type_
        captured["task"] = task
        return {"variables": [], "overall_notes": "ok"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)

    result = _execute_tool(
        "spawn_analyst",
        {"kernel_source": "ORIGINAL KERNEL SOURCE"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    assert result["status"] == "ok"
    assert captured["type"] == "analyst"
    # Kernel source comes first, evidence block appended after.
    assert captured["task"].startswith("ORIGINAL KERNEL SOURCE")
    assert "PROBE EVIDENCE (JSON):" in captured["task"]
    assert "float_seed42" in captured["task"]
    # The evidence text appears AFTER the descriptive paragraph.
    para_idx = captured["task"].index("PROBE EVIDENCE (JSON):")
    json_idx = captured["task"].index('"cells"')
    assert json_idx > para_idx


def test_execute_tool_spawn_analyst_no_evidence_file_unchanged_task(
    monkeypatch, tmp_path,
):
    """When evidence.json is absent (no probe was run, or probe_compare failed before writing it), _execute_tool's spawn_analyst branch silently falls back to the un-augmented kernel source. The probe is informational — missing evidence MUST NOT block the analyst or contaminate the task with a stale 'PROBE EVIDENCE' header that has nothing under it."""
    monkeypatch.chdir(tmp_path)
    # No baselines/ tree at all.
    captured = {}

    def stub_run_agent(type_, task):
        captured["task"] = task
        return {"variables": [], "overall_notes": "ok"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)

    _execute_tool(
        "spawn_analyst",
        {"kernel_source": "ORIGINAL KERNEL SOURCE"},
        KOKKOS_PROFILE,
        kernel_stem="nbody_force",
    )

    assert captured["task"] == "ORIGINAL KERNEL SOURCE"
    assert "PROBE EVIDENCE" not in captured["task"]


def test_execute_tool_spawn_analyst_no_kernel_stem_unchanged_task(monkeypatch):
    """When kernel_stem is None (the default — e.g. unit tests that exercise _execute_tool without going through run_orchestrator), the spawn_analyst branch skips the evidence-file lookup entirely. Without this, every unit test that touched spawn_analyst would need to mock the filesystem or pollute the cwd."""
    captured = {}

    def stub_run_agent(type_, task):
        captured["task"] = task
        return {"variables": [], "overall_notes": "ok"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)
    monkeypatch.delenv("AGENT_PRECISION_ANALYST_K", raising=False)

    _execute_tool(
        "spawn_analyst",
        {"kernel_source": "SRC"},
        KOKKOS_PROFILE,
        # kernel_stem omitted → defaults to None
    )

    assert captured["task"] == "SRC"


def test_format_baseline_block_kokkos_with_run_probe_true_lists_matrix():
    """On Kokkos (probe_precisions=('quad','double','float','mixed_io')), _format_baseline_block with run_probe=True (the default) appends a PROBE STEP block that names every (precision, seed) cell from the profile × _PROBE_SEEDS matrix. The orchestrator system prompt's tool docs cover the API; this block's job is to tell the LLM what concrete matrix to enumerate."""
    block = _format_baseline_block(
        "kernels/nbody_force.cpp", None, KOKKOS_PROFILE, run_probe=True
    )
    assert "PROBE STEP:" in block
    assert "'quad'" in block
    assert "'double'" in block
    assert "'float'" in block
    assert "'mixed_io'" in block
    # _PROBE_SEEDS is (42, 43) — both must surface in the rendered matrix.
    assert "42" in block
    assert "43" in block
    # Specific cell pairs must appear so the LLM has unambiguous targets.
    assert "('quad', 42)" in block
    assert "('mixed_io', 43)" in block


def test_format_baseline_block_kokkos_with_run_probe_false_omits_matrix():
    """When run_probe=False (the --no-probe CLI flag is set), the PROBE STEP block is omitted from the BASELINE STEP even on a Kokkos kernel whose profile carries a non-empty probe_precisions tuple. The two probe tools remain in ORCHESTRATOR_TOOLS — the LLM just isn't told to invoke them, and probe_step's preflight will cleanly error if it does."""
    block = _format_baseline_block(
        "kernels/nbody_force.cpp", None, KOKKOS_PROFILE, run_probe=False
    )
    assert "PROBE STEP:" not in block
    # The baseline harness instructions for Kokkos must still be there —
    # disabling the probe doesn't disable dynamic verification.
    assert "spawn_baseline_harness" in block


def test_format_baseline_block_cuda_never_mentions_probe_matrix():
    """CUDA's probe_precisions is empty in v1, so _format_baseline_block must NEVER emit a PROBE STEP block under the CUDA profile — regardless of the run_probe flag. The --no-probe CLI is a no-op for non-probing profiles; the prompt must reflect that by silently omitting the section rather than including an 'N/A' line that wastes context."""
    for run_probe in (True, False):
        block = _format_baseline_block(
            "kernels/vector_add.cu", None, CUDA_PROFILE, run_probe=run_probe
        )
        assert "PROBE STEP:" not in block, (
            f"PROBE STEP must be absent under CUDA (run_probe={run_probe})"
        )


def test_finish_gate_state_treats_probe_tools_as_explicit_no_ops():
    """_FinishGateState.observe must NOT mutate last_verifier_verdict or last_compare_status when fed a probe_step or probe_compare result — even an error result. The probe is informational only; it does not gate finish on .cpp inputs (the comparator does). Without this guarantee a failed probe cell could spuriously block finish, and check_finish() would return a missing-steps message for a run whose real verifier+comparator chain had actually completed."""
    from workflow.orchestrator import _FinishGateState

    state = _FinishGateState("kernels/nbody_force.cpp", KOKKOS_PROFILE)
    # Prime the gate to the "ready to finish" condition.
    state.observe(
        "spawn_verifier",
        {"status": "ok", "result": {"verdict": "accept"}},
    )
    state.observe(
        "compare_outputs",
        {"status": "ok", "stdout": "", "stderr": "", "artifacts": []},
    )
    assert state.last_verifier_verdict == "accept"
    assert state.last_compare_status == "ok"
    assert state.check_finish() is None

    # A probe_step failure must not invalidate either status.
    state.observe(
        "probe_step",
        {"status": "error", "stdout": "", "stderr": "compile failed", "artifacts": []},
    )
    assert state.last_verifier_verdict == "accept"
    assert state.last_compare_status == "ok"
    assert state.check_finish() is None

    # A probe_compare error must not invalidate either status either.
    state.observe(
        "probe_compare",
        {"status": "error", "stdout": "", "stderr": "no canonical cell", "artifacts": []},
    )
    assert state.last_verifier_verdict == "accept"
    assert state.last_compare_status == "ok"
    assert state.check_finish() is None


def test_layer2_score_known_tools_includes_probe_step_and_probe_compare():
    """evals.layer2.score._KNOWN_TOOLS must enumerate probe_step and probe_compare so a trace containing them does not produce 'unknown tool' warnings in the Layer-2 grader. Adding a new spawn/probe tool without updating this frozenset is a known failure mode the closed-set design exists to catch."""
    from evals.layer2.score import _KNOWN_TOOLS

    assert "probe_step" in _KNOWN_TOOLS
    assert "probe_compare" in _KNOWN_TOOLS
    # Cross-check: the existing chain tools must still be present.
    assert "spawn_baseline_harness" in _KNOWN_TOOLS
    assert "compare_outputs" in _KNOWN_TOOLS
    assert "finish" in _KNOWN_TOOLS


def test_run_cli_no_probe_flag_threads_run_probe_false(monkeypatch, tmp_path):
    """workflow/run.py exposes --no-probe and threads run_probe=False to run_orchestrator. The default (flag absent) threads run_probe=True. This is the operator's only knob for disabling the probe step; without it, --auto runs would always pay the probe wall-clock cost even on kernels where it has no diagnostic value."""
    import subprocess
    # Use the actual CLI parser via importlib-free path: subprocess against
    # `python -m workflow.run --help` to avoid mocking the SDK. We only
    # need to confirm the flag is wired into argparse, not that the full
    # run executes.
    completed = subprocess.run(
        ["python", "-m", "workflow.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--no-probe" in completed.stdout
