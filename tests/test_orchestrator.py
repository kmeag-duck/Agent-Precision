"""Tests for workflow.orchestrator.

Covers _hitl_pause, _execute_tool, and the run_orchestrator loop
(happy path, rejection, quit, stop-without-tool).
"""

import json

import pytest

from workflow import orchestrator
from workflow.languages import CUDA_PROFILE, KOKKOS_PROFILE
from workflow.orchestrator import (
    DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_TOOLS,
    _execute_tool,
    _format_baseline_block,
    _format_tolerance_block,
    _hitl_pause,
    run_orchestrator,
)

from .conftest import FakeResponse, TextBlock, ToolUseBlock


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
        '"source":"advisor_unknown_defaulted"}'
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

    result = run_orchestrator("path/to/kernel.cpp", "kernel source body")

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
        '"source":"advisor_unknown_defaulted"}'
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

    result = run_orchestrator("k.cpp", "src")

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

    assert run_orchestrator("k.cpp", "src") is None


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

    assert run_orchestrator("k.cpp", "src") is None


# ---------- Orchestrator prompt: vocabulary matches the agents it routes to ----------


def test_orchestrator_prompt_names_all_three_methods_and_rework():
    """The orchestrator prompt names downcast, emulate, keep, and the rework block, so its task-prompt translation step has the right vocabulary to pass to the rewriter."""
    for token in ("downcast", "emulate", "keep", "rework"):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_names_precision_advisor_and_tolerance_kinds():
    """The orchestrator prompt names spawn_precision_advisor, sig_figs, decimal_digits, and precision_budget so the LLM knows when to call the advisor and how to thread tolerance into downstream task prompts."""
    for token in (
        "spawn_precision_advisor",
        "sig_figs",
        "decimal_digits",
        "precision_budget",
    ):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_states_advisor_unknown_fallback():
    """The orchestrator prompt documents the fallback to {sig_figs, 6, advisor_unknown_defaulted} when the precision_advisor returns kind='unknown', matching DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN."""
    assert "advisor_unknown_defaulted" in ORCHESTRATOR_SYSTEM_PROMPT
    # The constant and the prompt must agree on the fallback shape.
    assert DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN == {
        "kind": "sig_figs",
        "value": 6,
        "source": "advisor_unknown_defaulted",
    }


def test_orchestrator_prompt_forbids_finish_without_accept():
    """The orchestrator prompt explicitly forbids calling finish unless the most recent verifier returned accept; this rule lives in the prompt, not in code."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "may not call finish" in text or "may not call `finish`" in text
    assert "accept" in text


# ---------- _format_tolerance_block ----------


def test_format_tolerance_block_none_tells_orchestrator_to_call_advisor():
    """With tolerance=None, the rendered block instructs the orchestrator to call spawn_precision_advisor first and documents both the advisor-returns-tolerance and advisor-unknown-fallback paths."""
    block = _format_tolerance_block(None)
    assert "not specified" in block
    assert "spawn_precision_advisor" in block
    assert "advisor_unknown_defaulted" in block


def test_format_tolerance_block_user_cli_forbids_advisor_call():
    """With a user-supplied tolerance, the rendered block contains the kind/value/source verbatim and tells the orchestrator NOT to call spawn_precision_advisor."""
    block = _format_tolerance_block(
        {"kind": "sig_figs", "value": 7, "source": "user_cli"}
    )
    assert "sig_figs" in block
    assert "7" in block
    assert "user_cli" in block
    assert "do NOT call" in block or "do not call" in block.lower()


# ---------- _execute_tool: spawn_precision_advisor + spawn_verifier(tolerance_json) ----------


def test_execute_tool_dispatches_spawn_precision_advisor(monkeypatch):
    """_execute_tool routes spawn_precision_advisor to run_agent('precision_advisor', kernel_source) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "kind": "sig_figs",
            "value": 6,
            "rationale": "stubbed",
            "confidence": "medium",
            "alternative": "",
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_precision_advisor", {"kernel_source": "SOURCE"}, KOKKOS_PROFILE
    )

    assert result["status"] == "ok"
    assert result["result"]["kind"] == "sig_figs"
    assert calls == [("precision_advisor", "SOURCE")]


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
    """When a user-supplied tolerance is passed, the first user message embeds it verbatim and the orchestrator is told NOT to call spawn_precision_advisor."""
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
    # advisor must NOT be invited when tolerance is user-supplied
    assert "do NOT call" in first_user or "do not call" in first_user.lower()


def test_run_orchestrator_no_tolerance_invites_advisor_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """When tolerance=None, the first user message instructs the orchestrator to call spawn_precision_advisor first."""
    # First-user-message-only test: short-circuit on turn 1 with a
    # text+end_turn response so the loop exits before engaging the gate.
    fake = fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="(test stop)")],
            stop_reason="end_turn",
        ),
    ])
    _scripted_input(monkeypatch, [])

    run_orchestrator("k.cpp", "src", tolerance=None)

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "spawn_precision_advisor" in first_user
    assert "not specified" in first_user


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

    run_orchestrator("path/to/nbody_force.cpp", "src")

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
        "path/to/vector_add.cpp", "src", kernel_name="vector_add"
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "TARGET KERNEL: vector_add" in first_user
    assert "KERNEL STEM: vector_add" in first_user


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

    run_orchestrator("path/to/vector_add.cu", "src")

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
    # Must mention the retry-bias suggestion (spawn_analyst, not
    # spawn_rewriter, on comparator error) so the model has a clear
    # next move when the gate blocks finish.
    assert "spawn_analyst" in text


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

    result = run_orchestrator("path/to/kstem.cpp", "src")
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

    result = run_orchestrator("path/to/kstem.cpp", "src")
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

    result = run_orchestrator("path/to/kstem.cpp", "src")
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

    result = run_orchestrator("path/to/vector_add.cu", "src")
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
