"""Tests for workflow.orchestrator.

Covers _hitl_pause, _execute_tool, and the run_orchestrator loop
(happy path, rejection, quit, stop-without-tool).
"""

import json

import pytest

from workflow import orchestrator
from workflow.orchestrator import (
    DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN,
    ORCHESTRATOR_SYSTEM_PROMPT,
    _execute_tool,
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
        _execute_tool("not_a_tool", {})


def test_execute_tool_dispatches_spawn_analyst(monkeypatch):
    """_execute_tool routes spawn_analyst to run_agent('analyst', kernel_source) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"variables": [], "overall_notes": "stubbed"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_analyst", {"kernel_source": "SOURCE"})

    assert result == {
        "status": "ok",
        "result": {"variables": [], "overall_notes": "stubbed"},
    }
    assert calls == [("analyst", "SOURCE")]


def test_execute_tool_dispatches_spawn_rewriter(monkeypatch):
    """_execute_tool routes spawn_rewriter to run_agent('rewriter', task_prompt) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"rewritten_code": "code", "summary_of_changes": "..."}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_rewriter", {"task_prompt": "PROMPT"})

    assert result["status"] == "ok"
    assert result["result"]["rewritten_code"] == "code"
    assert calls == [("rewriter", "PROMPT")]


# ---------- run_orchestrator: happy path ----------


def test_run_orchestrator_happy_path(monkeypatch, fake_anthropic):
    """run_orchestrator drives analyst -> rewriter -> finish end-to-end with HITL approvals."""
    # Three orchestrator turns: spawn_analyst, spawn_rewriter, finish.
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
        raise AssertionError(f"unexpected agent: {type_}")

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    # Three HITL prompts, all approved.
    _scripted_input(monkeypatch, ["y", "y", "y"])

    result = run_orchestrator("path/to/kernel.cpp", "kernel source body")

    assert result == {"rewritten_code": "FINAL", "notes": "done"}
    assert len(fake.messages.calls) == 3

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
    monkeypatch, fake_anthropic
):
    """A HITL 'n' rejects the tool call without invoking run_agent and feeds {'status':'rejected_by_user'} back to the orchestrator."""
    fake = fake_anthropic([
        # Turn 1: orchestrator proposes spawn_analyst — user rejects.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
        # Turn 2: after seeing rejection, orchestrator just calls finish.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_2",
                name="finish",
                input={"rewritten_code": "X", "notes": "gave up"},
            )],
        ),
    ])

    # If the analyst is actually invoked, this test should fail loudly.
    def fail_run_agent(type_, task):
        raise AssertionError(
            f"run_agent should not be called after rejection; got {type_!r}"
        )

    monkeypatch.setattr(orchestrator, "run_agent", fail_run_agent)
    _scripted_input(monkeypatch, ["n", "y"])  # reject, then approve finish

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
        "spawn_precision_advisor", {"kernel_source": "SOURCE"}
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
        },
    )

    assert result["status"] == "ok"
    assert captured["type"] == "verifier"
    # the task must contain all four pieces
    assert "ORIG" in captured["task"]
    assert "REW" in captured["task"]
    assert "ANALYST VERDICT" in captured["task"]
    assert "TOLERANCE" in captured["task"]
    assert "user_cli" in captured["task"]


# ---------- run_orchestrator: tolerance plumbing in the initial user message ----------


def test_run_orchestrator_user_cli_tolerance_appears_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """When a user-supplied tolerance is passed, the first user message embeds it verbatim and the orchestrator is told NOT to call spawn_precision_advisor."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])  # approve finish

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
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])

    run_orchestrator("k.cpp", "src", tolerance=None)

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "spawn_precision_advisor" in first_user
    assert "not specified" in first_user
