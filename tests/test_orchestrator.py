"""Tests for workflow.orchestrator.

Covers _hitl_pause, _execute_tool, and the run_orchestrator loop
(happy path, rejection, quit, stop-without-tool).
"""

import json

import pytest

from workflow import orchestrator
from workflow.orchestrator import (
    _execute_tool,
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
